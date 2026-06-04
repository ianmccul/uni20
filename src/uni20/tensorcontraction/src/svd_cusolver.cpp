#include <uni20/tensorcontraction/svd.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#if UNI20_TENSORCONTRACTION_HAS_CUSOLVER

#include "Arranger.hpp"
#include "Swapper.hpp"
#include "svd_split_kernels.hpp"

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction::detail
{

namespace
{

template <typename T, auto Destroy> struct CusolverResourceDeleter
{
    void operator()(T resource) const noexcept
    {
      if (resource != nullptr)
      {
        Destroy(resource);
      }
    }
};

using CusolverHandle = std::unique_ptr<std::remove_pointer_t<cusolverDnHandle_t>,
                                       CusolverResourceDeleter<cusolverDnHandle_t, cusolverDnDestroy>>;
using CudaStream =
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CusolverResourceDeleter<cudaStream_t, cudaStreamDestroy>>;

struct CusolverThreadContext
{
    CusolverHandle handle;
    CudaStream stream;
};

struct DeviceBuffer
{
    void* ptr = nullptr;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes)
    {
      if (bytes != 0)
      {
        this->check(cudaMalloc(&ptr, bytes), "cudaMalloc");
      }
    }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        ptr = other.ptr;
        other.ptr = nullptr;
      }
      return *this;
    }
    ~DeviceBuffer() { this->release(); }

    template <typename T> T* as() const noexcept { return static_cast<T*>(ptr); }

  private:
    static void check(cudaError_t status, char const* name)
    {
      if (status != cudaSuccess)
      {
        throw std::runtime_error(std::string(name) + " failed: " + cudaGetErrorString(status));
      }
    }

    void release() noexcept
    {
      if (ptr != nullptr)
      {
        cudaFree(ptr);
        ptr = nullptr;
      }
    }
};

void check_cuda(cudaError_t status, char const* name)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string(name) + " failed: " + cudaGetErrorString(status));
  }
}

char const* cusolver_status_name(cusolverStatus_t status)
{
  switch (status)
  {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "CUSOLVER_STATUS_INVALID_LICENSE";
    case CUSOLVER_STATUS_INVALID_WORKSPACE:
      return "CUSOLVER_STATUS_INVALID_WORKSPACE";
    default:
      return "CUSOLVER_STATUS_UNKNOWN";
  }
}

void check_cusolver(cusolverStatus_t status, char const* name)
{
  if (status != CUSOLVER_STATUS_SUCCESS)
  {
    throw std::runtime_error(std::string(name) + " failed: " + cusolver_status_name(status));
  }
}

bool cusolver_svd_disabled()
{
  auto const* value = std::getenv("UNI20_TENSORCONTRACTION_CUSOLVER_SVD");
  if (value == nullptr)
  {
    return false;
  }
  auto const text = std::string(value);
  return text.empty() || text == "0" || text == "OFF" || text == "off" || text == "false" || text == "FALSE";
}

bool cuda_device_available()
{
  int count = 0;
  auto const status = cudaGetDeviceCount(&count);
  if (status == cudaSuccess)
  {
    return count > 0;
  }

  // Leave the CUDA runtime usable for a later fallback path in this process.
  cudaGetLastError();
  return false;
}

CusolverThreadContext& cusolver_context_for_current_device()
{
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  thread_local std::unordered_map<int, CusolverThreadContext> contexts;
  if (auto found = contexts.find(device); found != contexts.end())
  {
    return found->second;
  }

  cusolverDnHandle_t raw_handle = nullptr;
  check_cusolver(cusolverDnCreate(&raw_handle), "cusolverDnCreate");
  cudaStream_t raw_stream = nullptr;
  check_cuda(cudaStreamCreate(&raw_stream), "cudaStreamCreate");
  CusolverThreadContext context{.handle = CusolverHandle(raw_handle), .stream = CudaStream(raw_stream)};
  check_cusolver(cusolverDnSetStream(context.handle.get(), context.stream.get()), "cusolverDnSetStream");
  auto [inserted, _] = contexts.emplace(device, std::move(context));
  return inserted->second;
}

int checked_cuda_int(std::size_t value, char const* name)
{
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    throw std::length_error(std::string(name) + " exceeds cuSOLVER integer range");
  }
  return static_cast<int>(value);
}

struct SvdSolverShape
{
    std::size_t rows = 0;
    std::size_t cols = 0;
    bool transposed = false;
    std::size_t solver_rows = 0;
    std::size_t solver_cols = 0;
    std::size_t minmn = 0;
    int m = 0;
    int n = 0;
    int lda = 0;
    int ldu = 0;
    int ldvt = 0;
};

auto make_svd_solver_shape(std::size_t rows, std::size_t cols) -> SvdSolverShape
{
  SvdSolverShape shape;
  shape.rows = rows;
  shape.cols = cols;
  shape.transposed = rows < cols;
  shape.solver_rows = shape.transposed ? cols : rows;
  shape.solver_cols = shape.transposed ? rows : cols;
  shape.minmn = std::min(rows, cols);
  shape.m = checked_cuda_int(shape.solver_rows, "SVD row count");
  shape.n = checked_cuda_int(shape.solver_cols, "SVD column count");
  shape.lda = std::max(1, shape.m);
  shape.ldu = std::max(1, shape.m);
  shape.ldvt = std::max(1, shape.n);
  return shape;
}

auto make_column_major_copy(MatrixFamily const& matrix, bool transpose) -> std::vector<double>
{
  auto const block = matrix.block(0);
  auto const input = matrix.values(0);
  std::vector<double> result(block.rows * block.cols);
  if (!transpose)
  {
    for (std::size_t row = 0; row < block.rows; ++row)
    {
      for (std::size_t col = 0; col < block.cols; ++col)
      {
        result[col * block.rows + row] = input[row * block.cols + col];
      }
    }
    return result;
  }

  for (std::size_t row = 0; row < block.rows; ++row)
  {
    for (std::size_t col = 0; col < block.cols; ++col)
    {
      result[row * block.cols + col] = input[row * block.cols + col];
    }
  }
  return result;
}

auto copy_cusolver_result(std::span<double const> singular_values, std::span<double const> u,
                          std::span<double const> vt, std::size_t rows, std::size_t cols, bool transposed,
                          SvdOptions options) -> SingleBlockSvd
{
  auto const minmn = std::min(rows, cols);
  auto const full_rank = singular_rank(singular_values);
  auto const kept = kept_singular_count(singular_values, options);
  auto const discarded_weight = discarded_singular_weight(singular_values, kept);

  MatrixFamily result_u(std::vector<MatrixFamily::Block>{MatrixFamily::Block{rows, kept}});
  MatrixFamily result_vt(std::vector<MatrixFamily::Block>{MatrixFamily::Block{kept, cols}});
  auto result_u_values = result_u.values(0);
  auto result_vt_values = result_vt.values(0);

  std::vector<double> kept_singular_values(singular_values.begin(), singular_values.begin() + kept);
  if (!transposed)
  {
    for (std::size_t row = 0; row < rows; ++row)
    {
      for (std::size_t rank = 0; rank < kept; ++rank)
      {
        result_u_values[row * kept + rank] = u[rank * rows + row];
      }
    }
    for (std::size_t rank = 0; rank < kept; ++rank)
    {
      for (std::size_t col = 0; col < cols; ++col)
      {
        result_vt_values[rank * cols + col] = vt[col * minmn + rank];
      }
    }
  }
  else
  {
    for (std::size_t row = 0; row < rows; ++row)
    {
      for (std::size_t rank = 0; rank < kept; ++rank)
      {
        result_u_values[row * kept + rank] = vt[row * minmn + rank];
      }
    }
    for (std::size_t rank = 0; rank < kept; ++rank)
    {
      for (std::size_t col = 0; col < cols; ++col)
      {
        result_vt_values[rank * cols + col] = u[rank * cols + col];
      }
    }
  }

  return SingleBlockSvd{.u = std::move(result_u),
                        .singular_values = std::move(kept_singular_values),
                        .vt = std::move(result_vt),
                        .discarded_weight = discarded_weight,
                        .full_rank = full_rank};
}

auto make_block_major_matrix_family(std::span<double const> values, std::size_t block_count, std::size_t rows,
                                    std::size_t cols) -> MatrixFamily
{
  std::vector<MatrixFamily::Block> blocks(block_count, MatrixFamily::Block{rows, cols});
  MatrixFamily family(blocks);
  auto const block_size = rows * cols;
  for (std::size_t block = 0; block < block_count; ++block)
  {
    family.assign(block, values.subspan(block * block_size, block_size));
  }
  return family;
}

auto run_split_cusolver_from_device_input(DeviceBuffer& device_a, SvdSolverShape const& shape,
                                          SingleBlockSvdSplitLayout layout, SvdAbsorbSingularValues absorb,
                                          SvdOptions options, cusolverDnHandle_t handle,
                                          cudaStream_t stream) -> std::optional<SingleBlockSvdSplit>
{
  std::vector<double> host_singular_values(shape.minmn);
  int lwork = 0;
  check_cusolver(cusolverDnDgesvd_bufferSize(handle, shape.m, shape.n, &lwork), "cusolverDnDgesvd_bufferSize");

  DeviceBuffer device_singular_values(host_singular_values.size() * sizeof(double));
  DeviceBuffer device_u(shape.solver_rows * shape.minmn * sizeof(double));
  DeviceBuffer device_vt(shape.minmn * shape.solver_cols * sizeof(double));
  DeviceBuffer device_work(static_cast<std::size_t>(std::max(1, lwork)) * sizeof(double));
  DeviceBuffer device_info(sizeof(int));

  signed char jobu = 'S';
  signed char jobvt = 'S';
  check_cusolver(cusolverDnDgesvd(handle, jobu, jobvt, shape.m, shape.n, device_a.as<double>(), shape.lda,
                                  device_singular_values.as<double>(), device_u.as<double>(), shape.ldu,
                                  device_vt.as<double>(), shape.ldvt, device_work.as<double>(), lwork, nullptr,
                                  device_info.as<int>()),
                 "cusolverDnDgesvd");

  int info = 0;
  check_cuda(cudaMemcpyAsync(host_singular_values.data(), device_singular_values.as<double>(),
                             host_singular_values.size() * sizeof(double), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync singular values to host");
  check_cuda(cudaMemcpyAsync(&info, device_info.as<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync SVD info to host");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  if (info < 0)
  {
    throw std::invalid_argument("cuSOLVER dgesvd rejected argument " + std::to_string(-info));
  }
  if (info > 0)
  {
    return std::nullopt;
  }

  auto const full_rank = singular_rank(host_singular_values);
  auto const kept = kept_singular_count(host_singular_values, options);
  auto const discarded_weight = discarded_singular_weight(host_singular_values, kept);
  auto const left_size = layout.left_physical_dim * layout.left_bond_dim * kept;
  auto const right_size = layout.right_physical_dim * kept * layout.right_bond_dim;
  std::vector<double> host_left(left_size);
  std::vector<double> host_right(right_size);

  if (kept != 0)
  {
    DeviceBuffer device_left(host_left.size() * sizeof(double));
    DeviceBuffer device_right(host_right.size() * sizeof(double));
    launch_svd_split_kernels(device_u.as<double>(), device_singular_values.as<double>(), device_vt.as<double>(),
                             device_left.as<double>(), device_right.as<double>(), layout.left_bond_dim,
                             layout.left_physical_dim, layout.right_physical_dim, layout.right_bond_dim, kept,
                             shape.minmn, shape.transposed, absorb == SvdAbsorbSingularValues::Left, stream);
    check_cuda(cudaGetLastError(), "launch_svd_split_kernels");
    check_cuda(cudaMemcpyAsync(host_left.data(), device_left.as<double>(), host_left.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync split left blocks to host");
    check_cuda(cudaMemcpyAsync(host_right.data(), device_right.as<double>(), host_right.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync split right blocks to host");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }

  std::vector<double> kept_singular_values(host_singular_values.begin(), host_singular_values.begin() + kept);
  auto left = make_block_major_matrix_family(host_left, layout.left_physical_dim, layout.left_bond_dim, kept);
  auto right = make_block_major_matrix_family(host_right, layout.right_physical_dim, kept, layout.right_bond_dim);
  return SingleBlockSvdSplit{.left = std::move(left),
                             .right = std::move(right),
                             .spectrum = SvdSpectrum{.singular_values = std::move(kept_singular_values),
                                                     .discarded_weight = discarded_weight,
                                                     .full_rank = full_rank}};
}

struct SectorSingularValue
{
    std::size_t sector = 0;
    std::size_t rank = 0;
    double value = 0.0;
};

struct ResidentSectorSvd
{
    SvdSolverShape shape;
    DeviceBuffer device_a;
    DeviceBuffer device_singular_values;
    DeviceBuffer device_u;
    DeviceBuffer device_vt;
    DeviceBuffer device_work;
    DeviceBuffer device_info;
    std::vector<double> singular_values;
    int info = 0;
};

struct ResidentSourceBuffer
{
    int device_id = -1;
    std::shared_ptr<tensor::GpuBuffer> buffer;
};

void validate_resident_block_sparse_svd_inputs(MatrixFamily const& vector, ResidentBlockSparseSvdPlan const& plan,
                                               SvdOptions options)
{
  if (options.max_rank == 0)
  {
    throw std::invalid_argument("resident block-sparse SVD requires a positive max_rank");
  }
  if (options.cutoff < 0.0 || std::isnan(options.cutoff))
  {
    throw std::invalid_argument("resident block-sparse SVD requires a finite non-negative cutoff");
  }
  if (plan.sectors.empty())
  {
    throw std::invalid_argument("resident block-sparse SVD requires at least one sector");
  }

  for (std::size_t sector_index = 0; sector_index < plan.sectors.size(); ++sector_index)
  {
    auto const& sector = plan.sectors[sector_index];
    if (sector.row_dim == 0 || sector.col_dim == 0)
    {
      throw std::invalid_argument("resident block-sparse SVD sector dimensions must be non-empty");
    }
    if (sector.source_terms.empty())
    {
      throw std::invalid_argument("resident block-sparse SVD sector has no source terms");
    }
    if (sector.left_terms.empty() || sector.right_terms.empty())
    {
      throw std::invalid_argument("resident block-sparse SVD sector has no output terms");
    }

    for (auto const& term : sector.source_terms)
    {
      if (term.source_block >= vector.size())
      {
        throw std::invalid_argument("resident block-sparse SVD source term block index is out of range");
      }
      auto const block = vector.block(term.source_block);
      if (term.row_offset > sector.row_dim || block.rows > sector.row_dim - term.row_offset ||
          term.col_offset > sector.col_dim || block.cols > sector.col_dim - term.col_offset)
      {
        throw std::invalid_argument("resident block-sparse SVD source term does not fit in its sector");
      }
    }

    for (auto const& term : sector.left_terms)
    {
      if (term.extent == 0 || term.offset > sector.row_dim || term.extent > sector.row_dim - term.offset)
      {
        throw std::invalid_argument("resident block-sparse SVD left output term does not fit in its sector");
      }
    }
    for (auto const& term : sector.right_terms)
    {
      if (term.extent == 0 || term.offset > sector.col_dim || term.extent > sector.col_dim - term.offset)
      {
        throw std::invalid_argument("resident block-sparse SVD right output term does not fit in its sector");
      }
    }
  }
}

auto select_resident_sector_ranks(std::span<std::vector<double> const> singular_values,
                                  SvdOptions options) -> std::pair<std::vector<std::size_t>, SvdSpectrum>
{
  std::vector<SectorSingularValue> candidates;
  std::vector<SectorSingularValue> positive_values;
  std::size_t full_rank = 0;
  for (std::size_t sector = 0; sector < singular_values.size(); ++sector)
  {
    for (std::size_t rank = 0; rank < singular_values[sector].size(); ++rank)
    {
      double const value = singular_values[sector][rank];
      if (value > 0.0)
      {
        ++full_rank;
        positive_values.push_back(SectorSingularValue{.sector = sector, .rank = rank, .value = value});
      }
      if (value > options.cutoff)
      {
        candidates.push_back(SectorSingularValue{.sector = sector, .rank = rank, .value = value});
      }
    }
  }
  if (candidates.empty() && !positive_values.empty())
  {
    candidates.push_back(*std::max_element(positive_values.begin(), positive_values.end(),
                                           [](auto const& lhs, auto const& rhs) { return lhs.value < rhs.value; }));
  }
  std::sort(candidates.begin(), candidates.end(), [](auto const& lhs, auto const& rhs) {
    if (lhs.value != rhs.value)
    {
      return lhs.value > rhs.value;
    }
    if (lhs.sector != rhs.sector)
    {
      return lhs.sector < rhs.sector;
    }
    return lhs.rank < rhs.rank;
  });
  if (candidates.size() > options.max_rank)
  {
    candidates.resize(options.max_rank);
  }

  std::vector<std::size_t> ranks(singular_values.size(), 0);
  for (auto const& singular : candidates)
  {
    ranks[singular.sector] = std::max(ranks[singular.sector], singular.rank + 1);
  }

  SvdSpectrum spectrum;
  spectrum.full_rank = full_rank;
  std::vector<double> kept_values;
  for (std::size_t sector = 0; sector < singular_values.size(); ++sector)
  {
    for (std::size_t rank = 0; rank < ranks[sector]; ++rank)
    {
      kept_values.push_back(singular_values[sector][rank]);
    }
    for (std::size_t rank = ranks[sector]; rank < singular_values[sector].size(); ++rank)
    {
      double const value = singular_values[sector][rank];
      spectrum.discarded_weight += value * value;
    }
  }
  std::sort(kept_values.begin(), kept_values.end(), std::greater<>{});
  spectrum.singular_values = std::move(kept_values);
  return {std::move(ranks), std::move(spectrum)};
}

auto make_resident_sector_svd(DeviceBuffer device_a, SvdSolverShape shape, cusolverDnHandle_t handle,
                              cudaStream_t stream) -> ResidentSectorSvd
{
  int lwork = 0;
  check_cusolver(cusolverDnDgesvd_bufferSize(handle, shape.m, shape.n, &lwork), "cusolverDnDgesvd_bufferSize");

  ResidentSectorSvd result;
  result.shape = shape;
  result.device_a = std::move(device_a);
  result.singular_values.resize(shape.minmn);
  result.device_singular_values = DeviceBuffer(result.singular_values.size() * sizeof(double));
  result.device_u = DeviceBuffer(shape.solver_rows * shape.minmn * sizeof(double));
  result.device_vt = DeviceBuffer(shape.minmn * shape.solver_cols * sizeof(double));
  result.device_work = DeviceBuffer(static_cast<std::size_t>(std::max(1, lwork)) * sizeof(double));
  result.device_info = DeviceBuffer(sizeof(int));

  signed char jobu = 'S';
  signed char jobvt = 'S';
  check_cusolver(cusolverDnDgesvd(handle, jobu, jobvt, shape.m, shape.n, result.device_a.as<double>(), shape.lda,
                                  result.device_singular_values.as<double>(), result.device_u.as<double>(), shape.ldu,
                                  result.device_vt.as<double>(), shape.ldvt, result.device_work.as<double>(), lwork,
                                  nullptr, result.device_info.as<int>()),
                 "cusolverDnDgesvd");

  check_cuda(cudaMemcpyAsync(result.singular_values.data(), result.device_singular_values.as<double>(),
                             result.singular_values.size() * sizeof(double), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync resident block-sparse SVD singular values to host");
  check_cuda(cudaMemcpyAsync(&result.info, result.device_info.as<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync resident block-sparse SVD info to host");
  return result;
}

auto make_resident_output_blocks(ResidentBlockSparseSvdPlan const& plan, std::span<std::size_t const> ranks)
    -> std::pair<std::vector<MatrixFamily::Block>, std::vector<MatrixFamily::Block>>
{
  std::vector<MatrixFamily::Block> left_blocks;
  std::vector<MatrixFamily::Block> right_blocks;
  for (std::size_t sector_index = 0; sector_index < plan.sectors.size(); ++sector_index)
  {
    auto const rank = ranks[sector_index];
    if (rank == 0)
    {
      continue;
    }
    auto const& sector = plan.sectors[sector_index];
    for (auto const& term : sector.left_terms)
    {
      left_blocks.push_back(MatrixFamily::Block{.rows = term.extent, .cols = rank});
    }
    for (auto const& term : sector.right_terms)
    {
      right_blocks.push_back(MatrixFamily::Block{.rows = rank, .cols = term.extent});
    }
  }
  return {std::move(left_blocks), std::move(right_blocks)};
}

void register_resident_output(MatrixFamily& family, int target_device, tensor::Swapper& swapper)
{
  for (auto const& matrix : raw_matrices(family))
  {
    swapper.registerGpuAllocation(matrix, target_device);
  }
}

auto resident_output_buffers(MatrixFamily& family, int target_device,
                             tensor::Swapper& swapper) -> std::vector<std::shared_ptr<tensor::GpuBuffer>>
{
  std::vector<std::shared_ptr<tensor::GpuBuffer>> result;
  result.reserve(family.size());
  for (auto const& matrix : raw_matrices(family))
  {
    auto [device_id, buffer] = swapper.getPreStoreBufferOrNone(matrix);
    if (device_id != target_device || buffer == nullptr)
    {
      throw std::logic_error("resident block-sparse SVD failed to allocate an output block on the target device");
    }
    result.push_back(std::move(buffer));
  }
  return result;
}

} // namespace

std::optional<SingleBlockSvd> single_block_svd_cusolver(MatrixFamily const& matrix, SvdOptions options)
{
  validate_single_block_svd_inputs(matrix, options);
  if (cusolver_svd_disabled() || !cuda_device_available())
  {
    return std::nullopt;
  }

  auto const block = matrix.block(0);
  auto const rows = block.rows;
  auto const cols = block.cols;
  auto const shape = make_svd_solver_shape(rows, cols);
  auto const minmn = std::min(rows, cols);

  auto& context = cusolver_context_for_current_device();
  auto const handle = context.handle.get();
  auto const stream = context.stream.get();

  auto host_a = make_column_major_copy(matrix, shape.transposed);
  std::vector<double> host_singular_values(minmn);
  std::vector<double> host_u(shape.solver_rows * minmn);
  std::vector<double> host_vt(minmn * shape.solver_cols);
  int lwork = 0;
  check_cusolver(cusolverDnDgesvd_bufferSize(handle, shape.m, shape.n, &lwork), "cusolverDnDgesvd_bufferSize");

  DeviceBuffer device_a(host_a.size() * sizeof(double));
  DeviceBuffer device_singular_values(host_singular_values.size() * sizeof(double));
  DeviceBuffer device_u(host_u.size() * sizeof(double));
  DeviceBuffer device_vt(host_vt.size() * sizeof(double));
  DeviceBuffer device_work(static_cast<std::size_t>(std::max(1, lwork)) * sizeof(double));
  DeviceBuffer device_info(sizeof(int));

  check_cuda(cudaMemcpyAsync(device_a.as<double>(), host_a.data(), host_a.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync host matrix to device");

  signed char jobu = 'S';
  signed char jobvt = 'S';
  check_cusolver(cusolverDnDgesvd(handle, jobu, jobvt, shape.m, shape.n, device_a.as<double>(), shape.lda,
                                  device_singular_values.as<double>(), device_u.as<double>(), shape.ldu,
                                  device_vt.as<double>(), shape.ldvt, device_work.as<double>(), lwork, nullptr,
                                  device_info.as<int>()),
                 "cusolverDnDgesvd");

  int info = 0;
  check_cuda(cudaMemcpyAsync(host_singular_values.data(), device_singular_values.as<double>(),
                             host_singular_values.size() * sizeof(double), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync singular values to host");
  check_cuda(cudaMemcpyAsync(host_u.data(), device_u.as<double>(), host_u.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync U to host");
  check_cuda(cudaMemcpyAsync(host_vt.data(), device_vt.as<double>(), host_vt.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync Vt to host");
  check_cuda(cudaMemcpyAsync(&info, device_info.as<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync SVD info to host");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  if (info < 0)
  {
    throw std::invalid_argument("cuSOLVER dgesvd rejected argument " + std::to_string(-info));
  }
  if (info > 0)
  {
    return std::nullopt;
  }

  return copy_cusolver_result(host_singular_values, host_u, host_vt, rows, cols, shape.transposed, options);
}

std::optional<SingleBlockSvdSplit> single_block_svd_split_cusolver(MatrixFamily const& matrix,
                                                                   SingleBlockSvdSplitLayout layout,
                                                                   SvdAbsorbSingularValues absorb, SvdOptions options)
{
  validate_single_block_svd_split_inputs(matrix, layout, options);
  if (cusolver_svd_disabled() || !cuda_device_available())
  {
    return std::nullopt;
  }

  auto const block = matrix.block(0);
  auto const rows = block.rows;
  auto const cols = block.cols;
  auto const shape = make_svd_solver_shape(rows, cols);

  auto& context = cusolver_context_for_current_device();
  auto const handle = context.handle.get();
  auto const stream = context.stream.get();

  auto host_a = make_column_major_copy(matrix, shape.transposed);
  DeviceBuffer device_a(host_a.size() * sizeof(double));

  check_cuda(cudaMemcpyAsync(device_a.as<double>(), host_a.data(), host_a.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync host matrix to device");

  return run_split_cusolver_from_device_input(device_a, shape, layout, absorb, options, handle, stream);
}

std::optional<SingleBlockSvdSplit> single_block_svd_split_resident_cusolver(MatrixFamily const& vector,
                                                                            SingleBlockSvdSplitLayout layout,
                                                                            SvdAbsorbSingularValues absorb,
                                                                            SvdOptions options,
                                                                            VectorAlgebraEngine& algebra)
{
  validate_resident_svd_split_inputs(vector, layout, options);
  if (cusolver_svd_disabled() || algebra.uses_host_backend() || !cuda_device_available())
  {
    return std::nullopt;
  }

  auto& arranger = algebra.resident_arranger();
  auto& swapper = arranger.residentSwapper();
  auto const& matrices = raw_matrices(vector);
  std::vector<std::pair<int, std::shared_ptr<tensor::GpuBuffer>>> sources;
  sources.reserve(matrices.size());
  for (auto const& matrix : matrices)
  {
    auto [device_id, buffer] = swapper.getPreStoreBufferOrNone(matrix);
    if (buffer == nullptr || !buffer->contentValid())
    {
      return std::nullopt;
    }
    sources.emplace_back(device_id, std::move(buffer));
  }

  auto const target_device = sources.front().first;
  check_cuda(cudaSetDevice(target_device), "cudaSetDevice");
  auto& context = cusolver_context_for_current_device();
  auto const handle = context.handle.get();
  auto const stream = context.stream.get();

  auto const rows = layout.left_bond_dim * layout.left_physical_dim;
  auto const cols = layout.right_physical_dim * layout.right_bond_dim;
  auto const shape = make_svd_solver_shape(rows, cols);
  DeviceBuffer device_a(rows * cols * sizeof(double));
  std::vector<DeviceBuffer> staging_buffers;
  staging_buffers.reserve(sources.size());

  for (std::size_t block = 0; block < matrices.size(); ++block)
  {
    auto const source_device = sources[block].first;
    auto const& source_buffer = sources[block].second;
    source_buffer->waitBeforeRead(stream);
    double const* source_ptr = source_buffer->getPtr();
    if (source_device != target_device)
    {
      staging_buffers.emplace_back(matrices[block].sizeInByte());
      auto* staging = staging_buffers.back().as<double>();
      check_cuda(cudaMemcpyPeerAsync(staging, target_device, source_buffer->getPtr(), source_device,
                                     matrices[block].sizeInByte(), stream),
                 "cudaMemcpyPeerAsync resident SVD block staging");
      source_ptr = staging;
    }

    auto const left_physical = block / layout.right_physical_dim;
    auto const right_physical = block % layout.right_physical_dim;
    launch_pack_svd_input_block_kernel(source_ptr, device_a.as<double>(), layout.left_bond_dim, layout.right_bond_dim,
                                       layout.left_physical_dim, layout.right_physical_dim, left_physical,
                                       right_physical, shape.transposed, stream);
  }
  check_cuda(cudaGetLastError(), "launch_pack_svd_input_block_kernel");

  return run_split_cusolver_from_device_input(device_a, shape, layout, absorb, options, handle, stream);
}

std::optional<ResidentBlockSparseSvdSplit>
block_sparse_svd_split_resident_cusolver(MatrixFamily const& vector, ResidentBlockSparseSvdPlan const& plan,
                                         SvdAbsorbSingularValues absorb, SvdOptions options,
                                         VectorAlgebraEngine& algebra)
{
  validate_resident_block_sparse_svd_inputs(vector, plan, options);
  if (cusolver_svd_disabled() || algebra.uses_host_backend() || !cuda_device_available())
  {
    return std::nullopt;
  }

  auto& arranger = algebra.resident_arranger();
  arranger.ensureMemoryPoolsInitialized();
  auto& swapper = arranger.residentSwapper();
  auto const& matrices = raw_matrices(vector);
  std::vector<ResidentSourceBuffer> sources(matrices.size());
  int target_device = -1;
  for (std::size_t block = 0; block < matrices.size(); ++block)
  {
    auto [device_id, buffer] = swapper.getPreStoreBufferOrNone(matrices[block]);
    if (buffer == nullptr || !buffer->contentValid())
    {
      return std::nullopt;
    }
    if (target_device < 0)
    {
      target_device = device_id;
    }
    sources[block] = ResidentSourceBuffer{.device_id = device_id, .buffer = std::move(buffer)};
  }
  if (target_device < 0)
  {
    throw std::invalid_argument("resident block-sparse SVD requires at least one resident source block");
  }

  check_cuda(cudaSetDevice(target_device), "cudaSetDevice");
  auto& context = cusolver_context_for_current_device();
  auto const handle = context.handle.get();
  auto const stream = context.stream.get();

  std::vector<DeviceBuffer> staging_buffers;
  std::vector<ResidentSectorSvd> sector_svds;
  std::vector<std::shared_ptr<tensor::GpuBuffer>> read_buffers;
  staging_buffers.reserve(vector.size());
  sector_svds.reserve(plan.sectors.size());
  read_buffers.reserve(vector.size());

  for (auto const& sector : plan.sectors)
  {
    auto const shape = make_svd_solver_shape(sector.row_dim, sector.col_dim);
    DeviceBuffer device_a(sector.row_dim * sector.col_dim * sizeof(double));
    check_cuda(cudaMemsetAsync(device_a.as<double>(), 0, sector.row_dim * sector.col_dim * sizeof(double), stream),
               "cudaMemsetAsync resident block-sparse SVD sector");

    for (auto const& term : sector.source_terms)
    {
      auto const& source = sources[term.source_block];
      auto const matrix = matrices[term.source_block];
      source.buffer->waitBeforeRead(stream);
      bool seen_source_buffer = false;
      for (auto const& buffer : read_buffers)
      {
        seen_source_buffer = seen_source_buffer || buffer->getId() == source.buffer->getId();
      }
      if (!seen_source_buffer)
      {
        read_buffers.push_back(source.buffer);
      }

      double const* source_ptr = source.buffer->getPtr();
      if (source.device_id != target_device)
      {
        staging_buffers.emplace_back(matrix.sizeInByte());
        auto* staging = staging_buffers.back().as<double>();
        check_cuda(cudaMemcpyPeerAsync(staging, target_device, source.buffer->getPtr(), source.device_id,
                                       matrix.sizeInByte(), stream),
                   "cudaMemcpyPeerAsync resident block-sparse SVD source staging");
        source_ptr = staging;
      }

      auto const block = vector.block(term.source_block);
      launch_pack_svd_input_subblock_kernel(source_ptr, device_a.as<double>(), block.rows, block.cols, sector.row_dim,
                                            sector.col_dim, term.row_offset, term.col_offset, shape.transposed, stream);
    }
    check_cuda(cudaGetLastError(), "launch_pack_svd_input_subblock_kernel");
    sector_svds.push_back(make_resident_sector_svd(std::move(device_a), shape, handle, stream));
  }

  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize resident block-sparse SVD singular values");
  std::vector<std::vector<double>> singular_values;
  singular_values.reserve(sector_svds.size());
  for (auto& svd : sector_svds)
  {
    if (svd.info < 0)
    {
      throw std::invalid_argument("cuSOLVER dgesvd rejected argument " + std::to_string(-svd.info));
    }
    if (svd.info > 0)
    {
      return std::nullopt;
    }
    singular_values.push_back(svd.singular_values);
  }

  auto [ranks, spectrum] = select_resident_sector_ranks(singular_values, options);
  auto [left_blocks, right_blocks] = make_resident_output_blocks(plan, ranks);
  MatrixFamily left(left_blocks);
  MatrixFamily right(right_blocks);
  register_resident_output(left, target_device, swapper);
  register_resident_output(right, target_device, swapper);
  auto left_buffers = resident_output_buffers(left, target_device, swapper);
  auto right_buffers = resident_output_buffers(right, target_device, swapper);

  std::size_t left_block = 0;
  std::size_t right_block = 0;
  bool const absorb_left = absorb == SvdAbsorbSingularValues::Left;
  for (std::size_t sector_index = 0; sector_index < plan.sectors.size(); ++sector_index)
  {
    auto const rank = ranks[sector_index];
    if (rank == 0)
    {
      continue;
    }
    auto const& sector = plan.sectors[sector_index];
    auto const& svd = sector_svds[sector_index];
    for (auto const& term : sector.left_terms)
    {
      auto& buffer = left_buffers.at(left_block++);
      buffer->waitBeforeWrite(stream);
      launch_scatter_svd_left_subblock_kernel(svd.device_u.as<double>(), svd.device_singular_values.as<double>(),
                                              svd.device_vt.as<double>(), buffer->getPtr(), term.extent, sector.row_dim,
                                              svd.shape.minmn, term.offset, rank, svd.shape.transposed, absorb_left,
                                              stream);
    }
    for (auto const& term : sector.right_terms)
    {
      auto& buffer = right_buffers.at(right_block++);
      buffer->waitBeforeWrite(stream);
      launch_scatter_svd_right_subblock_kernel(svd.device_u.as<double>(), svd.device_singular_values.as<double>(),
                                               svd.device_vt.as<double>(), buffer->getPtr(), term.extent,
                                               sector.col_dim, svd.shape.minmn, term.offset, rank, svd.shape.transposed,
                                               absorb_left, stream);
    }
  }
  check_cuda(cudaGetLastError(), "resident block-sparse SVD scatter kernels");

  auto completion = swapper.deviceContext(target_device).recordCompletionEvent(stream);
  for (auto const& buffer : read_buffers)
  {
    buffer->publishRead(completion);
  }
  for (auto const& buffer : left_buffers)
  {
    buffer->publishWrite(completion);
  }
  for (auto const& buffer : right_buffers)
  {
    buffer->publishWrite(completion);
  }
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize resident block-sparse SVD scatter");

  return ResidentBlockSparseSvdSplit{.left = std::move(left),
                                     .right = std::move(right),
                                     .spectrum = std::move(spectrum),
                                     .sector_ranks = std::move(ranks)};
}

} // namespace uni20::tensorcontraction::detail

#endif
