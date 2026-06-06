#include <uni20/tensorcontraction/vector_algebra.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"
#include "Utils.h"
#include "vector_algebra_kernels.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction
{

struct VectorAlgebraEngine::Impl
{
    std::unique_ptr<tensor::Swapper> swapper;
    std::unique_ptr<tensor::Arranger> arranger;
    bool upload_from_host = true;
    bool sync_results_to_host = false;

    Impl();

    [[nodiscard]] bool host_backend() const { return arranger == nullptr; }

    struct CoalescedGroup
    {
        int deviceId = 0;
        std::vector<std::size_t> blocks;
        std::size_t valueCount = 0;
    };

    [[nodiscard]] auto group_matrices(MatrixFamily const& x,
                                      CoalescedGroup const& group) const -> std::vector<tensor::Matrix>
    {
      auto const& matrices = raw_matrices(x);
      std::vector<tensor::Matrix> groupMatrices;
      groupMatrices.reserve(group.blocks.size());
      for (std::size_t block : group.blocks)
      {
        groupMatrices.push_back(matrices[block]);
      }
      return groupMatrices;
    }

    [[nodiscard]] auto coalesced_groups(MatrixFamily const& x) const -> std::vector<CoalescedGroup>
    {
      struct GroupBuild
      {
          int deviceId = 0;
          void* allocationBase = nullptr;
          std::size_t allocationBytes = 0;
          std::vector<std::size_t> blocks;
          std::size_t valueCount = 0;
      };

      std::vector<GroupBuild> builds;
      if (arranger == nullptr || x.empty())
      {
        return {};
      }

      auto& swapper = arranger->residentSwapper();
      auto const& matrices = raw_matrices(x);
      for (std::size_t block = 0; block < matrices.size(); ++block)
      {
        auto [deviceId, buffer] = swapper.getPreStoreBufferOrNone(matrices[block]);
        if (buffer == nullptr)
        {
          return {};
        }

        void* const allocationBase = buffer->allocationBasePtr();
        std::size_t const allocationBytes = buffer->allocationSizeInByte();
        auto* group = [&]() -> GroupBuild* {
          for (auto& candidate : builds)
          {
            if (candidate.deviceId == deviceId && candidate.allocationBase == allocationBase)
            {
              return &candidate;
            }
          }
          GroupBuild build;
          build.deviceId = deviceId;
          build.allocationBase = allocationBase;
          build.allocationBytes = allocationBytes;
          builds.push_back(std::move(build));
          return &builds.back();
        }();
        if (group->allocationBytes != allocationBytes)
        {
          return {};
        }
        group->blocks.push_back(block);
        group->valueCount += matrices[block].size();
      }

      std::vector<CoalescedGroup> groups;
      groups.reserve(builds.size());
      for (auto const& build : builds)
      {
        std::size_t offsetBytes = 0;
        for (std::size_t block : build.blocks)
        {
          auto [currentDevice, currentBuffer] = swapper.getPreStoreBufferOrNone(matrices[block]);
          if (currentBuffer == nullptr || currentDevice != build.deviceId ||
              currentBuffer->allocationBasePtr() != build.allocationBase ||
              currentBuffer->allocationOffsetInByte() != offsetBytes)
          {
            return {};
          }

          offsetBytes += matrices[block].sizeInByte();
          if (offsetBytes > build.allocationBytes)
          {
            return {};
          }
        }
        if (build.blocks.empty() || offsetBytes != build.allocationBytes)
        {
          return {};
        }

        CoalescedGroup group{.deviceId = build.deviceId, .blocks = build.blocks, .valueCount = build.valueCount};
        if (!swapper.preStoreBuffersAreCoalesced(this->group_matrices(x, group), build.deviceId))
        {
          return {};
        }
        groups.push_back(std::move(group));
      }

      return groups;
    }

    [[nodiscard]] bool compatible_groups(std::vector<CoalescedGroup> const& lhs,
                                         std::vector<CoalescedGroup> const& rhs) const
    {
      if (lhs.size() != rhs.size())
      {
        return false;
      }
      for (std::size_t group = 0; group < lhs.size(); ++group)
      {
        if (lhs[group].deviceId != rhs[group].deviceId || lhs[group].blocks != rhs[group].blocks ||
            lhs[group].valueCount != rhs[group].valueCount)
        {
          return false;
        }
      }
      return true;
    }

    bool relayout_like(MatrixFamily const& source, MatrixFamily const& target, bool preserve_target_content)
    {
      auto const groups = this->coalesced_groups(source);
      if (groups.empty())
      {
        return false;
      }
      auto& swapper = arranger->residentSwapper();
      for (auto const& group : groups)
      {
        swapper.ensurePreStoreCoalescedOnDevice(this->group_matrices(target, group), group.deviceId,
                                                preserve_target_content);
      }
      return true;
    }

    void synchronize_if_requested(MatrixFamily& x)
    {
      if (sync_results_to_host)
      {
        arranger->synchronizeCoalescedLinearAlgebraToHost(raw_matrices(x), x.coalesced_values());
      }
    }

    bool zero_slab(MatrixFamily& x)
    {
      if (x.empty())
      {
        this->synchronize_if_requested(x);
        return true;
      }
      auto const groups = this->coalesced_groups(x);
      if (groups.empty())
      {
        return false;
      }

      auto& swapper = arranger->residentSwapper();
      for (auto const& group : groups)
      {
        auto access = swapper.createSlabAccessPlan(this->group_matrices(x, group), group.deviceId,
                                                   tensor::Swapper::SlabAccessKind::Write);
        CUDA_CALL(cudaMemsetAsync(access.data(), 0, access.sizeInByte(), access.stream()));
      }
      this->synchronize_if_requested(x);
      return true;
    }

    bool copy_slab(MatrixFamily const& source, MatrixFamily& target)
    {
      if (source.coalesced_values().size() != target.coalesced_values().size())
      {
        throw std::invalid_argument("TensorContraction slab copy has mismatched storage sizes");
      }
      if (source.empty())
      {
        this->synchronize_if_requested(target);
        return true;
      }

      auto const sourceGroups = this->coalesced_groups(source);
      auto const targetGroups = this->coalesced_groups(target);
      if (sourceGroups.empty() || !this->compatible_groups(sourceGroups, targetGroups))
      {
        return false;
      }

      auto& swapper = arranger->residentSwapper();
      for (std::size_t groupIndex = 0; groupIndex < sourceGroups.size(); ++groupIndex)
      {
        auto const& group = sourceGroups[groupIndex];
        auto read_buffers =
            swapper.collectCoalescedPreStoreBuffers(this->group_matrices(source, group), group.deviceId);
        auto write_buffers = swapper.collectCoalescedPreStoreBuffers(
            this->group_matrices(target, targetGroups[groupIndex]), group.deviceId);
        if (read_buffers.empty() || write_buffers.empty())
        {
          return false;
        }
        auto* source_base = read_buffers.front()->allocationBasePtr();
        auto* target_base = write_buffers.front()->allocationBasePtr();
        if (source_base != target_base)
        {
          auto access = swapper.createAccessPlan(std::move(read_buffers), std::move(write_buffers), group.deviceId);
          CUDA_CALL(cudaMemcpyAsync(target_base, source_base, group.valueCount * sizeof(double),
                                    cudaMemcpyDeviceToDevice, access.stream()));
        }
      }
      this->synchronize_if_requested(target);
      return true;
    }

    bool scale_slab(MatrixFamily& x, double alpha)
    {
      if (x.empty())
      {
        this->synchronize_if_requested(x);
        return true;
      }
      auto const groups = this->coalesced_groups(x);
      if (groups.empty())
      {
        return false;
      }

      auto& swapper = arranger->residentSwapper();
      for (auto const& group : groups)
      {
        auto access = swapper.createSlabAccessPlan(this->group_matrices(x, group), group.deviceId,
                                                   tensor::Swapper::SlabAccessKind::Write);
        detail::launch_slab_scale_kernel(static_cast<double*>(access.data()), group.valueCount, alpha, access.stream());
        CUDA_CALL(cudaGetLastError());
      }
      this->synchronize_if_requested(x);
      return true;
    }

    bool axpy_slab(double alpha, MatrixFamily const& x, MatrixFamily& y)
    {
      if (x.coalesced_values().size() != y.coalesced_values().size())
      {
        throw std::invalid_argument("TensorContraction slab axpy has mismatched storage sizes");
      }
      if (x.empty())
      {
        this->synchronize_if_requested(y);
        return true;
      }

      auto const xGroups = this->coalesced_groups(x);
      auto const yGroups = this->coalesced_groups(y);
      if (xGroups.empty() || !this->compatible_groups(xGroups, yGroups))
      {
        return false;
      }

      auto& swapper = arranger->residentSwapper();
      for (std::size_t groupIndex = 0; groupIndex < xGroups.size(); ++groupIndex)
      {
        auto const& group = xGroups[groupIndex];
        auto read_buffers = swapper.collectCoalescedPreStoreBuffers(this->group_matrices(x, group), group.deviceId);
        auto write_buffers =
            swapper.collectCoalescedPreStoreBuffers(this->group_matrices(y, yGroups[groupIndex]), group.deviceId);
        if (read_buffers.empty() || write_buffers.empty())
        {
          return false;
        }
        auto* x_base = static_cast<double const*>(read_buffers.front()->allocationBasePtr());
        auto* y_base = static_cast<double*>(write_buffers.front()->allocationBasePtr());
        auto access = swapper.createAccessPlan(std::move(read_buffers), std::move(write_buffers), group.deviceId);
        detail::launch_slab_axpy_kernel(x_base, y_base, group.valueCount, alpha, access.stream());
        CUDA_CALL(cudaGetLastError());
      }
      this->synchronize_if_requested(y);
      return true;
    }

    auto dot_slab(MatrixFamily const& lhs, MatrixFamily const& rhs) -> std::optional<double>
    {
      if (lhs.coalesced_values().size() != rhs.coalesced_values().size())
      {
        throw std::invalid_argument("TensorContraction slab dot has mismatched storage sizes");
      }
      if (lhs.empty())
      {
        return 0.0;
      }

      auto const lhsGroups = this->coalesced_groups(lhs);
      auto const rhsGroups = this->coalesced_groups(rhs);
      if (lhsGroups.empty() || !this->compatible_groups(lhsGroups, rhsGroups))
      {
        return std::nullopt;
      }

      auto& swapper = arranger->residentSwapper();
      double result = 0.0;
      for (std::size_t groupIndex = 0; groupIndex < lhsGroups.size(); ++groupIndex)
      {
        auto const& group = lhsGroups[groupIndex];
        if (group.valueCount > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
          return std::nullopt;
        }

        auto read_buffers = swapper.collectCoalescedPreStoreBuffers(this->group_matrices(lhs, group), group.deviceId);
        auto rhs_buffers =
            swapper.collectCoalescedPreStoreBuffers(this->group_matrices(rhs, rhsGroups[groupIndex]), group.deviceId);
        if (read_buffers.empty() || rhs_buffers.empty())
        {
          return std::nullopt;
        }
        auto const* lhs_base = static_cast<double const*>(read_buffers.front()->allocationBasePtr());
        auto const* rhs_base = static_cast<double const*>(rhs_buffers.front()->allocationBasePtr());
        read_buffers.insert(read_buffers.end(), rhs_buffers.begin(), rhs_buffers.end());

        double partial = 0.0;
        auto access = swapper.createBlasAccessPlan(std::move(read_buffers), {}, group.deviceId);
        auto device_result = swapper.deviceContext(group.deviceId).acquireScratch(sizeof(double), access.stream());
        CUBLAS_CALL(cublasSetPointerMode(access.handle(), CUBLAS_POINTER_MODE_DEVICE));
        CUBLAS_CALL(cublasDdot(access.handle(), static_cast<int>(group.valueCount), lhs_base, 1, rhs_base, 1,
                               device_result.as<double>()));
        CUBLAS_CALL(cublasSetPointerMode(access.handle(), CUBLAS_POINTER_MODE_HOST));
        CUDA_CALL(cudaMemcpyAsync(&partial, device_result.as<double>(), sizeof(double), cudaMemcpyDeviceToHost,
                                  access.stream()));
        CUDA_CALL(cudaStreamSynchronize(access.stream()));
        result += partial;
      }
      return result;
    }

    void localize(MatrixFamily const& x, bool upload_from_host, bool refresh_existing)
    {
      // MatrixFamily remains the public host-visible container.  The
      // TensorContraction runtime uses pre-store buffers as the current GPU
      // resident representation for Lanczos-local vector algebra.
      if (arranger != nullptr)
      {
        arranger->localizeCoalescedForLinearAlgebra(raw_matrices(x), x.coalesced_values(), upload_from_host,
                                                    refresh_existing);
      }
    }
};

namespace
{

double broadcast_from_rank_zero(double value)
{
  // The current TensorContraction runtime is lockstep MPI.  Scalar Krylov
  // decisions must be identical on every rank, so rank 0 is the authority for
  // reductions exposed back to the algorithm layer.
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0)
  {
    return value;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size > 1)
  {
    MPI_Bcast(&value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  return value;
}

bool use_host_vector_algebra_backend()
{
  auto const* backend = std::getenv("UNI20_TENSORCONTRACTION_BACKEND");
  if (backend == nullptr)
  {
    return false;
  }
  return std::string(backend) == "host" || std::string(backend) == "cpu";
}

} // namespace

VectorAlgebraEngine::Impl::Impl()
{
  if (use_host_vector_algebra_backend())
  {
    return;
  }
  swapper = std::make_unique<tensor::Swapper>();
  arranger = std::make_unique<tensor::Arranger>(*swapper);
}

VectorAlgebraEngine::VectorAlgebraEngine() : impl_(std::make_unique<Impl>()) {}

VectorAlgebraEngine::VectorAlgebraEngine(VectorAlgebraEngine&&) noexcept = default;
VectorAlgebraEngine& VectorAlgebraEngine::operator=(VectorAlgebraEngine&&) noexcept = default;
VectorAlgebraEngine::~VectorAlgebraEngine() = default;

double VectorAlgebraEngine::dot(MatrixFamily const& lhs, MatrixFamily const& rhs)
{
  validate_compatible_vector_shapes(lhs, rhs);
  if (impl_->host_backend())
  {
    return uni20::tensorcontraction::dot(lhs, rhs);
  }

  if (!impl_->upload_from_host)
  {
    // Resident mode assumes the latest vector data is already in pre-store
    // buffers.  localize(..., false) only guarantees GPU allocation; it does
    // not overwrite resident data from stale host storage.
    impl_->localize(lhs, false, false);
    impl_->localize(rhs, false, false);
  }
  else
  {
    impl_->localize(lhs, true, false);
    impl_->localize(rhs, true, false);
  }

  if (auto result = impl_->dot_slab(lhs, rhs); result.has_value())
  {
    return broadcast_from_rank_zero(*result);
  }
  if (impl_->relayout_like(lhs, rhs, /*preserve_target_content=*/true))
  {
    if (auto result = impl_->dot_slab(lhs, rhs); result.has_value())
    {
      return broadcast_from_rank_zero(*result);
    }
  }

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  if (lhs_matrices.empty())
  {
    return broadcast_from_rank_zero(0.0);
  }

  auto& swapper = impl_->arranger->residentSwapper();
  int const device_count = swapper.getDeviceCount();
  std::vector<double> device_results(static_cast<std::size_t>(device_count), 0.0);
  std::vector<tensor::Matrix> device_partials;
  device_partials.reserve(static_cast<std::size_t>(device_count));
  for (int device = 0; device < device_count; ++device)
  {
    device_partials.emplace_back(&device_results[static_cast<std::size_t>(device)], 1, 1);
    swapper.registerGpuAllocation(device_partials.back(), device);
    impl_->arranger->compileZeroForLinearAlgebra(device_partials.back(), false);
  }

  for (std::size_t block = 0; block < lhs_matrices.size(); ++block)
  {
    auto const [device, buffer] = swapper.getPreStoreBufferOrNone(lhs_matrices[block]);
    if (buffer == nullptr)
    {
      throw std::logic_error("TensorContraction dot input block was not localized to a GPU device");
    }
    impl_->arranger->compileInnerProductAccumulateForLinearAlgebra(device_partials[static_cast<std::size_t>(device)],
                                                                   lhs_matrices[block], rhs_matrices[block]);
  }
  for (auto partial : device_partials)
  {
    impl_->arranger->compileSyncForLinearAlgebra(partial);
  }
  impl_->arranger->doLinearAlgebra();

  double result = 0.0;
  for (double value : device_results)
  {
    result += value;
  }
  for (auto partial : device_partials)
  {
    swapper.clear(partial);
  }
  return broadcast_from_rank_zero(result);
}

double VectorAlgebraEngine::norm2(MatrixFamily const& x) { return this->dot(x, x); }

double VectorAlgebraEngine::norm(MatrixFamily const& x) { return std::sqrt(this->norm2(x)); }

void VectorAlgebraEngine::set_host_synchronization(bool enabled)
{
  impl_->upload_from_host = enabled;
  impl_->sync_results_to_host = enabled;
}

void VectorAlgebraEngine::localize(MatrixFamily& x) { impl_->localize(x, false, false); }

void VectorAlgebraEngine::upload(MatrixFamily& x) { impl_->localize(x, true, true); }

void VectorAlgebraEngine::synchronize(MatrixFamily& x)
{
  if (impl_->host_backend())
  {
    return;
  }
  // Explicit escape hatch for boundaries that still consume host storage, such
  // as the current EffectiveHamiltonianOperator adapter.
  impl_->arranger->synchronizeCoalescedLinearAlgebraToHost(raw_matrices(x), x.coalesced_values());
}

void VectorAlgebraEngine::release(MatrixFamily const& x) noexcept
{
  if (impl_->host_backend())
  {
    return;
  }

  try
  {
    auto& swapper = impl_->arranger->residentSwapper();
    for (auto const& matrix : raw_matrices(x))
    {
      swapper.clear(matrix);
    }
  }
  catch (...)
  {
    // This function is used from cleanup paths.  Leaking here is preferable to
    // terminating during stack unwinding; explicit diagnostics can use Swapper
    // counters when needed.
  }
}

void VectorAlgebraEngine::zero(MatrixFamily& x)
{
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::zero(x);
    return;
  }
  impl_->localize(x, false, false);
  if (impl_->zero_slab(x))
  {
    return;
  }
  for (auto const& matrix : raw_matrices(x))
  {
    impl_->arranger->compileZeroForLinearAlgebra(matrix, impl_->sync_results_to_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::copy(MatrixFamily const& source, MatrixFamily& target)
{
  validate_compatible_vector_shapes(source, target);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::copy(source, target);
    return;
  }

  if (!impl_->upload_from_host)
  {
    impl_->localize(source, false, false);
  }
  else
  {
    impl_->localize(source, true, false);
  }
  if (!impl_->relayout_like(source, target, /*preserve_target_content=*/false))
  {
    impl_->localize(target, false, false);
  }

  if (impl_->copy_slab(source, target))
  {
    return;
  }

  double one = 1.0;
  auto const& source_matrices = raw_matrices(source);
  auto const& target_matrices = raw_matrices(target);
  for (std::size_t block = 0; block < source_matrices.size(); ++block)
  {
    impl_->arranger->compileZeroForLinearAlgebra(target_matrices[block], impl_->sync_results_to_host);
    impl_->arranger->compileAddAccuForLinearAlgebra(target_matrices[block], source_matrices[block], &one,
                                                    impl_->sync_results_to_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::scale(MatrixFamily& x, double alpha)
{
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::scale(x, alpha);
    return;
  }
  if (!impl_->upload_from_host)
  {
    impl_->localize(x, false, false);
  }
  else
  {
    impl_->localize(x, true, false);
  }
  if (impl_->scale_slab(x, alpha))
  {
    return;
  }
  for (auto const& matrix : raw_matrices(x))
  {
    impl_->arranger->compileScalarMulForLinearAlgebra(matrix, &alpha, impl_->sync_results_to_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::axpy(double alpha, MatrixFamily const& x, MatrixFamily& y)
{
  validate_compatible_vector_shapes(x, y);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::axpy(alpha, x, y);
    return;
  }

  if (!impl_->upload_from_host)
  {
    impl_->localize(x, false, false);
    impl_->localize(y, false, false);
  }
  else
  {
    impl_->localize(x, true, false);
    impl_->localize(y, true, false);
  }

  auto const& x_matrices = raw_matrices(x);
  auto const& y_matrices = raw_matrices(y);
  if (impl_->axpy_slab(alpha, x, y))
  {
    return;
  }
  if (impl_->relayout_like(x, y, /*preserve_target_content=*/true) && impl_->axpy_slab(alpha, x, y))
  {
    return;
  }
  for (std::size_t block = 0; block < x_matrices.size(); ++block)
  {
    impl_->arranger->compileAddAccuForLinearAlgebra(y_matrices[block], x_matrices[block], &alpha,
                                                    impl_->sync_results_to_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::gemm_each(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result)
{
  validate_compatible_gemm_shapes(lhs, rhs, result);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::gemm_each(lhs, rhs, result);
    return;
  }

  if (!impl_->upload_from_host)
  {
    impl_->localize(lhs, false, false);
    impl_->localize(rhs, false, false);
    impl_->localize(result, false, false);
  }
  else
  {
    impl_->localize(lhs, true, false);
    impl_->localize(rhs, true, false);
    impl_->localize(result, false, false);
  }

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  auto const& result_matrices = raw_matrices(result);
  for (std::size_t block = 0; block < lhs_matrices.size(); ++block)
  {
    impl_->arranger->compileMatMulForLinearAlgebra(result_matrices[block], lhs_matrices[block], rhs_matrices[block],
                                                   impl_->sync_results_to_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::gemm_each_to_resident(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result)
{
  validate_compatible_gemm_shapes(lhs, rhs, result);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::gemm_each(lhs, rhs, result);
    return;
  }

  impl_->localize(lhs, true, true);
  impl_->localize(rhs, true, true);
  impl_->localize(result, false, false);

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  auto const& result_matrices = raw_matrices(result);
  for (std::size_t block = 0; block < lhs_matrices.size(); ++block)
  {
    impl_->arranger->compileMatMulForLinearAlgebra(result_matrices[block], lhs_matrices[block], rhs_matrices[block],
                                                   false);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::gemm_selected_to_resident(MatrixFamily const& lhs, MatrixFamily const& rhs,
                                                    MatrixFamily& result,
                                                    std::span<std::size_t const> lhs_block_for_result,
                                                    std::span<std::size_t const> rhs_block_for_result)
{
  validate_compatible_selected_gemm_shapes(lhs, rhs, result, lhs_block_for_result, rhs_block_for_result);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::gemm_selected(lhs, rhs, result, lhs_block_for_result, rhs_block_for_result);
    return;
  }

  impl_->localize(lhs, true, true);
  impl_->localize(rhs, true, true);
  impl_->localize(result, false, false);

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  auto const& result_matrices = raw_matrices(result);
  for (std::size_t block = 0; block < result_matrices.size(); ++block)
  {
    impl_->arranger->compileMatMulForLinearAlgebra(result_matrices[block], lhs_matrices[lhs_block_for_result[block]],
                                                   rhs_matrices[rhs_block_for_result[block]], false);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::gemm_sparse_selected_to_resident(MatrixFamily const& lhs, MatrixFamily const& rhs,
                                                           MatrixFamily& result,
                                                           std::span<std::size_t const> lhs_block_for_product,
                                                           std::span<std::size_t const> rhs_block_for_product,
                                                           std::span<std::size_t const> result_block_for_product)
{
  validate_compatible_sparse_selected_gemm_shapes(lhs, rhs, result, lhs_block_for_product, rhs_block_for_product,
                                                  result_block_for_product);

  std::vector<bool> result_seen(result.size(), false);
  for (std::size_t result_index : result_block_for_product)
  {
    if (result_seen[result_index])
    {
      throw std::invalid_argument(
          "TensorContraction sparse selected GEMM does not yet support duplicate result blocks");
    }
    result_seen[result_index] = true;
  }

  if (impl_->host_backend())
  {
    uni20::tensorcontraction::zero(result);
    for (std::size_t product = 0; product < result_block_for_product.size(); ++product)
    {
      auto const lhs_index = lhs_block_for_product[product];
      auto const rhs_index = rhs_block_for_product[product];
      auto const result_index = result_block_for_product[product];
      auto const lhs_block = lhs.block(lhs_index);
      auto const rhs_block = rhs.block(rhs_index);
      auto const result_block = result.block(result_index);
      auto const lhs_values = lhs.values(lhs_index);
      auto const rhs_values = rhs.values(rhs_index);
      auto result_values = result.values(result_index);
      for (std::size_t row = 0; row < lhs_block.rows; ++row)
      {
        for (std::size_t inner = 0; inner < lhs_block.cols; ++inner)
        {
          auto const lhs_value = lhs_values[row * lhs_block.cols + inner];
          for (std::size_t col = 0; col < rhs_block.cols; ++col)
          {
            result_values[row * result_block.cols + col] += lhs_value * rhs_values[inner * rhs_block.cols + col];
          }
        }
      }
    }
    return;
  }

  impl_->localize(lhs, true, true);
  impl_->localize(rhs, true, true);
  impl_->localize(result, false, false);

  auto const& result_matrices = raw_matrices(result);
  for (auto const& matrix : result_matrices)
  {
    impl_->arranger->compileZeroForLinearAlgebra(matrix, false);
  }

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  for (std::size_t product = 0; product < result_block_for_product.size(); ++product)
  {
    impl_->arranger->compileMatMulForLinearAlgebra(result_matrices[result_block_for_product[product]],
                                                   lhs_matrices[lhs_block_for_product[product]],
                                                   rhs_matrices[rhs_block_for_product[product]], false);
  }
  impl_->arranger->doLinearAlgebra();
}

double VectorAlgebraEngine::normalize(MatrixFamily& x)
{
  double const x_norm = this->norm(x);
  if (x_norm == 0.0)
  {
    throw std::invalid_argument("TensorContraction cannot normalize a zero vector");
  }
  this->scale(x, 1.0 / x_norm);
  return x_norm;
}

bool VectorAlgebraEngine::uses_host_backend() const { return impl_->host_backend(); }

tensor::Arranger& VectorAlgebraEngine::resident_arranger()
{
  if (impl_->arranger == nullptr)
  {
    throw std::logic_error("TensorContraction resident runtime is not available on the host backend");
  }
  return *impl_->arranger;
}

tensor::Swapper& VectorAlgebraEngine::resident_swapper()
{
  if (impl_->swapper == nullptr)
  {
    throw std::logic_error("TensorContraction resident runtime is not available on the host backend");
  }
  return *impl_->swapper;
}

} // namespace uni20::tensorcontraction
