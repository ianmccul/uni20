#include <uni20/async/cuda_task.hpp>
#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cublas/cublas_error_presentation.hpp>
#include <uni20/backend/cublas/gemm.hpp>
#include <uni20/backend/cublas/task_awaiters.hpp>
#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/common/presentation.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/async/matrix_product.hpp>
#include <uni20/linalg/backends/cublas/gemm.hpp>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/tensor.hpp>

#include "../../linalg/gemm_conformance.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

template <class Tensor> class CudaMatrixView {
  public:
    using tensor_type = Tensor;
    using element_type = typename tensor_type::element_type;
    using storage_policy = typename tensor_type::storage_policy;
    using index_type = uni20::index_type;
    using extents_type = stdex::dextents<index_type, 2>;
    using mapping_type = stdex::layout_stride::mapping<extents_type>;
    using accessor_type = typename tensor_type::device_accessor_type;
    using const_accessor_type = typename tensor_type::const_device_accessor_type;
    using descriptor_type = uni20::cuda::CudaBufferView<element_type>;
    using const_descriptor_type = uni20::cuda::CudaBufferView<element_type const>;
    using device_mdspan_type =
        uni20::device_mdspan<element_type, extents_type, stdex::layout_stride, accessor_type, descriptor_type>;
    using const_device_mdspan_type = uni20::device_mdspan<element_type const, extents_type, stdex::layout_stride,
                                                          const_accessor_type, const_descriptor_type>;

    CudaMatrixView(tensor_type& tensor, std::size_t offset, index_type rows, index_type cols,
                   std::array<index_type, 2> strides)
        : tensor_(&tensor), offset_(offset), mapping_(extents_type{rows, cols}, strides)
    {
      auto const span_size = static_cast<std::size_t>(mapping_.required_span_size());
      CHECK(offset_ <= tensor.storage().size() && span_size <= tensor.storage().size() - offset_);
    }

    [[nodiscard]] static constexpr auto backend_selector() noexcept { return storage_policy::backend_selector(); }

    [[nodiscard]] auto device_mdspan() noexcept -> device_mdspan_type
    {
      auto span = tensor_->device_mdspan();
      return device_mdspan_type(span.data_descriptor().offset_by(offset_), mapping_, span.accessor());
    }

    [[nodiscard]] auto device_mdspan() const noexcept -> const_device_mdspan_type
    {
      auto const& tensor = std::as_const(*tensor_);
      auto span = tensor.device_mdspan();
      return const_device_mdspan_type(span.data_descriptor().offset_by(offset_), mapping_, span.accessor());
    }

    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return mapping_.extents(); }
    [[nodiscard]] auto extent(std::size_t axis) const noexcept { return mapping_.extents().extent(axis); }
    [[nodiscard]] auto mapping() const noexcept -> mapping_type const& { return mapping_; }
    [[nodiscard]] std::size_t size() const noexcept { return mapping_.required_span_size(); }

  private:
    tensor_type* tensor_;
    std::size_t offset_;
    mapping_type mapping_;
};

template <class ElementType> struct UnrecognizedCudaAccessor
{
    using element_type = ElementType;
    using reference = uni20::cuda::CudaBufferView<element_type>;
    using data_handle_type = reference;
    using offset_policy = UnrecognizedCudaAccessor;

    [[nodiscard]] constexpr reference access(data_handle_type handle, std::size_t offset) const noexcept
    {
      return handle.offset_by(offset);
    }

    [[nodiscard]] constexpr data_handle_type offset(data_handle_type handle, std::size_t offset) const noexcept
    {
      return handle.offset_by(offset);
    }
};

using cuda_test_extents = stdex::dextents<uni20::index_type, 2>;
using writable_cuda_span =
    uni20::device_mdspan<double, cuda_test_extents, stdex::layout_left, uni20::cuda::CudaPointerAccessor<double>,
                         uni20::cuda::CudaBufferView<double>>;
using readable_cuda_span =
    uni20::device_mdspan<double const, cuda_test_extents, stdex::layout_left,
                         uni20::cuda::CudaPointerAccessor<double const>, uni20::cuda::CudaBufferView<double const>>;
using unrecognized_cuda_span =
    uni20::device_mdspan<double const, cuda_test_extents, stdex::layout_left, UnrecognizedCudaAccessor<double const>,
                         uni20::cuda::CudaBufferView<double const>>;

template <class Input>
concept cublas_gemm_accepts_input_accessor =
    requires(writable_cuda_span& output, Input& input, readable_cuda_span& rhs) {
      uni20::linalg::detail::cublas_backend::try_gemm(output, 1.0, input, rhs, 0.0);
    };

static_assert(cublas_gemm_accepts_input_accessor<readable_cuda_span>);
static_assert(!cublas_gemm_accepts_input_accessor<unrecognized_cuda_span>);

class CublasExecutionTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess)
      {
        GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      }
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
      device_ = 0;
    }

    int device_count_ = 0;
    int device_ = -1;
};

class CudaGemmPlatform {
  public:
    static constexpr std::size_t expected_backend_candidates = 1;

    explicit CudaGemmPlatform(int device) : resources_({.device = uni20::cuda::Device::get(device), .stream_count = 2})
    {}

    template <class Scalar, class Layout> [[nodiscard]] auto make_matrix(uni20::index_type rows, uni20::index_type cols)
    {
      return uni20::CudaMatrix<Scalar, Layout>(resources_, rows, cols);
    }

    template <class Scalar>
    [[nodiscard]] auto make_strided_matrix(uni20::index_type rows, uni20::index_type cols,
                                           std::array<uni20::index_type, 2> strides)
    {
      using matrix_type = uni20::Tensor<Scalar, 2, uni20::CudaStorage, stdex::layout_stride>;
      using extents_type = typename matrix_type::extents_type;
      typename matrix_type::mapping_type mapping(extents_type{rows, cols}, strides);
      typename matrix_type::storage_type storage(resources_, mapping.required_span_size());
      return matrix_type::adopt_storage(std::move(mapping), std::move(storage));
    }

    template <class Tensor>
    void write_physical(Tensor& tensor, std::vector<uni20::tensor_element_t<Tensor>> const& values)
    {
      auto span = tensor.device_mdspan();
      auto view = span.data_descriptor();
      auto& buffer = view.buffer();
      uni20::cuda::ScopedDevice device_scope(buffer.device().ordinal());
      ASSERT_EQ(values.size(), static_cast<std::size_t>(span.mapping().required_span_size()));
      ASSERT_LE(view.element_offset(), buffer.size());
      ASSERT_LE(values.size(), buffer.size() - view.element_offset());
      auto stream = resources_.streams().acquire();
      {
        auto write = buffer.write_synchronized_with(stream);
        ASSERT_EQ(cudaMemcpyAsync(write.data() + view.element_offset(), values.data(),
                                  values.size() * sizeof(values[0]), cudaMemcpyHostToDevice, stream.native_handle()),
                  cudaSuccess);
      }
      stream.synchronize();
    }

    template <class Tensor>
    [[nodiscard]] auto read_physical(Tensor const& tensor) -> std::vector<uni20::tensor_element_t<Tensor>>
    {
      auto span = tensor.device_mdspan();
      auto view = span.data_descriptor();
      auto const& buffer = view.buffer();
      uni20::cuda::ScopedDevice device_scope(buffer.device().ordinal());
      std::vector<uni20::tensor_element_t<Tensor>> values(span.mapping().required_span_size());
      CHECK(view.element_offset() <= buffer.size() && values.size() <= buffer.size() - view.element_offset());
      auto stream = resources_.streams().acquire();
      {
        auto read = buffer.read_synchronized_with(stream);
        EXPECT_EQ(cudaMemcpyAsync(values.data(), read.data() + view.element_offset(), values.size() * sizeof(values[0]),
                                  cudaMemcpyDeviceToHost, stream.native_handle()),
                  cudaSuccess);
      }
      stream.synchronize();
      return values;
    }

    [[nodiscard]] auto resources() noexcept -> uni20::cuda::DeviceResources& { return resources_; }

    template <class Selector, class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
    [[nodiscard]] auto kernel_type_candidates(Selector const& selector, OutputMdspan& output, Scalar alpha,
                                              LhsMdspan& lhs, RhsMdspan& rhs, Scalar beta)
    {
      return uni20::linalg::kernel_type_candidates(selector, uni20::linalg::gemm_op{}, output, alpha, lhs, rhs, beta);
    }

    template <class Backend, class OutputTensor, class Scalar, class LhsTensor, class RhsTensor>
    void gemm(Backend const& backend, OutputTensor& output, Scalar alpha, LhsTensor const& lhs, RhsTensor const& rhs,
              Scalar beta)
    {
      uni20::linalg::gemm(backend, output, alpha, lhs, rhs, beta);
    }

  private:
    uni20::cuda::DeviceResources resources_;
};

static_assert(!std::is_copy_constructible_v<uni20::cublas::ExecutionLease>);
static_assert(std::is_move_constructible_v<uni20::cublas::ExecutionLease>);
static_assert(uni20::async::CudaTaskAwaitable<
              decltype(uni20::cublas::acquire_execution(std::declval<uni20::cublas::ExecutionPool&>()))>);

TEST(CublasErrorTest, RendersThroughPresentationLayer)
{
  uni20::cublas::CublasError error(CUBLAS_STATUS_INVALID_VALUE, "cublasDgemm", 1);
  auto policy = uni20::presentation::plain_policy();
  policy.glyphs = uni20::presentation::glyph_set::ascii;
  auto const rendered = uni20::presentation::render_plain(uni20::cublas::diagnostic_report(error), policy);

  EXPECT_NE(rendered.find("[FAIL] cuBLAS 'cublasDgemm' failed"), std::string::npos) << rendered;
  EXPECT_NE(rendered.find("CUBLAS_STATUS_INVALID_VALUE"), std::string::npos) << rendered;
  EXPECT_NE(rendered.find("Device"), std::string::npos) << rendered;
}

TEST(CublasErrorTest, CheckedProviderFailureRaisesStructuredException)
{
  bool const previous_errors_abort = trace::get_formatting_options().errors_abort();
  trace::get_formatting_options().set_errors_abort(false);
  try
  {
    uni20::cublas::check(CUBLAS_STATUS_INVALID_VALUE, "cublasDgemm", 1);
    ADD_FAILURE() << "failed cuBLAS status should raise CublasError";
  }
  catch (uni20::cublas::CublasError const& error)
  {
    EXPECT_EQ(error.status(), CUBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(error.operation(), "cublasDgemm");
    EXPECT_EQ(error.device(), 1);
  }
  catch (...)
  {
    ADD_FAILURE() << "failed cuBLAS status raised the wrong exception type";
  }
  trace::get_formatting_options().set_errors_abort(previous_errors_abort);
}

template <class Scalar> void check_column_major_gemm(int device)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device), .stream_count = 2});
  uni20::cublas::ExecutionPool executions(resources.streams(), 1);
  uni20::cuda::CudaBuffer<Scalar> lhs(resources, 6);
  uni20::cuda::CudaBuffer<Scalar> rhs(resources, 6);
  uni20::cuda::CudaBuffer<Scalar> output(resources, 4);

  // Column-major matrices: lhs is 2x3, rhs is 3x2.
  std::array<Scalar, 6> const host_lhs{Scalar{1}, Scalar{4}, Scalar{2}, Scalar{5}, Scalar{3}, Scalar{6}};
  std::array<Scalar, 6> const host_rhs{Scalar{7}, Scalar{9}, Scalar{11}, Scalar{8}, Scalar{10}, Scalar{12}};
  std::array<Scalar, 4> host_output{};

  auto execution = executions.acquire();
  {
    auto lhs_write = lhs.write_synchronized_with(execution.stream());
    auto rhs_write = rhs.write_synchronized_with(execution.stream());
    ASSERT_EQ(cudaMemcpyAsync(lhs_write.data(), host_lhs.data(), lhs_write.size_bytes(), cudaMemcpyHostToDevice,
                              execution.stream().native_handle()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(rhs_write.data(), host_rhs.data(), rhs_write.size_bytes(), cudaMemcpyHostToDevice,
                              execution.stream().native_handle()),
              cudaSuccess);
  }
  {
    auto lhs_read = lhs.read_synchronized_with(execution.stream());
    auto rhs_read = rhs.read_synchronized_with(execution.stream());
    auto output_write = output.write_synchronized_with(execution.stream());

    uni20::linalg::blas::BlasWritableMatrix<Scalar> output_matrix{
        .data = output_write.data(), .rows = 2, .cols = 2, .leading_dimension = 2};
    uni20::linalg::blas::BlasReadableMatrix<Scalar> lhs_matrix{
        .data = lhs_read.data(), .rows = 2, .cols = 3, .leading_dimension = 2};
    uni20::linalg::blas::BlasReadableMatrix<Scalar> rhs_matrix{
        .data = rhs_read.data(), .rows = 3, .cols = 2, .leading_dimension = 3};
    uni20::linalg::cublas::gemm(execution, output_matrix, Scalar{1}, lhs_matrix, rhs_matrix, Scalar{});
  }
  {
    auto output_read = output.read_synchronized_with(execution.stream());
    ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output_read.data(), output_read.size_bytes(), cudaMemcpyDeviceToHost,
                              execution.stream().native_handle()),
              cudaSuccess);
  }
  execution.stream().synchronize();

  EXPECT_EQ(host_output[0], Scalar{58});
  EXPECT_EQ(host_output[1], Scalar{139});
  EXPECT_EQ(host_output[2], Scalar{64});
  EXPECT_EQ(host_output[3], Scalar{154});

  execution.release();
  resources.streams().synchronize();
}

template <class Tensor> void upload_tensor(Tensor& tensor, std::span<uni20::tensor_element_t<Tensor> const> values)
{
  ASSERT_EQ(values.size(), tensor.size());
  uni20::cuda::ScopedDevice device_scope(tensor.storage().device().ordinal());
  auto stream = tensor.storage().resources().streams().acquire();
  {
    auto write = tensor.storage().write_synchronized_with(stream);
    ASSERT_EQ(cudaMemcpyAsync(write.data(), values.data(), write.size_bytes(), cudaMemcpyHostToDevice,
                              stream.native_handle()),
              cudaSuccess);
  }
  stream.synchronize();
}

template <class Tensor> auto download_tensor(Tensor const& tensor) -> std::vector<uni20::tensor_element_t<Tensor>>
{
  std::vector<uni20::tensor_element_t<Tensor>> values(tensor.size());
  uni20::cuda::ScopedDevice device_scope(tensor.storage().device().ordinal());
  auto stream = tensor.storage().resources().streams().acquire();
  {
    auto read = tensor.storage().read_synchronized_with(stream);
    EXPECT_EQ(
        cudaMemcpyAsync(values.data(), read.data(), read.size_bytes(), cudaMemcpyDeviceToHost, stream.native_handle()),
        cudaSuccess);
  }
  stream.synchronize();
  return values;
}

template <class Scalar> void check_complex_conjugate_transpose_gemm(int device)
{
  using real_type = typename Scalar::value_type;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device), .stream_count = 1});
  uni20::cublas::ExecutionPool executions(resources.streams(), 1);
  uni20::cuda::CudaBuffer<Scalar> lhs(resources, 2);
  uni20::cuda::CudaBuffer<Scalar> rhs(resources, 2);
  uni20::cuda::CudaBuffer<Scalar> output(resources, 1);

  std::array<Scalar, 2> const host_lhs{Scalar{real_type{1}, real_type{2}}, Scalar{real_type{3}, real_type{-1}}};
  std::array<Scalar, 2> const host_rhs{Scalar{real_type{2}, real_type{-1}}, Scalar{real_type{-1}, real_type{4}}};
  Scalar host_output{};

  auto execution = executions.acquire();
  {
    auto lhs_write = lhs.write_synchronized_with(execution.stream());
    auto rhs_write = rhs.write_synchronized_with(execution.stream());
    ASSERT_EQ(cudaMemcpyAsync(lhs_write.data(), host_lhs.data(), lhs_write.size_bytes(), cudaMemcpyHostToDevice,
                              execution.stream().native_handle()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(rhs_write.data(), host_rhs.data(), rhs_write.size_bytes(), cudaMemcpyHostToDevice,
                              execution.stream().native_handle()),
              cudaSuccess);
  }
  {
    auto lhs_read = lhs.read_synchronized_with(execution.stream());
    auto rhs_read = rhs.read_synchronized_with(execution.stream());
    auto output_write = output.write_synchronized_with(execution.stream());

    uni20::cublas::gemm(execution, 'C', 'N', 1, 1, 2, Scalar{real_type{1}}, lhs_read.data(), 2, rhs_read.data(), 2,
                        Scalar{}, output_write.data(), 1);
  }
  {
    auto output_read = output.read_synchronized_with(execution.stream());
    ASSERT_EQ(cudaMemcpyAsync(&host_output, output_read.data(), output_read.size_bytes(), cudaMemcpyDeviceToHost,
                              execution.stream().native_handle()),
              cudaSuccess);
  }
  execution.stream().synchronize();

  EXPECT_EQ(host_output, Scalar(real_type{-7}, real_type{6}));

  execution.release();
  resources.streams().synchronize();
}

TEST_F(CublasExecutionTest, ReusesHandleWithAnotherIdleStream)
{
  uni20::cuda::StreamPool streams({.device = device_, .stream_count = 2});
  uni20::cublas::ExecutionPool executions(streams, 1);

  auto first = executions.acquire();
  auto retained_first_stream = first.stream();
  cudaStream_t const first_native_stream = retained_first_stream.native_handle();
  first.release();
  retained_first_stream.synchronize();

  EXPECT_EQ(executions.idle_handle_count(), 1);
  EXPECT_EQ(streams.leased_stream_count(), 1);
  EXPECT_EQ(streams.idle_stream_count(), 1);

  auto second = executions.acquire();
  EXPECT_NE(second.stream().native_handle(), first_native_stream);

  second.release();
  retained_first_stream = {};
  streams.synchronize();
  EXPECT_EQ(executions.idle_handle_count(), 1);
  EXPECT_EQ(streams.idle_stream_count(), 2);
}

TEST_F(CublasExecutionTest, AsyncAcquisitionReservesHandleBeforeWaitingForStream)
{
  uni20::cuda::StreamPool streams({.device = device_, .stream_count = 1});
  uni20::cublas::ExecutionPool executions(streams, 1);
  auto occupied_stream = streams.acquire();

  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(device_));
  bool acquired = false;
  auto task = [](uni20::cublas::ExecutionPool& pool, bool& result) static -> uni20::async::CudaTask {
    auto execution = co_await uni20::cublas::acquire_execution(pool);
    result = static_cast<bool>(execution);
    co_return;
  }(executions, acquired);

  scheduler.schedule(std::move(task), device_);
  scheduler.run();

  EXPECT_FALSE(acquired);
  EXPECT_EQ(executions.idle_handle_count(), 0);

  occupied_stream = {};
  streams.synchronize();
  scheduler.run_all();

  EXPECT_TRUE(acquired);
  streams.synchronize();
  EXPECT_EQ(executions.idle_handle_count(), 1);
  EXPECT_EQ(streams.idle_stream_count(), 1);
}

TEST_F(CublasExecutionTest, AsyncTensorAssignAndAddProductUseCudaTaskLowering)
{
  using matrix_type = uni20::CudaMatrix<double>;
  using async_matrix_type = uni20::async::Async<matrix_type>;
  auto runtime =
      uni20::cuda::initialize({.device_ordinals = {device_}, .default_device = device_, .streams_per_device = 2});
  auto& resources = runtime.device_resources(device_);
  matrix_type lhs_value(resources, 2, 3);
  matrix_type rhs_value(resources, 3, 2);
  matrix_type output_value(resources, 2, 2);
  std::array<double, 6> const lhs_values{1, 4, 2, 5, 3, 6};
  std::array<double, 6> const rhs_values{7, 9, 11, 8, 10, 12};
  std::array<double, 4> const initial_output{1, 2, 3, 4};
  upload_tensor(lhs_value, std::span<double const>{lhs_values});
  upload_tensor(rhs_value, std::span<double const>{rhs_values});
  upload_tensor(output_value, std::span<double const>{initial_output});

  async_matrix_type lhs(std::move(lhs_value));
  async_matrix_type rhs(std::move(rhs_value));
  async_matrix_type assigned;
  async_matrix_type updated(std::move(output_value));
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(device_));
  uni20::async::ScopedScheduler scoped(&scheduler);

  uni20::linalg::assign_product(assigned, lhs, rhs);
  EXPECT_EQ(download_tensor(assigned.get_wait(scheduler)), (std::vector<double>{58, 139, 64, 154}));

  uni20::linalg::add_product(updated, lhs, rhs, 2.0);
  EXPECT_EQ(download_tensor(updated.get_wait(scheduler)), (std::vector<double>{117, 280, 131, 312}));
}

TEST_F(CublasExecutionTest, AsyncOutputRemainsPendingUntilCublasSubmissionCompletes)
{
  using matrix_type = uni20::CudaMatrix<double>;
  using async_matrix_type = uni20::async::Async<matrix_type>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 2});
  matrix_type lhs_value(resources, 1, 1);
  matrix_type rhs_value(resources, 1, 1);
  matrix_type output_value(resources, 1, 1);
  std::array<double, 1> const lhs_values{6};
  std::array<double, 1> const rhs_values{7};
  upload_tensor(lhs_value, std::span<double const>{lhs_values});
  upload_tensor(rhs_value, std::span<double const>{rhs_values});

  auto& executions = uni20::cublas::execution_pool(resources);
  std::vector<uni20::cublas::ExecutionLease> occupied;
  occupied.reserve(executions.handle_count());
  for (std::size_t i = 0; i < executions.handle_count(); ++i)
  {
    occupied.push_back(executions.acquire());
  }

  async_matrix_type lhs(std::move(lhs_value));
  async_matrix_type rhs(std::move(rhs_value));
  async_matrix_type output(std::move(output_value));
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(device_));
  uni20::async::ScopedScheduler scoped(&scheduler);

  uni20::linalg::gemm(output, 1.0, lhs, rhs, 0.0);
  auto output_reader = output.read();
  scheduler.run_all();
  EXPECT_FALSE(output_reader.await_ready());

  occupied.clear();
  resources.streams().synchronize();
  scheduler.run_all();

  EXPECT_TRUE(output_reader.await_ready());
  EXPECT_EQ(download_tensor(output_reader.get_wait(scheduler)), (std::vector<double>{42}));
}

TEST_F(CublasExecutionTest, AsyncEmptyOutputDoesNotWaitForCublasExecutionResources)
{
  using matrix_type = uni20::CudaMatrix<double>;
  using async_matrix_type = uni20::async::Async<matrix_type>;
  auto runtime =
      uni20::cuda::initialize({.device_ordinals = {device_}, .default_device = device_, .streams_per_device = 1});
  auto& resources = runtime.device_resources(device_);
  matrix_type lhs_value(resources, 0, 3);
  matrix_type rhs_value(resources, 3, 2);

  auto& executions = uni20::cublas::execution_pool(resources);
  std::vector<uni20::cublas::ExecutionLease> occupied;
  occupied.reserve(executions.handle_count());
  for (std::size_t i = 0; i < executions.handle_count(); ++i)
  {
    occupied.push_back(executions.acquire());
  }

  async_matrix_type lhs(std::move(lhs_value));
  async_matrix_type rhs(std::move(rhs_value));
  async_matrix_type output;
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(device_));
  uni20::async::ScopedScheduler scoped(&scheduler);

  uni20::linalg::assign_product(output, lhs, rhs);
  auto output_reader = output.read();
  scheduler.run_all();
  ASSERT_TRUE(output_reader.await_ready());

  occupied.clear();
  resources.streams().synchronize();
  auto const& result = output_reader.get_wait(scheduler);
  EXPECT_EQ(result.rows(), 0);
  EXPECT_EQ(result.cols(), 2);
  EXPECT_EQ(executions.idle_handle_count(), executions.handle_count());
}

TEST_F(CublasExecutionTest, AsyncTensorProductMigratesToOperandDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "test requires at least two CUDA devices";

  using matrix_type = uni20::CudaMatrix<double>;
  using async_matrix_type = uni20::async::Async<matrix_type>;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .default_device = 0, .streams_per_device = 1});
  auto& resources = runtime.device_resources(1);
  matrix_type lhs_value(resources, 1, 1);
  matrix_type rhs_value(resources, 1, 1);
  std::array<double, 1> const lhs_values{6};
  std::array<double, 1> const rhs_values{7};
  upload_tensor(lhs_value, std::span<double const>{lhs_values});
  upload_tensor(rhs_value, std::span<double const>{rhs_values});

  async_matrix_type lhs(std::move(lhs_value));
  async_matrix_type rhs(std::move(rhs_value));
  async_matrix_type output;
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);

  uni20::linalg::assign_product(output, lhs, rhs);

  auto const& result = output.get_wait(scheduler);
  EXPECT_EQ(result.storage().device().ordinal(), 1);
  EXPECT_EQ(download_tensor(result), (std::vector<double>{42}));
}

TEST_F(CublasExecutionTest, TensorProductMigratesConcreteOutputToOperandDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "test requires at least two CUDA devices";

  using matrix_type = uni20::CudaMatrix<double>;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .default_device = 0, .streams_per_device = 2});
  auto& output_resources = runtime.device_resources(0);
  auto& input_resources = runtime.device_resources(1);
  matrix_type lhs(input_resources, 1, 1);
  matrix_type rhs(input_resources, 1, 1);
  matrix_type output(output_resources, 2, 2);
  std::array<double, 1> const lhs_values{6};
  std::array<double, 1> const rhs_values{7};
  upload_tensor(lhs, std::span<double const>{lhs_values});
  upload_tensor(rhs, std::span<double const>{rhs_values});

  uni20::linalg::assign_product(output, lhs, rhs);

  EXPECT_EQ(output.rows(), 1);
  EXPECT_EQ(output.cols(), 1);
  EXPECT_EQ(output.storage().device().ordinal(), 1);
  EXPECT_EQ(download_tensor(output), (std::vector<double>{42}));
}

TEST_F(CublasExecutionTest, TensorProductResizePreservesCompatibleDeviceResources)
{
  using matrix_type = uni20::CudaMatrix<double>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 2});
  matrix_type lhs(resources, 1, 1);
  matrix_type rhs(resources, 1, 1);
  matrix_type output(resources, 2, 2);
  std::array<double, 1> const lhs_values{6};
  std::array<double, 1> const rhs_values{7};
  upload_tensor(lhs, std::span<double const>{lhs_values});
  upload_tensor(rhs, std::span<double const>{rhs_values});

  uni20::linalg::assign_product(output, lhs, rhs);

  EXPECT_EQ(&output.storage().resources(), &resources);
  EXPECT_EQ(output.rows(), 1);
  EXPECT_EQ(output.cols(), 1);
  EXPECT_EQ(download_tensor(output), (std::vector<double>{42}));
}

TEST_F(CublasExecutionTest, TensorProductDeviceMismatchDeclinesBeforeReplacingOutput)
{
  if (device_count_ < 2) GTEST_SKIP() << "test requires at least two CUDA devices";

  using matrix_type = uni20::CudaMatrix<double>;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .default_device = 0, .streams_per_device = 2});
  auto& device_zero_resources = runtime.device_resources(0);
  auto& device_one_resources = runtime.device_resources(1);
  matrix_type lhs(device_one_resources, 1, 1);
  matrix_type rhs(device_zero_resources, 1, 1);
  matrix_type output(device_zero_resources, 1, 1);
  std::array<double, 1> const lhs_values{6};
  std::array<double, 1> const rhs_values{7};
  std::array<double, 1> const output_values{5};
  upload_tensor(lhs, std::span<double const>{lhs_values});
  upload_tensor(rhs, std::span<double const>{rhs_values});
  upload_tensor(output, std::span<double const>{output_values});

  EXPECT_THROW(uni20::linalg::assign_product(uni20::linalg::CublasBackend{}, output, lhs, rhs),
               uni20::linalg::KernelDispatchError);

  EXPECT_EQ(output.storage().device().ordinal(), 0);
  EXPECT_EQ(download_tensor(output), (std::vector<double>{5}));
}

TEST_F(CublasExecutionTest, ComputesColumnMajorRealGemm)
{
  check_column_major_gemm<float>(device_);
  check_column_major_gemm<double>(device_);
}

TEST_F(CublasExecutionTest, ComputesComplexConjugateTransposeGemm)
{
  check_complex_conjugate_transpose_gemm<uni20::cfloat>(device_);
  check_complex_conjugate_transpose_gemm<uni20::cdouble>(device_);
}

TEST_F(CublasExecutionTest, ZeroInnerExtentScalesOutputWithNullInputs)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 1});
  uni20::cublas::ExecutionPool executions(resources.streams(), 1);
  uni20::cuda::CudaBuffer<double> output(resources, 4);
  std::array<double, 4> const initial{1.0, 2.0, 3.0, 4.0};
  std::array<double, 4> result{};

  auto execution = executions.acquire();
  {
    auto write = output.write_synchronized_with(execution.stream());
    ASSERT_EQ(cudaMemcpyAsync(write.data(), initial.data(), write.size_bytes(), cudaMemcpyHostToDevice,
                              execution.stream().native_handle()),
              cudaSuccess);
  }
  {
    auto write = output.write_synchronized_with(execution.stream());
    uni20::cublas::gemm(execution, 'N', 'N', 2, 2, 0, 1.0, static_cast<double const*>(nullptr), 2,
                        static_cast<double const*>(nullptr), 1, 3.0, write.data(), 2);
  }
  {
    auto read = output.read_synchronized_with(execution.stream());
    ASSERT_EQ(cudaMemcpyAsync(result.data(), read.data(), read.size_bytes(), cudaMemcpyDeviceToHost,
                              execution.stream().native_handle()),
              cudaSuccess);
  }
  execution.stream().synchronize();

  EXPECT_EQ(result, (std::array<double, 4>{3.0, 6.0, 9.0, 12.0}));
}

TEST_F(CublasExecutionTest, TensorGemmDispatchesFromColumnMajorCudaTensorViews)
{
  using matrix_type = uni20::CudaMatrix<double>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 2});
  matrix_type lhs(resources, 2, 3);
  matrix_type rhs(resources, 3, 2);
  matrix_type output(resources, 2, 2);

  std::array<double, 6> const lhs_values{1, 4, 2, 5, 3, 6};
  std::array<double, 6> const rhs_values{7, 9, 11, 8, 10, 12};
  upload_tensor(lhs, std::span<double const>{lhs_values});
  upload_tensor(rhs, std::span<double const>{rhs_values});

  uni20::linalg::gemm(output, 1.0, lhs, rhs, 0.0);

  EXPECT_EQ(download_tensor(output), (std::vector<double>{58, 139, 64, 154}));

  uni20::linalg::gemm(output, 1.0, lhs, rhs, 1.0);

  EXPECT_EQ(download_tensor(output), (std::vector<double>{116, 278, 128, 308}));
}

TEST_F(CublasExecutionTest, OperationDispatchAcceptsTensorViewsRatherThanDeviceMdspans)
{
  using matrix_type = uni20::CudaMatrix<double>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 2});
  matrix_type lhs(resources, 2, 3);
  matrix_type rhs(resources, 3, 2);
  matrix_type output(resources, 2, 2);
  auto output_span = output.device_mdspan();
  auto lhs_span = lhs.device_mdspan();
  auto rhs_span = rhs.device_mdspan();

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CublasBackend{}, uni20::linalg::gemm_op{}, output_span,
                                                 1.0, lhs_span, rhs_span, 0.0),
            uni20::linalg::KernelTypeAcceptance::no);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CublasBackend{}, uni20::linalg::gemm_op{}, output, 1.0,
                                                 lhs, rhs, 0.0),
            uni20::linalg::KernelTypeAcceptance::maybe);
}

TEST_F(CublasExecutionTest, TensorGemmConformanceScalarsAndCanonicalLayouts)
{
  CudaGemmPlatform platform(device_);
  uni20::test::gemm_conformance::check_all_scalar_and_layout_cases(platform);
}

TEST_F(CublasExecutionTest, TensorGemmConformancePaddedLeadingDimensions)
{
  CudaGemmPlatform platform(device_);
  uni20::test::gemm_conformance::check_all_padded_layout_cases(platform);
}

TEST_F(CublasExecutionTest, TensorGemmConformanceConjugatingInputs)
{
  CudaGemmPlatform platform(device_);
  uni20::test::gemm_conformance::check_all_conjugating_input_cases(platform);
}

TEST_F(CublasExecutionTest, TensorGemmSupportsOffsetTransposeView)
{
  CudaGemmPlatform platform(device_);
  auto lhs_storage = platform.make_matrix<double, uni20::ColumnMajor>(4, 4);
  auto rhs_storage = platform.make_matrix<double, uni20::ColumnMajor>(4, 4);
  auto output_storage = platform.make_matrix<double, uni20::ColumnMajor>(4, 4);
  CudaMatrixView lhs(lhs_storage, 2, 2, 3, {3, 1});
  CudaMatrixView rhs(rhs_storage, 4, 3, 2, {1, 4});
  CudaMatrixView output(output_storage, 3, 2, 2, {1, 3});

  std::vector<double> const lhs_values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> const rhs_values{7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  std::vector<double> const output_values{1.0, 2.0, 3.0, 4.0};
  double const alpha = 2.0;
  double const beta = 3.0;
  auto const expected = uni20::test::gemm_conformance::reference_gemm<double>(output_values, lhs_values, rhs_values, 2,
                                                                              3, 2, alpha, beta);

  uni20::test::gemm_conformance::write_logical(platform, lhs, lhs_values);
  uni20::test::gemm_conformance::write_logical(platform, rhs, rhs_values);
  uni20::test::gemm_conformance::expect_each_candidate(platform, output, alpha, lhs, rhs, beta, output_values,
                                                       expected);
}

TEST_F(CublasExecutionTest, TensorGemmSupportsOffsetConjugateTransposeView)
{
  using scalar_type = uni20::complex<double>;
  CudaGemmPlatform platform(device_);
  auto lhs_storage = platform.make_matrix<scalar_type, uni20::ColumnMajor>(4, 4);
  auto rhs_storage = platform.make_matrix<scalar_type, uni20::ColumnMajor>(4, 4);
  auto output_storage = platform.make_matrix<scalar_type, uni20::ColumnMajor>(4, 4);
  CudaMatrixView lhs(lhs_storage, 2, 2, 3, {3, 1});
  CudaMatrixView rhs(rhs_storage, 4, 3, 2, {1, 4});
  CudaMatrixView output(output_storage, 3, 2, 2, {1, 3});

  std::vector<scalar_type> const lhs_values{{1.0, 2.0}, {2.0, -1.0}, {3.0, 1.0}, {4.0, -2.0}, {5.0, 3.0}, {6.0, -1.0}};
  std::vector<scalar_type> const rhs_values{{1.0, -1.0}, {2.0, 1.0}, {3.0, 2.0}, {4.0, -1.0}, {5.0, 1.0}, {6.0, 2.0}};
  std::vector<scalar_type> const output_values{{1.0, 1.0}, {2.0, -1.0}, {3.0, 2.0}, {4.0, -2.0}};
  std::vector<scalar_type> conjugated_lhs_values;
  conjugated_lhs_values.reserve(lhs_values.size());
  for (auto const value : lhs_values)
  {
    conjugated_lhs_values.push_back(uni20::conj(value));
  }
  scalar_type const alpha{2.0, -1.0};
  scalar_type const beta{-1.0, 0.5};
  auto const expected = uni20::test::gemm_conformance::reference_gemm<scalar_type>(output_values, conjugated_lhs_values,
                                                                                   rhs_values, 2, 3, 2, alpha, beta);

  uni20::test::gemm_conformance::write_logical(platform, lhs, lhs_values);
  uni20::test::gemm_conformance::write_logical(platform, rhs, rhs_values);
  auto conjugated_lhs = uni20::conj(lhs);
  uni20::test::gemm_conformance::expect_each_candidate(platform, output, alpha, conjugated_lhs, rhs, beta,
                                                       output_values, expected);
}

TEST_F(CublasExecutionTest, UnsupportedLayoutDeclinesBeforeAcquiringExecutionResources)
{
  CudaGemmPlatform platform(device_);
  auto lhs_storage = platform.make_matrix<double, uni20::ColumnMajor>(4, 4);
  auto rhs = platform.make_matrix<double, uni20::ColumnMajor>(2, 2);
  auto output = platform.make_matrix<double, uni20::ColumnMajor>(2, 2);
  CudaMatrixView lhs(lhs_storage, 0, 2, 2, {2, 5});
  std::vector<double> const initial_output{1.0, 2.0, 3.0, 4.0};
  uni20::test::gemm_conformance::write_logical(platform, output, initial_output);

  auto& executions = uni20::cublas::execution_pool(platform.resources());
  platform.resources().streams().synchronize();
  auto const idle_handles = executions.idle_handle_count();
  auto const idle_streams = platform.resources().streams().idle_stream_count();
  auto const leased_streams = platform.resources().streams().leased_stream_count();

  EXPECT_EQ(uni20::linalg::detail::cublas_backend::try_gemm(output.device_mdspan(), 1.0, lhs.device_mdspan(),
                                                            rhs.device_mdspan(), 0.0),
            uni20::linalg::KernelAttempt::unsupported_layout);
  EXPECT_EQ(executions.idle_handle_count(), idle_handles);
  EXPECT_EQ(platform.resources().streams().idle_stream_count(), idle_streams);
  EXPECT_EQ(platform.resources().streams().leased_stream_count(), leased_streams);
  EXPECT_EQ(uni20::test::gemm_conformance::read_logical(platform, output), initial_output);
}

TEST_F(CublasExecutionTest, TensorGemmRejectsExactOutputInputAlias)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::CudaMatrix<double> output(resources, 2, 2);
        uni20::CudaMatrix<double> rhs(resources, 2, 2);
        uni20::linalg::gemm(output, 1.0, output, rhs, 0.0);
      },
      "must not share a CUDA buffer");
}

TEST_F(CublasExecutionTest, TensorGemmRejectsPartialOutputInputAlias)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::CudaMatrix<double> shared_storage(resources, 4, 4);
        uni20::CudaMatrix<double> rhs(resources, 2, 2);
        CudaMatrixView output(shared_storage, 0, 2, 2, {1, 3});
        CudaMatrixView lhs(shared_storage, 1, 2, 2, {1, 3});
        uni20::linalg::gemm(output, 1.0, lhs, rhs, 0.0);
      },
      "must not share a CUDA buffer");
}

TEST_F(CublasExecutionTest, TensorGemmRejectsCrossDeviceOperands)
{
  if (device_count_ < 2) GTEST_SKIP() << "test requires at least two CUDA devices";

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources device0({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::cuda::DeviceResources device1({.device = uni20::cuda::Device::get(1), .stream_count = 1});
        uni20::CudaMatrix<double> output(device0, 2, 2);
        uni20::CudaMatrix<double> lhs(device0, 2, 2);
        uni20::CudaMatrix<double> rhs(device1, 2, 2);
        uni20::linalg::gemm(output, 1.0, lhs, rhs, 0.0);
      },
      "operands must use one CUDA device");
}

TEST_F(CublasExecutionTest, TensorGemmNormalizesRowMajorCudaOutput)
{
  using matrix_type = uni20::CudaMatrix<double, uni20::RowMajor>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 2});
  matrix_type lhs(resources, 2, 3);
  matrix_type rhs(resources, 3, 2);
  matrix_type output(resources, 2, 2);

  std::array<double, 6> const lhs_values{1, 2, 3, 4, 5, 6};
  std::array<double, 6> const rhs_values{7, 8, 9, 10, 11, 12};
  upload_tensor(lhs, std::span<double const>{lhs_values});
  upload_tensor(rhs, std::span<double const>{rhs_values});

  uni20::linalg::gemm(output, 1.0, lhs, rhs, 0.0);

  EXPECT_EQ(download_tensor(output), (std::vector<double>{58, 64, 139, 154}));
}

TEST_F(CublasExecutionTest, TensorGemmLowersConjugatingCudaAccessor)
{
  using scalar_type = uni20::cdouble;
  using row_matrix_type = uni20::CudaMatrix<scalar_type, uni20::RowMajor>;
  using column_matrix_type = uni20::CudaMatrix<scalar_type>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 2});
  row_matrix_type lhs(resources, 1, 2);
  column_matrix_type rhs(resources, 2, 1);
  column_matrix_type output(resources, 1, 1);

  std::array<scalar_type, 2> const lhs_values{scalar_type{1, 2}, scalar_type{3, -1}};
  std::array<scalar_type, 2> const rhs_values{scalar_type{2, -1}, scalar_type{-1, 4}};
  upload_tensor(lhs, std::span<scalar_type const>{lhs_values});
  upload_tensor(rhs, std::span<scalar_type const>{rhs_values});

  auto conjugated_lhs = uni20::conj(lhs);
  uni20::linalg::gemm(output, scalar_type{1}, conjugated_lhs, rhs, scalar_type{});

  EXPECT_EQ(download_tensor(output), (std::vector<scalar_type>{scalar_type{-7, 6}}));
}

TEST_F(CublasExecutionTest, TensorGemmEmptyOutputSucceedsBeforeOperandStaging)
{
  using matrix_type = uni20::CudaMatrix<double, uni20::RowMajor>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 1});
  matrix_type lhs(resources, 0, 3);
  matrix_type rhs(resources, 3, 2);
  matrix_type output(resources, 0, 2);

  EXPECT_EQ(uni20::linalg::detail::cublas_backend::try_gemm(output.device_mdspan(), 1.0, lhs.device_mdspan(),
                                                            rhs.device_mdspan(), 0.0),
            uni20::linalg::KernelAttempt::success);
}

TEST_F(CublasExecutionTest, TensorGemmScalesOutputForZeroInnerExtent)
{
  using matrix_type = uni20::CudaMatrix<double, uni20::RowMajor>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 1});
  matrix_type lhs(resources, 2, 0);
  matrix_type rhs(resources, 0, 2);
  matrix_type output(resources, 2, 2);
  std::array<double, 4> const output_values{1.0, 2.0, 3.0, 4.0};
  upload_tensor(output, std::span<double const>{output_values});

  uni20::linalg::gemm(output, 1.0, lhs, rhs, 3.0);
  EXPECT_EQ(download_tensor(output), (std::vector<double>{3.0, 6.0, 9.0, 12.0}));
}

} // namespace
