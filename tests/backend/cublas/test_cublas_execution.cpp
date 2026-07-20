#include <uni20/async/cuda_task.hpp>
#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cublas/cublas_error_presentation.hpp>
#include <uni20/backend/cublas/gemm.hpp>
#include <uni20/backend/cublas/task_awaiters.hpp>
#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/common/presentation.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/backends/cublas/gemm.hpp>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/tensor.hpp>

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
    uni20::linalg::cublas::gemm(execution, output_matrix, lhs_matrix, rhs_matrix, Scalar{1}, Scalar{});
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

TEST_F(CublasExecutionTest, TensorGemmDispatchesFromColumnMajorCudaMdspans)
{
  using matrix_type = uni20::CudaAsyncMatrix<double>;
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

TEST_F(CublasExecutionTest, TensorGemmNormalizesRowMajorCudaOutput)
{
  using matrix_type = uni20::CudaAsyncMatrix<double, uni20::RowMajor>;
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
  using row_matrix_type = uni20::CudaAsyncMatrix<scalar_type, uni20::RowMajor>;
  using column_matrix_type = uni20::CudaAsyncMatrix<scalar_type>;
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
  using matrix_type = uni20::CudaAsyncMatrix<double, uni20::RowMajor>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 1});
  matrix_type lhs(resources, 0, 3);
  matrix_type rhs(resources, 3, 2);
  matrix_type output(resources, 0, 2);

  EXPECT_EQ(uni20::linalg::cublas::try_gemm(output.mdspan(), 1.0, lhs.mdspan(), rhs.mdspan(), 0.0),
            uni20::linalg::KernelAttempt::success);
}

TEST_F(CublasExecutionTest, TensorGemmDeclinesZeroInnerExtentBeforeOperandStaging)
{
  using matrix_type = uni20::CudaAsyncMatrix<double, uni20::RowMajor>;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(device_), .stream_count = 1});
  matrix_type lhs(resources, 2, 0);
  matrix_type rhs(resources, 0, 2);
  matrix_type output(resources, 2, 2);
  std::array<double, 4> const output_values{1.0, 2.0, 3.0, 4.0};
  upload_tensor(output, std::span<double const>{output_values});

  EXPECT_EQ(uni20::linalg::cublas::try_gemm(output.mdspan(), 1.0, lhs.mdspan(), rhs.mdspan(), 3.0),
            uni20::linalg::KernelAttempt::unsupported_instance);
  EXPECT_EQ(download_tensor(output), (std::vector<double>{1.0, 2.0, 3.0, 4.0}));
}

} // namespace
