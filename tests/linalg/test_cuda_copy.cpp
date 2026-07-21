#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <concepts>
#include <stdexcept>
#include <thread>
#include <utility>

namespace
{

using host_matrix_type = uni20::Tensor<double, 2>;
using cuda_matrix_type = uni20::CudaTensor<double, 2>;
using row_major_host_matrix_type = uni20::RowMajorTensor<double, 2>;

using namespace std::chrono_literals;

struct CopyGate
{
    std::atomic<bool> open = false;
    std::atomic<bool> entered = false;
};

void CUDART_CB wait_for_copy_gate(void* raw_gate)
{
  auto& gate = *static_cast<CopyGate*>(raw_gate);
  gate.entered.store(true, std::memory_order_release);
  while (!gate.open.load(std::memory_order_acquire))
  {
    std::this_thread::yield();
  }
}

void CUDART_CB set_copy_flag(void* raw_flag)
{
  static_cast<std::atomic<bool>*>(raw_flag)->store(true, std::memory_order_release);
}

bool wait_until(std::atomic<bool> const& flag)
{
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::yield();
  }
  return flag.load(std::memory_order_acquire);
}

class ErrorModeGuard {
  public:
    ErrorModeGuard() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ~ErrorModeGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};

class CudaCopyTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess)
      {
        GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      }
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
    }

    static host_matrix_type make_matrix()
    {
      host_matrix_type matrix(2, 3);
      matrix[0, 0] = 1.0;
      matrix[1, 0] = 2.0;
      matrix[0, 1] = 3.0;
      matrix[1, 1] = 4.0;
      matrix[0, 2] = 5.0;
      matrix[1, 2] = 6.0;
      return matrix;
    }

    static void expect_matrix(host_matrix_type const& matrix)
    {
      ASSERT_EQ(matrix.rows(), 2);
      ASSERT_EQ(matrix.cols(), 3);
      EXPECT_DOUBLE_EQ((matrix[0, 0]), 1.0);
      EXPECT_DOUBLE_EQ((matrix[1, 0]), 2.0);
      EXPECT_DOUBLE_EQ((matrix[0, 1]), 3.0);
      EXPECT_DOUBLE_EQ((matrix[1, 1]), 4.0);
      EXPECT_DOUBLE_EQ((matrix[0, 2]), 5.0);
      EXPECT_DOUBLE_EQ((matrix[1, 2]), 6.0);
    }

    int device_count_ = 0;
};

TEST_F(CudaCopyTest, PageableHostRoundTripUsesExplicitTransferFunctions)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto host = make_matrix();

  auto device = uni20::to_device(host, 0);
  auto result = uni20::to_host(device);

  EXPECT_EQ(device.storage().device().ordinal(), 0);
  expect_matrix(result);
}

TEST_F(CudaCopyTest, FixedOutputPageableTransfersResizeAndRoundTrip)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = make_matrix();
  cuda_matrix_type device(runtime.device_resources(0), 1, 1);
  host_matrix_type result(1, 1);

  uni20::copy(device, source);
  uni20::copy(result, device);

  EXPECT_EQ(device.rows(), source.rows());
  EXPECT_EQ(device.cols(), source.cols());
  expect_matrix(result);
}

TEST_F(CudaCopyTest, SameDeviceCopyUsesCudaReferenceFallback)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type destination(runtime.device_resources(0), 2, 3);

  uni20::copy(destination, source);

  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, SameDeviceCopyOrdersPendingSourceAndDestinationReuse)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 3});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type destination(runtime.device_resources(0), 2, 3);
  CopyGate gate;

  {
    auto stream = runtime.device_resources(0).streams().acquire();
    auto source_predecessor = source.storage().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_copy_gate, &gate),
                       "cudaLaunchHostFunc same-device source predecessor", 0);
  }

  uni20::copy(destination, source);

  std::atomic<bool> destination_reader_completed = false;
  {
    auto stream = runtime.device_resources(0).streams().acquire();
    auto destination_reader = destination.storage().read_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_copy_flag, &destination_reader_completed),
                       "cudaLaunchHostFunc same-device destination reader", 0);
  }

  bool const predecessor_entered = wait_until(gate.entered);
  if (!predecessor_entered) gate.open.store(true, std::memory_order_release);
  ASSERT_TRUE(predecessor_entered);
  EXPECT_FALSE(destination_reader_completed.load(std::memory_order_acquire));

  gate.open.store(true, std::memory_order_release);
  runtime.device_resources(0).streams().synchronize();
  EXPECT_TRUE(destination_reader_completed.load(std::memory_order_acquire));
  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, RowMajorRoundTripPreservesLogicalOrder)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  row_major_host_matrix_type host(2, 3);
  host[0, 0] = 1.0;
  host[0, 1] = 2.0;
  host[0, 2] = 3.0;
  host[1, 0] = 4.0;
  host[1, 1] = 5.0;
  host[1, 2] = 6.0;

  auto result = uni20::to_host(uni20::to_device(host, 0));

  static_assert(std::same_as<typename decltype(result)::layout_type, stdex::layout_right>);
  EXPECT_DOUBLE_EQ((result[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((result[0, 2]), 3.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 4.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 5.0);
  EXPECT_DOUBLE_EQ((result[1, 2]), 6.0);
}

TEST_F(CudaCopyTest, ZeroExtentRoundTripIsAnEmptySuccessfulTransfer)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  host_matrix_type host(3, 0);

  auto device = uni20::to_device(host, 0);
  auto result = uni20::to_host(device);

  EXPECT_EQ(device.rows(), 3);
  EXPECT_EQ(device.cols(), 0);
  EXPECT_EQ(result.rows(), 3);
  EXPECT_EQ(result.cols(), 0);
}

TEST_F(CudaCopyTest, PeerCopyPreservesValues)
{
  if (device_count_ < 2) GTEST_SKIP() << "peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);

  auto destination = uni20::to_device(source, 1);

  EXPECT_EQ(destination.storage().device().ordinal(), 1);
  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, PeerCopyPreservesSourceAndDestinationLedgers)
{
  if (device_count_ < 2) GTEST_SKIP() << "peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 3});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type destination(runtime.device_resources(1), 2, 3);
  CopyGate gate;

  {
    auto stream = runtime.device_resources(1).streams().acquire();
    auto destination_predecessor = destination.storage().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(1);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_copy_gate, &gate),
                       "cudaLaunchHostFunc peer destination predecessor", 1);
  }

  uni20::copy(destination, source);

  std::atomic<bool> source_writer_completed = false;
  {
    auto stream = runtime.device_resources(0).streams().acquire();
    auto source_writer = source.storage().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_copy_flag, &source_writer_completed),
                       "cudaLaunchHostFunc peer source writer", 0);
  }

  std::atomic<bool> destination_reader_completed = false;
  {
    auto stream = runtime.device_resources(1).streams().acquire();
    auto destination_reader = destination.storage().read_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(1);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_copy_flag, &destination_reader_completed),
                       "cudaLaunchHostFunc peer destination reader", 1);
  }

  bool const predecessor_entered = wait_until(gate.entered);
  if (!predecessor_entered) gate.open.store(true, std::memory_order_release);
  ASSERT_TRUE(predecessor_entered);
  EXPECT_FALSE(source_writer_completed.load(std::memory_order_acquire));
  EXPECT_FALSE(destination_reader_completed.load(std::memory_order_acquire));

  gate.open.store(true, std::memory_order_release);
  runtime.device_resources(0).streams().synchronize();
  runtime.device_resources(1).streams().synchronize();
  EXPECT_TRUE(source_writer_completed.load(std::memory_order_acquire));
  EXPECT_TRUE(destination_reader_completed.load(std::memory_order_acquire));
  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, AsyncCopyConstructsOutputAndCarriesDeviceCompletion)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<cuda_matrix_type> input = uni20::to_device(make_matrix(), 0);
  uni20::async::Async<cuda_matrix_type> output;

  uni20::copy(output, input);

  auto const& device_result = output.get_wait(scheduler);
  expect_matrix(uni20::to_host(device_result));
}

TEST_F(CudaCopyTest, AsyncPeerCopyRunsOnDestinationDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "async peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<cuda_matrix_type> input = uni20::to_device(make_matrix(), 0);
  uni20::async::Async<cuda_matrix_type> output = cuda_matrix_type(runtime.device_resources(1), 2, 3);

  uni20::copy(output, input);

  auto const& device_result = output.get_wait(scheduler);
  EXPECT_EQ(device_result.storage().device().ordinal(), 1);
  expect_matrix(uni20::to_host(device_result));
}

TEST_F(CudaCopyTest, AsyncCopyRejectsOutputInputEpochAliasing)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::Async<cuda_matrix_type> tensor = uni20::to_device(make_matrix(), 0);
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::copy(tensor, tensor), std::runtime_error);
}

} // namespace
