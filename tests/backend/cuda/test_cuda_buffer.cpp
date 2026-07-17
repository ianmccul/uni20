#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/backend/cuda/cuda_error.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace
{

using namespace std::chrono_literals;

struct BufferGate
{
    std::atomic<bool> open = false;
    std::atomic<bool> entered = false;
};

void CUDART_CB wait_for_buffer_gate(void* raw_gate)
{
  auto& gate = *static_cast<BufferGate*>(raw_gate);
  gate.entered.store(true, std::memory_order_release);
  while (!gate.open.load(std::memory_order_acquire))
  {
    std::this_thread::yield();
  }
}

void CUDART_CB set_buffer_flag(void* raw_flag)
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

bool wait_until_ready(uni20::cuda::Completion const& completion)
{
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (!completion.ready() && std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::yield();
  }
  return completion.ready();
}

class CudaBufferTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      int device_count = 0;
      cudaError_t const status = cudaGetDeviceCount(&device_count);
      if (status != cudaSuccess)
      {
        GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      }
      if (device_count == 0)
      {
        GTEST_SKIP() << "no CUDA devices are available";
      }
    }
};

} // namespace

static_assert(!std::is_copy_constructible_v<uni20::cuda::Buffer<>>);
static_assert(std::is_move_constructible_v<uni20::cuda::Buffer<>>);
static_assert(!std::is_copy_constructible_v<uni20::cuda::ReadBuffer<std::byte>>);
static_assert(std::is_move_constructible_v<uni20::cuda::ReadBuffer<std::byte>>);
static_assert(!std::is_copy_constructible_v<uni20::cuda::WriteBuffer<std::byte>>);
static_assert(std::is_move_constructible_v<uni20::cuda::WriteBuffer<std::byte>>);

TEST_F(CudaBufferTest, OwnsAndMovesDeviceAllocation)
{
  uni20::cuda::DeviceContext context({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  uni20::cuda::Buffer<> source(context, 4096);
  std::byte* address = nullptr;
  {
    auto stream = context.streams().acquire();
    auto access = source.write(stream);
    address = access.data();
  }
  context.streams().synchronize();

  ASSERT_NE(address, nullptr);
  EXPECT_EQ(source.size(), 4096U);
  EXPECT_EQ(source.size_bytes(), 4096U);
  EXPECT_EQ(source.device(), context.device());

  uni20::cuda::Buffer<> destination(std::move(source));
  {
    auto stream = context.streams().acquire();
    auto access = destination.write(stream);
    EXPECT_EQ(access.data(), address);
  }
  EXPECT_EQ(destination.size_bytes(), 4096U);
  EXPECT_EQ(source.size(), 0U);
  EXPECT_TRUE(source.empty());
}

TEST_F(CudaBufferTest, RepeatedWriteWaitsForPreviousWriter)
{
  uni20::cuda::DeviceContext context({.device = uni20::cuda::Device::get(0), .stream_count = 2});
  uni20::cuda::Buffer<> buffer(context, 4096);

  BufferGate gate;
  {
    auto stream = context.streams().acquire();
    auto producer = buffer.write(stream);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_buffer_gate, &gate),
                       "cudaLaunchHostFunc buffer producer gate", 0);
  }

  std::atomic<bool> consumer_completed = false;
  {
    auto stream = context.streams().acquire();
    auto consumer = buffer.write(stream);
    uni20::cuda::check(cudaMemsetAsync(consumer.data(), 0, consumer.size_bytes(), stream.native_handle()),
                       "cudaMemsetAsync dependent buffer", 0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_buffer_flag, &consumer_completed),
                       "cudaLaunchHostFunc dependent buffer flag", 0);
  }

  bool const producer_entered = wait_until(gate.entered);
  if (!producer_entered)
  {
    gate.open.store(true, std::memory_order_release);
  }
  ASSERT_TRUE(producer_entered);
  EXPECT_FALSE(consumer_completed.load(std::memory_order_acquire));

  gate.open.store(true, std::memory_order_release);
  context.streams().synchronize();
  EXPECT_TRUE(consumer_completed.load(std::memory_order_acquire));
}

TEST_F(CudaBufferTest, IndependentBuffersDoNotAcquireAFalseDependency)
{
  uni20::cuda::DeviceContext context({.device = uni20::cuda::Device::get(0), .stream_count = 2});
  uni20::cuda::Buffer<> blocked_buffer(context, 4096);
  uni20::cuda::Buffer<> independent_buffer(context, 4096);

  BufferGate gate;
  {
    auto stream = context.streams().acquire();
    auto blocked = blocked_buffer.write(stream);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_buffer_gate, &gate),
                       "cudaLaunchHostFunc independent-buffer gate", 0);
  }

  uni20::cuda::Completion independent_completion;
  {
    auto stream = context.streams().acquire();
    auto independent = independent_buffer.write(stream);
    uni20::cuda::check(cudaMemsetAsync(independent.data(), 0, independent.size_bytes(), stream.native_handle()),
                       "cudaMemsetAsync independent buffer", 0);
    independent_completion = stream.record_completion();
  }

  bool const blocked_entered = wait_until(gate.entered);
  bool const independent_ready = blocked_entered && wait_until_ready(independent_completion);
  gate.open.store(true, std::memory_order_release);

  ASSERT_TRUE(blocked_entered);
  EXPECT_TRUE(independent_ready);
  context.streams().synchronize();
}

TEST_F(CudaBufferTest, ConcurrentReadersOverlapAndWriterWaitsForOutstandingReaders)
{
  uni20::cuda::DeviceContext context({.device = uni20::cuda::Device::get(0), .stream_count = 3});
  uni20::cuda::Buffer<> source(context, 4096);
  uni20::cuda::Buffer<> first_output(context, 4096);
  uni20::cuda::Buffer<> second_output(context, 4096);

  BufferGate gate;
  {
    auto stream = context.streams().acquire();
    auto source_read = source.read(stream);
    auto output_write = first_output.write(stream);
    uni20::cuda::check(cudaMemcpyAsync(output_write.data(), source_read.data(), source_read.size_bytes(),
                                       cudaMemcpyDeviceToDevice, stream.native_handle()),
                       "cudaMemcpyAsync first concurrent reader", 0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_buffer_gate, &gate),
                       "cudaLaunchHostFunc concurrent-reader gate", 0);
  }

  uni20::cuda::Completion second_completion;
  {
    auto stream = context.streams().acquire();
    auto source_read = source.read(stream);
    auto output_write = second_output.write(stream);
    uni20::cuda::check(cudaMemcpyAsync(output_write.data(), source_read.data(), source_read.size_bytes(),
                                       cudaMemcpyDeviceToDevice, stream.native_handle()),
                       "cudaMemcpyAsync second concurrent reader", 0);
    second_completion = stream.record_completion();
  }

  bool const first_entered = wait_until(gate.entered);
  bool const second_completed_independently = first_entered && wait_until_ready(second_completion);

  uni20::cuda::Completion writer_completion;
  {
    auto stream = context.streams().acquire();
    auto writer = source.write(stream);
    uni20::cuda::check(cudaMemsetAsync(writer.data(), 0, writer.size_bytes(), stream.native_handle()),
                       "cudaMemsetAsync after concurrent readers", 0);
    writer_completion = stream.record_completion();
  }

  if (!first_entered)
  {
    gate.open.store(true, std::memory_order_release);
  }
  ASSERT_TRUE(first_entered);
  EXPECT_TRUE(second_completed_independently);
  EXPECT_FALSE(writer_completion.ready());

  gate.open.store(true, std::memory_order_release);
  writer_completion.synchronize();
  EXPECT_TRUE(writer_completion.ready());
}

TEST_F(CudaBufferTest, ScopedAccessSynchronizesDuringStackUnwinding)
{
  uni20::cuda::DeviceContext context({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  uni20::cuda::Buffer<> buffer(context, 4096);

  EXPECT_THROW(
      {
        auto stream = context.streams().acquire();
        auto access = buffer.write(stream);
        uni20::cuda::check(cudaMemsetAsync(access.data(), 0, access.size_bytes(), stream.native_handle()),
                           "cudaMemsetAsync exception cleanup", 0);
        throw std::runtime_error("test submission failure");
      },
      std::runtime_error);

  context.streams().synchronize();
  EXPECT_EQ(context.streams().idle_stream_count(), 1U);
  {
    auto stream = context.streams().acquire();
    auto access = buffer.write(stream);
  }
  context.streams().synchronize();
}
