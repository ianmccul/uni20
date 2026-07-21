#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/backend/cuda/cuda_error.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
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
      GTEST_FLAG_SET(death_test_style, "threadsafe");
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

static_assert(!std::is_copy_constructible_v<uni20::cuda::CudaBuffer<>>);
static_assert(std::is_move_constructible_v<uni20::cuda::CudaBuffer<>>);
static_assert(!std::is_copy_constructible_v<uni20::cuda::ReadAccess<std::byte>>);
static_assert(std::is_move_constructible_v<uni20::cuda::ReadAccess<std::byte>>);
static_assert(std::is_move_assignable_v<uni20::cuda::ReadAccess<std::byte>>);
static_assert(!std::is_copy_constructible_v<uni20::cuda::WriteAccess<std::byte>>);
static_assert(std::is_move_constructible_v<uni20::cuda::WriteAccess<std::byte>>);
static_assert(std::is_move_assignable_v<uni20::cuda::WriteAccess<std::byte>>);
static_assert(!std::is_copy_constructible_v<uni20::cuda::BlockingReadAccess<std::byte>>);
static_assert(std::is_move_constructible_v<uni20::cuda::BlockingReadAccess<std::byte>>);
static_assert(!std::is_copy_constructible_v<uni20::cuda::BlockingWriteAccess<std::byte>>);
static_assert(std::is_move_constructible_v<uni20::cuda::BlockingWriteAccess<std::byte>>);

TEST_F(CudaBufferTest, OwnsAndMovesDeviceAllocation)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  uni20::cuda::CudaBuffer<> source(resources, 4096);
  std::byte* address = nullptr;
  {
    auto stream = resources.streams().acquire();
    auto access = source.write_synchronized_with(stream);
    address = access.data();
  }
  resources.streams().synchronize();

  ASSERT_NE(address, nullptr);
  EXPECT_EQ(source.size(), 4096U);
  EXPECT_EQ(source.size_bytes(), 4096U);
  EXPECT_EQ(source.device(), resources.device());

  uni20::cuda::CudaBuffer<> destination(std::move(source));
  {
    auto stream = resources.streams().acquire();
    auto access = destination.write_synchronized_with(stream);
    EXPECT_EQ(access.data(), address);
  }
  EXPECT_EQ(destination.size_bytes(), 4096U);
  EXPECT_EQ(source.size(), 0U);
  EXPECT_TRUE(source.empty());
}

TEST_F(CudaBufferTest, RepeatedWriteWaitsForPreviousWriter)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 2});
  uni20::cuda::CudaBuffer<> buffer(resources, 4096);

  BufferGate gate;
  {
    auto stream = resources.streams().acquire();
    auto producer = buffer.write_synchronized_with(stream);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_buffer_gate, &gate),
                       "cudaLaunchHostFunc buffer producer gate", 0);
  }

  std::atomic<bool> consumer_completed = false;
  {
    auto stream = resources.streams().acquire();
    auto consumer = buffer.write_synchronized_with(stream);
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
  resources.streams().synchronize();
  EXPECT_TRUE(consumer_completed.load(std::memory_order_acquire));
}

TEST_F(CudaBufferTest, IndependentBuffersDoNotAcquireAFalseDependency)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 2});
  uni20::cuda::CudaBuffer<> blocked_buffer(resources, 4096);
  uni20::cuda::CudaBuffer<> independent_buffer(resources, 4096);

  BufferGate gate;
  {
    auto stream = resources.streams().acquire();
    auto blocked = blocked_buffer.write_synchronized_with(stream);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_buffer_gate, &gate),
                       "cudaLaunchHostFunc independent-buffer gate", 0);
  }

  uni20::cuda::Completion independent_completion;
  {
    auto stream = resources.streams().acquire();
    auto independent = independent_buffer.write_synchronized_with(stream);
    uni20::cuda::check(cudaMemsetAsync(independent.data(), 0, independent.size_bytes(), stream.native_handle()),
                       "cudaMemsetAsync independent buffer", 0);
    independent_completion = stream.record_completion();
  }

  bool const blocked_entered = wait_until(gate.entered);
  bool const independent_ready = blocked_entered && wait_until_ready(independent_completion);
  gate.open.store(true, std::memory_order_release);

  ASSERT_TRUE(blocked_entered);
  EXPECT_TRUE(independent_ready);
  resources.streams().synchronize();
}

TEST_F(CudaBufferTest, ConcurrentReadersOverlapAndWriterWaitsForOutstandingReaders)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 3});
  uni20::cuda::CudaBuffer<> source(resources, 4096);
  uni20::cuda::CudaBuffer<> first_output(resources, 4096);
  uni20::cuda::CudaBuffer<> second_output(resources, 4096);

  BufferGate gate;
  {
    auto stream = resources.streams().acquire();
    auto source_read = source.read_synchronized_with(stream);
    auto output_write = first_output.write_synchronized_with(stream);
    uni20::cuda::check(cudaMemcpyAsync(output_write.data(), source_read.data(), source_read.size_bytes(),
                                       cudaMemcpyDeviceToDevice, stream.native_handle()),
                       "cudaMemcpyAsync first concurrent reader", 0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_buffer_gate, &gate),
                       "cudaLaunchHostFunc concurrent-reader gate", 0);
  }

  uni20::cuda::Completion second_completion;
  {
    auto stream = resources.streams().acquire();
    auto source_read = source.read_synchronized_with(stream);
    auto output_write = second_output.write_synchronized_with(stream);
    uni20::cuda::check(cudaMemcpyAsync(output_write.data(), source_read.data(), source_read.size_bytes(),
                                       cudaMemcpyDeviceToDevice, stream.native_handle()),
                       "cudaMemcpyAsync second concurrent reader", 0);
    second_completion = stream.record_completion();
  }

  bool const first_entered = wait_until(gate.entered);
  bool const second_completed_independently = first_entered && wait_until_ready(second_completion);

  uni20::cuda::Completion writer_completion;
  {
    auto stream = resources.streams().acquire();
    auto writer = source.write_synchronized_with(stream);
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
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  uni20::cuda::CudaBuffer<> buffer(resources, 4096);

  EXPECT_THROW(
      {
        auto stream = resources.streams().acquire();
        auto access = buffer.write_synchronized_with(stream);
        uni20::cuda::check(cudaMemsetAsync(access.data(), 0, access.size_bytes(), stream.native_handle()),
                           "cudaMemsetAsync exception cleanup", 0);
        throw std::runtime_error("test submission failure");
      },
      std::runtime_error);

  resources.streams().synchronize();
  EXPECT_EQ(resources.streams().idle_stream_count(), 1U);
  {
    auto stream = resources.streams().acquire();
    auto access = buffer.write_synchronized_with(stream);
  }
  resources.streams().synchronize();
}

TEST_F(CudaBufferTest, BlockingAccessSupportsSynchronousHostTransfers)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  uni20::cuda::CudaBuffer<int> buffer(resources, 1);
  int const expected = 42;
  {
    auto write = buffer.blocking_write_access();
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaMemcpy(write.data(), &expected, sizeof(expected), cudaMemcpyHostToDevice),
                       "cudaMemcpy blocking buffer write", 0);
  }

  int result = 0;
  {
    auto read = buffer.blocking_read_access();
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaMemcpy(&result, read.data(), sizeof(result), cudaMemcpyDeviceToHost),
                       "cudaMemcpy blocking buffer read", 0);
  }

  EXPECT_EQ(result, expected);
}

TEST_F(CudaBufferTest, StreamAccessWhileBlockingWriterIsLiveFails)
{
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::cuda::CudaBuffer<> buffer(resources, 4096);
        auto writer = buffer.blocking_write_access();
        auto stream = resources.streams().acquire();
        (void)buffer.read_synchronized_with(stream);
      },
      "cannot acquire CUDA read access while a write access is live");
}

TEST_F(CudaBufferTest, BlockingWriteWhileStreamReaderIsLiveFails)
{
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::cuda::CudaBuffer<> buffer(resources, 4096);
        auto stream = resources.streams().acquire();
        auto reader = buffer.read_synchronized_with(stream);
        (void)buffer.blocking_write_access();
      },
      "cannot acquire blocking CUDA write access while another access is live");
}

TEST_F(CudaBufferTest, MultipleReadAccessesRemainValidUntilExplicitRelease)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 2});
  uni20::cuda::CudaBuffer<> buffer(resources, 4096);
  auto first_stream = resources.streams().acquire();
  auto second_stream = resources.streams().acquire();

  auto first = buffer.read_synchronized_with(first_stream);
  auto second = buffer.read_synchronized_with(second_stream);

  first.release();
  first.release();
  second.release();

  auto writer = buffer.write_synchronized_with(first_stream);
  writer.release();
  writer.release();
  resources.streams().synchronize();
}

TEST_F(CudaBufferTest, MovingAccessTransfersExactlyOneLiveToken)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 2});
  uni20::cuda::CudaBuffer<> buffer(resources, 4096);
  auto first_stream = resources.streams().acquire();
  auto second_stream = resources.streams().acquire();

  auto first = buffer.read_synchronized_with(first_stream);
  auto second = buffer.read_synchronized_with(second_stream);
  auto moved = std::move(first);
  first.release();

  moved = std::move(second);
  second.release();
  moved.release();

  auto writer = buffer.write_synchronized_with(first_stream);
  writer.release();
  resources.streams().synchronize();
}

TEST_F(CudaBufferTest, ReadAccessWhileWriterIsLiveFails)
{
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::cuda::CudaBuffer<> buffer(resources, 4096);
        auto stream = resources.streams().acquire();
        auto writer = buffer.write_synchronized_with(stream);
        (void)buffer.read_synchronized_with(stream);
      },
      "cannot acquire CUDA read access while a write access is live");
}

TEST_F(CudaBufferTest, AccessSynchronizedWithEmptyStreamFails)
{
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::cuda::CudaBuffer<> buffer(resources, 4096);
        (void)buffer.read_synchronized_with(uni20::cuda::Stream{});
      },
      "cannot synchronize CUDA buffer access with an empty stream");
}

TEST_F(CudaBufferTest, WriteAccessWhileReaderIsLiveFails)
{
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::cuda::CudaBuffer<> buffer(resources, 4096);
        auto stream = resources.streams().acquire();
        auto reader = buffer.read_synchronized_with(stream);
        (void)buffer.write_synchronized_with(stream);
      },
      "cannot acquire CUDA write access while another access is live");
}

TEST_F(CudaBufferTest, WriteAccessWhileWriterIsLiveFails)
{
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::cuda::CudaBuffer<> buffer(resources, 4096);
        auto stream = resources.streams().acquire();
        auto writer = buffer.write_synchronized_with(stream);
        (void)buffer.write_synchronized_with(stream);
      },
      "cannot acquire CUDA write access while another access is live");
}

TEST_F(CudaBufferTest, MovingBufferWhileAccessIsLiveFails)
{
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::cuda::CudaBuffer<> buffer(resources, 4096);
        auto stream = resources.streams().acquire();
        auto reader = buffer.read_synchronized_with(stream);
        uni20::cuda::CudaBuffer<> moved(std::move(buffer));
      },
      "cannot move a CUDA buffer while access guards are live");
}

TEST_F(CudaBufferTest, DestroyingBufferWhileAccessIsLiveFails)
{
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        auto buffer = std::make_unique<uni20::cuda::CudaBuffer<>>(resources, 4096);
        auto stream = resources.streams().acquire();
        auto reader = buffer->read_synchronized_with(stream);
        buffer.reset();
      },
      "cannot destroy or reset a CUDA buffer while access guards are live");
}

TEST_F(CudaBufferTest, SynchronizingBufferWhileAccessIsLiveFails)
{
  EXPECT_DEATH(
      {
        uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
        uni20::cuda::CudaBuffer<> buffer(resources, 4096);
        auto stream = resources.streams().acquire();
        auto reader = buffer.read_synchronized_with(stream);
        buffer.synchronize();
      },
      "cannot synchronize a CUDA buffer while access guards are live");
}

TEST_F(CudaBufferTest, ForeignDeviceStreamCarriesBufferCompletionsInBothDirections)
{
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2)
  {
    GTEST_SKIP() << "test requires at least two CUDA devices";
  }

  uni20::cuda::DeviceResources source_context({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  uni20::cuda::DeviceResources foreign_context({.device = uni20::cuda::Device::get(1), .stream_count = 1});
  uni20::cuda::CudaBuffer<> buffer(source_context, 4096);
  auto source_stream = source_context.streams().acquire();
  auto foreign_stream = foreign_context.streams().acquire();

  BufferGate gate;
  std::atomic<bool> foreign_read_completed = false;
  std::atomic<bool> successor_completed = false;

  auto producer = buffer.write_synchronized_with(source_stream);
  {
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(source_stream.native_handle(), wait_for_buffer_gate, &gate),
                       "cudaLaunchHostFunc cross-device producer gate", 0);
  }
  producer.release();

  auto foreign_reader = buffer.read_synchronized_with(foreign_stream);
  {
    uni20::cuda::ScopedDevice device(1);
    uni20::cuda::check(cudaLaunchHostFunc(foreign_stream.native_handle(), set_buffer_flag, &foreign_read_completed),
                       "cudaLaunchHostFunc cross-device reader flag", 1);
  }
  foreign_reader.release();

  auto successor = buffer.write_synchronized_with(source_stream);
  {
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(source_stream.native_handle(), set_buffer_flag, &successor_completed),
                       "cudaLaunchHostFunc cross-device successor flag", 0);
  }
  successor.release();

  bool const producer_entered = wait_until(gate.entered);
  if (!producer_entered)
  {
    gate.open.store(true, std::memory_order_release);
  }
  ASSERT_TRUE(producer_entered);
  EXPECT_FALSE(foreign_read_completed.load(std::memory_order_acquire));
  EXPECT_FALSE(successor_completed.load(std::memory_order_acquire));

  gate.open.store(true, std::memory_order_release);
  source_stream.synchronize();
  foreign_stream.synchronize();
  EXPECT_TRUE(foreign_read_completed.load(std::memory_order_acquire));
  EXPECT_TRUE(successor_completed.load(std::memory_order_acquire));
}
