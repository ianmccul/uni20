#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/backend/cuda/cuda_error_presentation.hpp>
#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/common/presentation.hpp>
#include <uni20/common/trace.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <type_traits>

namespace
{

using namespace std::chrono_literals;

struct Gate
{
    std::atomic<bool> open = false;
    std::atomic<bool> entered = false;
};

void CUDART_CB wait_for_gate(void* raw_gate)
{
  auto& gate = *static_cast<Gate*>(raw_gate);
  gate.entered.store(true, std::memory_order_release);
  while (!gate.open.load(std::memory_order_acquire))
  {
    std::this_thread::yield();
  }
}

void CUDART_CB set_flag(void* raw_flag)
{
  static_cast<std::atomic<bool>*>(raw_flag)->store(true, std::memory_order_release);
}

void wait_until(std::atomic<bool> const& flag)
{
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::yield();
  }
  ASSERT_TRUE(flag.load(std::memory_order_acquire));
}

class ScopedThrowErrors {
  public:
    ScopedThrowErrors() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ScopedThrowErrors(ScopedThrowErrors const&) = delete;
    ScopedThrowErrors& operator=(ScopedThrowErrors const&) = delete;

    ~ScopedThrowErrors() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};

class CudaDeviceTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess)
      {
        GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      }
      if (device_count_ == 0)
      {
        GTEST_SKIP() << "no CUDA devices are available";
      }
    }

    int device_count_ = 0;
};

} // namespace

static_assert(std::is_copy_constructible_v<uni20::cuda::Stream>);
static_assert(std::is_move_constructible_v<uni20::cuda::Stream>);
static_assert(std::is_copy_constructible_v<uni20::cuda::Completion>);
static_assert(std::is_trivially_copyable_v<uni20::cuda::Device>);
static_assert(sizeof(uni20::cuda::Device) == sizeof(int));
static_assert(!std::is_copy_constructible_v<uni20::cuda::Runtime>);
static_assert(!std::is_move_constructible_v<uni20::cuda::Runtime>);

TEST(CudaRuntimeErrorTest, PreservesStructuredRuntimeStatus)
{
  ScopedThrowErrors throw_errors;
  try
  {
    uni20::cuda::check(cudaErrorInvalidValue, "test CUDA operation", 1);
    FAIL() << "expected CudaRuntimeError";
  }
  catch (uni20::cuda::CudaRuntimeError const& error)
  {
    EXPECT_EQ(error.status(), cudaErrorInvalidValue);
    EXPECT_EQ(error.operation(), "test CUDA operation");
    EXPECT_EQ(error.device(), 1);
    EXPECT_FALSE(error.error_name().empty());
    EXPECT_FALSE(error.reason().empty());
    EXPECT_TRUE(error.source_location().has_value());
  }
}

TEST(CudaRuntimeErrorTest, RendersThroughPresentationLayer)
{
  uni20::cuda::CudaRuntimeError error(cudaErrorInvalidValue, "test CUDA operation", 1);
  auto policy = uni20::presentation::plain_policy();
  policy.glyphs = uni20::presentation::glyph_set::ascii;
  auto const rendered = uni20::presentation::render_plain(uni20::cuda::diagnostic_report(error), policy);

  EXPECT_NE(rendered.find("[FAIL] CUDA operation 'test CUDA operation' failed"), std::string::npos) << rendered;
  EXPECT_NE(rendered.find("Status"), std::string::npos) << rendered;
  EXPECT_NE(rendered.find("cudaErrorInvalidValue"), std::string::npos) << rendered;
  EXPECT_NE(rendered.find("Device"), std::string::npos) << rendered;
}

TEST_F(CudaDeviceTest, ScopedDeviceRestoresPreviousDevice)
{
  if (device_count_ < 2)
  {
    GTEST_SKIP() << "the device-restoration test requires two CUDA devices";
  }

  uni20::cuda::check(cudaSetDevice(0), "cudaSetDevice test setup", 0);
  {
    uni20::cuda::ScopedDevice guard(1);
    int current_device = -1;
    uni20::cuda::check(cudaGetDevice(&current_device), "cudaGetDevice test", 1);
    EXPECT_EQ(current_device, 1);
  }

  int restored_device = -1;
  uni20::cuda::check(cudaGetDevice(&restored_device), "cudaGetDevice restored test", 0);
  EXPECT_EQ(restored_device, 0);
}

TEST_F(CudaDeviceTest, ScopedRuntimeInstallsCanonicalDeviceResources)
{
  ASSERT_FALSE(uni20::cuda::is_initialized());
  {
    auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .default_device = 0, .streams_per_device = 3});

    ASSERT_TRUE(uni20::cuda::is_initialized());
    EXPECT_EQ(&uni20::cuda::runtime(), &runtime);
    EXPECT_TRUE(runtime.has_device(0));
    EXPECT_FALSE(runtime.has_device(device_count_));
    EXPECT_EQ(runtime.default_device(), 0);

    auto& resources = runtime.device_resources(0);
    EXPECT_EQ(&uni20::cuda::device_resources(0), &resources);
    EXPECT_EQ(&uni20::cuda::device_resources(), &resources);
    EXPECT_EQ(resources.device().ordinal(), 0);
    EXPECT_EQ(resources.streams().size(), 3);
    EXPECT_EQ(resources.live_allocation_count(), 0);

    {
      uni20::cuda::CudaBuffer<double> buffer(4);
      EXPECT_EQ(&buffer.resources(), &resources);
      EXPECT_EQ(resources.live_allocation_count(), 1);
    }
    EXPECT_EQ(resources.live_allocation_count(), 0);
  }
  EXPECT_FALSE(uni20::cuda::is_initialized());
}

TEST_F(CudaDeviceTest, ScopedRuntimeMayBeInstalledAgainAfterShutdown)
{
  {
    auto first = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
    EXPECT_EQ(&uni20::cuda::runtime(), &first);
  }
  {
    auto second = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
    EXPECT_EQ(&uni20::cuda::runtime(), &second);
    EXPECT_EQ(uni20::cuda::device_resources().streams().size(), 2);
  }
}

TEST_F(CudaDeviceTest, RejectsASecondConcurrentRuntimeInstallation)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto first = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
        auto second = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
      },
      "CUDA runtime is already installed");
}

TEST_F(CudaDeviceTest, RejectsShutdownWhileExternalBuffersRemainAlive)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
        auto* leaked_buffer = new uni20::cuda::CudaBuffer<double>(4);
        (void)leaked_buffer;
      },
      "allocations still borrow");
}

TEST_F(CudaDeviceTest, DiscoversAndCachesCapabilities)
{
  EXPECT_EQ(uni20::cuda::Device::count(), device_count_);

  auto const devices = uni20::cuda::Device::enumerate();
  ASSERT_EQ(devices.size(), static_cast<std::size_t>(device_count_));
  for (std::size_t ordinal = 0; ordinal < devices.size(); ++ordinal)
  {
    EXPECT_EQ(devices[ordinal].ordinal(), static_cast<int>(ordinal));
    auto const& capabilities = devices[ordinal].capabilities();
    EXPECT_FALSE(capabilities.name.empty());
    EXPECT_GT(capabilities.total_global_memory, 0U);
    EXPECT_GT(capabilities.compute_capability_major, 0);
    EXPECT_GT(capabilities.multiprocessor_count, 0);
    EXPECT_GT(capabilities.warp_size, 0);
    EXPECT_GT(capabilities.max_threads_per_block, 0);
    EXPECT_GT(capabilities.max_threads_per_multiprocessor, 0);
  }

  auto const device = uni20::cuda::Device::get(0);
  auto const same_device = uni20::cuda::Device::get(0);
  EXPECT_EQ(device, same_device);
  EXPECT_EQ(&device.capabilities(), &same_device.capabilities());
}

TEST_F(CudaDeviceTest, DiscoveryDoesNotChangeCurrentDevice)
{
  int const selected_device = device_count_ > 1 ? 1 : 0;
  uni20::cuda::ScopedDevice selected(selected_device);

  auto const discovered = uni20::cuda::Device::get(0);
  EXPECT_EQ(discovered.ordinal(), 0);

  int current_device = -1;
  uni20::cuda::check(cudaGetDevice(&current_device), "cudaGetDevice after discovery", selected_device);
  EXPECT_EQ(current_device, selected_device);
}

TEST_F(CudaDeviceTest, CurrentReturnsSelectedDevice)
{
  int const selected_device = device_count_ > 1 ? 1 : 0;
  uni20::cuda::ScopedDevice selected(selected_device);
  EXPECT_EQ(uni20::cuda::Device::current().ordinal(), selected_device);
}

TEST_F(CudaDeviceTest, InvalidOrdinalRaisesStructuredError)
{
  ScopedThrowErrors throw_errors;
  try
  {
    (void)uni20::cuda::Device::get(device_count_);
    FAIL() << "expected CudaRuntimeError";
  }
  catch (uni20::cuda::CudaRuntimeError const& error)
  {
    EXPECT_EQ(error.status(), cudaErrorInvalidDevice);
    EXPECT_EQ(error.operation(), "cudaGetDeviceProperties");
    EXPECT_EQ(error.device(), device_count_);
  }
}

TEST_F(CudaDeviceTest, StreamReturnsToPoolAtScopeExit)
{
  uni20::cuda::StreamPool pool({.device = 0, .stream_count = 1});
  {
    auto stream = pool.try_acquire();
    ASSERT_TRUE(stream.has_value());
    EXPECT_EQ(pool.idle_stream_count(), 0);
    EXPECT_EQ(pool.leased_stream_count(), 1);
  }

  pool.synchronize();
  EXPECT_EQ(pool.idle_stream_count(), 1);
  EXPECT_EQ(pool.leased_stream_count(), 0);
}

TEST_F(CudaDeviceTest, SubmittedStreamReturnsOnlyAfterActualCompletion)
{
  uni20::cuda::StreamPool pool({.device = 0, .stream_count = 1});
  auto stream = pool.try_acquire();
  ASSERT_TRUE(stream.has_value());

  Gate gate;
  uni20::cuda::check(cudaLaunchHostFunc(stream->native_handle(), wait_for_gate, &gate), "cudaLaunchHostFunc test gate",
                     0);
  auto completion = stream->record_completion();
  stream.reset();

  wait_until(gate.entered);
  EXPECT_FALSE(pool.try_acquire().has_value());
  EXPECT_EQ(pool.pending_stream_count(), 1);
  EXPECT_FALSE(completion.ready());

  gate.open.store(true, std::memory_order_release);
  pool.synchronize();

  EXPECT_TRUE(completion.ready());
  EXPECT_EQ(pool.pending_stream_count(), 0);
  EXPECT_EQ(pool.idle_stream_count(), 1);
  auto next = pool.try_acquire();
  ASSERT_TRUE(next.has_value());
  next.reset();
  pool.synchronize();
}

TEST_F(CudaDeviceTest, BlockingAcquireWaitsForActuallyIdleStream)
{
  uni20::cuda::StreamPool pool({.device = 0, .stream_count = 1});
  auto producer = pool.acquire();

  Gate gate;
  uni20::cuda::check(cudaLaunchHostFunc(producer.native_handle(), wait_for_gate, &gate),
                     "cudaLaunchHostFunc blocking acquire gate", 0);
  auto completion = producer.record_completion();
  producer = {};

  std::atomic<bool> acquired = false;
  std::jthread waiter([&] {
    auto stream = pool.acquire();
    acquired.store(true, std::memory_order_release);
  });

  wait_until(gate.entered);
  EXPECT_FALSE(acquired.load(std::memory_order_acquire));

  gate.open.store(true, std::memory_order_release);
  waiter.join();
  EXPECT_TRUE(acquired.load(std::memory_order_acquire));
  EXPECT_TRUE(completion.ready());
}

TEST_F(CudaDeviceTest, DependenciesUseCompletionsAcrossIndependentStreams)
{
  uni20::cuda::StreamPool pool({.device = 0, .stream_count = 2});
  auto producer = pool.try_acquire();
  auto consumer = pool.try_acquire();
  ASSERT_TRUE(producer.has_value());
  ASSERT_TRUE(consumer.has_value());

  Gate producer_gate;
  uni20::cuda::check(cudaLaunchHostFunc(producer->native_handle(), wait_for_gate, &producer_gate),
                     "cudaLaunchHostFunc producer gate", 0);
  auto producer_completion = producer->record_completion();
  producer.reset();

  std::atomic<bool> consumer_completed = false;
  consumer->wait_on(producer_completion);
  uni20::cuda::check(cudaLaunchHostFunc(consumer->native_handle(), set_flag, &consumer_completed),
                     "cudaLaunchHostFunc consumer flag", 0);
  auto consumer_completion = consumer->record_completion();
  consumer.reset();

  wait_until(producer_gate.entered);
  EXPECT_FALSE(consumer_completed.load(std::memory_order_acquire));
  EXPECT_EQ(pool.pending_stream_count(), 2);

  producer_gate.open.store(true, std::memory_order_release);
  consumer_completion.synchronize();
  pool.synchronize();

  EXPECT_TRUE(consumer_completed.load(std::memory_order_acquire));
  EXPECT_TRUE(producer_completion.ready());
  EXPECT_TRUE(consumer_completion.ready());
  EXPECT_EQ(pool.idle_stream_count(), 2);
}

TEST_F(CudaDeviceTest, CompletionMayBeDestroyedAfterWaitIsEnqueued)
{
  uni20::cuda::StreamPool pool({.device = 0, .stream_count = 2});
  auto producer = pool.acquire();
  auto consumer = pool.acquire();

  Gate producer_gate;
  uni20::cuda::check(cudaLaunchHostFunc(producer.native_handle(), wait_for_gate, &producer_gate),
                     "cudaLaunchHostFunc producer gate", 0);

  std::atomic<bool> consumer_completed = false;
  {
    auto completion = producer.record_completion();
    EXPECT_EQ(completion.device(), producer.device());
    consumer.wait_on(completion);
  }

  uni20::cuda::check(cudaLaunchHostFunc(consumer.native_handle(), set_flag, &consumer_completed),
                     "cudaLaunchHostFunc consumer flag", 0);
  wait_until(producer_gate.entered);
  EXPECT_FALSE(consumer_completed.load(std::memory_order_acquire));

  producer_gate.open.store(true, std::memory_order_release);
  consumer.synchronize();
  EXPECT_TRUE(consumer_completed.load(std::memory_order_acquire));
}

TEST_F(CudaDeviceTest, CompletionSupportsCrossDeviceWait)
{
  if (device_count_ < 2)
  {
    GTEST_SKIP() << "the cross-device wait test requires two CUDA devices";
  }

  uni20::cuda::StreamPool producer_pool({.device = 0, .stream_count = 1});
  uni20::cuda::StreamPool consumer_pool({.device = 1, .stream_count = 1});
  auto producer = producer_pool.acquire();
  auto consumer = consumer_pool.acquire();
  auto completion = producer.record_completion();

  EXPECT_EQ(completion.device(), 0);
  consumer.wait_on(completion);
  consumer.synchronize();
  EXPECT_TRUE(completion.ready());
}
