#include <uni20/async/async.hpp>
#include <uni20/async/tbb_cuda_scheduler.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <latch>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using uni20::async::Async;
using uni20::async::AsyncTask;
using uni20::async::CudaTask;
using uni20::async::ReadBuffer;
using uni20::async::TbbCudaScheduler;
using uni20::async::TbbCudaSchedulerOptions;
using uni20::async::WriteBuffer;

template <typename Scheduler, typename Task>
concept PubliclySchedules = requires(Scheduler& scheduler, Task&& task) { scheduler.schedule(std::move(task)); };

template <typename Scheduler, typename Task>
concept PubliclySchedulesOnDevice =
    requires(Scheduler& scheduler, Task&& task) { scheduler.schedule(std::move(task), 0); };

static_assert(PubliclySchedules<TbbCudaScheduler, AsyncTask>);
static_assert(PubliclySchedules<TbbCudaScheduler, CudaTask>);
static_assert(PubliclySchedulesOnDevice<TbbCudaScheduler, CudaTask>);

struct CudaTaskTestError
{};

class ConcurrentActivationGate {
  public:
    [[nodiscard]] bool arrive_and_wait(std::chrono::milliseconds timeout)
    {
      std::unique_lock lock(mutex_);
      ++arrivals_;
      ready_.notify_all();

      // The timeout only turns missing concurrency into a test failure instead
      // of leaving one arena participant blocked indefinitely.
      if (!ready_.wait_for(lock, timeout, [this] { return arrivals_ == 2 || timed_out_; }))
      {
        timed_out_ = true;
        ready_.notify_all();
      }
      return !timed_out_;
    }

  private:
    std::mutex mutex_;
    std::condition_variable ready_;
    int arrivals_ = 0;
    bool timed_out_ = false;
};

struct ActivationObservation
{
    std::thread::id thread{};
    cudaError_t status = cudaErrorUnknown;
    int device = -1;
    bool overlapped = false;
};

class TbbCudaSchedulerTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess)
      {
        GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      }
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";

      ASSERT_EQ(cudaGetDevice(&original_device_), cudaSuccess);
      target_device_ = device_count_ > 1 ? (original_device_ + 1) % device_count_ : original_device_;
    }

    void expect_calling_device_restored() const
    {
      int current_device = -1;
      ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
      EXPECT_EQ(current_device, original_device_);
    }

    int device_count_ = 0;
    int original_device_ = -1;
    int target_device_ = -1;
};

TEST_F(TbbCudaSchedulerTest, RunsOnBoundDeviceAndRestoresCallingThread)
{
  TbbCudaScheduler scheduler({.host_max_concurrency = 2, .cuda_max_concurrency_per_device = 2});
  int observed_device = -1;
  cudaError_t status = cudaErrorUnknown;

  auto task = [](int& observed, cudaError_t& result) static -> CudaTask {
    result = cudaGetDevice(&observed);
    co_return;
  }(observed_device, status);

  scheduler.schedule(std::move(task), target_device_);
  scheduler.run_all();

  EXPECT_EQ(status, cudaSuccess);
  EXPECT_EQ(observed_device, target_device_);
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, RunsUnboundCudaTaskOnConfiguredDefaultDevice)
{
  TbbCudaScheduler scheduler({.host_max_concurrency = 2,
                              .cuda_max_concurrency_per_device = 2,
                              .default_cuda_device = target_device_});
  int observed_device = -1;

  auto task = [](int& observed) static -> CudaTask {
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed)), static_cast<int>(cudaSuccess));
    co_return;
  }(observed_device);

  auto const handle = task.handle();
  scheduler.schedule(std::move(task));
  EXPECT_FALSE(uni20::async::cuda_promise(handle).device());
  scheduler.run_all();

  EXPECT_EQ(observed_device, target_device_);
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, SetDeviceMigratesBetweenCudaArenas)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  int const other_device = (original_device_ + 1) % device_count_;
  TbbCudaScheduler scheduler({.host_max_concurrency = 2,
                              .cuda_max_concurrency_per_device = 2,
                              .default_cuda_device = original_device_});
  Async<std::array<int, 3>> output;

  auto task = [](int other_device, int original_device, WriteBuffer<std::array<int, 3>> output) static -> CudaTask {
    std::array<int, 3> observed{-1, -1, -1};
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed[0])), static_cast<int>(cudaSuccess));
    co_await uni20::cuda::set_device(other_device);
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed[1])), static_cast<int>(cudaSuccess));
    co_await uni20::cuda::set_device(original_device);
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed[2])), static_cast<int>(cudaSuccess));
    co_await output = observed;
  }(other_device, original_device_, output.write());

  scheduler.schedule(std::move(task));
  EXPECT_EQ(output.get_wait(scheduler), (std::array{original_device_, other_device, original_device_}));
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, RejectsTaskBoundToUnenrolledDevice)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  int const other_device = (target_device_ + 1) % device_count_;
  TbbCudaScheduler scheduler(
      std::vector{uni20::cuda::Device::get(target_device_)},
      {.host_max_concurrency = 1, .cuda_max_concurrency_per_device = 1, .default_cuda_device = target_device_});
  EXPECT_DEATH(
      {
        auto task = []() static -> CudaTask { co_return; }();
        uni20::async::cuda_promise(task.handle()).bind_device(other_device);
        static_cast<void>(task.set_scheduler(&scheduler));
      },
      "scheduler does not accept task route");
}

TEST_F(TbbCudaSchedulerTest, GetWaitDrivesCudaTaskFromUnifiedScheduler)
{
  TbbCudaScheduler scheduler({.host_max_concurrency = 1, .cuda_max_concurrency_per_device = 1});
  Async<int> output;

  auto task = [](WriteBuffer<int> output) static -> CudaTask {
    int device = -1;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    co_await output = device;
  }(output.write());

  scheduler.schedule(std::move(task), target_device_);
  EXPECT_EQ(output.get_wait(scheduler), target_device_);
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, HostParentAwaitsExplicitlyBoundCudaChild)
{
  TbbCudaScheduler scheduler({.host_max_concurrency = 2, .cuda_max_concurrency_per_device = 2});
  Async<int> output;
  int observed_device = -1;

  auto child = [](int& observed) static -> CudaTask {
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed)), static_cast<int>(cudaSuccess));
    co_return;
  }(observed_device);
  uni20::async::cuda_promise(child.handle()).bind_device(target_device_);
  ASSERT_TRUE(child.set_scheduler(&scheduler));

  auto parent = [](CudaTask child, WriteBuffer<int> output, int const* observed) static -> AsyncTask {
    co_await child;
    co_await output = *observed;
  }(std::move(child), output.write(), &observed_device);

  scheduler.schedule(std::move(parent));
  EXPECT_EQ(output.get_wait(scheduler), target_device_);
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, HostParentReceivesCudaChildException)
{
  TbbCudaScheduler scheduler({.host_max_concurrency = 2, .cuda_max_concurrency_per_device = 2});
  Async<int> output;

  auto child = []() static -> CudaTask {
    throw CudaTaskTestError{};
    co_return;
  }();
  uni20::async::cuda_promise(child.handle()).bind_device(target_device_);
  ASSERT_TRUE(child.set_scheduler(&scheduler));

  auto parent = [](CudaTask child, WriteBuffer<int> output) static -> AsyncTask {
    bool caught = false;
    try
    {
      co_await child;
    }
    catch (CudaTaskTestError const&)
    {
      caught = true;
    }
    co_await output = caught ? 1 : 0;
  }(std::move(child), output.write());

  scheduler.schedule(std::move(parent));
  EXPECT_EQ(output.get_wait(scheduler), 1);
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, CudaCancellationPropagatesAfterBufferSuspension)
{
  TbbCudaScheduler scheduler({.host_max_concurrency = 2, .cuda_max_concurrency_per_device = 2});
  Async<int> input;
  Async<int> output;
  auto input_writer = input.write();

  auto task = [](ReadBuffer<int> input, WriteBuffer<int> output) static -> CudaTask {
    int const value = co_await input.or_cancel();
    co_await output = value;
  }(input.read(), output.write());

  scheduler.schedule(std::move(task), target_device_);
  scheduler.run_all();
  input_writer.release();
  scheduler.run_all();

  EXPECT_THROW((void)output.get_wait(scheduler), uni20::async::buffer_read_uninitialized);
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, ConcurrentArenaActivationsSelectBoundDeviceBeforeAndAfterResumption)
{
  using namespace std::chrono_literals;

  TbbCudaScheduler scheduler({.host_max_concurrency = 2, .cuda_max_concurrency_per_device = 2});
  std::array<Async<int>, 2> inputs;
  std::array<ActivationObservation, 2> initial;
  std::array<ActivationObservation, 2> resumed;
  ConcurrentActivationGate initial_gate;
  ConcurrentActivationGate resumed_gate;
  std::thread::id const calling_thread = std::this_thread::get_id();

  auto task = [](std::size_t index, ReadBuffer<int> input, std::array<ActivationObservation, 2>* initial,
                 std::array<ActivationObservation, 2>* resumed, ConcurrentActivationGate* initial_gate,
                 ConcurrentActivationGate* resumed_gate) static -> CudaTask {
    (*initial)[index].thread = std::this_thread::get_id();
    (*initial)[index].status = cudaGetDevice(&(*initial)[index].device);
    (*initial)[index].overlapped = initial_gate->arrive_and_wait(2s);

    static_cast<void>(co_await input);

    (*resumed)[index].thread = std::this_thread::get_id();
    (*resumed)[index].status = cudaGetDevice(&(*resumed)[index].device);
    (*resumed)[index].overlapped = resumed_gate->arrive_and_wait(2s);
  };

  scheduler.schedule(task(0, inputs[0].read(), &initial, &resumed, &initial_gate, &resumed_gate), target_device_);
  scheduler.schedule(task(1, inputs[1].read(), &initial, &resumed, &initial_gate, &resumed_gate), target_device_);
  scheduler.run_all();

  EXPECT_TRUE(initial[0].overlapped);
  EXPECT_TRUE(initial[1].overlapped);
  EXPECT_NE(initial[0].thread, initial[1].thread);
  EXPECT_TRUE(initial[0].thread != calling_thread || initial[1].thread != calling_thread)
      << "expected at least one initial activation on a oneTBB worker";
  for (auto const& observation : initial)
  {
    EXPECT_EQ(observation.status, cudaSuccess);
    EXPECT_EQ(observation.device, target_device_);
  }
  this->expect_calling_device_restored();

  scheduler.schedule([](WriteBuffer<int> output) static -> AsyncTask { co_await output = 1; }(inputs[0].write()));
  scheduler.schedule([](WriteBuffer<int> output) static -> AsyncTask { co_await output = 2; }(inputs[1].write()));
  scheduler.run_all();

  EXPECT_TRUE(resumed[0].overlapped);
  EXPECT_TRUE(resumed[1].overlapped);
  EXPECT_NE(resumed[0].thread, resumed[1].thread);
  EXPECT_TRUE(resumed[0].thread != calling_thread || resumed[1].thread != calling_thread)
      << "expected at least one resumed activation on a oneTBB worker";
  for (auto const& observation : resumed)
  {
    EXPECT_EQ(observation.status, cudaSuccess);
    EXPECT_EQ(observation.device, target_device_);
  }
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, SaturatedArenaAcceptsResumptionWithoutPublisherParticipation)
{
  using namespace std::chrono_literals;

  TbbCudaScheduler scheduler({.host_max_concurrency = 1, .cuda_max_concurrency_per_device = 1});
  Async<int> input;
  Async<std::array<int, 2>> output;
  std::latch waiting_for_input{1};
  std::latch blocker_started{1};
  std::latch release_blocker{1};

  scheduler.schedule(
      [](ReadBuffer<int> input, WriteBuffer<std::array<int, 2>> output, std::latch* waiting) static -> CudaTask {
        waiting->count_down();
        int const value = co_await input;
        int device = -1;
        CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
        co_await output = std::array{value, device};
      }(input.read(), output.write(), &waiting_for_input),
      target_device_);
  waiting_for_input.wait();

  scheduler.schedule(
      [](std::latch* started, std::latch* release) static -> CudaTask {
        started->count_down();
        release->wait();
        co_return;
      }(&blocker_started, &release_blocker),
      target_device_);
  blocker_started.wait();

  std::binary_semaphore publication_returned{0};
  std::atomic<int> publisher_device_before{-1};
  std::atomic<int> publisher_device_after{-1};
  std::jthread publisher([&] {
    CHECK_EQUAL(static_cast<int>(cudaSetDevice(original_device_)), static_cast<int>(cudaSuccess));
    int device = -1;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    publisher_device_before.store(device, std::memory_order_relaxed);

    scheduler.schedule([](WriteBuffer<int> input) static -> AsyncTask { co_await input = 42; }(input.write()));

    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    publisher_device_after.store(device, std::memory_order_relaxed);
    publication_returned.release();
  });

  if (!publication_returned.try_acquire_for(2s))
  {
    release_blocker.count_down();
    publisher.join();
    FAIL() << "TbbCudaScheduler rescheduling waited for saturated-arena admission";
  }

  EXPECT_EQ(publisher_device_before.load(std::memory_order_relaxed), original_device_);
  EXPECT_EQ(publisher_device_after.load(std::memory_order_relaxed), original_device_);

  // This wait is sequenced after rescheduling returned, so the deferred
  // activation must already belong to the scheduler's task group.
  release_blocker.count_down();
  scheduler.run_all();
  publisher.join();

  EXPECT_EQ(output.get_wait(scheduler), (std::array{42, target_device_}));
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, OneSchedulerRoutesResumptionsAcrossMultipleDeviceArenas)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  using Observations = std::array<int, 5>;
  TbbCudaScheduler scheduler({.host_max_concurrency = 2, .cuda_max_concurrency_per_device = 2});
  std::vector<std::unique_ptr<Async<int>>> first_inputs;
  std::vector<std::unique_ptr<Async<int>>> second_inputs;
  std::vector<std::unique_ptr<Async<Observations>>> outputs;
  first_inputs.reserve(device_count_);
  second_inputs.reserve(device_count_);
  outputs.reserve(device_count_);

  for (int device = 0; device < device_count_; ++device)
  {
    first_inputs.push_back(std::make_unique<Async<int>>());
    second_inputs.push_back(std::make_unique<Async<int>>());
    outputs.push_back(std::make_unique<Async<Observations>>());

    auto task = [](ReadBuffer<int> first_input, ReadBuffer<int> second_input,
                   WriteBuffer<Observations> output) static -> CudaTask {
      int device_before_suspend = -1;
      int device_after_first_resume = -1;
      int device_after_second_resume = -1;
      cudaError_t status = cudaGetDevice(&device_before_suspend);
      CHECK_EQUAL(static_cast<int>(status), static_cast<int>(cudaSuccess));
      int const first_value = co_await first_input;
      status = cudaGetDevice(&device_after_first_resume);
      CHECK_EQUAL(static_cast<int>(status), static_cast<int>(cudaSuccess));
      int const second_value = co_await second_input;
      status = cudaGetDevice(&device_after_second_resume);
      CHECK_EQUAL(static_cast<int>(status), static_cast<int>(cudaSuccess));
      co_await output = Observations{first_value, second_value, device_before_suspend, device_after_first_resume,
                                     device_after_second_resume};
    }(first_inputs.back()->read(), second_inputs.back()->read(), outputs.back()->write());
    scheduler.schedule(std::move(task), device);
    this->expect_calling_device_restored();
  }

  scheduler.run_all();
  this->expect_calling_device_restored();

  auto publish = [](WriteBuffer<int> output, int value) static -> AsyncTask { co_await output = value; };
  for (int device = device_count_ - 1; device >= 0; --device)
  {
    scheduler.schedule(publish(first_inputs[device]->write(), 100 + device));
  }
  scheduler.run_all();
  this->expect_calling_device_restored();

  for (int device = 0; device < device_count_; ++device)
  {
    scheduler.schedule(publish(second_inputs[device]->write(), 200 + device));
  }
  scheduler.run_all();
  this->expect_calling_device_restored();

  for (int device = 0; device < device_count_; ++device)
  {
    EXPECT_EQ(outputs[device]->get_wait(scheduler), (Observations{100 + device, 200 + device, device, device, device}));
  }
}

TEST_F(TbbCudaSchedulerTest, CrossDeviceChildReturnsThroughSharedScheduler)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  int const parent_device = original_device_;
  int const child_device = (parent_device + 1) % device_count_;
  TbbCudaScheduler scheduler({.host_max_concurrency = 2, .cuda_max_concurrency_per_device = 2});
  std::array<int, 3> observed{-1, -1, -1};

  auto child = [](int* observed) static -> CudaTask {
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(observed)), static_cast<int>(cudaSuccess));
    co_return;
  }(&observed[1]);
  uni20::async::cuda_promise(child.handle()).bind_device(child_device);
  ASSERT_TRUE(child.set_scheduler(&scheduler));

  auto parent = [](CudaTask child, std::array<int, 3>* observed) static -> CudaTask {
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&(*observed)[0])), static_cast<int>(cudaSuccess));
    co_await child;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&(*observed)[2])), static_cast<int>(cudaSuccess));
  }(std::move(child), &observed);

  scheduler.schedule(std::move(parent), parent_device);
  scheduler.run_all();

  EXPECT_EQ(observed, (std::array{parent_device, child_device, parent_device}));
  this->expect_calling_device_restored();
}

} // namespace
