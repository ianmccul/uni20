#include <uni20/async/async.hpp>
#include <uni20/async/cuda_task.hpp>
#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using uni20::async::Async;
using uni20::async::AsyncTask;
using uni20::async::BasicTask;
using uni20::async::cuda_promise;
using uni20::async::CudaTask;
using uni20::async::DebugCudaScheduler;
using uni20::async::DebugScheduler;
using uni20::async::DebugSchedulerOptions;
using uni20::async::ICudaScheduler;
using uni20::async::IScheduler;
using uni20::async::ReadBuffer;
using uni20::async::WriteBuffer;

template <typename Scheduler, typename Task>
concept PubliclySchedules = requires(Scheduler& scheduler, Task&& task) { scheduler.schedule(std::move(task)); };

template <typename Scheduler, typename Task>
concept PubliclySchedulesOnDevice =
    requires(Scheduler& scheduler, Task&& task) { scheduler.schedule(std::move(task), 0); };

struct CudaTaskTestError
{};

struct ObserveCudaPromiseDevice
{
    int& device;

    [[nodiscard]] constexpr bool await_ready() const noexcept { return false; }

    BasicTask await_suspend(BasicTask task) const noexcept
    {
      device = uni20::async::cuda_promise(task.handle()).device().value_or(-1);
      return task;
    }

    constexpr void await_resume() const noexcept {}
};

class DebugCudaSchedulerTest : public ::testing::Test {
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

    int device_count_ = 0;
    int original_device_ = -1;
    int target_device_ = -1;
};

static_assert(std::derived_from<AsyncTask, BasicTask>);
static_assert(std::derived_from<CudaTask, BasicTask>);
static_assert(!std::same_as<AsyncTask::promise_type, CudaTask::promise_type>);
static_assert(std::derived_from<AsyncTask::promise_type, uni20::async::TaskPromiseBase>);
static_assert(std::derived_from<CudaTask::promise_type, uni20::async::TaskPromiseBase>);
static_assert(std::derived_from<ICudaScheduler, IScheduler>);
static_assert(!std::convertible_to<AsyncTask, CudaTask>);
static_assert(!std::convertible_to<CudaTask, AsyncTask>);
static_assert(PubliclySchedules<DebugScheduler, AsyncTask>);
static_assert(!PubliclySchedules<DebugScheduler, CudaTask>);
static_assert(PubliclySchedules<DebugCudaScheduler, AsyncTask>);
static_assert(PubliclySchedules<DebugCudaScheduler, CudaTask>);
static_assert(PubliclySchedulesOnDevice<DebugCudaScheduler, CudaTask>);
static_assert(std::is_constructible_v<DebugCudaScheduler, DebugSchedulerOptions>);
static_assert(uni20::async::CudaTaskAwaitable<decltype(uni20::cuda::set_device(0))>);

TEST_F(DebugCudaSchedulerTest, RunsCudaTaskOnBoundDeviceAndRestoresCallingThread)
{
  DebugCudaScheduler scheduler;
  int observed_device = -1;
  cudaError_t status = cudaErrorUnknown;

  auto task = [](int& observed, cudaError_t& result) static -> CudaTask {
    result = cudaGetDevice(&observed);
    co_return;
  }(observed_device, status);

  auto const handle = task.handle();
  EXPECT_EQ(handle.domain(), uni20::async::TaskDomain::cuda);
  EXPECT_FALSE(uni20::async::cuda_promise(handle).device());
  scheduler.schedule(std::move(task), target_device_);
  EXPECT_EQ(uni20::async::cuda_promise(handle).device(), target_device_);
  scheduler.run();

  EXPECT_EQ(status, cudaSuccess);
  EXPECT_EQ(observed_device, target_device_);
  EXPECT_TRUE(scheduler.done());

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, original_device_);
}

TEST_F(DebugCudaSchedulerTest, RunsUnboundCudaTaskOnDefaultDeviceWithoutBindingPromise)
{
  DebugCudaScheduler scheduler(uni20::cuda::Device::get(target_device_));
  int observed_device = -1;

  auto task = [](int& observed) static -> CudaTask {
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed)), static_cast<int>(cudaSuccess));
    co_return;
  }(observed_device);

  auto const handle = task.handle();
  scheduler.schedule(std::move(task));
  EXPECT_FALSE(cuda_promise(handle).device());
  scheduler.run_all();

  EXPECT_EQ(observed_device, target_device_);
  EXPECT_TRUE(scheduler.done());

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, original_device_);
}

TEST_F(DebugCudaSchedulerTest, SetDeviceEstablishesAffinityAtSuspension)
{
  DebugCudaScheduler scheduler(uni20::cuda::Device::get(original_device_));
  Async<std::array<int, 3>> output;

  auto task = [](int selected_device, WriteBuffer<std::array<int, 3>> output) static -> CudaTask {
    std::array<int, 3> observed{-1, -1, -1};
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed[0])), static_cast<int>(cudaSuccess));
    co_await uni20::cuda::set_device(selected_device);
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed[1])), static_cast<int>(cudaSuccess));
    co_await ObserveCudaPromiseDevice{observed[2]};
    co_await output = observed;
  }(target_device_, output.write());

  scheduler.schedule(std::move(task));
  EXPECT_EQ(output.get_wait(scheduler), (std::array{original_device_, target_device_, target_device_}));
  EXPECT_TRUE(scheduler.done());
}

TEST_F(DebugCudaSchedulerTest, SetDeviceAlreadyMatchingEstablishedAffinityDoesNotReschedule)
{
  DebugCudaScheduler scheduler(uni20::cuda::Device::get(target_device_));
  bool completed = false;

  auto task = [](int device, bool& completed) static -> CudaTask {
    co_await uni20::cuda::set_device(device);
    completed = true;
  }(target_device_, completed);

  scheduler.schedule(std::move(task), target_device_);
  scheduler.run();

  EXPECT_TRUE(completed);
  EXPECT_TRUE(scheduler.done());
}

TEST_F(DebugCudaSchedulerTest, SetDeviceRejectsInvalidDeviceRoute)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  int const invalid_device = device_count_;
  DebugCudaScheduler scheduler(uni20::cuda::Device::get(target_device_));
  auto task = []() static -> CudaTask { co_return; }();
  ASSERT_TRUE(task.set_scheduler(&scheduler));

  EXPECT_DEATH(uni20::async::cuda_promise(task.handle()).select_device(invalid_device),
               "scheduler does not accept task route");

  EXPECT_FALSE(uni20::async::cuda_promise(task.handle()).device());
  task.set_cancel_on_resume();
}

TEST_F(DebugCudaSchedulerTest, FirstSelectionMatchingDefaultActivationDoesNotReschedule)
{
  DebugCudaScheduler scheduler(uni20::cuda::Device::get(target_device_));
  std::array<int, 2> observed{-1, -1};
  bool completed = false;

  auto task = [](int device, std::array<int, 2>& observed, bool& completed) static -> CudaTask {
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed[0])), static_cast<int>(cudaSuccess));
    co_await uni20::cuda::set_device(device);
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed[1])), static_cast<int>(cudaSuccess));
    completed = true;
  }(target_device_, observed, completed);

  scheduler.schedule(std::move(task));
  scheduler.run();

  EXPECT_EQ(observed, (std::array{target_device_, target_device_}));
  EXPECT_TRUE(completed);
  EXPECT_TRUE(scheduler.done());
}

TEST_F(DebugCudaSchedulerTest, SetDeviceDifferentFromCurrentActivationReschedules)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  int const other_device = (original_device_ + 1) % device_count_;
  DebugCudaScheduler scheduler(uni20::cuda::Device::get(original_device_));
  std::array<int, 2> observed{-1, -1};
  bool completed = false;

  auto task = [](int device, std::array<int, 2>& observed, bool& completed) static -> CudaTask {
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed[0])), static_cast<int>(cudaSuccess));
    co_await uni20::cuda::set_device(device);
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&observed[1])), static_cast<int>(cudaSuccess));
    completed = true;
  }(other_device, observed, completed);

  scheduler.schedule(std::move(task));
  scheduler.run();

  EXPECT_EQ(observed[0], original_device_);
  EXPECT_EQ(observed[1], -1);
  EXPECT_FALSE(completed);
  EXPECT_FALSE(scheduler.done());

  scheduler.run();

  EXPECT_EQ(observed[1], other_device);
  EXPECT_TRUE(completed);
  EXPECT_TRUE(scheduler.done());
}

TEST_F(DebugCudaSchedulerTest, SetDeviceCanMigrateAStartedTaskMoreThanOnce)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  int const other_device = (original_device_ + 1) % device_count_;
  DebugCudaScheduler scheduler(uni20::cuda::Device::get(original_device_));
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
}

TEST_F(DebugCudaSchedulerTest, SuspendedTaskReschedulesOntoItsCudaDevice)
{
  DebugCudaScheduler scheduler;
  Async<int> input;
  Async<std::array<int, 3>> output;

  auto task = [](ReadBuffer<int> input, WriteBuffer<std::array<int, 3>> output) static -> CudaTask {
    int device_before_suspend = -1;
    cudaError_t status = cudaGetDevice(&device_before_suspend);
    CHECK_EQUAL(static_cast<int>(status), static_cast<int>(cudaSuccess));
    int const value = co_await input;
    int device_after_resume = -1;
    status = cudaGetDevice(&device_after_resume);
    CHECK_EQUAL(static_cast<int>(status), static_cast<int>(cudaSuccess));
    co_await output = std::array{value, device_before_suspend, device_after_resume};
  }(input.read(), output.write());

  scheduler.schedule(std::move(task), target_device_);
  scheduler.run_all();
  EXPECT_TRUE(scheduler.done());

  auto writer = [](WriteBuffer<int> output) static -> AsyncTask { co_await output = 17; }(input.write());
  scheduler.schedule(std::move(writer));
  scheduler.run_all();

  EXPECT_EQ(output.get_wait(scheduler), (std::array{17, target_device_, target_device_}));

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, original_device_);
}

TEST_F(DebugCudaSchedulerTest, GetWaitDrivesCudaTaskFromUnifiedQueue)
{
  DebugCudaScheduler scheduler;
  Async<int> output;

  auto task = [](WriteBuffer<int> output) static -> CudaTask {
    int device = -1;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    co_await output = device;
  }(output.write());

  scheduler.schedule(std::move(task), target_device_);
  EXPECT_EQ(output.get_wait(scheduler), target_device_);
  EXPECT_TRUE(scheduler.done());

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, original_device_);
}

TEST_F(DebugCudaSchedulerTest, OneSchedulerRestoresDevicesAcrossMultiDeviceResumption)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  using Observations = std::array<int, 5>;
  DebugCudaScheduler scheduler;
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
  }

  auto expect_calling_device_restored = [this] {
    int current_device = -1;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, original_device_);
  };

  // Every CUDA task reaches its first suspension through the shared queue.
  scheduler.run_all();
  EXPECT_TRUE(scheduler.done());
  expect_calling_device_restored();

  auto publish = [](WriteBuffer<int> output, int value) static -> AsyncTask { co_await output = value; };
  for (int device = device_count_ - 1; device >= 0; --device)
  {
    scheduler.schedule(publish(first_inputs[device]->write(), 100 + device));
  }
  scheduler.run_all();
  expect_calling_device_restored();

  for (int device = 0; device < device_count_; ++device)
  {
    scheduler.schedule(publish(second_inputs[device]->write(), 200 + device));
  }
  scheduler.run_all();
  expect_calling_device_restored();

  for (int device = 0; device < device_count_; ++device)
  {
    EXPECT_EQ(outputs[device]->get_wait(scheduler), (Observations{100 + device, 200 + device, device, device, device}));
  }
}

TEST_F(DebugCudaSchedulerTest, CpuParentReturnsToCpuSchedulerAfterCudaChild)
{
  DebugCudaScheduler scheduler;
  std::vector<int> events;

  auto child = [](std::vector<int>& events) static -> CudaTask {
    int current_device = -1;
    cudaError_t const status = cudaGetDevice(&current_device);
    CHECK_EQUAL(static_cast<int>(status), static_cast<int>(cudaSuccess));
    events.push_back(10 + current_device);
    co_return;
  }(events);
  cuda_promise(child.handle()).bind_device(target_device_);
  ASSERT_TRUE(child.set_scheduler(&scheduler));

  auto parent = [](CudaTask child, std::vector<int>& events) static -> AsyncTask {
    events.push_back(1);
    co_await child;
    events.push_back(2);
  }(std::move(child), events);

  scheduler.schedule(std::move(parent));
  scheduler.run();
  EXPECT_EQ(events, (std::vector<int>{1}));

  scheduler.run();
  EXPECT_EQ(events, (std::vector<int>{1, 10 + target_device_}));

  scheduler.run();
  EXPECT_EQ(events, (std::vector<int>{1, 10 + target_device_, 2}));

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, original_device_);
}

TEST_F(DebugCudaSchedulerTest, GetWaitDrivesHostCudaHostContinuationChain)
{
  DebugCudaScheduler scheduler;
  Async<std::array<int, 3>> output;
  std::array<int, 3> observed_devices{-1, -1, -1};

  auto child = [](int* observed_device) static -> CudaTask {
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(observed_device)), static_cast<int>(cudaSuccess));
    co_return;
  }(&observed_devices[1]);
  cuda_promise(child.handle()).bind_device(target_device_);
  ASSERT_TRUE(child.set_scheduler(&scheduler));

  auto parent = [](CudaTask child, std::array<int, 3>* observed_devices,
                   WriteBuffer<std::array<int, 3>> output) static -> AsyncTask {
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&(*observed_devices)[0])), static_cast<int>(cudaSuccess));
    co_await child;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&(*observed_devices)[2])), static_cast<int>(cudaSuccess));
    co_await output = *observed_devices;
  }(std::move(child), &observed_devices, output.write());

  scheduler.schedule(std::move(parent));

  EXPECT_EQ(output.get_wait(scheduler), (std::array{original_device_, target_device_, original_device_}));
  EXPECT_TRUE(scheduler.done());
}

TEST_F(DebugCudaSchedulerTest, CpuParentReceivesCudaChildExceptionOnCpuScheduler)
{
  DebugCudaScheduler scheduler;
  std::vector<int> events;

  auto child = []() static -> CudaTask {
    throw CudaTaskTestError{};
    co_return;
  }();
  cuda_promise(child.handle()).bind_device(target_device_);
  ASSERT_TRUE(child.set_scheduler(&scheduler));

  auto parent = [](CudaTask child, std::vector<int>& events) static -> AsyncTask {
    events.push_back(1);
    try
    {
      co_await child;
    }
    catch (CudaTaskTestError const&)
    {
      events.push_back(2);
    }
  }(std::move(child), events);

  scheduler.schedule(std::move(parent));
  scheduler.run();
  EXPECT_EQ(events, (std::vector<int>{1}));

  scheduler.run();
  EXPECT_EQ(events, (std::vector<int>{1}));

  scheduler.run();
  EXPECT_EQ(events, (std::vector<int>{1, 2}));
}

TEST_F(DebugCudaSchedulerTest, ExplicitlyBoundAsyncChildRunsInHostDomain)
{
  DebugCudaScheduler scheduler;
  std::vector<int> observed_devices;

  auto child = [](std::vector<int>& observed) static -> AsyncTask {
    int device = -1;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    observed.push_back(device);
    co_return;
  }(observed_devices);
  ASSERT_TRUE(child.set_scheduler(&scheduler));

  auto parent = [](AsyncTask child, std::vector<int>& observed) static -> CudaTask {
    int device = -1;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    observed.push_back(device);
    co_await child;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    observed.push_back(device);
  }(std::move(child), observed_devices);

  scheduler.schedule(std::move(parent), target_device_);
  scheduler.run();
  scheduler.run();
  scheduler.run();

  EXPECT_EQ(observed_devices, (std::vector<int>{target_device_, original_device_, target_device_}));
  EXPECT_TRUE(scheduler.done());
}

TEST_F(DebugCudaSchedulerTest, CudaParentDirectlyTransfersToSameDeviceCudaChild)
{
  DebugCudaScheduler scheduler;
  std::vector<int> events;
  int child_promise_device = -1;

  auto child = [](std::vector<int>& events, int& promise_device) static -> CudaTask {
    co_await ObserveCudaPromiseDevice{promise_device};
    int current_device = -1;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&current_device)), static_cast<int>(cudaSuccess));
    events.push_back(10 + current_device);
    co_return;
  }(events, child_promise_device);

  auto parent = [](CudaTask child, std::vector<int>& events) static -> CudaTask {
    events.push_back(1);
    co_await child;
    events.push_back(2);
  }(std::move(child), events);

  scheduler.schedule(std::move(parent), target_device_);
  scheduler.run();

  EXPECT_EQ(events, (std::vector<int>{1, 10 + target_device_, 2}));
  EXPECT_EQ(child_promise_device, target_device_);
  EXPECT_TRUE(scheduler.done());
}

TEST_F(DebugCudaSchedulerTest, UnboundCudaParentAndChildUseDefaultDeviceWithoutAcquiringAffinity)
{
  DebugCudaScheduler scheduler(uni20::cuda::Device::get(target_device_));
  std::vector<int> events;
  int child_promise_device = -2;

  auto child = [](std::vector<int>& events, int& promise_device) static -> CudaTask {
    co_await ObserveCudaPromiseDevice{promise_device};
    int current_device = -1;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&current_device)), static_cast<int>(cudaSuccess));
    events.push_back(10 + current_device);
  }(events, child_promise_device);

  auto parent = [](CudaTask child, std::vector<int>& events) static -> CudaTask {
    events.push_back(1);
    co_await child;
    events.push_back(2);
  }(std::move(child), events);

  scheduler.schedule(std::move(parent));
  scheduler.run_all();

  EXPECT_EQ(events, (std::vector<int>{1, 10 + target_device_, 2}));
  EXPECT_EQ(child_promise_device, -1);
}

TEST_F(DebugCudaSchedulerTest, CrossDeviceCudaChildReturnsThroughSharedScheduler)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  int const parent_device = original_device_;
  int const child_device = (parent_device + 1) % device_count_;
  DebugCudaScheduler scheduler;
  std::vector<int> observed_devices;

  auto child = [](std::vector<int>& observed) static -> CudaTask {
    int device = -1;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    observed.push_back(device);
    co_return;
  }(observed_devices);
  cuda_promise(child.handle()).bind_device(child_device);
  ASSERT_TRUE(child.set_scheduler(&scheduler));

  auto parent = [](CudaTask child, std::vector<int>& observed) static -> CudaTask {
    int device = -1;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    observed.push_back(device);
    co_await child;
    CHECK_EQUAL(static_cast<int>(cudaGetDevice(&device)), static_cast<int>(cudaSuccess));
    observed.push_back(device);
  }(std::move(child), observed_devices);

  scheduler.schedule(std::move(parent), parent_device);
  scheduler.run();
  EXPECT_EQ(observed_devices, (std::vector<int>{parent_device}));

  scheduler.run();
  EXPECT_EQ(observed_devices, (std::vector<int>{parent_device, child_device}));

  scheduler.run();
  EXPECT_EQ(observed_devices, (std::vector<int>{parent_device, child_device, parent_device}));
}

TEST_F(DebugCudaSchedulerTest, CudaPromisePropagatesUnhandledExceptionToWriter)
{
  DebugCudaScheduler scheduler;
  Async<int> output;

  auto task = [](WriteBuffer<int> output) static -> CudaTask {
    (void)output;
    throw CudaTaskTestError{};
    co_return;
  }(output.write());

  scheduler.schedule(std::move(task), target_device_);
  scheduler.run_all();

  EXPECT_THROW((void)output.get_wait(scheduler), CudaTaskTestError);
}

TEST_F(DebugCudaSchedulerTest, CudaPromisePropagatesCancellationAfterBufferSuspension)
{
  DebugCudaScheduler scheduler;
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
}

} // namespace
