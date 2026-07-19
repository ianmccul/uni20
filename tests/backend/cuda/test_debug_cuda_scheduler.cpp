#include <uni20/async/async.hpp>
#include <uni20/async/cuda_task.hpp>
#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/backend/cuda/device.hpp>

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
using uni20::async::CudaTask;
using uni20::async::DebugCudaScheduler;
using uni20::async::DebugScheduler;
using uni20::async::ReadBuffer;
using uni20::async::WriteBuffer;

template <typename Scheduler, typename Task>
concept PubliclySchedules = requires(Scheduler& scheduler, Task&& task) { scheduler.schedule(std::move(task)); };

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
static_assert(std::same_as<AsyncTask::promise_type, CudaTask::promise_type>);
static_assert(!std::convertible_to<AsyncTask, CudaTask>);
static_assert(!std::convertible_to<CudaTask, AsyncTask>);
static_assert(PubliclySchedules<DebugScheduler, AsyncTask>);
static_assert(!PubliclySchedules<DebugScheduler, CudaTask>);
static_assert(PubliclySchedules<DebugCudaScheduler, CudaTask>);
static_assert(!PubliclySchedules<DebugCudaScheduler, AsyncTask>);

TEST_F(DebugCudaSchedulerTest, RunsCudaTaskOnBoundDeviceAndRestoresCallingThread)
{
  DebugCudaScheduler scheduler(target_device_);
  int observed_device = -1;
  cudaError_t status = cudaErrorUnknown;

  auto task = [](int& observed, cudaError_t& result) static -> CudaTask {
    result = cudaGetDevice(&observed);
    co_return;
  }(observed_device, status);

  scheduler.schedule(std::move(task));
  scheduler.run();

  EXPECT_EQ(status, cudaSuccess);
  EXPECT_EQ(observed_device, target_device_);
  EXPECT_TRUE(scheduler.done());

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, original_device_);
}

TEST_F(DebugCudaSchedulerTest, SuspendedTaskReschedulesOntoItsCudaDevice)
{
  DebugCudaScheduler cuda_scheduler(target_device_);
  DebugScheduler cpu_scheduler;
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

  cuda_scheduler.schedule(std::move(task));
  cuda_scheduler.run_all();
  EXPECT_TRUE(cuda_scheduler.done());

  auto writer = [](WriteBuffer<int> output) static -> AsyncTask { co_await output = 17; }(input.write());
  cpu_scheduler.schedule(std::move(writer));
  cpu_scheduler.run_all();

  EXPECT_FALSE(cuda_scheduler.done());
  cuda_scheduler.run_all();
  EXPECT_EQ(output.get_wait(cpu_scheduler), (std::array{17, target_device_, target_device_}));

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, original_device_);
}

TEST_F(DebugCudaSchedulerTest, MultipleDeviceSchedulersRestoreTheirDevicesAcrossOutOfOrderResumption)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  using Observations = std::array<int, 5>;
  std::vector<std::unique_ptr<DebugCudaScheduler>> cuda_schedulers;
  std::vector<std::unique_ptr<Async<int>>> first_inputs;
  std::vector<std::unique_ptr<Async<int>>> second_inputs;
  std::vector<std::unique_ptr<Async<Observations>>> outputs;
  cuda_schedulers.reserve(device_count_);
  first_inputs.reserve(device_count_);
  second_inputs.reserve(device_count_);
  outputs.reserve(device_count_);

  for (int device = 0; device < device_count_; ++device)
  {
    cuda_schedulers.push_back(std::make_unique<DebugCudaScheduler>(device));
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
    cuda_schedulers.back()->schedule(std::move(task));
  }

  auto expect_calling_device_restored = [this] {
    int current_device = -1;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, original_device_);
  };

  // Start in reverse device order. Every task reaches its first suspension.
  for (int device = device_count_ - 1; device >= 0; --device)
  {
    cuda_schedulers[device]->run_all();
    EXPECT_TRUE(cuda_schedulers[device]->done());
    expect_calling_device_restored();
  }

  DebugScheduler cpu_scheduler;
  auto publish = [](WriteBuffer<int> output, int value) static -> AsyncTask { co_await output = value; };
  for (int device = device_count_ - 1; device >= 0; --device)
  {
    cpu_scheduler.schedule(publish(first_inputs[device]->write(), 100 + device));
  }
  cpu_scheduler.run_all();

  // Resume in rotated order. Every task samples its device and suspends again.
  for (int offset = 0; offset < device_count_; ++offset)
  {
    int const device = (offset + 1) % device_count_;
    EXPECT_FALSE(cuda_schedulers[device]->done());
    cuda_schedulers[device]->run_all();
    EXPECT_TRUE(cuda_schedulers[device]->done());
    expect_calling_device_restored();
  }

  for (int device = 0; device < device_count_; ++device)
  {
    cpu_scheduler.schedule(publish(second_inputs[device]->write(), 200 + device));
  }
  cpu_scheduler.run_all();

  // Finish in reverse order, different from the second wakeup order.
  for (int device = device_count_ - 1; device >= 0; --device)
  {
    EXPECT_FALSE(cuda_schedulers[device]->done());
    cuda_schedulers[device]->run_all();
    EXPECT_TRUE(cuda_schedulers[device]->done());
    expect_calling_device_restored();
  }

  for (int device = 0; device < device_count_; ++device)
  {
    EXPECT_EQ(outputs[device]->get_wait(cpu_scheduler),
              (Observations{100 + device, 200 + device, device, device, device}));
  }
}

TEST_F(DebugCudaSchedulerTest, CpuParentReturnsToCpuSchedulerAfterCudaChild)
{
  DebugCudaScheduler cuda_scheduler(target_device_);
  DebugScheduler cpu_scheduler;
  std::vector<int> events;

  auto child = [](std::vector<int>& events) static -> CudaTask {
    int current_device = -1;
    cudaError_t const status = cudaGetDevice(&current_device);
    CHECK_EQUAL(static_cast<int>(status), static_cast<int>(cudaSuccess));
    events.push_back(10 + current_device);
    co_return;
  }(events);
  ASSERT_TRUE(child.set_scheduler(&cuda_scheduler));

  auto parent = [](CudaTask child, std::vector<int>& events) static -> AsyncTask {
    events.push_back(1);
    co_await child;
    events.push_back(2);
  }(std::move(child), events);

  cpu_scheduler.schedule(std::move(parent));
  cpu_scheduler.run_all();
  EXPECT_EQ(events, (std::vector<int>{1}));

  cuda_scheduler.run_all();
  EXPECT_EQ(events, (std::vector<int>{1, 10 + target_device_}));

  cpu_scheduler.run_all();
  EXPECT_EQ(events, (std::vector<int>{1, 10 + target_device_, 2}));

  int restored_device = -1;
  ASSERT_EQ(cudaGetDevice(&restored_device), cudaSuccess);
  EXPECT_EQ(restored_device, original_device_);
}

} // namespace
