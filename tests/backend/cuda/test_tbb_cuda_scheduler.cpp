#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_cuda_scheduler.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <memory>
#include <utility>
#include <vector>

namespace
{

using uni20::async::Async;
using uni20::async::AsyncTask;
using uni20::async::CudaTask;
using uni20::async::DebugScheduler;
using uni20::async::ReadBuffer;
using uni20::async::TbbCudaScheduler;
using uni20::async::WriteBuffer;

template <typename Scheduler, typename Task>
concept PubliclySchedules = requires(Scheduler& scheduler, Task&& task) { scheduler.schedule(std::move(task)); };

static_assert(PubliclySchedules<TbbCudaScheduler, CudaTask>);
static_assert(!PubliclySchedules<TbbCudaScheduler, AsyncTask>);

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
  TbbCudaScheduler scheduler(target_device_, 2);
  int observed_device = -1;
  cudaError_t status = cudaErrorUnknown;

  auto task = [](int& observed, cudaError_t& result) static -> CudaTask {
    result = cudaGetDevice(&observed);
    co_return;
  }(observed_device, status);

  scheduler.schedule(std::move(task));
  scheduler.run_all();

  EXPECT_EQ(status, cudaSuccess);
  EXPECT_EQ(observed_device, target_device_);
  this->expect_calling_device_restored();
}

TEST_F(TbbCudaSchedulerTest, MultipleDeviceArenasRestoreDevicesAcrossOutOfOrderResumption)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  using Observations = std::array<int, 5>;
  std::vector<std::unique_ptr<TbbCudaScheduler>> cuda_schedulers;
  std::vector<std::unique_ptr<Async<int>>> first_inputs;
  std::vector<std::unique_ptr<Async<int>>> second_inputs;
  std::vector<std::unique_ptr<Async<Observations>>> outputs;
  cuda_schedulers.reserve(device_count_);
  first_inputs.reserve(device_count_);
  second_inputs.reserve(device_count_);
  outputs.reserve(device_count_);

  for (int device = 0; device < device_count_; ++device)
  {
    cuda_schedulers.push_back(std::make_unique<TbbCudaScheduler>(device, 2));
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
    this->expect_calling_device_restored();
  }

  // Wait for every first activation in reverse device order.
  for (int device = device_count_ - 1; device >= 0; --device)
  {
    cuda_schedulers[device]->run_all();
    this->expect_calling_device_restored();
  }

  DebugScheduler cpu_scheduler;
  auto publish = [](WriteBuffer<int> output, int value) static -> AsyncTask { co_await output = value; };
  for (int device = device_count_ - 1; device >= 0; --device)
  {
    cpu_scheduler.schedule(publish(first_inputs[device]->write(), 100 + device));
  }
  cpu_scheduler.run_all();

  // Wait for the first resumptions in rotated order. Each task suspends again.
  for (int offset = 0; offset < device_count_; ++offset)
  {
    int const device = (offset + 1) % device_count_;
    cuda_schedulers[device]->run_all();
    this->expect_calling_device_restored();
  }

  for (int device = 0; device < device_count_; ++device)
  {
    cpu_scheduler.schedule(publish(second_inputs[device]->write(), 200 + device));
  }
  cpu_scheduler.run_all();

  // Wait for final resumptions in reverse order.
  for (int device = device_count_ - 1; device >= 0; --device)
  {
    cuda_schedulers[device]->run_all();
    this->expect_calling_device_restored();
  }

  for (int device = 0; device < device_count_; ++device)
  {
    EXPECT_EQ(outputs[device]->get_wait(cpu_scheduler),
              (Observations{100 + device, 200 + device, device, device, device}));
  }
}

TEST_F(TbbCudaSchedulerTest, NestedDeviceArenaRestoresOuterArenaDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires at least two CUDA devices";

  int const outer_device = original_device_;
  int const inner_device = (outer_device + 1) % device_count_;
  TbbCudaScheduler outer_scheduler(outer_device, 2);
  TbbCudaScheduler inner_scheduler(inner_device, 2);
  Async<std::array<int, 2>> output;

  auto task = [](TbbCudaScheduler* inner, WriteBuffer<std::array<int, 2>> output) static -> CudaTask {
    int device_before_nested_arena = -1;
    int device_after_nested_arena = -1;
    cudaError_t status = cudaGetDevice(&device_before_nested_arena);
    CHECK_EQUAL(static_cast<int>(status), static_cast<int>(cudaSuccess));
    inner->run_all();
    status = cudaGetDevice(&device_after_nested_arena);
    CHECK_EQUAL(static_cast<int>(status), static_cast<int>(cudaSuccess));
    co_await output = std::array{device_before_nested_arena, device_after_nested_arena};
  }(&inner_scheduler, output.write());

  outer_scheduler.schedule(std::move(task));
  outer_scheduler.run_all();

  DebugScheduler cpu_scheduler;
  EXPECT_EQ(output.get_wait(cpu_scheduler), (std::array{outer_device, outer_device}));
  this->expect_calling_device_restored();
}

} // namespace
