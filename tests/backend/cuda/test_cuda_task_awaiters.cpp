#include <uni20/async/cuda_task.hpp>
#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/resource_pool.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace
{

class CudaTaskAwaiterTest : public ::testing::Test {
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

    int device_count_ = 0;
};

static_assert(uni20::async::CudaTaskAwaitable<
              decltype(uni20::cuda::acquire_resource(std::declval<uni20::cuda::ResourcePool<int>&>()))>);
static_assert(
    uni20::async::CudaTaskAwaitable<decltype(uni20::cuda::acquire_stream(std::declval<uni20::cuda::StreamPool&>()))>);

TEST_F(CudaTaskAwaiterTest, ResourceAcquisitionSuspendsUntilLeaseRelease)
{
  uni20::cuda::ResourcePool<int> pool(std::vector<int>{17});
  auto occupied = pool.acquire();
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  int observed = 0;

  auto task = [](uni20::cuda::ResourcePool<int>& resource_pool, int& result) static -> uni20::async::CudaTask {
    auto lease = co_await uni20::cuda::acquire_resource(resource_pool);
    result = lease.get();
    co_return;
  }(pool, observed);

  scheduler.schedule(std::move(task), 0);
  scheduler.run();
  EXPECT_EQ(observed, 0);
  EXPECT_EQ(pool.idle_count(), 0);

  occupied.release();
  scheduler.run_all();
  EXPECT_EQ(observed, 17);
  EXPECT_EQ(pool.idle_count(), 1);
}

TEST_F(CudaTaskAwaiterTest, StreamAcquisitionSuspendsUntilStreamIsIdle)
{
  uni20::cuda::StreamPool pool({.device = 0, .stream_count = 1});
  auto occupied = pool.acquire();
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  bool acquired = false;

  auto task = [](uni20::cuda::StreamPool& stream_pool, bool& result) static -> uni20::async::CudaTask {
    auto stream = co_await uni20::cuda::acquire_stream(stream_pool);
    result = static_cast<bool>(stream);
    co_return;
  }(pool, acquired);

  scheduler.schedule(std::move(task), 0);
  scheduler.run();
  EXPECT_FALSE(acquired);
  EXPECT_EQ(pool.idle_stream_count(), 0);

  occupied = {};
  pool.synchronize();
  scheduler.run_all();
  EXPECT_TRUE(acquired);
  pool.synchronize();
  EXPECT_EQ(pool.idle_stream_count(), 1);
}

} // namespace
