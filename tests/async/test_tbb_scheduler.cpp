#include "async/async.hpp"
#include "async/debug_scheduler.hpp"
#include "async/reverse_value.hpp"
#include "async/tbb_scheduler.hpp"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>

using namespace uni20::async;

TEST(TbbScheduler, BasicSchedule)
{
  TbbScheduler sched{2};
  ScopedScheduler guard(&sched);

  std::atomic<bool> ran{false};

  auto task = []() -> AsyncTask { co_return; }();
  sched.schedule(std::move(task));

  sched.run_all();
  EXPECT_TRUE(true); // just smoke-test that it didn't deadlock
}

TEST(TbbScheduler, AsyncArithmetic)
{
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  Async<int> a = 1;
  Async<int> b = 2;
  Async<int> c = a + b;

  EXPECT_EQ(c.get_wait(), 3);
}

TEST(TbbScheduler, CoroutineAndAsync)
{
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  auto task = []() -> AsyncTask {
    Async<int> x = 10;
    Async<int> y = 32;
    Async<int> z = x + y;
    EXPECT_EQ(z.get_wait(), 42);
    co_return;
  }();

  sched.schedule(std::move(task));
  sched.run_all();
}

TEST(TbbScheduler, ManyTasks)
{
  TbbScheduler sched{4};

  std::atomic<int> counter{0};

  for (int i = 0; i < 100; i++)
  {
    sched.schedule([](std::atomic<int>& c) -> AsyncTask {
      c.fetch_add(1, std::memory_order_relaxed);
      co_return;
    }(counter));
  }

  sched.run_all();
  EXPECT_EQ(counter.load(), 100);
}

TEST(TbbScheduler, Parallelism)
{
  // This  test is not strictly deterministic but should be robust enough
  // (with 4 threads, runtime should be ~100–150 ms instead of 400 ms).
  TbbScheduler sched{4};
  using clock = std::chrono::steady_clock;

  auto start = clock::now();
  for (int i = 0; i < 8; i++)
  {
    sched.schedule([]() -> AsyncTask {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      co_return;
    }());
  }
  sched.run_all();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start).count();

  // With 4 threads, should take significantly less than 8*50ms sequential
  EXPECT_LT(elapsed, 400);
}

TEST(TbbScheduler, ReverseValue)
{
  // Test a case where we are guaranteed that dependencies are non-trivial
  TbbScheduler sched{4};
  set_global_scheduler(&sched);

  ReverseValue<int> rv;
  Async<int> v;
  async_assign(rv.value().read(), v.write());

  // At this point, v is not ready: rv hasn’t been written yet.
  // get_wait() must suspend/resume under the scheduler.
  std::thread writer([&] {
    // Small delay ensures the consumer suspends first
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    rv = 99;
  });

  EXPECT_EQ(v.read().get_wait(), 99);
  writer.join();
}

TEST(TbbScheduler, PausePreventsExecutionUntilResume)
{
  TbbScheduler sched{2};
  ScopedScheduler guard(&sched);

  sched.pause();

  std::atomic<int> direct_counter{0};
  std::atomic<int> async_counter{0};
  Async<int> value;

  constexpr int kDirectTasks = 3;
  for (int i = 0; i < kDirectTasks; ++i)
  {
    sched.schedule([](std::atomic<int>* counter) -> AsyncTask {
      counter->fetch_add(1, std::memory_order_relaxed);
      co_return;
    }(&direct_counter));
  }

  sched.schedule([](ReadBuffer<int> read_buffer, std::atomic<int>* counter) -> AsyncTask {
    auto& result = co_await read_buffer;
    counter->fetch_add(result, std::memory_order_relaxed);
    co_return;
  }(value.read(), &async_counter));

  constexpr int kWrittenValue = 42;
  constexpr int kDelayMs = 20;
  sched.schedule([](WriteBuffer<int> write_buffer, int value_to_write, int delay_ms) -> AsyncTask {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    auto& out = co_await write_buffer;
    out = value_to_write;
    co_return;
  }(value.write(), kWrittenValue, kDelayMs));

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(direct_counter.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(async_counter.load(std::memory_order_relaxed), 0);

  sched.resume();
  sched.run_all();

  EXPECT_EQ(direct_counter.load(std::memory_order_relaxed), kDirectTasks);
  EXPECT_EQ(async_counter.load(std::memory_order_relaxed), kWrittenValue);
}
