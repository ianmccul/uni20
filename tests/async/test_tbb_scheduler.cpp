#include "async/async.hpp"
#include "async/debug_scheduler.hpp"
#include "async/tbb_scheduler.hpp"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>

using namespace uni20::async;

TEST(TbbScheduler, BasicSchedule)
{
  ScopedTbbConcurrency tbb{2};
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
  ScopedTbbConcurrency tbb{4};
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  Async<int> a = 1;
  Async<int> b = 2;
  Async<int> c = a + b;

  EXPECT_EQ(c.get_wait(), 3);
}

TEST(TbbScheduler, CoroutineAndAsync)
{
  ScopedTbbConcurrency tbb{4};
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
  ScopedTbbConcurrency tbb{4};
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
  ScopedTbbConcurrency tbb{4};
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
