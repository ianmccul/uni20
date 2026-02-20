#include "async/async.hpp"
#include "async/async_ops.hpp"
#include "async/debug_scheduler.hpp"
#include "async/dual.hpp"
#include "async/dual_toys.hpp"
#include "async/reverse_value.hpp"
#include "async/tbb_scheduler.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

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

TEST(TbbScheduler, AsyncAccumulationGetWait)
{
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  Async<int> x = 0;
  constexpr int iterations = 64;
  for (int i = 0; i < iterations; ++i)
  {
    x += 1;
  }

  // Regression coverage: a historical bug dropped coroutines in linear chains
  // of tasks, so the final get_wait() never observed all increments.
  EXPECT_EQ(x.get_wait(), iterations);
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
  async_assign(rv.last_value().read(), v.write());

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
  std::atomic<int> writer_runs{0};
  std::atomic<int> reader_runs{0};
  Async<int> value;

  constexpr int kDirectTasks = 3;
  for (int i = 0; i < kDirectTasks; ++i)
  {
    sched.schedule([](std::atomic<int>* counter) -> AsyncTask {
      counter->fetch_add(1, std::memory_order_relaxed);
      co_return;
    }(&direct_counter));
  }

  constexpr int kWrittenValue = 42;
  constexpr int kDelayMs = 20;
  sched.schedule(
      [](WriteBuffer<int> write_buffer, int value_to_write, int delay_ms, std::atomic<int>* runs) -> AsyncTask {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        runs->fetch_add(1, std::memory_order_relaxed);
        auto& out = co_await write_buffer;
        out = value_to_write;
        co_return;
      }(value.write(), kWrittenValue, kDelayMs, &writer_runs));

  sched.schedule([](ReadBuffer<int> read_buffer, std::atomic<int>* counter, std::atomic<int>* runs) -> AsyncTask {
    runs->fetch_add(1, std::memory_order_relaxed);
    auto& result = co_await read_buffer;
    counter->fetch_add(result, std::memory_order_relaxed);
    co_return;
  }(value.read(), &async_counter, &reader_runs));

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(direct_counter.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(async_counter.load(std::memory_order_relaxed), 0);
  {
    auto paused_read = value.read();
    EXPECT_FALSE(paused_read.await_ready());
    paused_read.release();
  }

  sched.resume();
  sched.run_all();

  auto const direct_result = direct_counter.load(std::memory_order_relaxed);
  auto const async_result = async_counter.load(std::memory_order_relaxed);
  auto const writer_result = writer_runs.load(std::memory_order_relaxed);
  auto const reader_result = reader_runs.load(std::memory_order_relaxed);
  EXPECT_EQ(direct_result, kDirectTasks);
  EXPECT_EQ(async_result, kWrittenValue) << "direct=" << direct_result << ", writers=" << writer_result
                                         << ", readers=" << reader_result;
  EXPECT_EQ(writer_result, 1);
  EXPECT_EQ(reader_result, 1);
}

TEST(TbbScheduler, StressLongAsyncChain)
{
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  Async<int> value = 0;
  constexpr int kChainLength = 4096;

  for (int i = 0; i < kChainLength; ++i)
  {
    value += 1;
  }

  EXPECT_EQ(value.get_wait(), kChainLength);
  sched.run_all();
}

TEST(TbbScheduler, StressConcurrentProducers)
{
  TbbScheduler sched{6};

  std::atomic<int> counter{0};

  constexpr int kProducerThreads = 6;
  constexpr int kTasksPerThread = 512;

  std::vector<std::thread> producers;
  producers.reserve(kProducerThreads);

  for (int t = 0; t < kProducerThreads; ++t)
  {
    producers.emplace_back([&sched, &counter] {
      for (int i = 0; i < kTasksPerThread; ++i)
      {
        sched.schedule([](std::atomic<int>* target) -> AsyncTask {
          target->fetch_add(1, std::memory_order_relaxed);
          co_return;
        }(&counter));
      }
    });
  }

  for (auto& producer : producers)
  {
    producer.join();
  }

  sched.run_all();

  EXPECT_EQ(counter.load(std::memory_order_relaxed), kProducerThreads * kTasksPerThread);
}

TEST(TbbScheduler, DualBackpropStress)
{
  // DebugScheduler sched; //{4};
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  double const base_value = 0.375;
  Dual<double> x = base_value;
  Dual<double> total = 0.0;

  constexpr int kTerms = 128;
  double expected_value = 0.0;
  double expected_grad = 0.0;

  for (int term_index = 0; term_index < kTerms; ++term_index)
  {
    double const shift = static_cast<double>(term_index) * 0.0025;
    Dual<double> term = sin(x + shift) * cos(x - shift);
    Dual<double> new_total = total + term;
    total = new_total;

    double const plus = base_value + shift;
    double const minus = base_value - shift;
    double const term_value = std::sin(plus) * std::cos(minus);
    expected_value += term_value;
    double const derivative = std::cos(plus) * std::cos(minus) - std::sin(plus) * std::sin(minus);
    expected_grad += derivative;
  }

  double const actual_value = total.value.get_wait();
  EXPECT_NEAR(actual_value, expected_value, 1e-9);

  total.grad = 1.0;
  sched.run_all();

  double const actual_grad = x.grad.final().get_wait();
  EXPECT_NEAR(actual_grad, expected_grad, 1e-9);

  sched.run_all();
}
