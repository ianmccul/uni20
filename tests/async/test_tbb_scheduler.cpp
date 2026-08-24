#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/reverse_value.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/async/var.hpp>
#include <uni20/async/var_toys.hpp>
#include <uni20/config.hpp>

#include "../common/env_var_guard.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <latch>
#include <oneapi/tbb/global_control.h>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace uni20::async;

namespace
{
class ErrorModeGuard {
  public:
    ErrorModeGuard() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ~ErrorModeGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};
} // namespace

#if UNI20_DEBUG_ASYNC_TASKS
namespace
{
using uni20::test::EnvVarGuard;

std::filesystem::path make_temp_dir(std::string_view name)
{
  auto const stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  auto path = std::filesystem::temp_directory_path() / (std::string(name) + "-" + std::to_string(stamp));
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::string read_file(std::filesystem::path const& path)
{
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

bool wait_for_dot_file(std::filesystem::path const& dir, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline)
  {
    if (std::filesystem::exists(dir))
    {
      for (auto const& entry : std::filesystem::directory_iterator(dir))
      {
        if (entry.path().extension() == ".dot" &&
            read_file(entry.path()).find("digraph uni20_async_dag") != std::string::npos)
          return true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

} // namespace
#endif

TEST(TbbScheduler, AsyncArithmetic)
{
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  Async<int> a = 1;
  Async<int> b = 2;
  Async<int> c = a + b;

  EXPECT_EQ(c.get_wait(), 3);
}

TEST(TbbScheduler, LightweightBatchExecutesEveryIndexExactlyOnce)
{
  TbbScheduler scheduler{4};
  std::array<std::atomic<int>, 64> counts{};

  scheduler.execute_batch(counts.size(),
                          [&](std::size_t index) { counts[index].fetch_add(1, std::memory_order_relaxed); });

  for (auto const& count : counts)
    EXPECT_EQ(count.load(std::memory_order_relaxed), 1);
}

TEST(TbbScheduler, EmptyLightweightBatchDoesNotInvokeCallable)
{
  TbbScheduler scheduler{4};
  std::atomic<bool> invoked{false};

  scheduler.execute_batch(0, [&](std::size_t) { invoked.store(true, std::memory_order_relaxed); });

  EXPECT_FALSE(invoked.load(std::memory_order_relaxed));
}

TEST(TbbScheduler, LightweightBatchPropagatesException)
{
  TbbScheduler scheduler{4};

  EXPECT_THROW(scheduler.execute_batch(64,
                                       [](std::size_t index) {
                                         if (index == 31) throw std::runtime_error("batch failure");
                                       }),
               std::runtime_error);
}

TEST(TbbScheduler, OneParticipantLightweightBatchItemMayScheduleAndWait)
{
  oneapi::tbb::global_control single_participant(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  TbbScheduler scheduler{1};
  ScopedScheduler scoped(&scheduler);
  Async<int> result;

  scheduler.execute_batch(1, [&](std::size_t) {
    scheduler.schedule([](WriteBuffer<int> output) static -> AsyncTask {
      co_await output = 42;
      co_return;
    }(result.write()));
    EXPECT_EQ(result.get_wait(), 42);
  });
}

#if UNI20_DEBUG_ASYNC_TASKS
TEST(TbbScheduler, ServicesQueuedGraphvizDumpRequests)
{
  auto dir = make_temp_dir("uni20-dag-tbb-request-test");
  EnvVarGuard output_dir("UNI20_DEBUG_DAG_OUTPUT_DIR", dir.string());
  EnvVarGuard prefix("UNI20_DEBUG_DAG_FILE_PREFIX", "tbb");

  uni20::TaskRegistry::request_graphviz_dump();
  TbbScheduler sched{2};
  sched.run_all();

  EXPECT_TRUE(wait_for_dot_file(dir, std::chrono::milliseconds(200)));
  std::filesystem::remove_all(dir);
}
#endif

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

TEST(TbbScheduler, WatchdogDefaultFollowsAsyncDebugMode)
{
  using namespace std::chrono_literals;

  TbbSchedulerWaitOptions const options;
#if UNI20_ASYNC_DEBUG
  ASSERT_TRUE(options.watchdog_timeout.has_value());
  EXPECT_EQ(*options.watchdog_timeout, 5s);
#else
  EXPECT_FALSE(options.watchdog_timeout.has_value());
#endif
}

TEST(TbbScheduler, OneParticipantTopLevelGetWaitExecutesDependency)
{
  oneapi::tbb::global_control single_participant(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  TbbScheduler sched{1};
  ScopedScheduler guard(&sched);

  Async<int> result;
  sched.schedule([](WriteBuffer<int> output) static -> AsyncTask {
    co_await output = 42;
    co_return;
  }(result.write()));

  EXPECT_EQ(result.get_wait(), 42);
}

TEST(TbbScheduler, SaturatedArenaAcceptsAdmissionAndResumptionWithoutParticipation)
{
  using namespace std::chrono_literals;

  TbbScheduler scheduler{1};
  Async<int> input;
  Async<int> output;
  std::latch waiting_for_input{1};
  std::latch blocker_started{1};
  std::latch release_blocker{1};
  std::atomic<int> completed{0};

  scheduler.schedule([](ReadBuffer<int> input, WriteBuffer<int> output, std::latch* waiting) static -> AsyncTask {
    waiting->count_down();
    int const value = co_await input;
    co_await output = value;
  }(input.read(), output.write(), &waiting_for_input));
  waiting_for_input.wait();

  scheduler.schedule([](std::latch* started, std::latch* release) static -> AsyncTask {
    started->count_down();
    release->wait();
    co_return;
  }(&blocker_started, &release_blocker));
  blocker_started.wait();

  std::binary_semaphore submission_returned{0};
  std::jthread submitter([&] {
    scheduler.schedule([](std::atomic<int>* count) static -> AsyncTask {
      count->fetch_add(1, std::memory_order_relaxed);
      co_return;
    }(&completed));

    DebugScheduler publisher_scheduler;
    publisher_scheduler.schedule(
        [](WriteBuffer<int> input) static -> AsyncTask { co_await input = 17; }(input.write()));
    publisher_scheduler.run_all();
    submission_returned.release();
  });

  if (!submission_returned.try_acquire_for(2s))
  {
    release_blocker.count_down();
    submitter.join();
    FAIL() << "TbbScheduler admission or resumption waited for saturated-arena participation";
  }

  release_blocker.count_down();
  scheduler.run_all();
  submitter.join();
  EXPECT_EQ(completed.load(std::memory_order_relaxed), 1);

  DebugScheduler result_scheduler;
  EXPECT_EQ(output.get_wait(result_scheduler), 17);
}

TEST(TbbScheduler, OneParticipantNestedGetWaitSupportsRepeatedSuspension)
{
  using namespace std::chrono_literals;

  ErrorModeGuard error_mode;
  oneapi::tbb::global_control single_participant(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  TbbScheduler sched{1, {.watchdog_timeout = 500ms}};
  ScopedScheduler guard(&sched);

  Async<int> result;
  sched.schedule([](WriteBuffer<int> output) static -> AsyncTask {
    Async<int> first = 1;
    Async<int> first_result = first + 2;
    int const first_value = first_result.get_wait();

    Async<int> second = first_value;
    Async<int> second_result = second * 4;
    int const second_value = second_result.get_wait();

    co_await output = second_value;
    co_return;
  }(result.write()));

  EXPECT_EQ(result.get_wait(), 12);
}

TEST(TbbScheduler, IdleWaitAcceptsWorkSubmittedBeforeWatchdogDeadline)
{
  using namespace std::chrono_literals;

  oneapi::tbb::global_control single_participant(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  TbbScheduler sched{1, {.watchdog_timeout = 500ms}};
  ScopedScheduler guard(&sched);

  Async<int> result;
  auto writer = result.write();
  std::jthread submitter([&sched, writer = std::move(writer)]() mutable {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    sched.schedule([](WriteBuffer<int> output) static -> AsyncTask {
      co_await output = 17;
      co_return;
    }(std::move(writer)));
  });

  EXPECT_EQ(result.get_wait(), 17);
}

TEST(TbbScheduler, RunnableWorkGenerationResetsIdleWatchdog)
{
  using namespace std::chrono_literals;

  oneapi::tbb::global_control single_participant(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  TbbScheduler sched{1, {.watchdog_timeout = 400ms}};
  ScopedScheduler guard(&sched);

  Async<int> result;
  auto writer = result.write();
  std::jthread submitter([&sched, writer = std::move(writer)]() mutable {
    std::this_thread::sleep_for(200ms);
    sched.schedule([]() static -> AsyncTask { co_return; }());

    std::this_thread::sleep_for(300ms);
    sched.schedule([](WriteBuffer<int> output) static -> AsyncTask {
      co_await output = 23;
      co_return;
    }(std::move(writer)));
  });

  EXPECT_EQ(result.get_wait(), 23);
}

TEST(TbbScheduler, IdleWaitWatchdogRaisesTimeout)
{
  using namespace std::chrono_literals;

  ErrorModeGuard error_mode;
  oneapi::tbb::global_control single_participant(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  TbbScheduler sched{1, {.watchdog_timeout = 20ms}};
  ScopedScheduler guard(&sched);

  Async<int> result;
  auto pending_writer = result.write();
  EXPECT_THROW((void)result.get_wait(), async_wait_timeout);
  pending_writer.release();
}

TEST(TbbScheduler, CoroutineAndAsync)
{
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  auto task = []() static -> AsyncTask {
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
    sched.schedule([](std::atomic<int>& c) static -> AsyncTask {
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
    sched.schedule([]() static -> AsyncTask {
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
  async_assign(v.write(), rv.last_value().read());

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
    sched.schedule([](std::atomic<int>* counter) static -> AsyncTask {
      counter->fetch_add(1, std::memory_order_relaxed);
      co_return;
    }(&direct_counter));
  }

  constexpr int kWrittenValue = 42;
  constexpr int kDelayMs = 20;
  sched.schedule(
      [](WriteBuffer<int> write_buffer, int value_to_write, int delay_ms, std::atomic<int>* runs) static -> AsyncTask {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        runs->fetch_add(1, std::memory_order_relaxed);
        co_await write_buffer = value_to_write;
        co_return;
      }(value.write(), kWrittenValue, kDelayMs, &writer_runs));

  sched.schedule(
      [](ReadBuffer<int> read_buffer, std::atomic<int>* counter, std::atomic<int>* runs) static -> AsyncTask {
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
        sched.schedule([](std::atomic<int>* target) static -> AsyncTask {
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

TEST(TbbScheduler, VarBackpropStress)
{
  // DebugScheduler sched; //{4};
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  double const base_value = 0.375;
  Var<double> x = base_value;
  Var<double> total = 0.0;

  constexpr int kTerms = 128;
  double expected_value = 0.0;
  double expected_grad = 0.0;

  for (int term_index = 0; term_index < kTerms; ++term_index)
  {
    double const shift = static_cast<double>(term_index) * 0.0025;
    Var<double> term = sin(x + shift) * cos(x - shift);
    Var<double> new_total = total + term;
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
