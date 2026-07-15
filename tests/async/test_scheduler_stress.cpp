#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/reverse_value.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/async/var.hpp>
#include <uni20/async/var_toys.hpp>
#include <vector>

using namespace uni20::async;

namespace
{
void update_max(std::atomic<int>& target, int value)
{
  int expected = target.load(std::memory_order_relaxed);
  while (expected < value && !target.compare_exchange_weak(expected, value, std::memory_order_relaxed))
  {
    // expected updated with current value on failure
  }
}

struct AdStressFailure
{
    std::size_t branch;
};

struct AdStressCounters
{
    std::atomic<std::size_t> successful_forwards{0};
    std::atomic<std::size_t> failed_forwards{0};
    std::atomic<std::size_t> reverse_bodies{0};
};

Var<double> tracked_branch(Var<double> input, bool fail, std::size_t branch, AdStressCounters* counters)
{
  Var<double> result;

  schedule([](ReadBuffer<double> input, WriteBuffer<double> output, bool fail, std::size_t branch,
              AdStressCounters* counters) static -> AsyncTask {
    auto input_buffer = co_await input.transfer();
    double const value = input_buffer.get();
    input_buffer.release();

    if (fail)
    {
      counters->failed_forwards.fetch_add(1, std::memory_order_relaxed);
      throw AdStressFailure{branch};
    }

    counters->successful_forwards.fetch_add(1, std::memory_order_relaxed);
    co_await output = value;
  }(input.value.read(), result.value.write(), fail, branch, counters));

  schedule([](ReadBuffer<double> input_gradient, WriteBuffer<double> output_gradient,
              AdStressCounters* counters) static -> AsyncTask {
    auto gradient = co_await input_gradient.transfer().or_cancel();
    double const value = gradient.get();
    gradient.release();
    counters->reverse_bodies.fetch_add(1, std::memory_order_relaxed);
    co_await output_gradient += value;
  }(result.grad.input(), input.grad.output(), counters));

  return result;
}

Var<double> make_dead_expression(Var<double>& input, std::size_t branch)
{
  double const scale = 0.2 + 0.025 * static_cast<double>(branch % 5);
  double const bias = 0.01 * static_cast<double>(static_cast<int>(branch % 7) - 3);
  switch (branch % 6)
  {
    case 0:
      return sin(input);
    case 1:
      return cos(input);
    case 2:
      return scale * input;
    case 3:
      return input * scale;
    case 4:
      return input + bias;
    default:
      return input * input;
  }
}

struct AdStressSummary
{
    double maximum_value_error = 0.0;
    double maximum_gradient_error = 0.0;
    std::size_t expected_successful_forwards = 0;
    std::size_t expected_failed_forwards = 0;
    std::size_t successful_forwards = 0;
    std::size_t failed_forwards = 0;
    std::size_t reverse_bodies = 0;
};

AdStressSummary run_var_pruning_stress(int concurrency, std::size_t iterations, std::size_t depth,
                                       std::size_t branches_per_level, std::size_t failures_per_level)
{
  TbbScheduler scheduler{concurrency};
  ScopedScheduler scoped_scheduler(&scheduler);
  AdStressCounters counters;
  AdStressSummary summary;

  for (std::size_t iteration = 0; iteration < iterations; ++iteration)
  {
    scheduler.pause();
    {
      double const initial_value = 0.15 + 0.01 * static_cast<double>(iteration);
      double expected_value = initial_value;
      double expected_gradient = 1.0;
      std::vector<Var<double>> chain;
      chain.reserve(depth + 1);
      chain.emplace_back(initial_value);

      for (std::size_t level = 0; level < depth; ++level)
      {
        double const scale = 0.65 + 0.025 * static_cast<double>((level + iteration) % 7);
        double const bias = 0.02 * static_cast<double>(static_cast<int>((3 * level + iteration) % 9) - 4);
        double const argument = scale * expected_value + bias;
        expected_value = std::sin(argument);
        expected_gradient *= scale * std::cos(argument);
        chain.emplace_back(sin(scale * chain.back() + bias));

        for (std::size_t branch = 0; branch < branches_per_level; ++branch)
        {
          bool const fail = ((branch + level + iteration) % branches_per_level) < failures_per_level;
          std::size_t const branch_id = (iteration * depth + level) * branches_per_level + branch;
          auto unused = tracked_branch(make_dead_expression(chain.back(), branch_id), fail, branch_id, &counters);
          static_cast<void>(unused);
        }
      }

      chain.back().grad = 1.0;
      for (std::size_t level = 1; level + 1 < chain.size(); ++level)
        chain[level].grad.finalize();
      auto value = chain.back().value.read();
      auto gradient = chain.front().grad.backprop().read();
      scheduler.run_all();

      summary.maximum_value_error =
          std::max(summary.maximum_value_error, std::abs(value.get_wait(scheduler) - expected_value));
      summary.maximum_gradient_error =
          std::max(summary.maximum_gradient_error, std::abs(gradient.get_wait(scheduler) - expected_gradient));
    }
    scheduler.run_all();
  }

  summary.expected_failed_forwards = iterations * depth * failures_per_level;
  summary.expected_successful_forwards = iterations * depth * (branches_per_level - failures_per_level);
  summary.successful_forwards = counters.successful_forwards.load(std::memory_order_relaxed);
  summary.failed_forwards = counters.failed_forwards.load(std::memory_order_relaxed);
  summary.reverse_bodies = counters.reverse_bodies.load(std::memory_order_relaxed);
  return summary;
}

} // namespace

TEST(TbbSchedulerStress, LinearChainCompletes)
{
  constexpr int kChainLength = 20;

  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  Async<int> current = 0;

  // Build a long dependency chain where each task increments the previous
  // value. All work is sequenced so the scheduler must advance through every
  // node without stalling.
  for (int i = 0; i < kChainLength; ++i)
  {
    Async<int> next = current + 1;
    current = std::move(next);
  }

  sched.run_all();

  EXPECT_EQ(current.get_wait(), kChainLength);
}

TEST(TbbSchedulerStress, BalancedReductionProducesExpectedSum)
{
  constexpr int kLeafCount = 1 << 9; // 512 leaves, 511 internal nodes

  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  std::vector<Async<int>> level;
  level.reserve(kLeafCount);
  // Seed the reduction tree with constant leaves.
  for (int i = 0; i < kLeafCount; ++i)
  {
    level.emplace_back(1);
  }
  std::vector<Async<int>> next_level;

  std::atomic<int> executed{0};

  // Pairwise combine the current level into the next until a single root
  // remains. Each combine task records that it executed so we can confirm all
  // internal nodes ran.
  while (level.size() > 1)
  {
    next_level.clear();
    next_level.reserve((level.size() + 1) / 2);
    for (std::size_t i = 0; i + 1 < level.size(); i += 2)
    {
      Async<int> combined;
      schedule([](ReadBuffer<int> lhs, ReadBuffer<int> rhs, WriteBuffer<int> out,
                  std::atomic<int>* counter) static -> AsyncTask {
        auto const& lhs_value = co_await lhs;
        auto const& rhs_value = co_await rhs;
        co_await out = lhs_value + rhs_value;
        counter->fetch_add(1, std::memory_order_relaxed);
        co_return;
      }(level[i].read(), level[i + 1].read(), combined.write(), &executed));
      next_level.push_back(std::move(combined));
    }
    if (level.size() % 2 == 1)
    {
      next_level.push_back(level.back());
    }
    level = std::move(next_level);
  }

  sched.run_all();

  ASSERT_EQ(level.size(), 1U);
  EXPECT_EQ(level.front().get_wait(), kLeafCount);
  EXPECT_EQ(executed.load(std::memory_order_relaxed), kLeafCount - 1);
}

TEST(TbbSchedulerStress, BalancedReductionShowsParallelism)
{
  constexpr int kLeafCount = 128;
  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  std::vector<Async<int>> level;
  level.reserve(kLeafCount);
  // Introduce an initial delay at the leaves so the scheduler has work ready
  // before the reduction fan-in begins.
  for (int i = 0; i < kLeafCount; ++i)
  {
    Async<int> leaf;
    schedule([](WriteBuffer<int> out) static -> AsyncTask {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      co_await out = 1;
      co_return;
    }(leaf.write()));
    level.push_back(std::move(leaf));
  }
  std::vector<Async<int>> next_level;

  std::atomic<int> active{0};
  std::atomic<int> max_active{0};

  // Combine leaves in parallel while measuring how many reduction tasks run
  // concurrently. The artificial sleeps widen the window for overlap.
  while (level.size() > 1)
  {
    next_level.clear();
    next_level.reserve((level.size() + 1) / 2);
    for (std::size_t i = 0; i + 1 < level.size(); i += 2)
    {
      Async<int> combined;
      schedule([](ReadBuffer<int> lhs, ReadBuffer<int> rhs, WriteBuffer<int> out, std::atomic<int>* active_tasks,
                  std::atomic<int>* peak_tasks) static -> AsyncTask {
        int current = active_tasks->fetch_add(1, std::memory_order_relaxed) + 1;
        update_max(*peak_tasks, current);
        auto const& lhs_value = co_await lhs;
        auto const& rhs_value = co_await rhs;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        co_await out = lhs_value + rhs_value;
        active_tasks->fetch_sub(1, std::memory_order_relaxed);
        co_return;
      }(level[i].read(), level[i + 1].read(), combined.write(), &active, &max_active));
      next_level.push_back(std::move(combined));
    }
    if (level.size() % 2 == 1)
    {
      next_level.push_back(level.back());
    }
    level = std::move(next_level);
  }

  sched.run_all();

  ASSERT_EQ(level.size(), 1U);
  EXPECT_EQ(level.front().get_wait(), kLeafCount);
  EXPECT_GE(max_active.load(std::memory_order_relaxed), 2);
}

TEST(TbbSchedulerStress, ReverseValueWideAggregation)
{
  constexpr int kLeafCount = 1024;

  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  ReverseValue<double> root;

  std::vector<ReverseValue<double>> plus_nodes;
  plus_nodes.reserve(kLeafCount / 2);
  std::vector<ReverseValue<double>> minus_nodes;
  minus_nodes.reserve(kLeafCount / 2);

  double expected = 1.0;

  // Attach thousands of inputs to the root ReverseValue, mixing direct
  // ReverseValue links with scalar Async values. The expected forward value is
  // computed in parallel so we can validate the final reverse accumulation.
  for (int i = 0; i < kLeafCount; ++i)
  {
    int const branch = i % 3;
    if (branch == 0)
    {
      plus_nodes.emplace_back();
      ReverseValue<double>& node = plus_nodes.back();
      root += node;
      double const value = static_cast<double>((i % 7) + 1);
      node = value;
      expected += value;
    }
    else if (branch == 1)
    {
      minus_nodes.emplace_back();
      ReverseValue<double>& node = minus_nodes.back();
      root -= node;
      double const value = static_cast<double>((i % 11) + 1);
      node = value;
      expected -= value;
    }
    else
    {
      double const scalar = static_cast<double>((i % 5) + 1);
      Async<double> value = scalar;
      root += value;
      expected += scalar;
    }
  }

  auto root_output = root.output();
  root_output.write(1.0);

  sched.run_all();

  EXPECT_DOUBLE_EQ(root.final().get_wait(), expected);
}

TEST(TbbSchedulerStress, ReverseValueLayeredAggregation)
{
  constexpr int kIntermediateCount = 512;
  constexpr int kFanOut = 4;

  TbbScheduler sched{4};
  ScopedScheduler guard(&sched);

  ReverseValue<int> root;
  std::vector<ReverseValue<int>> intermediates;
  intermediates.reserve(kIntermediateCount);

  int expected = 1;

  // Construct a layered graph: the root sums many intermediate ReverseValue
  // nodes, each of which itself aggregates several Async leaves and a direct
  // value. This stresses the reverse accumulation order across wide fan-in
  // levels.
  for (int i = 0; i < kIntermediateCount; ++i)
  {
    intermediates.emplace_back();
    ReverseValue<int>& node = intermediates.back();
    root += node;

    for (int j = 0; j < kFanOut; ++j)
    {
      int const scalar = (i + j) % 5 + 1;
      Async<int> value = scalar;
      node += value;
      expected += scalar;
    }

    int const direct = (i % 9) + 1;
    node = direct;
    expected += direct;
  }

  auto root_output = root.output();
  root_output.write(1);

  sched.run_all();

  EXPECT_EQ(root.final().get_wait(), expected);
}

TEST(TbbSchedulerStress, VarPrunesUnusedSuccessfulAndFailedBranches)
{
  for (int const concurrency : {1, 4})
  {
    SCOPED_TRACE(concurrency);
    auto const summary = run_var_pruning_stress(concurrency, 3, 8, 6, 2);
    EXPECT_LE(summary.maximum_value_error, 1e-13);
    EXPECT_LE(summary.maximum_gradient_error, 1e-13);
    EXPECT_EQ(summary.successful_forwards, summary.expected_successful_forwards);
    EXPECT_EQ(summary.failed_forwards, summary.expected_failed_forwards);
    EXPECT_EQ(summary.reverse_bodies, 0U);
  }
}

TEST(TbbSchedulerStress, VarValueReaderOutlivesProductDescriptor)
{
  TbbScheduler scheduler{4};
  ScopedScheduler scoped_scheduler(&scheduler);
  scheduler.pause();

  Var<double> input = 0.25;
  auto product = std::make_unique<Var<double>>(0.5 * input);
  auto product_value = product->value.read();
  product.reset();
  scheduler.run_all();

  EXPECT_DOUBLE_EQ(product_value.get_wait(scheduler), 0.125);
}

TEST(TbbSchedulerStress, VarObservedFailureSurvivesConcurrentPruning)
{
  TbbScheduler scheduler{4};
  ScopedScheduler scoped_scheduler(&scheduler);
  AdStressCounters counters;
  constexpr std::size_t kObservedBranch = 10'000;
  bool caught_expected_failure = false;

  scheduler.pause();
  {
    Var<double> input = 0.25;
    for (std::size_t branch = 0; branch < 32; ++branch)
    {
      auto unused = tracked_branch(sin(input), true, branch, &counters);
      static_cast<void>(unused);
    }

    auto observed = tracked_branch(cos(input), true, kObservedBranch, &counters);
    auto observed_value = observed.value.read();
    scheduler.run_all();

    try
    {
      static_cast<void>(observed_value.get_wait(scheduler));
    }
    catch (AdStressFailure const& failure)
    {
      caught_expected_failure = failure.branch == kObservedBranch;
    }
  }
  scheduler.run_all();

  EXPECT_TRUE(caught_expected_failure);
  EXPECT_EQ(counters.successful_forwards.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(counters.failed_forwards.load(std::memory_order_relaxed), 33U);
  EXPECT_EQ(counters.reverse_bodies.load(std::memory_order_relaxed), 0U);
}
