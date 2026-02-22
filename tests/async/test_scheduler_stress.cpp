#include "async/async.hpp"
#include "async/async_ops.hpp"
#include "async/debug_scheduler.hpp"
#include "async/reverse_value.hpp"
#include "async/tbb_scheduler.hpp"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
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
                  std::atomic<int> * counter) static->AsyncTask {
        auto const& lhs_value = co_await lhs;
        auto const& rhs_value = co_await rhs;
        co_await out.emplace(lhs_value + rhs_value);
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
    schedule([](WriteBuffer<int> out) static->AsyncTask {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      co_await out.emplace(1);
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
      schedule([](ReadBuffer<int> lhs, ReadBuffer<int> rhs, WriteBuffer<int> out, std::atomic<int> * active_tasks,
                  std::atomic<int> * peak_tasks) static->AsyncTask {
        int current = active_tasks->fetch_add(1, std::memory_order_relaxed) + 1;
        update_max(*peak_tasks, current);
        auto const& lhs_value = co_await lhs;
        auto const& rhs_value = co_await rhs;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        co_await out.emplace(lhs_value + rhs_value);
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
