#include "async/debug_scheduler.hpp"
#include "async/tbb_numa_scheduler.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <vector>

#include <oneapi/tbb/info.h>

using namespace uni20::async;

TEST(TbbNumaScheduler, RoundRobinScheduling)
{
  auto system_nodes = oneapi::tbb::info::numa_nodes();
  if (system_nodes.size() <= 1)
  {
    GTEST_SKIP() << "Requires at least two NUMA nodes";
  }

  TbbNumaScheduler scheduler{system_nodes};

  constexpr std::size_t kRounds = 3;
  const std::size_t task_count = scheduler.numa_nodes().size() * kRounds;

  for (std::size_t i = 0; i < task_count; ++i)
  {
    scheduler.schedule([]() static->AsyncTask { co_return; }());
  }

  scheduler.run_all();

  std::vector<std::size_t> counts;
  counts.reserve(scheduler.numa_nodes().size());
  for (int node : scheduler.numa_nodes())
  {
    counts.push_back(scheduler.scheduled_count_for(node));
  }

  auto [min_it, max_it] = std::minmax_element(counts.begin(), counts.end());
  ASSERT_NE(min_it, counts.end());
  EXPECT_LE(*max_it - *min_it, 1U);
}

TEST(TbbNumaScheduler, HonorsPreferredNumaNode)
{
  auto system_nodes = oneapi::tbb::info::numa_nodes();
  TbbNumaScheduler scheduler{system_nodes};

  ASSERT_FALSE(scheduler.numa_nodes().empty());
  int preferred = scheduler.numa_nodes().front();
  std::size_t before = scheduler.scheduled_count_for(preferred);

  auto task = []() static->AsyncTask { co_return; }();
  task.set_preferred_numa_node(preferred);
  scheduler.schedule(std::move(task));
  scheduler.run_all();

  EXPECT_EQ(scheduler.scheduled_count_for(preferred), before + 1);

  for (int node : scheduler.numa_nodes())
  {
    if (node == preferred) continue;
    EXPECT_EQ(scheduler.scheduled_count_for(node), 0U);
  }
}

TEST(TbbNumaScheduler, RunAllDrainsArenas)
{
  TbbNumaScheduler scheduler;
  ScopedScheduler guard(&scheduler);

  std::atomic<int> counter{0};
  constexpr int kTasks = 16;

  for (int i = 0; i < kTasks; ++i)
  {
    scheduler.schedule([](std::atomic<int>* counter) static->AsyncTask {
      counter->fetch_add(1, std::memory_order_relaxed);
      co_return;
    }(&counter));
  }

  scheduler.run_all();

  EXPECT_EQ(counter.load(std::memory_order_relaxed), kTasks);
}
