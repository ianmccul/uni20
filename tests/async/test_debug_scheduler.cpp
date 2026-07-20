#include <uni20/async/debug_scheduler.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace
{

using uni20::async::AsyncTask;
using uni20::async::DebugScheduler;
using uni20::async::DebugSchedulerOptions;
using uni20::async::DebugSchedulerOrder;

std::vector<int> execution_order(DebugSchedulerOptions options, int task_count = 8)
{
  DebugScheduler scheduler(options);
  std::vector<int> result;
  result.reserve(static_cast<std::size_t>(task_count));

  auto record = [](std::vector<int>* output, int value) static -> AsyncTask {
    output->push_back(value);
    co_return;
  };

  for (int value = 0; value < task_count; ++value)
  {
    scheduler.schedule(record(&result, value));
  }
  scheduler.run_all();
  return result;
}

TEST(DebugSchedulerTest, DefaultUsesReverseSubmissionOrder)
{
  EXPECT_EQ(execution_order({}), (std::vector{7, 6, 5, 4, 3, 2, 1, 0}));
}

TEST(DebugSchedulerTest, FifoUsesSubmissionOrder)
{
  EXPECT_EQ(execution_order({.order = DebugSchedulerOrder::fifo}), (std::vector{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(DebugSchedulerTest, ReverseUsesReverseSubmissionOrder)
{
  EXPECT_EQ(execution_order({.order = DebugSchedulerOrder::reverse}), (std::vector{7, 6, 5, 4, 3, 2, 1, 0}));
}

TEST(DebugSchedulerTest, RandomOrderIsAReproduciblePermutation)
{
  DebugSchedulerOptions const options{.order = DebugSchedulerOrder::random, .random_seed = 0x5eed};
  auto const first = execution_order(options);
  auto const second = execution_order(options);
  EXPECT_EQ(first, second);

  auto sorted = first;
  std::ranges::sort(sorted);
  std::vector<int> expected(sorted.size());
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(sorted, expected);
  EXPECT_NE(first, expected);
  std::ranges::reverse(expected);
  EXPECT_NE(first, expected);
}

} // namespace
