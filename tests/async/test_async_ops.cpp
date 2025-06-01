#include "async/async.hpp"
#include "async/async_ops.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include "gtest/gtest.h"

using namespace uni20::async;

TEST(AsyncOpsTest, AddTwoAsyncInts)
{
  DebugScheduler sched;
  set_global_scheduler(&sched); // installs into global `schedule()` dispatch

  Async<int> a = 5;
  Async<int> b = 7;
  Async<int> c = a + b; // launches a coroutine via operator+

  EXPECT_EQ(c.get_wait(), 12);
}

TEST(AsyncOpsTest, AddMixedTypesIntDouble)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> a = 4;
  Async<double> b = 1.5;
  auto c = a + b; // Should deduce Async<double>

  EXPECT_DOUBLE_EQ(c.get_wait(), 5.5);
}

TEST(AsyncOpsTest, AddAsyncAndScalar)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> a = 10;
  auto c = a + 2.5; // should be Async<double>

  EXPECT_DOUBLE_EQ(c.get_wait(), 12.5);
}

TEST(AsyncOpsTest, AddScalarAndAsync)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<float> b = 3.5f;
  auto c = 1 + b; // should be Async<float>

  EXPECT_FLOAT_EQ(c.get_wait(), 4.5f);
}

TEST(AsyncOpsTest, BasicArithmeticOps)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> a = 6;
  Async<double> b = 2.0;

  auto sum = a + b;  // 8.0
  auto diff = a - b; // 4.0
  auto prod = a * b; // 12.0
  auto quot = a / b; // 3.0

  Async<double> x = 1.0;
  x += sum;  // 9.0
  x -= diff; // 5.0
  x *= prod; // 60.0
  x /= quot; // 20.0

  EXPECT_DOUBLE_EQ(x.get_wait(), 20.0);
}
