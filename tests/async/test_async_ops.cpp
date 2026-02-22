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

TEST(AsyncOpsTest, UnaryNegation)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> value = 21;
  auto negated_value = -value;
  EXPECT_EQ(negated_value.get_wait(), -21);

  Async<int> lhs = 4;
  Async<int> rhs = 6;
  auto summed_async = lhs + rhs; // produces result through coroutine path
  auto negated_sum = -summed_async;
  EXPECT_EQ(negated_sum.get_wait(), -10);
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

TEST(AsyncOpsTest, MoveOnlyType)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  using Ptr = std::unique_ptr<std::string>;
  Async<Ptr> dst;

  Ptr src = std::make_unique<std::string>("test-move");
  async_move(std::move(src), dst);

  Ptr result = dst.move_from_wait(); // Must return by value
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, "test-move");
}

TEST(AsyncOpsTest, AsyncAssignReadWriteSameAsyncDoesNotDeadlock)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> value = 9;
  auto src = value.read();
  auto dst = value.write();
  async_assign(std::move(src), std::move(dst));

  sched.run_all();
  EXPECT_EQ(value.get_wait(), 9);
}

TEST(AsyncBasicTest, EpochQueueResetOnAssignment)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> a;
  int count1 = 0, count2 = 0;
  int v1, v2;

  // This test demonstrates that
  // a = 5;
  // a += 10;
  //
  // will run simultaneously with
  // a = 10;
  // a += 20;

  a = 5;
  schedule([](ReadBuffer<int> a, int& count, int& v) -> AsyncTask {
    v = co_await a;
    ++count;
  }(a.read(), count1, v1));
  a += 10;
  schedule([](ReadBuffer<int> a, int& count, int& v) -> AsyncTask {
    v = co_await a;
    ++count;
  }(a.read(), count1, v1));

  a = 10;
  schedule([](ReadBuffer<int> a, int& count, int& v) -> AsyncTask {
    v = co_await a;
    ++count;
  }(a.read(), count2, v2));
  a += 20;
  schedule([](ReadBuffer<int> a, int& count, int& v) -> AsyncTask {
    v = co_await a;
    ++count;
  }(a.read(), count2, v2));

  // initial state; no tasks have run yet
  EXPECT_EQ(count1, 0);
  EXPECT_EQ(count2, 0);

  // there should be exactly two runnable tasks
  sched.run();
  EXPECT_EQ(count1, 1);
  EXPECT_EQ(count2, 1);

  EXPECT_EQ(v1, 5);
  EXPECT_EQ(v2, 10);

  // next set of tasks should be a += 10 and a += 20, to separate variables
  sched.run();
  EXPECT_EQ(count1, 1);
  EXPECT_EQ(count2, 1);

  EXPECT_EQ(v1, 5);
  EXPECT_EQ(v2, 10);

  // next set of tasks is our second round of immmediate coroutines
  sched.run();
  EXPECT_EQ(count1, 2);
  EXPECT_EQ(count2, 2);

  EXPECT_EQ(v1, 15);
  EXPECT_EQ(v2, 30);
}

TEST(AsyncBasicTest, EpochQueueResetOnAssignmentAsync)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> a;
  int count1 = 0, count2 = 0;
  int v1, v2;

  Async<int> aa = 5;

  // This test demonstrates that
  // Async<int> aa = 5;
  // a = aa;
  // a += 10;
  //
  // will run simultaneously with
  // aa = 10;
  // a = aa;
  // a += 20;

  a = aa;
  schedule([](ReadBuffer<int> a, int& count, int& v) -> AsyncTask {
    v = co_await a;
    ++count;
  }(a.read(), count1, v1));
  a += 10;
  schedule([](ReadBuffer<int> a, int& count, int& v) -> AsyncTask {
    v = co_await a;
    ++count;
  }(a.read(), count1, v1));

  aa = 10;
  a = aa;
  schedule([](ReadBuffer<int> a, int& count, int& v) -> AsyncTask {
    v = co_await a;
    ++count;
  }(a.read(), count2, v2));
  a += 20;
  schedule([](ReadBuffer<int> a, int& count, int& v) -> AsyncTask {
    v = co_await a;
    ++count;
  }(a.read(), count2, v2));

  // initial state; no tasks have run yet
  EXPECT_EQ(count1, 0);
  EXPECT_EQ(count2, 0);

  // there should be exactly two runnable tasks, the initial assignments
  sched.run();

  // there should be exactly two runnable tasks, the first two coroutines
  sched.run();
  EXPECT_EQ(count1, 1);
  EXPECT_EQ(count2, 1);

  EXPECT_EQ(v1, 5);
  EXPECT_EQ(v2, 10);

  // next set of tasks should be a += 10 and a += 20, to separate variables
  sched.run();
  EXPECT_EQ(count1, 1);
  EXPECT_EQ(count2, 1);

  EXPECT_EQ(v1, 5);
  EXPECT_EQ(v2, 10);

  // next set of tasks is our second round of immmediate coroutines
  sched.run();
  EXPECT_EQ(count1, 2);
  EXPECT_EQ(count2, 2);

  EXPECT_EQ(v1, 15);
  EXPECT_EQ(v2, 30);
}
