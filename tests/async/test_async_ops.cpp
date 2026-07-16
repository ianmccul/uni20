#include "gtest/gtest.h"
#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/debug_scheduler.hpp>

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

namespace async_test_types
{
struct AssignmentAware
{
    explicit AssignmentAware(int value_in) : value(value_in) { ++constructions; }
    AssignmentAware(AssignmentAware const& other) : value(other.value) { ++constructions; }
    AssignmentAware(AssignmentAware&& other) noexcept : value(other.value) { ++constructions; }
    ~AssignmentAware() { ++destructions; }

    AssignmentAware& operator=(int new_value)
    {
      value = new_value;
      ++assignments;
      return *this;
    }

    static void reset()
    {
      constructions = 0;
      assignments = 0;
      destructions = 0;
    }

    int value;
    inline static int constructions = 0;
    inline static int assignments = 0;
    inline static int destructions = 0;
};
} // namespace async_test_types

TEST(AsyncOpsTest, WriteProxyAssignmentConstructsThenUsesUnderlyingAssignment)
{
  async_test_types::AssignmentAware::reset();

  DebugScheduler sched;
  set_global_scheduler(&sched);

  {
    Async<async_test_types::AssignmentAware> value;

    schedule([](WriteBuffer<async_test_types::AssignmentAware> out) static -> AsyncTask {
      co_await out = 42;
      co_return;
    }(value.write()));
    schedule([](WriteBuffer<async_test_types::AssignmentAware> out) static -> AsyncTask {
      co_await out = 77;
      co_return;
    }(value.write()));

    sched.run_all();

    EXPECT_EQ(value.get_wait().value, 77);
    EXPECT_EQ(async_test_types::AssignmentAware::constructions, 1);
    EXPECT_EQ(async_test_types::AssignmentAware::assignments, 1);
    EXPECT_EQ(async_test_types::AssignmentAware::destructions, 0);
  }

  EXPECT_EQ(async_test_types::AssignmentAware::destructions, 1);
}

TEST(AsyncOpsTest, WriteProxyEmplaceExplicitlyReconstructs)
{
  async_test_types::AssignmentAware::reset();

  DebugScheduler sched;
  set_global_scheduler(&sched);

  {
    Async<async_test_types::AssignmentAware> value;

    schedule([](WriteBuffer<async_test_types::AssignmentAware> out) static -> AsyncTask {
      auto writer = co_await out;
      writer.emplace(3);
      writer.emplace(9);
      co_return;
    }(value.write()));

    sched.run_all();

    EXPECT_EQ(value.get_wait().value, 9);
    EXPECT_EQ(async_test_types::AssignmentAware::constructions, 2);
    EXPECT_EQ(async_test_types::AssignmentAware::assignments, 0);
    EXPECT_EQ(async_test_types::AssignmentAware::destructions, 1);
  }

  EXPECT_EQ(async_test_types::AssignmentAware::destructions, 2);
}

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

TEST(AsyncOpsTest, BinaryOperationRejectsOutputInputQueueAliasAtSubmission)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> output = 3;
  Async<int> rhs = 4;
  ErrorModeGuard const error_mode;

  EXPECT_THROW(async_binary_op(output, output, rhs, std::plus<>{}), std::runtime_error);
  EXPECT_EQ(output.get_wait(), 3);
}

TEST(AsyncOpsTest, CompoundOperationRejectsRhsQueueAliasAtSubmission)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> value = 3;
  ErrorModeGuard const error_mode;

  EXPECT_THROW(value += value, std::runtime_error);
  EXPECT_EQ(value.get_wait(), 3);
}

TEST(AsyncOpsTest, CompoundOperationExecutionFailurePropagatesThroughOutput)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> value = 3;
  Async<int> rhs = 4;

  async_compound_op(value, rhs, [](int&, int) { throw std::runtime_error("compound failure"); });
  sched.run_all();

  EXPECT_THROW((void)value.get_wait(), std::runtime_error);
}

TEST(AsyncOpsTest, MoveOnlyType)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  using Ptr = std::unique_ptr<std::string>;
  Async<Ptr> dst;

  Ptr src = std::make_unique<std::string>("test-move");
  async_move(dst, std::move(src));

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
  async_assign(std::move(dst), std::move(src));

  sched.run_all();
  EXPECT_EQ(value.get_wait(), 9);
}
