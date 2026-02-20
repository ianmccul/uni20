// tests/test_future_value.cpp
#include "async/async_ops.hpp"
#include "async/debug_scheduler.hpp"
#include "async/reverse_value.hpp"
#include "async/scheduler.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>

using namespace uni20::async;

TEST(ReverseValue, BasicWriteRead)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  ReverseValue<int> fv;
  Async<int> v;
  async_assign(fv.last_value().read(), v.write());
  fv = 42;
  EXPECT_EQ(v.read().get_wait(), 42);
}

TEST(ReverseValue, MoveOnlyType)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  using Ptr = std::unique_ptr<std::string>;
  ReverseValue<Ptr> fv;
  Async<Ptr> v;
  async_move(std::move(fv.last_value()), v);
  // auto str_ptr = std::move(fv.last_value().move_from());
  fv = std::make_unique<std::string>("hello");
  EXPECT_EQ(*v.get_wait(), "hello");
}

TEST(ReverseValue, AssignFromAsync)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> a = 99;
  ReverseValue<int> fv;
  fv = a;

  sched.run_all(); // drive coroutine

  EXPECT_EQ(fv.final().get_wait(), 99);
}

TEST(ReverseValue, ChainAccumulationWithCancellation)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  // Exercise the cancellation-safe branch when no upstream gradient is supplied.
  ReverseValue<double> canceled;
  ReverseValue<double> cancel_add;
  ReverseValue<double> cancel_sub;
  Async<double> cancel_async = 3.0;

  canceled += cancel_async; // Async operand
  canceled += cancel_add;   // ReverseValue operand
  canceled -= cancel_sub;   // ReverseValue operand

  cancel_add = 1.75;
  cancel_sub = 0.5;

  auto canceled_input = canceled.output();
  canceled_input.release();

  sched.run_all();
  double const canceled_expected = 3.0 + 1.75 - 0.5;
  EXPECT_DOUBLE_EQ(canceled.final().get_wait(), canceled_expected);

  // Compose a gradient chain that mixes Async and ReverseValue operands.
  ReverseValue<double> base;
  ReverseValue<double> rv_add;
  ReverseValue<double> rv_sub;
  Async<double> async_add = 1.25;
  Async<double> async_sub = 0.75;

  base += async_add;
  base -= async_sub;
  base += rv_add;
  base -= rv_sub;

  rv_add = 2.0;
  rv_sub = 0.5;

  auto base_input = base.output();
  base_input.write(1.0);

  sched.run_all();

  double const base_expected = 1.0 + 1.25 - 0.75 + 2.0 - 0.5;
  EXPECT_DOUBLE_EQ(base.final().get_wait(), base_expected);
}
