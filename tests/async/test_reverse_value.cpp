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
  async_assign(fv.final_grad(), v.write());
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
  async_move(std::move(fv.value()), v);
  // auto str_ptr = std::move(fv.value().move_from());
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

  EXPECT_EQ(fv.final_grad().get_wait(), 99);
}
