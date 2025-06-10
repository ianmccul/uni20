// tests/test_future_value.cpp
#include "async/async_ops.hpp"
#include "async/debug_scheduler.hpp"
#include "async/future_value.hpp"
#include "async/scheduler.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>

using namespace uni20::async;

TEST(FutureValue, BasicWriteRead)
{
  FutureValue<int> fv;
  fv = 42;
  EXPECT_EQ(fv.read().get_wait(), 42);
}

TEST(FutureValue, MoveOnlyType)
{
  using Ptr = std::unique_ptr<std::string>;
  FutureValue<Ptr> fv;
  fv = std::make_unique<std::string>("hello");
  auto str_ptr = fv.value().move_from_wait();
  EXPECT_EQ(*str_ptr, "hello");
}

TEST(FutureValue, AssignFromAsync)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> a = 99;
  FutureValue<int> fv;
  fv = a;

  sched.run_all(); // drive coroutine

  EXPECT_EQ(fv.read().get_wait(), 99);
}

TEST(FutureValue, DeferWrite)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> a;
  Defer d(a);

  a += 10;
  d = 5;

  EXPECT_EQ(a.get_wait(), 15);
}
