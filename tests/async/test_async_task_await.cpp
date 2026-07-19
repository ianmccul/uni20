#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>

#include <string>
#include <vector>

using namespace uni20;
using namespace uni20::async;

namespace
{

struct NestedTaskError
{};

} // namespace

/// \brief A coroutine that forwards one value from a read buffer to a write buffer.
AsyncTask assign_task(ReadBuffer<int> readBuf, WriteBuffer<int> writeBuf, int& count)
{
  auto& val = co_await readBuf;
  co_await writeBuf = val;
  ++count; // count this coroutine
  co_return;
}

TEST(AsyncTaskAwaitTest, AsyncTaskAwait_NestedAssignment)
{
  Async<int> a = 123;
  Async<int> b;

  int count = 0;
  DebugScheduler sched;

  auto outer = [](ReadBuffer<int> a, WriteBuffer<int> b, int& count) static -> AsyncTask {
    auto task = assign_task(a, std::move(b), count);
    co_await task;
    ++count; // count this coroutine
    co_return;
  }(a.read(), b.write(), count);

  sched.schedule(std::move(outer));
  sched.run_all();

  EXPECT_EQ(count, 2); // both inner and outer should have run

  auto result = b.get_wait(sched);
  EXPECT_EQ(result, 123);
}

TEST(AsyncTaskAwaitTest, AsyncTaskAwait_IntermediateChannel)
{
  DebugScheduler sched;
  int count = 0;

  Async<int> input = 5;
  Async<int> output;

  // Stage 1: compute an intermediate result into a temporary async channel.
  auto kernel = [](ReadBuffer<int> a, WriteBuffer<int> b, int& count) static -> AsyncTask {
    auto& val = co_await a;
    co_await b = val * 2;
    ++count;
    co_return;
  };

  auto outer = [](ReadBuffer<int> in, WriteBuffer<int> final_out, auto kernel_fn, int& count) static -> AsyncTask {
    // This test checks `co_await AsyncTask` sequencing with a local intermediate channel.
    // The outer coroutine awaits an inner coroutine that writes `tmp`, then consumes `tmp`
    // and writes `final_out`.
    Async<int> tmp;

    // Awaiting the nested task must complete stage 1 before stage 2 starts.
    co_await kernel_fn(in, tmp.write(), count);
    EXPECT_EQ(count, 1); // the inner coroutine must have finished once we get here

    // Stage 2: consume the intermediate channel and write the final result.
    auto mid = co_await tmp.read();
    co_await final_out = mid + 1;
    ++count;
    co_return;
  }(input.read(), output.write(), kernel, count);

  sched.schedule(std::move(outer));
  sched.run_all();

  auto result = output.get_wait(sched); // Access directly, test already count
  EXPECT_EQ(result, 11);                // (5 * 2) + 1
  EXPECT_EQ(count, 2);                  // both kernel and outer
}

TEST(AsyncTaskAwaitTest, NestedTaskPreservesExplicitSchedulerAndReturnsToParentScheduler)
{
  DebugScheduler parent_scheduler;
  DebugScheduler child_scheduler;
  std::vector<std::string> events;

  auto child = [](std::vector<std::string>& events) static -> AsyncTask {
    events.emplace_back("child");
    co_return;
  }(events);
  ASSERT_TRUE(child.set_scheduler(&child_scheduler));

  auto parent = [](AsyncTask child, std::vector<std::string>& events) static -> AsyncTask {
    events.emplace_back("parent before");
    co_await child;
    events.emplace_back("parent after");
  }(std::move(child), events);

  parent_scheduler.schedule(std::move(parent));
  parent_scheduler.run_all();
  EXPECT_EQ(events, (std::vector<std::string>{"parent before"}));

  child_scheduler.run_all();
  EXPECT_EQ(events, (std::vector<std::string>{"parent before", "child"}));

  parent_scheduler.run_all();
  EXPECT_EQ(events, (std::vector<std::string>{"parent before", "child", "parent after"}));
}

TEST(AsyncTaskAwaitTest, NestedTaskOnSameExplicitSchedulerUsesSymmetricTransfer)
{
  DebugScheduler scheduler;
  std::vector<std::string> events;

  auto child = [](std::vector<std::string>& events) static -> AsyncTask {
    events.emplace_back("child");
    co_return;
  }(events);
  ASSERT_TRUE(child.set_scheduler(&scheduler));

  auto parent = [](AsyncTask child, std::vector<std::string>& events) static -> AsyncTask {
    events.emplace_back("parent before");
    co_await child;
    events.emplace_back("parent after");
  }(std::move(child), events);

  scheduler.schedule(std::move(parent));
  scheduler.run();

  EXPECT_TRUE(scheduler.done());
  EXPECT_EQ(events, (std::vector<std::string>{"parent before", "child", "parent after"}));
}

TEST(AsyncTaskAwaitTest, NestedTaskExceptionIsRethrownAtAwaitResume)
{
  DebugScheduler scheduler;
  bool caught = false;

  auto child = []() static -> AsyncTask {
    throw NestedTaskError{};
    co_return;
  }();

  auto parent = [](AsyncTask child, bool& caught) static -> AsyncTask {
    try
    {
      co_await child;
    }
    catch (NestedTaskError const&)
    {
      caught = true;
    }
  }(std::move(child), caught);

  scheduler.schedule(std::move(parent));
  scheduler.run_all();

  EXPECT_TRUE(caught);
}
