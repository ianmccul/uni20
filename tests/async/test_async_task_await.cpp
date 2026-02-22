#include "async/async.hpp"
#include "async/awaiters.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>

using namespace uni20;
using namespace uni20::async;

/// \brief A coroutine that forwards one value from a read buffer to a write buffer.
AsyncTask assign_task(ReadBuffer<int> readBuf, WriteBuffer<int> writeBuf, int& count)
{
  auto& val = co_await readBuf;
  co_await writeBuf.emplace(val);
  ++count; // count this coroutine
  co_return;
}

TEST(AsyncTaskAwaitTest, AsyncTaskAwait_NestedAssignment)
{
  Async<int> a = 123;
  Async<int> b;

  int count = 0;
  DebugScheduler sched;

  auto outer = [](ReadBuffer<int> a, WriteBuffer<int> b, int& count) static->AsyncTask
  {
    auto task = assign_task(a, std::move(b), count);
    co_await task;
    ++count; // count this coroutine
    co_return;
  }
  (a.read(), b.write(), count);

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
  auto kernel = [](ReadBuffer<int> a, WriteBuffer<int> b, int& count) static->AsyncTask
  {
    auto& val = co_await a;
    co_await b.emplace(val * 2);
    ++count;
    co_return;
  };

  auto outer = [](ReadBuffer<int> in, WriteBuffer<int> final_out, auto kernel_fn, int& count) static->AsyncTask
  {
    // This test checks `co_await AsyncTask` sequencing with a local intermediate channel.
    // The outer coroutine awaits an inner coroutine that writes `tmp`, then consumes `tmp`
    // and writes `final_out`.
    Async<int> tmp;

    // Awaiting the nested task must complete stage 1 before stage 2 starts.
    co_await kernel_fn(in, tmp.write(), count);
    EXPECT_EQ(count, 1); // the inner coroutine must have finished once we get here

    // Stage 2: consume the intermediate channel and emplace the final result.
    auto mid = co_await tmp.read();
    co_await final_out.emplace(mid + 1);
    ++count;
    co_return;
  }
  (input.read(), output.write(), kernel, count);

  sched.schedule(std::move(outer));
  sched.run_all();

  auto result = output.get_wait(sched); // Access directly, test already count
  EXPECT_EQ(result, 11);                // (5 * 2) + 1
  EXPECT_EQ(count, 2);                  // both kernel and outer
}
