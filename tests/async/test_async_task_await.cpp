#include "async/async.hpp"
#include "async/awaiters.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>

using namespace uni20;
using namespace uni20::async;

/// \brief A coroutine that tcountsfers a value from read to write
AsyncTask assign_task(ReadBuffer<int> readBuf, WriteBuffer<int> writeBuf, int& count)
{
  auto& val = co_await readBuf;
  auto& out = co_await writeBuf;
  out = val;
  ++count; // count this coroutine
  co_return;
}

TEST(AsyncTaskAwaitTest, AsyncTaskAwait_NestedAssignment)
{
  Async<int> a = 123;
  Async<int> b;

  int count = 0;
  DebugScheduler sched;

  auto outer = [](ReadBuffer<int> a, WriteBuffer<int> b, int& count) -> AsyncTask {
    auto task = assign_task(a, dup(b), count);
    co_await task;
    ++count; // count this coroutine
    co_return;
  }(a.read(), b.write(), count);

  sched.schedule(std::move(outer));
  sched.run_all();

  EXPECT_EQ(count, 2); // both inner and outer should have run

  auto result = b.value();
  EXPECT_EQ(result, 123);
}

TEST(AsyncTaskAwaitTest, AsyncTaskAwait_IntermediateChannel)
{
  DebugScheduler sched;
  int count = 0;

  Async<int> input = 5;
  Async<int> output;

  auto kernel = [](ReadBuffer<int> a, WriteBuffer<int> b, int& count) -> AsyncTask {
    auto& val = co_await a;
    auto& out = co_await b;
    out = val * 2;
    ++count;
    co_return;
  };

  auto outer = [](ReadBuffer<int> in, WriteBuffer<int> final_out, auto kernel_fn, int& count) -> AsyncTask {
    // Intermediate async temporary variable that will receive a value from the nested
    // coroutine and pass it on for further computation
    Async<int> tmp;

    // Run a nested coroutine to compute into tmp
    co_await kernel_fn(in, tmp.write(), count);
    EXPECT_EQ(count, 1); // the inner coroutine must have finished once we get here

    // Now continue in this coroutine: read tmp and write to output
    auto mid = co_await tmp.read();
    auto& out = co_await final_out;
    out = mid + 1;
    ++count;
    co_return;
  }(input.read(), output.write(), kernel, count);

  sched.schedule(std::move(outer));
  sched.run_all();

  auto result = output.value(); // Access directly, test already count
  EXPECT_EQ(result, 11);        // (5 * 2) + 1
  EXPECT_EQ(count, 2);          // both kernel and outer
}
