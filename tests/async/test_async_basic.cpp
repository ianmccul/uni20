#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>

using namespace uni20;
using namespace uni20::async;

TEST(AsyncBasicTest, WriteThenRead)
{
  Async<int> a;
  DebugScheduler sched;

  // Empty capture list: ensures safety if coroutine escapes the local scope (not possible here, but good style)
  auto writer = [](WriteBuffer<int> wbuf) -> AsyncTask {
    auto& w = co_await wbuf;
    w = 42;
    // Implicit RAII release
    co_return;
  }(a.write());
  sched.schedule(std::move(writer));
  sched.run_all();

  auto reader = [](ReadBuffer<int> rbuf) -> AsyncTask {
    auto& r = co_await rbuf;
    EXPECT_EQ(r, 42);
    // Implicit RAII release
    co_return;
  }(a.read());
  sched.schedule(std::move(reader));
  sched.run_all();
}

TEST(AsyncBasicTest, MultipleReaders)
{
  // Coroutines returned from immediately-invoked lambdas must NOT use capture lists.
  // Any captured variable — whether by reference or value — resides in the lambda's frame,
  // which is destroyed after the lambda returns. If the coroutine suspends and later resumes,
  // it may access freed memory and cause undefined behavior.
  // Instead, pass all external state as function parameters.
  Async<int> a = 99;
  DebugScheduler sched;

  std::vector<int> results(3);
  for (int i = 0; i < 3; ++i)
  {
    // Capture by reference is OK here since &results will out-live the coroutine
    sched.schedule([](int i, ReadBuffer<int> rbuf, std::vector<int>& results) -> AsyncTask {
      auto& r = co_await rbuf;
      // Safe to capture &results: it outlives all coroutines
      results[i] = r;
      co_return;
    }(i, a.read(), results));
  }
  sched.run_all();
  for (int val : results)
    EXPECT_EQ(val, 99);
}

TEST(AsyncBasicTest, WriterWaitsForReaders)
{
  int count = 0;
  Async<int> a = 7;
  DebugScheduler sched;

  // Schedule two readers that hold the value
  sched.schedule([](ReadBuffer<int> rbuf, int& count) -> AsyncTask {
    auto& r = co_await rbuf;
    EXPECT_EQ(r, 7);
    ++count;
    co_return;
  }(a.read(), count));
  sched.schedule([](ReadBuffer<int> rbuf, int& count) -> AsyncTask {
    auto& r = co_await rbuf;
    EXPECT_EQ(r, 7);
    ++count;
    co_return;
  }(a.read(), count));

  // Writer
  sched.schedule([](WriteBuffer<int> wbuf, int& count) -> AsyncTask {
    auto& w = co_await wbuf;
    w = 8;
    ++count;
    co_return;
  }(a.write(), count));

  // Schedule two new readers that should observe the updated value
  sched.schedule([](ReadBuffer<int> rbuf, int& count) -> AsyncTask {
    auto& r = co_await rbuf;
    EXPECT_EQ(r, 8);
    ++count;
    co_return;
  }(a.read(), count));
  sched.schedule([](ReadBuffer<int> rbuf, int& count) -> AsyncTask {
    auto& r = co_await rbuf;
    EXPECT_EQ(r, 8);
    ++count;
    co_return;
  }(a.read(), count));

  // run all of the tasks
  sched.run_all();

  // make sure that we have run all of the coroutines
  EXPECT_EQ(count, 5);
}

TEST(AsyncBasicTest, RAII_NoAwaitSafeDestruction)
{
  Async<int> a;

  // Construct but do not await
  auto r = a.read();
  auto w = a.write();
  AsyncTask task = []() -> AsyncTask { co_return; }();

  // Destructors must not throw or cause side effects
  // No explicit co_await or release
}

TEST(AsyncBasicTest, CopyConstructor)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> original = 42;

  // Copy constructor
  Async<int> copy = original;

  // Check that both are valid and contain the same value
  EXPECT_EQ(original.get_wait(), 42);
  EXPECT_EQ(copy.get_wait(), 42);

  // Mutate only the copy
  copy += 57;

  // The original should still hold 42
  EXPECT_EQ(original.get_wait(), 42);
  EXPECT_EQ(copy.get_wait(), 99);
}
