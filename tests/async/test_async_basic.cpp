#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>

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

TEST(AsyncBasicTest, InPlaceConstructsValue)
{
  Async<std::string> value(10, 'x');
  DebugScheduler sched;

  sched.schedule([](ReadBuffer<std::string> reader) static->AsyncTask {
    auto& str = co_await reader;
    EXPECT_EQ(str, std::string(10, 'x'));
    co_return;
  }(value.read()));

  sched.run_all();
}

TEST(AsyncBasicTest, InPlaceConstructsFromInitializerList)
{
  Async<std::vector<int>> value({1, 2, 3, 4});
  DebugScheduler sched;

  sched.schedule([](ReadBuffer<std::vector<int>> reader) static->AsyncTask {
    auto const& vec = co_await reader;
    EXPECT_EQ(vec.size(), std::size_t{4});
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);
    EXPECT_EQ(vec[3], 4);
    co_return;
  }(value.read()));

  sched.run_all();
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
  }(a.mutate(), count));

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

TEST(AsyncBasicTest, RAII_NoAwaitTriggersDeath)
{
  EXPECT_DEATH(
      [] {
        Async<int> a;

        auto r = a.read();
        auto w = a.write();
        AsyncTask task = []() static->AsyncTask { co_return; }
        ();
        (void)r;
        (void)w;
        (void)task;
      }(),
      "unexpected destruction");
}

TEST(AsyncBasicTest, WriteProxyReleasesEpochs)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  // Initialization path should use WriteBuffer to populate the first value.
  Async<int> uninitialized_value;

  auto init_writer = [](WriteBuffer<int> buf) -> AsyncTask {
    co_await buf = 42;
    co_return;
  }(uninitialized_value.write());

  auto init_reader = [](ReadBuffer<int> buf) -> AsyncTask {
    auto& value = co_await buf;
    EXPECT_EQ(value, 42);
    co_return;
  }(uninitialized_value.read());

  sched.schedule(std::move(init_writer));
  sched.schedule(std::move(init_reader));
  sched.run_all();

  // mutate() now also yields WriteBuffer and must observe the most recent initialized value.
  Async<int> initialized_value = 5;

  auto mutate_writer = [](WriteBuffer<int> buf) -> AsyncTask {
    co_await buf += 42;
    co_return;
  }(initialized_value.mutate());

  auto mutate_reader = [](ReadBuffer<int> buf) -> AsyncTask {
    auto value = co_await buf;
    EXPECT_EQ(value, 47);
    co_return;
  }(initialized_value.read());

  sched.schedule(std::move(mutate_writer));
  sched.schedule(std::move(mutate_reader));
  sched.run_all();
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

TEST(AsyncBasicTest, WriteCommitsAfterAwaitAndMove)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> mutable_value = 0;
  Async<int> write_only;

  auto mutate_task = [](WriteBuffer<int> buffer) -> AsyncTask {
    auto& ref = co_await buffer;
    ref = 17;
    auto moved = std::move(buffer);
    (void)moved;
    co_return;
  }(mutable_value.mutate());

  auto write_task = [](WriteBuffer<int> buffer) -> AsyncTask {
    auto& ref = co_await buffer;
    ref = 23;
    auto moved = std::move(buffer);
    (void)moved;
    co_return;
  }(write_only.write());

  sched.schedule(std::move(mutate_task));
  sched.schedule(std::move(write_task));
  sched.run_all();

  EXPECT_EQ(mutable_value.get_wait(), 17);
  EXPECT_EQ(write_only.get_wait(), 23);
}

TEST(AsyncBasicTest, WriterDisappearsExpectException)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> x = 10;

  int val = 0;

  {
    auto Buf = x.write(); // get a WriteBuffer, but don't use it
    schedule([](ReadBuffer<int> r, int& val) static->AsyncTask {
      try
      {
        val = co_await r;
      }
      catch (buffer_uninitialized const&)
      {
        val = 1;
      }
      catch (...)
      { // wrong exception
        val = 2;
      }
    }(x.read(), val));
  }
  sched.run_all();
  EXPECT_EQ(val, 1);
}

TEST(AsyncBasicTest, MutateDisappears)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> x = 10;

  int val = 0;

  {
    auto Buf = x.mutate(); // get a WriteBuffer from mutate(), but don't use it. Should be no problem.
    schedule([](ReadBuffer<int> r, int& val) static->AsyncTask { val = co_await r; }(x.read(), val));
  }
  sched.run_all();
  EXPECT_EQ(val, 10);
}

TEST(AsyncBasicTest, WriterDisappearsCancelTask)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> x = 10;

  int val = 0;

  {
    auto Buf = x.write(); // get a WriteBuffer, but don't use it

    // the .or_cancel() modifier cancels the coroutine if the buffer is invalid
    schedule([](ReadBuffer<int> r, int& val) static->AsyncTask {
      val = co_await r.or_cancel();
      val = 3; // should never run, since the coroutine will be cancelled
    }(x.read(), val));

    // And we should propogate the 'unwritten' state to the next task
    schedule([](ReadBuffer<int> r, int& val) static->AsyncTask {
      try
      {
        val = co_await r;
      }
      catch (buffer_uninitialized const&)
      {
        val = 1;
      }
      catch (...)
      { // wrong exception
        val = 2;
      }
    }(x.read(), val));
  }
  sched.run_all();
  EXPECT_EQ(val, 1);
}
