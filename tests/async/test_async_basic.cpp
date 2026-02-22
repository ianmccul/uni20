#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace uni20;
using namespace uni20::async;

namespace
{
void schedule_record_read(Async<int>& value, std::vector<int>& observed)
{
  schedule([](ReadBuffer<int> reader, std::vector<int> & out) static->AsyncTask {
    out.push_back(co_await reader);
    co_return;
  }(value.read(), observed));
}
} // namespace

TEST(AsyncBasicTest, WriteThenRead)
{
  Async<int> a;
  DebugScheduler sched;

  // Empty capture list: ensures safety if coroutine escapes the local scope (not possible here, but good style)
  auto writer = [](WriteBuffer<int> wbuf) static -> AsyncTask {
    co_await wbuf.emplace(42);
    co_return;
  }(a.write());
  sched.schedule(std::move(writer));
  sched.run_all();

  auto reader = [](ReadBuffer<int> rbuf) static -> AsyncTask {
    auto& r = co_await rbuf;
    EXPECT_EQ(r, 42);
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
    // Pass references explicitly as coroutine parameters so lifetimes are clear.
    sched.schedule([](int i, ReadBuffer<int> rbuf, std::vector<int>& results) static -> AsyncTask {
      auto& r = co_await rbuf;
      // results outlives all scheduled coroutines in this test.
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
  sched.schedule([](ReadBuffer<int> rbuf, int& count) static -> AsyncTask {
    auto& r = co_await rbuf;
    EXPECT_EQ(r, 7);
    ++count;
    co_return;
  }(a.read(), count));
  sched.schedule([](ReadBuffer<int> rbuf, int& count) static -> AsyncTask {
    auto& r = co_await rbuf;
    EXPECT_EQ(r, 7);
    ++count;
    co_return;
  }(a.read(), count));

  // Writer
  sched.schedule([](WriteBuffer<int> wbuf, int& count) static -> AsyncTask {
    auto& w = co_await wbuf;
    w = 8;
    ++count;
    co_return;
  }(a.write(), count));

  // Schedule two new readers that should observe the updated value
  sched.schedule([](ReadBuffer<int> rbuf, int& count) static -> AsyncTask {
    auto& r = co_await rbuf;
    EXPECT_EQ(r, 8);
    ++count;
    co_return;
  }(a.read(), count));
  sched.schedule([](ReadBuffer<int> rbuf, int& count) static -> AsyncTask {
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

TEST(AsyncBasicTest, EpochQueueResetOnAssignment)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> value;
  std::vector<int> first_branch;
  std::vector<int> second_branch;

  value = 5;
  schedule_record_read(value, first_branch);
  value += 10;
  schedule_record_read(value, first_branch);

  value = 10;
  schedule_record_read(value, second_branch);
  value += 20;
  schedule_record_read(value, second_branch);

  sched.run_all();

  EXPECT_EQ(first_branch, (std::vector<int>{5, 15}));
  EXPECT_EQ(second_branch, (std::vector<int>{10, 30}));
}

TEST(AsyncBasicTest, EpochQueueResetOnAssignmentAsync)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> value;
  Async<int> source = 5;
  std::vector<int> first_branch;
  std::vector<int> second_branch;

  value = source;
  schedule_record_read(value, first_branch);
  value += 10;
  schedule_record_read(value, first_branch);

  source = 10;
  value = source;
  schedule_record_read(value, second_branch);
  value += 20;
  schedule_record_read(value, second_branch);

  sched.run_all();

  EXPECT_EQ(first_branch, (std::vector<int>{5, 15}));
  EXPECT_EQ(second_branch, (std::vector<int>{10, 30}));
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

  auto init_writer = [](WriteBuffer<int> buf) static -> AsyncTask {
    co_await buf = 42;
    co_return;
  }(uninitialized_value.write());

  auto init_reader = [](ReadBuffer<int> buf) static -> AsyncTask {
    auto& value = co_await buf;
    EXPECT_EQ(value, 42);
    co_return;
  }(uninitialized_value.read());

  sched.schedule(std::move(init_writer));
  sched.schedule(std::move(init_reader));
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

  auto mutate_task = [](WriteBuffer<int> buffer) static->AsyncTask
  {
    auto& ref = co_await buffer;
    ref = 17;
    auto moved = std::move(buffer);
    (void)moved;
    co_return;
  }
  (mutable_value.write());

  auto write_task = [](WriteBuffer<int> buffer) static->AsyncTask
  {
    auto& ref = co_await buffer.emplace(23);
    EXPECT_EQ(ref, 23);
    auto moved = std::move(buffer);
    (void)moved;
    co_return;
  }
  (write_only.write());

  sched.schedule(std::move(mutate_task));
  sched.schedule(std::move(write_task));
  sched.run_all();

  EXPECT_EQ(mutable_value.get_wait(), 17);
  EXPECT_EQ(write_only.get_wait(), 23);
}

TEST(AsyncBasicTest, WriterAwaitOnUninitializedStorageHandledExceptionDoesNotPropagate)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> value;
  bool writer_saw_exception = false;
  int reader_status = 0;

  schedule([](WriteBuffer<int> writer, bool& saw_exception) static->AsyncTask {
    try
    {
      auto& ref = co_await writer;
      (void)ref;
    }
    catch (buffer_write_uninitialized const&)
    {
      saw_exception = true;
    }
    co_return;
  }(value.write(), writer_saw_exception));

  schedule([](ReadBuffer<int> reader, int& status) static->AsyncTask {
    try
    {
      (void)co_await reader;
      status = 0;
    }
    catch (buffer_write_uninitialized const&)
    {
      status = 1;
    }
    catch (buffer_read_uninitialized const&)
    {
      status = 2;
    }
    catch (...)
    {
      status = 3;
    }
    co_return;
  }(value.read(), reader_status));

  sched.run_all();

  EXPECT_TRUE(writer_saw_exception);
  EXPECT_EQ(reader_status, 2);
}

TEST(AsyncBasicTest, WriterAwaitOnUninitializedStorageUnhandledExceptionPropagates)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> value;
  int reader_status = 0;

  schedule([](WriteBuffer<int> writer) static->AsyncTask {
    auto& ref = co_await writer; // this will throw, since the buffer is uninitialized
    (void)ref;
    co_return;
  }(value.write()));

  // The exception will propagate into this coroutine
  schedule([](ReadBuffer<int> reader, int& status) static->AsyncTask {
    try
    {
      (void)co_await reader; // this will throw the pre-existing exception buffer_write_uninitialized
      status = 0;
    }
    catch (buffer_write_uninitialized const&)
    {
      status = 1;
    }
    catch (buffer_read_uninitialized const&)
    {
      status = 2;
    }
    catch (...)
    {
      status = 3;
    }
    co_return;
  }(value.read(), reader_status));

  sched.run_all();

  EXPECT_EQ(reader_status, 1);
  EXPECT_THROW((void)value.get_wait(), buffer_write_uninitialized);
}

TEST(AsyncBasicTest, UnhandledExceptionAutoPropagatesToAllWriteParameters)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> first;
  Async<int> second;

  schedule([](WriteBuffer<int> first_writer, WriteBuffer<int> second_writer) static->AsyncTask {
    (void)first_writer;
    (void)second_writer;
    throw std::runtime_error("auto-propagate");
    co_return;
  }(first.write(), second.write()));

  sched.run_all();

  EXPECT_THROW((void)first.get_wait(), std::runtime_error);
  EXPECT_THROW((void)second.get_wait(), std::runtime_error);
}

TEST(AsyncBasicTest, PropagateExceptionsToRoutesReadFailuresToWriters)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> source;
  Async<int> first;
  Async<int> second;

  schedule([](WriteBuffer<int> source_writer) static->AsyncTask {
    (void)source_writer;
    throw std::runtime_error("source read failure");
    co_return;
  }(source.write()));

  schedule([](ReadBuffer<int> reader, WriteBuffer<int> first_writer, WriteBuffer<int> second_writer) static->AsyncTask {
    co_await propagate_exceptions_to(first_writer, second_writer);
    (void)co_await reader;
    co_return;
  }(source.read(), first.write(), second.write()));

  sched.run_all();

  EXPECT_THROW((void)first.get_wait(), std::runtime_error);
  EXPECT_THROW((void)second.get_wait(), std::runtime_error);
}

TEST(AsyncBasicTest, PropagateExceptionsToReadBufferRoutesUnhandledException)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> source;
  Async<int> sink = 7;

  schedule([](WriteBuffer<int> source_writer) static->AsyncTask {
    (void)source_writer;
    throw std::runtime_error("source failure");
    co_return;
  }(source.write()));

  schedule([](ReadBuffer<int> source_reader, ReadBuffer<int> sink_reader) static->AsyncTask {
    // Explicitly route unhandled exceptions from this coroutine into a read sink.
    co_await propagate_exceptions_to(sink_reader);

    // Exercise read-sink copy/move registration and teardown before the exception path.
    {
      auto sink_copy = sink_reader;
      auto sink_moved = std::move(sink_copy);
      (void)sink_moved;
    }

    (void)co_await source_reader;
    co_return;
  }(source.read(), sink.read()));

  sched.run_all();

  EXPECT_THROW((void)sink.get_wait(), std::runtime_error);
}

TEST(AsyncBasicTest, PropagateExceptionsToDuplicateWriteSinkIsHarmless)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> source;
  Async<int> out;

  schedule([](WriteBuffer<int> source_writer) static->AsyncTask {
    (void)source_writer;
    throw std::runtime_error("source failure");
    co_return;
  }(source.write()));

  schedule([](ReadBuffer<int> source_reader, WriteBuffer<int> out_writer) static->AsyncTask {
    // out_writer is already auto-registered as a write sink by coroutine argument processing.
    // Explicit registration should remain a no-op from the caller's perspective.
    co_await propagate_exceptions_to(out_writer);
    co_await propagate_exceptions_to(out_writer, out_writer);
    (void)co_await source_reader;
    co_return;
  }(source.read(), out.write()));

  sched.run_all();

  EXPECT_THROW((void)out.get_wait(), std::runtime_error);
}

TEST(AsyncBasicTest, PropagateExceptionsToLocalSinkDestroyedDuringUnwindAborts)
{
  EXPECT_DEATH(
      [] {
        DebugScheduler sched;
        ScopedScheduler scoped(&sched);
        Async<int> out;
        schedule([](WriteBuffer<int> out_writer) static->AsyncTask {
          (void)out_writer;
          Async<int> local;
          {
            auto local_writer = local.write();
            co_await propagate_exceptions_to(local_writer);
            throw std::runtime_error("boom");
          }
          co_return;
        }(out.write()));
        sched.run_all();
      }(),
      "propagate_exceptions_to sink destroyed during exception unwinding");
}

TEST(AsyncBasicTest, WriteBufferDisappears)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> x = 10;

  int val = 0;

  {
    auto Buf = x.write(); // get a WriteBuffer from write(), but don't use it. Should be no problem.
    schedule([](ReadBuffer<int> r, int& val) static->AsyncTask { val = co_await r; }(x.read(), val));
  }
  sched.run_all();
  EXPECT_EQ(val, 10);
}

TEST(AsyncBasicTest, UninitializedCancelTask)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> x; // uninitialized

  int val = 0;

  {
    // the .or_cancel() modifier cancels the coroutine if the buffer is invalid
    schedule([](ReadBuffer<int> r, int& val) static->AsyncTask {
      val = co_await r.or_cancel();
      val = 4; // should never run, since the coroutine will be cancelled
    }(x.read(), val));

    // This doesn't affect subsequent accesses
    schedule([](WriteBuffer<int> w, int& val) static->AsyncTask {
      try
      {
        val = 1;
        co_await w.emplace(val + 1);
      }
      catch (...)
      { // wrong exception
        throw;
        val = 3;
      }
    }(x.write(), val));
  }

  // This one should succeed
  schedule([](ReadBuffer<int> r, int& val) static->AsyncTask { val = co_await r.or_cancel(); }(x.read(), val));

  sched.run_all();
  EXPECT_EQ(val, 2);
}
