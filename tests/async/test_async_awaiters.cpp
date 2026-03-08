#include <uni20/async/async.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <gtest/gtest.h>
#include <cstddef>

using namespace uni20;
using namespace uni20::async;

TEST(AsyncAwaitersTest, TryAwaitReady)
{
  Async<int> a = 123;
  DebugScheduler sched;
  int count = 0;

  auto task = [](int& count, ReadBuffer<int> rbuf) static->AsyncTask
  {
    auto opt = co_await try_await(rbuf);
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(*opt, 123);
    ++count;
    co_return;
  }
  (count, a.read());

  sched.schedule(std::move(task));
  sched.run_all();
  EXPECT_EQ(count, 1);
}

TEST(AsyncAwaitersTest, TryAwaitReadBufferBeforeAndAfterInitialization)
{
  Async<int> value;
  DebugScheduler sched;

  auto writer = [](WriteBuffer<int> writer) static->AsyncTask
  {
    co_await writer = 42;
    EXPECT_EQ(co_await writer, 42);
    co_return;
  }
  (value.write());

  auto reader = [](ReadBuffer<int> reader) static->AsyncTask
  {
    auto first = co_await try_await(reader);
    EXPECT_FALSE(first.has_value());

    auto& waited = co_await reader;
    EXPECT_EQ(waited, 42);

    auto second = co_await try_await(reader);
    EXPECT_TRUE(second.has_value());
    EXPECT_EQ(second->get(), 42);
    co_return;
  }
  (value.read());

  sched.schedule(std::move(reader));
  sched.run_all();
  sched.schedule(std::move(writer));
  sched.run_all();
}

TEST(AsyncAwaitersTest, TryAwaitFailsThenSucceeds)
{
  int count = 0;
  Async<int> a;
  DebugScheduler sched;

  auto writer = [](int& count, WriteBuffer<int> w) static->AsyncTask
  {
    co_await w = 99;
    EXPECT_EQ(co_await w, 99);
    ++count;
    co_return;
  }
  (count, a.write());

  auto task = [](int& count, ReadBuffer<int> rbuf) static->AsyncTask
  {
    auto opt = co_await try_await(rbuf);
    EXPECT_FALSE(opt.has_value());
    auto& val = co_await rbuf;
    EXPECT_EQ(val, 99);
    ++count;
    co_return;
  }
  (count, a.read());

  sched.schedule(std::move(task));
  sched.run_all();
  sched.schedule(std::move(writer));
  sched.run_all();
  EXPECT_EQ(count, 2);
}

TEST(AsyncAwaitersTest, AllAwaiterTwoBuffers)
{
  int count = 0;
  Async<int> a = 10;
  Async<int> b = 20;
  DebugScheduler sched;

  int sum = 0;
  auto task = [](ReadBuffer<int> a, ReadBuffer<int> b, int& count, int& sum) static->AsyncTask
  {
    auto [va, vb] = co_await all(a, b);
    sum = va + vb;
    ++count;
    co_return;
  }
  (a.read(), b.read(), count, sum);

  sched.schedule(std::move(task));
  sched.run_all();
  EXPECT_EQ(sum, 30);
  EXPECT_EQ(count, 1);
}

TEST(AsyncAwaitersTest, AllAwaiterBlockedThenUnblocked)
{
  int count = 0;
  Async<int> a;
  Async<int> b;
  DebugScheduler sched;

  auto writer_a = [](WriteBuffer<int> w, int& count) static->AsyncTask
  {
    co_await w = 42;
    EXPECT_EQ(co_await w, 42);
    ++count;
    co_return;
  }
  (a.write(), count);

  auto writer_b = [](WriteBuffer<int> w, int& count) static->AsyncTask
  {
    co_await w = 77;
    EXPECT_EQ(co_await w, 77);
    ++count;
    co_return;
  }
  (b.write(), count);

  auto task = [](ReadBuffer<int> a, ReadBuffer<int> b, int& count) static->AsyncTask
  {
    auto [va, vb] = co_await all(a, b);
    EXPECT_EQ(va, 42);
    EXPECT_EQ(vb, 77);
    ++count;
    co_return;
  }
  (a.read(), b.read(), count);

  sched.schedule(std::move(task));
  sched.run_all();
  EXPECT_EQ(count, 0);

  sched.schedule(std::move(writer_b));
  sched.run_all();
  EXPECT_EQ(count, 1);

  sched.schedule(std::move(writer_a));
  sched.run_all();
  EXPECT_EQ(count, 3);
}

TEST(AsyncAwaitersTest, AllAwaiterOneUnblocksThenSecond)
{
  int count = 0;
  Async<int> a;
  Async<int> b;
  DebugScheduler sched;

  auto writer_a = [](WriteBuffer<int> w, int& count) static->AsyncTask
  {
    co_await w = 42;
    EXPECT_EQ(co_await w, 42);
    ++count;
    co_return;
  }
  (a.write(), count);

  auto writer_b = [](WriteBuffer<int> w, int& count) static->AsyncTask
  {
    co_await w = 77;
    EXPECT_EQ(co_await w, 77);
    ++count;
    co_return;
  }
  (b.write(), count);

  auto task = [](ReadBuffer<int> a, ReadBuffer<int> b, int& count) static->AsyncTask
  {
    auto [va, vb] = co_await all(a, b);
    EXPECT_EQ(va, 42);
    EXPECT_EQ(vb, 77);
    ++count;
    co_return;
  }
  (a.read(), b.read(), count);

  sched.schedule(std::move(writer_b));
  sched.run_all();
  EXPECT_EQ(count, 1);

  // schedule the task, but it should still be blocked on writer_a
  sched.schedule(std::move(task));
  sched.run_all();
  EXPECT_EQ(count, 1);

  sched.schedule(std::move(writer_a));
  sched.run_all();
  EXPECT_EQ(count, 3);
}

TEST(AsyncAwaitersTest, AllAwaiterNoneBlocked)
{
  int count = 0;
  Async<int> a;
  Async<int> b;
  DebugScheduler sched;

  auto writer_a = [](WriteBuffer<int> w, int& count) static->AsyncTask
  {
    co_await w = 42;
    EXPECT_EQ(co_await w, 42);
    ++count;
    co_return;
  }
  (a.write(), count);

  auto writer_b = [](WriteBuffer<int> w, int& count) static->AsyncTask
  {
    co_await w = 77;
    EXPECT_EQ(co_await w, 77);
    ++count;
    co_return;
  }
  (b.write(), count);

  auto task = [](ReadBuffer<int> a, ReadBuffer<int> b, int& count) static->AsyncTask
  {
    auto [va, vb] = co_await all(a, b);
    EXPECT_EQ(va, 42);
    EXPECT_EQ(vb, 77);
    ++count;
    co_return;
  }
  (a.read(), b.read(), count);

  sched.schedule(std::move(writer_b));
  sched.run_all();
  EXPECT_EQ(count, 1);

  sched.schedule(std::move(writer_a));
  sched.run_all();
  EXPECT_EQ(count, 2);

  // schedule the task, it should be able to run immediately
  sched.schedule(std::move(task));
  sched.run_all();
  EXPECT_EQ(count, 3);
}

TEST(AsyncAwaitersTest, BufferAwaitersSupportRepeatedCoAwait)
{
  Async<int> value;
  DebugScheduler sched;

  int read_sum = 0;
  int maybe_value = 0;
  int cancel_sum = 0;
  bool reader_finished = false;

  auto writer = [](WriteBuffer<int> writer) static->AsyncTask
  {
    co_await writer = 1;
    EXPECT_EQ(co_await writer, 1);

    co_await writer = 2;
    EXPECT_EQ(co_await writer, 2);

    auto writable_1 = co_await writer;
    EXPECT_EQ(writable_1, 2);
    writable_1 = 3;

    auto writable_2 = co_await writer;
    EXPECT_EQ(writable_2, 3);

    auto& storage_1 = co_await writer.storage();
    auto& storage_2 = co_await writer.storage();
    EXPECT_EQ(&storage_1, &storage_2);
    EXPECT_TRUE(storage_1.constructed());
    auto* ptr = storage_1.get();
    EXPECT_NE(ptr, nullptr);
    if (ptr == nullptr) co_return;
    EXPECT_EQ(*ptr, 3);

    auto taken_1 = co_await writer.take();
    EXPECT_EQ(taken_1, 3);
    EXPECT_FALSE(storage_1.constructed());

    co_await writer = 5;
    EXPECT_EQ(co_await writer, 5);

    auto taken_2 = co_await writer.take();
    EXPECT_EQ(taken_2, 5);

    co_await writer = 7;
    EXPECT_EQ(co_await writer, 7);
    co_return;
  }
  (value.write());

  auto reader =
      [](ReadBuffer<int> reader, int& read_sum, int& maybe_value, int& cancel_sum, bool& finished) static->AsyncTask
  {
    auto const& first = co_await reader;
    auto const& second = co_await reader;
    read_sum = first + second;

    auto maybe_1 = co_await reader.maybe();
    auto maybe_2 = co_await reader.maybe();
    EXPECT_NE(maybe_1, nullptr);
    EXPECT_NE(maybe_2, nullptr);
    if (maybe_1 == nullptr || maybe_2 == nullptr) co_return;
    maybe_value = *maybe_1 + *maybe_2;

    auto cancel_1 = co_await reader.or_cancel();
    auto cancel_2 = co_await reader.or_cancel();
    cancel_sum = cancel_1 + cancel_2;

    finished = true;
    co_return;
  }
  (value.read(), read_sum, maybe_value, cancel_sum, reader_finished);

  sched.schedule(std::move(reader));
  sched.run_all();
  EXPECT_FALSE(reader_finished);

  sched.schedule(std::move(writer));
  sched.run_all();

  EXPECT_TRUE(reader_finished);
  EXPECT_EQ(read_sum, 14);
  EXPECT_EQ(maybe_value, 14);
  EXPECT_EQ(cancel_sum, 14);
}

TEST(AsyncAwaitersTest, MoveReadBufferReturnsOwningProxy)
{
  Async<int> value = 19;
  DebugScheduler sched;

  int result = 0;
  bool released = false;

  auto reader = [](ReadBuffer<int> reader, int& result, bool& released) static->AsyncTask
  {
    auto owned = co_await std::move(reader);
    result = owned.get();
    owned.release();
    released = true;
    co_return;
  }
  (value.read(), result, released);

  sched.schedule(std::move(reader));
  sched.run_all();

  EXPECT_TRUE(released);
  EXPECT_EQ(result, 19);
}

TEST(AsyncAwaitersTest, MoveReadBufferGetReleaseAllowsSameCoroutineWriteAcquire)
{
  Async<int> value = 23;
  DebugScheduler sched;

  bool writer_ready_after_release = false;
  int observed = 0;

  auto const task_fn = [](ReadBuffer<int> reader, WriteBuffer<int> writer, bool& writer_ready_after_release,
                          int& observed) static->AsyncTask
  {
    observed = (co_await std::move(reader)).get_release();
    writer_ready_after_release = writer.await_ready();
    co_await writer = observed + 1;
    co_return;
  };
  auto reader = value.read();
  auto writer = value.write();
  auto task = task_fn(std::move(reader), std::move(writer), writer_ready_after_release, observed);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(observed, 23);
  EXPECT_TRUE(writer_ready_after_release);
  EXPECT_EQ(value.get_wait(), 24);
}

TEST(AsyncAwaitersTest, MoveReadBufferMaybeReturnsOwningProxy)
{
  Async<int> value = 27;
  DebugScheduler sched;

  bool has_value = false;
  int read_value = 0;

  auto reader = [](ReadBuffer<int> reader, bool& has_value, int& read_value) static->AsyncTask
  {
    auto maybe = co_await std::move(reader).maybe();
    has_value = static_cast<bool>(maybe);
    if (maybe)
    {
      read_value = maybe->get();
      maybe->release();
    }
    co_return;
  }
  (value.read(), has_value, read_value);

  sched.schedule(std::move(reader));
  sched.run_all();

  EXPECT_TRUE(has_value);
  EXPECT_EQ(read_value, 27);
}

TEST(AsyncAwaitersTest, MoveReadBufferMaybeReportsCancelled)
{
  Async<int> value;
  auto writer = value.write();
  writer.release();

  DebugScheduler sched;
  bool has_value = true;

  auto reader = [](ReadBuffer<int> reader, bool& has_value) static->AsyncTask
  {
    auto maybe = co_await std::move(reader).maybe();
    has_value = static_cast<bool>(maybe);
    if (maybe) maybe->release();
    co_return;
  }
  (value.read(), has_value);

  sched.schedule(std::move(reader));
  sched.run_all();

  EXPECT_FALSE(has_value);
}

TEST(AsyncAwaitersTest, MoveReadBufferOrCancelReturnsOwningProxy)
{
  Async<int> value = 31;
  DebugScheduler sched;

  int result = 0;
  bool released = false;

  auto reader = [](ReadBuffer<int> reader, int& result, bool& released) static->AsyncTask
  {
    auto owned = co_await std::move(reader).or_cancel();
    result = owned.get();
    owned.release();
    released = true;
    co_return;
  }
  (value.read(), result, released);

  sched.schedule(std::move(reader));
  sched.run_all();

  EXPECT_TRUE(released);
  EXPECT_EQ(result, 31);
}

TEST(AsyncAwaitersTest, MoveReadBufferOrCancelThrowsOnCancelled)
{
  Async<int> value;
  auto writer = value.write();
  writer.release();

  DebugScheduler sched;
  bool cancelled = false;

  auto reader = [](ReadBuffer<int> reader, bool& cancelled) static->AsyncTask
  {
    try
    {
      auto owned = co_await std::move(reader).or_cancel();
      owned.release();
    }
    catch (task_cancelled const&)
    {
      cancelled = true;
    }
    co_return;
  }
  (value.read(), cancelled);

  sched.schedule(std::move(reader));
  sched.run_all();

  EXPECT_TRUE(cancelled);
}

TEST(AsyncAwaitersTest, MoveReadBufferKeepsRefcountStableDuringOwnershipTransfer)
{
  Async<int> value = 5;
  DebugScheduler sched;

  std::size_t count_before = 0;
  std::size_t count_during = 0;
  std::size_t count_after = 0;

  auto task = [](ReadBuffer<int> reader, shared_storage<int> const& storage, std::size_t& count_before,
                 std::size_t& count_during,
                 std::size_t& count_after) static->AsyncTask
  {
    count_before = storage.use_count();
    {
      auto owned = co_await std::move(reader);
      count_during = storage.use_count();
      (void)owned.get();
    }
    count_after = storage.use_count();
    co_return;
  }
  (value.read(), value.storage(), count_before, count_during, count_after);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(count_before, 2u);
  EXPECT_EQ(count_during, count_before);
  EXPECT_EQ(count_after, 1u);
}

TEST(AsyncAwaitersTest, MoveReadMaybeKeepsRefcountStableDuringOwnershipTransfer)
{
  Async<int> value = 6;
  DebugScheduler sched;

  std::size_t count_before = 0;
  std::size_t count_during = 0;
  std::size_t count_after = 0;
  bool saw_value = false;

  auto task = [](ReadBuffer<int> reader, shared_storage<int> const& storage, std::size_t& count_before,
                 std::size_t& count_during, std::size_t& count_after,
                 bool& saw_value) static->AsyncTask
  {
    count_before = storage.use_count();
    {
      auto maybe = co_await std::move(reader).maybe();
      if (maybe)
      {
        saw_value = true;
        count_during = storage.use_count();
        maybe->release();
      }
    }
    count_after = storage.use_count();
    co_return;
  }
  (value.read(), value.storage(), count_before, count_during, count_after, saw_value);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_TRUE(saw_value);
  EXPECT_EQ(count_before, 2u);
  EXPECT_EQ(count_during, count_before);
  EXPECT_EQ(count_after, 1u);
}

TEST(AsyncAwaitersTest, MoveReadOrCancelKeepsRefcountStableDuringOwnershipTransfer)
{
  Async<int> value = 7;
  DebugScheduler sched;

  std::size_t count_before = 0;
  std::size_t count_during = 0;
  std::size_t count_after = 0;

  auto task = [](ReadBuffer<int> reader, shared_storage<int> const& storage, std::size_t& count_before,
                 std::size_t& count_during,
                 std::size_t& count_after) static->AsyncTask
  {
    count_before = storage.use_count();
    {
      auto owned = co_await std::move(reader).or_cancel();
      count_during = storage.use_count();
      owned.release();
    }
    count_after = storage.use_count();
    co_return;
  }
  (value.read(), value.storage(), count_before, count_during, count_after);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(count_before, 2u);
  EXPECT_EQ(count_during, count_before);
  EXPECT_EQ(count_after, 1u);
}

TEST(AsyncAwaitersTest, MoveWriteBufferKeepsRefcountStableDuringOwnershipTransfer)
{
  Async<int> value;
  DebugScheduler sched;

  std::size_t count_before = 0;
  std::size_t count_during = 0;
  std::size_t count_after = 0;

  auto task = [](WriteBuffer<int> writer, shared_storage<int> const& storage, std::size_t& count_before,
                 std::size_t& count_during,
                 std::size_t& count_after) static->AsyncTask
  {
    count_before = storage.use_count();
    {
      auto owned = co_await std::move(writer);
      count_during = storage.use_count();
      owned = 11;
    }
    count_after = storage.use_count();
    co_return;
  }
  (value.write(), value.storage(), count_before, count_during, count_after);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(count_before, 2u);
  EXPECT_EQ(count_during, count_before);
  EXPECT_EQ(count_after, 1u);
  EXPECT_EQ(value.get_wait(), 11);
}

TEST(AsyncAwaitersTest, MoveWriteStorageKeepsRefcountStableDuringOwnershipTransfer)
{
  Async<int> value;
  DebugScheduler sched;

  std::size_t count_before = 0;
  std::size_t count_during = 0;
  std::size_t count_after = 0;

  auto task = [](WriteBuffer<int> writer, shared_storage<int> const& storage, std::size_t& count_before,
                 std::size_t& count_during,
                 std::size_t& count_after) static->AsyncTask
  {
    count_before = storage.use_count();
    auto owned_storage = co_await std::move(writer).storage();
    count_during = storage.use_count();
    owned_storage->emplace(13);
    owned_storage.release();
    count_after = storage.use_count();
    co_return;
  }
  (value.write(), value.storage(), count_before, count_during, count_after);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(count_before, 2u);
  EXPECT_EQ(count_during, count_before);
  EXPECT_EQ(count_after, 1u);
  EXPECT_EQ(value.get_wait(), 13);
}
