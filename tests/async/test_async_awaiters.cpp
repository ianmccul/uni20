#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/awaiters.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>

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
    auto& out = co_await writer.emplace(42);
    EXPECT_EQ(out, 42);
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
    auto& ref = co_await w.emplace(99);
    EXPECT_EQ(ref, 99);
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
    auto& ref = co_await w.emplace(42);
    EXPECT_EQ(ref, 42);
    ++count;
    co_return;
  }
  (a.write(), count);

  auto writer_b = [](WriteBuffer<int> w, int& count) static->AsyncTask
  {
    auto& ref = co_await w.emplace(77);
    EXPECT_EQ(ref, 77);
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
    auto& ref = co_await w.emplace(42);
    EXPECT_EQ(ref, 42);
    ++count;
    co_return;
  }
  (a.write(), count);

  auto writer_b = [](WriteBuffer<int> w, int& count) static->AsyncTask
  {
    auto& ref = co_await w.emplace(77);
    EXPECT_EQ(ref, 77);
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
    auto& ref = co_await w.emplace(42);
    EXPECT_EQ(ref, 42);
    ++count;
    co_return;
  }
  (a.write(), count);

  auto writer_b = [](WriteBuffer<int> w, int& count) static->AsyncTask
  {
    auto& ref = co_await w.emplace(77);
    EXPECT_EQ(ref, 77);
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
    auto& first = co_await writer.emplace(1);
    EXPECT_EQ(first, 1);

    auto& second = co_await writer.emplace(2);
    EXPECT_EQ(second, 2);

    auto& writable_1 = co_await writer;
    EXPECT_EQ(writable_1, 2);
    writable_1 = 3;

    auto& writable_2 = co_await writer;
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

    auto& rebuilt = co_await writer.emplace(5);
    EXPECT_EQ(rebuilt, 5);

    auto taken_2 = co_await writer.take();
    EXPECT_EQ(taken_2, 5);

    auto& final_value = co_await writer.emplace(7);
    EXPECT_EQ(final_value, 7);
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
