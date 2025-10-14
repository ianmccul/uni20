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

  auto task = [](int& count, ReadBuffer<int> rbuf) -> AsyncTask {
    auto opt = co_await try_await(rbuf);
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(*opt, 123);
    ++count;
    co_return;
  }(count, a.read());

  sched.schedule(std::move(task));
  sched.run_all();
  EXPECT_EQ(count, 1);
}

TEST(AsyncAwaitersTest, TryAwaitPrvalueDummyAwaiter)
{
  struct DummyAwaiter
  {
      bool* ready;
      int* value;

      bool await_ready() const noexcept { return *ready; }
      void await_suspend(AsyncTask) const noexcept {}
      int await_resume() const noexcept { return *value; }
  };

  static_assert(std::is_trivially_move_constructible_v<DummyAwaiter>);

  bool ready_flag = false;
  int produced_value = 0;
  DebugScheduler sched;

  auto task = [](bool& ready, int& produced) -> AsyncTask {
    auto first = co_await try_await(DummyAwaiter{&ready, &produced});
    EXPECT_FALSE(first.has_value());

    ready = true;
    produced = 42;

    auto second = co_await try_await(DummyAwaiter{&ready, &produced});
    EXPECT_TRUE(second.has_value());
    EXPECT_EQ(*second, 42);
    co_return;
  }(ready_flag, produced_value);

  sched.schedule(std::move(task));
  sched.run_all();
}

TEST(AsyncAwaitersTest, TryAwaitFailsThenSucceeds)
{
  int count = 0;
  Async<int> a;
  DebugScheduler sched;

  auto writer = [](int& count, WriteBuffer<int> w) -> AsyncTask {
    auto& ref = co_await w;
    ref = 99;
    ++count;
    co_return;
  }(count, a.write());

  auto task = [](int& count, ReadBuffer<int> rbuf) -> AsyncTask {
    auto opt = co_await try_await(rbuf);
    EXPECT_FALSE(opt.has_value());
    auto& val = co_await rbuf;
    EXPECT_EQ(val, 99);
    ++count;
    co_return;
  }(count, a.read());

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
  auto task = [](ReadBuffer<int> a, ReadBuffer<int> b, int& count, int& sum) -> AsyncTask {
    auto [va, vb] = co_await all(a, b);
    sum = va + vb;
    ++count;
    co_return;
  }(a.read(), b.read(), count, sum);

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

  auto writer_a = [](WriteBuffer<int> w, int& count) -> AsyncTask {
    auto& ref = co_await w;
    ref = 42;
    ++count;
    co_return;
  }(a.write(), count);

  auto writer_b = [](WriteBuffer<int> w, int& count) -> AsyncTask {
    auto& ref = co_await w;
    ref = 77;
    ++count;
    co_return;
  }(b.write(), count);

  auto task = [](ReadBuffer<int> a, ReadBuffer<int> b, int& count) -> AsyncTask {
    auto [va, vb] = co_await all(a, b);
    EXPECT_EQ(va, 42);
    EXPECT_EQ(vb, 77);
    ++count;
    co_return;
  }(a.read(), b.read(), count);

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

  auto writer_a = [](WriteBuffer<int> w, int& count) -> AsyncTask {
    auto& ref = co_await w;
    ref = 42;
    ++count;
    co_return;
  }(a.write(), count);

  auto writer_b = [](WriteBuffer<int> w, int& count) -> AsyncTask {
    auto& ref = co_await w;
    ref = 77;
    ++count;
    co_return;
  }(b.write(), count);

  auto task = [](ReadBuffer<int> a, ReadBuffer<int> b, int& count) -> AsyncTask {
    auto [va, vb] = co_await all(a, b);
    EXPECT_EQ(va, 42);
    EXPECT_EQ(vb, 77);
    ++count;
    co_return;
  }(a.read(), b.read(), count);

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

  auto writer_a = [](WriteBuffer<int> w, int& count) -> AsyncTask {
    auto& ref = co_await w;
    ref = 42;
    ++count;
    co_return;
  }(a.write(), count);

  auto writer_b = [](WriteBuffer<int> w, int& count) -> AsyncTask {
    auto& ref = co_await w;
    ref = 77;
    ++count;
    co_return;
  }(b.write(), count);

  auto task = [](ReadBuffer<int> a, ReadBuffer<int> b, int& count) -> AsyncTask {
    auto [va, vb] = co_await all(a, b);
    EXPECT_EQ(va, 42);
    EXPECT_EQ(vb, 77);
    ++count;
    co_return;
  }(a.read(), b.read(), count);

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
