#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <gtest/gtest.h>

using namespace uni20::async;

TEST(AsyncWriteTest, WriteValueCorrectly)
{
  DebugScheduler sched;
  Async<int> x;

  auto task = [](WriteBuffer<int> buffer) static->AsyncTask
  {
    Async<int> x;
    co_await write_to(x.write(), 42);
    co_await write_to(buffer.transfer(), co_await x.read());
    co_return;
  }
  (x.write());

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(x.get_wait(sched), 42);
}
