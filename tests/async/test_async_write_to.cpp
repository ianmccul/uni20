#include "async/async.hpp"
#include "async/awaiters.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>

using namespace uni20::async;

TEST(AsyncWriteTest, WriteValueCorrectly)
{
  DebugScheduler sched;
  Async<int> x;

  auto task = [](WriteBuffer<int> buffer) static->AsyncTask {
    Async<int> x;
    co_await write_to(x.write(), 42);
    co_await write_to(std::move(buffer), co_await x.read());
    co_return;
  }(x.write());

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(x.get_wait(sched), 42);
}
