#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>

using namespace uni20;
using namespace uni20::async;

struct DestructionObserver
{
    bool* destroyed_;
    explicit DestructionObserver(bool* flag) : destroyed_(flag) {}
    ~DestructionObserver() { *destroyed_ = true; }
};

TEST(AsyncDestroyTest, DestroyWaitingReader)
{
  Async<int> a;
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> result;

  bool was_destroyed = false;

  auto Reader = [](ReadBuffer<int> in, bool* flag) -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in;
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  {
    WriteBuffer<int> wb = result.write();
    wb.destroy_if_unwritten();

    schedule(Reader(result.read(), &was_destroyed));
    sched.run(); // ensure that Reader blocks at co_await
  }
  // when wb goes out of scope, this should trigger the cancellation of Reader

  EXPECT_EQ(was_destroyed, true);
}

TEST(AsyncDestroyTest, DestroyEnteringReader)
{
  Async<int> a;
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> result;

  bool was_destroyed = false;

  auto Reader = [](ReadBuffer<int> in, bool* flag) -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in;
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  {
    WriteBuffer<int> wb = result.write();
    wb.destroy_if_unwritten();
    schedule(Reader(result.read(), &was_destroyed));
    // do not run the scheduler yet
  }
  // when wb goes out of scope, this marks the EpochContext as cancelled
  EXPECT_EQ(was_destroyed, false);
  // so when we finally run the scheduler, the coroutine will get destroyed when it suspends
  sched.run();
  EXPECT_EQ(was_destroyed, true);
}

TEST(AsyncDestroyTest, DestroyNewReader)
{
  Async<int> a;
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> result;

  bool was_destroyed = false;

  auto Reader = [](ReadBuffer<int> in, bool* flag) -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in;
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  {
    WriteBuffer<int> wb = result.write();
    wb.destroy_if_unwritten();
    // do not run the scheduler yet
  }
  // when wb goes out of scope, this marks the EpochContext as cancelled
  EXPECT_EQ(was_destroyed, false);

  // Now schedule a new reader; this should appear in the sme Epoch
  schedule(Reader(result.read(), &was_destroyed));
  // so when we finally run the scheduler, the coroutine will get destroyed when it suspends
  sched.run();
  EXPECT_EQ(was_destroyed, true);
}

TEST(AsyncDestroyTest, DestroySubsequentReader)
{
  Async<int> a;
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> result;

  bool was_destroyed1 = false;
  bool was_destroyed2 = false;

  auto Reader = [](ReadBuffer<int> in, bool* flag) -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in;
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  {
    WriteBuffer<int> wb = result.write();
    wb.destroy_if_unwritten();
    // do not run the scheduler yet
  }
  // when wb goes out of scope, this marks the EpochContext as cancelled
  EXPECT_EQ(was_destroyed1, false);

  // Now schedule a new reader; this should appear in the sme Epoch
  schedule(Reader(result.read(), &was_destroyed1));

  // Now get another writer; this forces another epoch
  (void)result.write();
  // Now schedule another reader; this should inherit the destroy_if_unwritten() flag from the previous epoch
  schedule(Reader(result.read(), &was_destroyed2));

  // so when we finally run the scheduler, both coroutines should get destroyed
  sched.run();
  EXPECT_EQ(was_destroyed1, true);
  EXPECT_EQ(was_destroyed2, true);
}
