#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>

using namespace uni20;
using namespace uni20::async;

// These tests check the coroutine cancel behavior, making sure that
// local variables of the coroutine are destroyed.

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

  auto Reader = [](ReadBuffer<int> in, bool* flag) static -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in.or_cancel();
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  {
    WriteBuffer<int> wb = result.write();
    // wb.writer_require();

    schedule(Reader(result.read(), &was_destroyed));
    sched.run(); // ensure that Reader blocks at co_await
  }
  // when wb goes out of scope, this should trigger the cancellation of Reader
  EXPECT_EQ(was_destroyed, false);

  sched.run(); // we need to run again since the .destroy() will happen on resume

  EXPECT_EQ(was_destroyed, true);
}

TEST(AsyncDestroyTest, DestroyNewReader)
{
  Async<int> a;
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> result;

  bool was_destroyed = false;

  auto Reader = [](ReadBuffer<int> in, bool* flag) static -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in.or_cancel();
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  {
    WriteBuffer<int> wb = result.write();
    // wb.writer_require();
    //  do not run the scheduler yet
  }
  // when wb goes out of scope, this marks the EpochContext as cancelled
  EXPECT_EQ(was_destroyed, false);

  // Now schedule a new reader; this should appear in the sme Epoch
  schedule(Reader(result.read(), &was_destroyed));
  // so when we finally run the scheduler, the coroutine will get destroyed when it suspends
  sched.run_all();
  EXPECT_EQ(was_destroyed, true);
}

// This test is currently disabled. Arguiably, writer_require() should be transitive: if a writer is required at epoch
// n, but there is no writer, and also no writer a a subsequent epoch n+1, then readers at epoch n+1 should also
// cancel.

TEST(AsyncDestroyTest, DestroySubsequentReader)
{
  Async<int> a;
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> result;

  bool was_destroyed1 = false;
  bool was_destroyed2 = false;

  auto Reader = [](ReadBuffer<int> in, bool* flag) static -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in.or_cancel();
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  // get a write buffer.  when wb goes out of scope, this marks the EpochContext as cancelled
  (void)result.write();

  EXPECT_EQ(was_destroyed1, false);

  // Now schedule a new reader; this should appear in the same Epoch
  schedule(Reader(result.read(), &was_destroyed1));

  // Now get another writer; this forces another epoch.
  (void)result.write();
  // Now schedule another reader; this should inherit the writer_require() flag from the previous epoch
  schedule(Reader(result.read(), &was_destroyed2));

  // so when we finally run the scheduler, both coroutines should get destroyed
  sched.run_all();
  EXPECT_EQ(was_destroyed1, true);
  EXPECT_EQ(was_destroyed2, true);
}
