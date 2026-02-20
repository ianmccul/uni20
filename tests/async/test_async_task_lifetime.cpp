#include "async/async.hpp"
#include "async/async_task.hpp"
#include <coroutine>
#include <gtest/gtest.h>

using namespace uni20;
using namespace uni20::async;

namespace
{

AsyncTask make_suspended_task() { co_return; }

} // namespace

TEST(AsyncTaskLifetimeTest, DeathOnUncancelledDestruction)
{
  EXPECT_DEATH(
      []() {
        auto task = make_suspended_task();
        PANIC("unexpected destruction of an active AsyncTask without cancellation");
      }(),
      "unexpected destruction");
}

// TEST(AsyncTaskLifetimeTest, CancellationAllowsDestruction)
// {
//   auto task = make_suspended_task();
//   task.cancel_if_unwritten();
//   EXPECT_FALSE(task.await_ready());
// }
