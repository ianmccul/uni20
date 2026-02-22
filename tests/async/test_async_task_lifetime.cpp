#include "async/async.hpp"
#include "async/async_task.hpp"
#include <cstdlib>
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
      }(),
      "unexpected destruction of an active AsyncTask without cancellation");
}

TEST(AsyncTaskLifetimeTest, CancelOnResumeAllowsDestruction)
{
  EXPECT_EXIT(
      []() {
        {
          auto task = make_suspended_task();
          task.set_cancel_on_resume();
        }
        std::_Exit(0);
      }(),
      ::testing::ExitedWithCode(0),
      "");
}
