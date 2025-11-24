#include "async/async.hpp"
#include "async/async_task.hpp"
#include <coroutine>
#include <gtest/gtest.h>

using namespace uni20;
using namespace uni20::async;

namespace uni20::async
{
struct AsyncTaskTestAccess
{
  static void destroy_owned(AsyncTask& task) { task.destroy_owned_coroutine(); }
  static bool can_destroy(const AsyncTask& task) { return task.can_destroy_coroutine(task.h_); }
  static bool cancelled(const AsyncTask& task) { return task.cancel_.load(std::memory_order_acquire); }
};
}

using uni20::async::AsyncTaskTestAccess;

namespace
{

AsyncTask make_suspended_task()
{
  Async<int> a;
  co_await a.read();
  co_return;
}
} // namespace

TEST(AsyncTaskLifetimeTest, DeathOnUncancelledDestruction)
{
  EXPECT_DEATH(
      []() {
        auto task = make_suspended_task();
        task.h_.promise().mark_started();
        ASSERT_TRUE(task.h_);
        ASSERT_FALSE(task.h_.done());
        ASSERT_TRUE(task.h_.promise().has_started());
        ASSERT_FALSE(AsyncTaskTestAccess::cancelled(task));
        EXPECT_FALSE(task.await_ready());
        ASSERT_FALSE(AsyncTaskTestAccess::can_destroy(task));
        PANIC("unexpected destruction of an active AsyncTask without cancellation");
      }(),
      "unexpected destruction");
}

TEST(AsyncTaskLifetimeTest, CancellationAllowsDestruction)
{
  auto task = make_suspended_task();
  task.cancel_if_unwritten();
  EXPECT_FALSE(task.await_ready());
}
