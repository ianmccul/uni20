#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace uni20;
using namespace uni20::async;

TEST(AsyncMoveTest, MoveConstructKeepsStorage)
{
  Async<int> original = 7;
  Async<int> moved(std::move(original));

  DebugScheduler sched;
  sched.schedule([](ReadBuffer<int> reader) static -> AsyncTask {
    auto& value = co_await reader;
    EXPECT_EQ(value, 7);
    co_return;
  }(moved.read()));

  sched.run_all();
}

TEST(AsyncMoveTest, MoveAssignPreservesQueue)
{
  Async<int> lhs = 1;
  Async<int> rhs = 2;
  lhs = std::move(rhs);

  DebugScheduler sched;
  sched.schedule([](WriteBuffer<int> writer) static -> AsyncTask {
    auto& value = co_await writer;
    value = 9;
    co_return;
  }(lhs.mutate()));

  sched.schedule([](ReadBuffer<int> reader) static -> AsyncTask {
    auto& value = co_await reader;
    EXPECT_EQ(value, 9);
    co_return;
  }(lhs.read()));

  sched.run_all();
}

TEST(AsyncMoveTest, DeferredViewRetainsExternalOwner)
{
  auto backing = std::make_shared<int>(5);
  Async<int> view(deferred, backing);

  Async<int> moved_view(std::move(view));
  backing.reset();

  DebugScheduler sched;
  sched.schedule([](ReadBuffer<int> reader) static -> AsyncTask {
    auto& value = co_await reader;
    EXPECT_EQ(value, 5);
    co_return;
  }(moved_view.read()));

  sched.run_all();
}

TEST(AsyncMoveTest, AsyncMoveTransfersValue)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  using Ptr = std::unique_ptr<int>;
  Async<Ptr> src = std::make_unique<int>(42);
  Async<Ptr> dst;

  async_move(src, dst);

  sched.run_all();

  Ptr received = dst.move_from_wait();
  ASSERT_TRUE(received);
  EXPECT_EQ(*received, 42);

  Ptr source_remaining = src.move_from_wait();
  EXPECT_FALSE(source_remaining);

  reset_global_scheduler();
}

TEST(AsyncMoveTest, AsyncMoveFromRvalueAsync)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  using Ptr = std::unique_ptr<int>;
  Async<Ptr> dst;

  async_move(Async<Ptr>(std::make_unique<int>(11)), dst);

  sched.run_all();

  Ptr received = dst.move_from_wait();
  ASSERT_TRUE(received);
  EXPECT_EQ(*received, 11);

  reset_global_scheduler();
}

TEST(AsyncMoveTest, AsyncMoveFromValueBuffersOnStack)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  using Ptr = std::unique_ptr<int>;
  Ptr payload = std::make_unique<int>(7);
  Async<Ptr> dst;

  async_move(std::move(payload), dst);

  sched.run_all();

  Ptr received = dst.move_from_wait();
  ASSERT_TRUE(received);
  EXPECT_EQ(*received, 7);
  EXPECT_FALSE(payload);

  reset_global_scheduler();
}
