#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace uni20;
using namespace uni20::async;

namespace
{
struct NonDefault
{
  explicit NonDefault(int v_in) : v(v_in) {}
  int v;
};

struct MoveOnly
{
  explicit MoveOnly(std::unique_ptr<int> ptr_in) : ptr(std::move(ptr_in)) {}
  MoveOnly(MoveOnly&&) noexcept = default;
  MoveOnly& operator=(MoveOnly&&) noexcept = default;
  MoveOnly(MoveOnly const&) = delete;
  MoveOnly& operator=(MoveOnly const&) = delete;

  std::unique_ptr<int> ptr;
};

struct Counting
{
  Counting() = delete;
  explicit Counting(int v_in) : v(v_in) { ++constructed; }
  Counting(Counting&& other) noexcept : v(other.v) { ++constructed; }
  Counting& operator=(Counting&& other) noexcept
  {
    v = other.v;
    return *this;
  }
  Counting(Counting const&) = delete;
  Counting& operator=(Counting const&) = delete;
  ~Counting() = default;

  static void reset() { constructed = 0; }

  int v;
  inline static int constructed = 0;
};

struct DefaultConstructed
{
  DefaultConstructed() { ++constructed; }
  explicit DefaultConstructed(int v_in) : v(v_in) { ++constructed; }

  int v = 7;
  inline static int constructed = 0;

  static void reset() { constructed = 0; }
};
} // namespace

TEST(AsyncEmplaceTest, ConstructsNonDefaultInTask)
{
  Async<NonDefault> value;
  DebugScheduler sched;

  sched.schedule([](EmplaceBuffer<NonDefault> buffer) static -> AsyncTask {
    auto& obj = co_await std::move(buffer)(42);
    EXPECT_EQ(obj.v, 42);
    co_return;
  }(value.emplace()));

  sched.schedule([](ReadBuffer<NonDefault> reader) static -> AsyncTask {
    auto& obj = co_await reader;
    EXPECT_EQ(obj.v, 42);
    co_return;
  }(value.read()));

  sched.run_all();
}

TEST(AsyncEmplaceTest, ForwardsMoveOnlyArguments)
{
  Async<MoveOnly> value;
  DebugScheduler sched;

  auto ptr = std::make_unique<int>(99);
  sched.schedule(
    [](EmplaceBuffer<MoveOnly> buffer, std::unique_ptr<int> incoming) static -> AsyncTask {
      auto& obj = co_await std::move(buffer)(std::move(incoming));
      EXPECT_NE(obj.ptr, nullptr);
      EXPECT_EQ(*obj.ptr, 99);
      co_return;
    }(value.emplace(), std::move(ptr)));

  sched.schedule([](ReadBuffer<MoveOnly> reader) static -> AsyncTask {
    auto& obj = co_await reader;
    EXPECT_NE(obj.ptr, nullptr);
    EXPECT_EQ(*obj.ptr, 99);
    co_return;
  }(value.read()));

  sched.run_all();
}

TEST(AsyncEmplaceTest, DefersConstructionUntilAwait)
{
  Counting::reset();
  Async<Counting> value;
  DebugScheduler sched;

  EXPECT_EQ(Counting::constructed, 0);

  sched.schedule([](EmplaceBuffer<Counting> buffer) static -> AsyncTask {
    auto& obj = co_await std::move(buffer)(5);
    EXPECT_EQ(obj.v, 5);
    co_return;
  }(value.emplace()));

  sched.run_all();
  EXPECT_EQ(Counting::constructed, 1);
}

TEST(AsyncEmplaceTest, LazyDefaultConstructsForReads)
{
  DefaultConstructed::reset();
  Async<DefaultConstructed> value; // no immediate construction
  DebugScheduler sched;

  EXPECT_EQ(DefaultConstructed::constructed, 0);

  sched.schedule([](ReadBuffer<DefaultConstructed> reader) static -> AsyncTask {
    auto& ref = co_await reader;
    EXPECT_EQ(ref.v, 7);
    co_return;
  }(value.read()));

  EXPECT_EQ(DefaultConstructed::constructed, 1);

  sched.schedule([](ReadBuffer<DefaultConstructed> reader) static -> AsyncTask {
    auto& ref = co_await reader;
    EXPECT_EQ(ref.v, 7);
    co_return;
  }(value.read()));

  sched.run_all();
  EXPECT_EQ(DefaultConstructed::constructed, 1);
}

TEST(AsyncEmplaceTest, LazyDefaultConstructsForWriteBuffer)
{
  DefaultConstructed::reset();
  Async<DefaultConstructed> value;
  DebugScheduler sched;

  EXPECT_EQ(DefaultConstructed::constructed, 0);

  auto writer = value.write();
  EXPECT_EQ(DefaultConstructed::constructed, 1);

  sched.schedule([](WriteBuffer<DefaultConstructed> buffer) static -> AsyncTask {
    auto& ref = co_await buffer;
    ref.v = 13;
    co_return;
  }(std::move(writer)));

  sched.schedule([](ReadBuffer<DefaultConstructed> reader) static -> AsyncTask {
    auto& ref = co_await reader;
    EXPECT_EQ(ref.v, 13);
    co_return;
  }(value.read()));

  sched.run_all();
  EXPECT_EQ(DefaultConstructed::constructed, 1);
}

TEST(AsyncEmplaceTest, LazyDefaultConstructsForMutableBuffer)
{
  DefaultConstructed::reset();
  Async<DefaultConstructed> value;
  DebugScheduler sched;

  EXPECT_EQ(DefaultConstructed::constructed, 0);

  auto mutator = value.mutate();
  EXPECT_EQ(DefaultConstructed::constructed, 1);

  sched.schedule([](MutableBuffer<DefaultConstructed> buffer) static -> AsyncTask {
    auto& ref = co_await buffer;
    ref.v = 21;
    co_return;
  }(std::move(mutator)));

  sched.schedule([](ReadBuffer<DefaultConstructed> reader) static -> AsyncTask {
    auto& ref = co_await reader;
    EXPECT_EQ(ref.v, 21);
    co_return;
  }(value.read()));

  sched.run_all();
  EXPECT_EQ(DefaultConstructed::constructed, 1);
}

TEST(AsyncEmplaceTest, DeferredControlBlockAndQueueAreInitialized)
{
  Async<NonDefault> value;
  DebugScheduler sched;

  auto initial_control = value.value_ptr();
  auto queue = value.queue();
  ASSERT_TRUE(queue);
  EXPECT_FALSE(queue->has_pending_writers());
  EXPECT_EQ(initial_control.get(), nullptr);

  sched.schedule([](EmplaceBuffer<NonDefault> buffer, Async<NonDefault>& target) static -> AsyncTask {
    auto& ref = co_await std::move(buffer)(123);
    EXPECT_EQ(ref.v, 123);

    auto reader = target.read();
    auto& read_ref = co_await reader;
    EXPECT_EQ(read_ref.v, 123);
    co_return;
  }(value.emplace(), value));

  sched.run_all();

  auto control_after = value.value_ptr();
  EXPECT_NE(control_after.get(), nullptr);
  EXPECT_FALSE(initial_control.owner_before(control_after));
  EXPECT_FALSE(control_after.owner_before(initial_control));
}

TEST(AsyncEmplaceTest, EmplaceBufferCannotBeUsedTwice)
{
  Async<DefaultConstructed> value;

  auto buffer = value.emplace();
  auto awaiter = std::move(buffer)();
  EXPECT_THROW(std::move(buffer)(), std::logic_error);

  DebugScheduler sched;
  sched.schedule([](EmplaceAwaiter<DefaultConstructed> emplacer) static -> AsyncTask {
    auto& ref = co_await std::move(emplacer);
    EXPECT_EQ(ref.v, 7);
    co_return;
  }(std::move(awaiter)));

  sched.run_all();
}
