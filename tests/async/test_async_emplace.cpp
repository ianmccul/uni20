#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"

#include <gtest/gtest.h>

#include <memory>

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

struct CountedNonDefault
{
    static void reset()
    {
      constructions = 0;
      destructions = 0;
    }

    explicit CountedNonDefault(int v_in) : v(v_in) { ++constructions; }
    ~CountedNonDefault() { ++destructions; }

    CountedNonDefault(CountedNonDefault const&) = delete;
    CountedNonDefault& operator=(CountedNonDefault const&) = delete;
    CountedNonDefault(CountedNonDefault&&) = delete;
    CountedNonDefault& operator=(CountedNonDefault&&) = delete;

    int v;
    inline static int constructions = 0;
    inline static int destructions = 0;
};

struct CountedDefaultConstructible
{
    static void reset()
    {
      default_constructions = 0;
      value_constructions = 0;
      destructions = 0;
    }

    CountedDefaultConstructible() : v(7) { ++default_constructions; }

    explicit CountedDefaultConstructible(int v_in) : v(v_in) { ++value_constructions; }

    ~CountedDefaultConstructible() { ++destructions; }

    CountedDefaultConstructible(CountedDefaultConstructible const&) = delete;
    CountedDefaultConstructible& operator=(CountedDefaultConstructible const&) = delete;
    CountedDefaultConstructible(CountedDefaultConstructible&&) = delete;
    CountedDefaultConstructible& operator=(CountedDefaultConstructible&&) = delete;

    int v = 7;
    inline static int default_constructions = 0;
    inline static int value_constructions = 0;
    inline static int destructions = 0;
};
} // namespace

TEST(AsyncEmplaceTest, WriteBufferEmplaceConstructsNonDefaultInTask)
{
  Async<NonDefault> value;
  DebugScheduler sched;

  sched.schedule([](WriteBuffer<NonDefault> buffer) static->AsyncTask {
    auto& obj = co_await buffer.emplace(42);
    EXPECT_EQ(obj.v, 42);
    co_return;
  }(value.write()));

  sched.schedule([](ReadBuffer<NonDefault> reader) static->AsyncTask {
    auto& obj = co_await reader;
    EXPECT_EQ(obj.v, 42);
    co_return;
  }(value.read()));

  sched.run_all();
}

TEST(AsyncEmplaceTest, WriteBufferEmplaceForwardsMoveOnlyArguments)
{
  Async<MoveOnly> value;
  DebugScheduler sched;

  auto ptr = std::make_unique<int>(99);
  sched.schedule([](WriteBuffer<MoveOnly> buffer, std::unique_ptr<int> incoming) static->AsyncTask {
    auto& obj = co_await buffer.emplace(std::move(incoming));
    EXPECT_NE(obj.ptr, nullptr);
    EXPECT_EQ(*obj.ptr, 99);
    co_return;
  }(value.write(), std::move(ptr)));

  sched.schedule([](ReadBuffer<MoveOnly> reader) static->AsyncTask {
    auto& obj = co_await reader;
    EXPECT_NE(obj.ptr, nullptr);
    EXPECT_EQ(*obj.ptr, 99);
    co_return;
  }(value.read()));

  sched.run_all();
}

TEST(AsyncEmplaceTest, WriteBufferEmplaceDefersConstructionUntilAwait)
{
  CountedNonDefault::reset();

  {
    Async<CountedNonDefault> value;
    DebugScheduler sched;

    auto writer = value.write();
    EXPECT_EQ(CountedNonDefault::constructions, 0);

    sched.schedule([](WriteBuffer<CountedNonDefault> buffer) static->AsyncTask {
      auto& obj = co_await buffer.emplace(5);
      EXPECT_EQ(obj.v, 5);
      co_return;
    }(std::move(writer)));

    EXPECT_EQ(CountedNonDefault::constructions, 0);
    sched.run_all();
    EXPECT_EQ(CountedNonDefault::constructions, 1);
    EXPECT_EQ(CountedNonDefault::destructions, 0);
  }

  EXPECT_EQ(CountedNonDefault::destructions, 1);
}

TEST(AsyncEmplaceTest, WriteBufferEmplaceNeverDefaultConstructsAsyncValue)
{
  CountedDefaultConstructible::reset();

  {
    Async<CountedDefaultConstructible> value;
    DebugScheduler sched;

    EXPECT_EQ(CountedDefaultConstructible::default_constructions, 0);

    auto writer = value.write();
    EXPECT_EQ(CountedDefaultConstructible::default_constructions, 0);

    sched.schedule([](WriteBuffer<CountedDefaultConstructible> buffer) static->AsyncTask {
      auto& obj = co_await buffer.emplace(13);
      EXPECT_EQ(obj.v, 13);
      co_return;
    }(std::move(writer)));

    sched.schedule([](ReadBuffer<CountedDefaultConstructible> reader) static->AsyncTask {
      auto& ref = co_await reader;
      EXPECT_EQ(ref.v, 13);
      co_return;
    }(value.read()));

    sched.run_all();

    EXPECT_EQ(CountedDefaultConstructible::default_constructions, 0);
    EXPECT_EQ(CountedDefaultConstructible::value_constructions, 1);
    EXPECT_EQ(CountedDefaultConstructible::destructions, 0);
  }

  EXPECT_EQ(CountedDefaultConstructible::destructions, 1);
}

TEST(AsyncEmplaceTest, WriteBufferEmplaceReplacesObjectOnRepeatedCalls)
{
  CountedDefaultConstructible::reset();

  {
    Async<CountedDefaultConstructible> value;
    DebugScheduler sched;

    sched.schedule([](WriteBuffer<CountedDefaultConstructible> buffer) static->AsyncTask {
      auto& first = co_await buffer.emplace(1);
      EXPECT_EQ(first.v, 1);
      auto& second = co_await buffer.emplace(2);
      EXPECT_EQ(second.v, 2);
      co_return;
    }(value.write()));

    sched.schedule([](ReadBuffer<CountedDefaultConstructible> reader) static->AsyncTask {
      auto& ref = co_await reader;
      EXPECT_EQ(ref.v, 2);
      co_return;
    }(value.read()));

    sched.run_all();

    EXPECT_EQ(CountedDefaultConstructible::default_constructions, 0);
    EXPECT_EQ(CountedDefaultConstructible::value_constructions, 2);
    EXPECT_EQ(CountedDefaultConstructible::destructions, 1);
  }

  EXPECT_EQ(CountedDefaultConstructible::destructions, 2);
}

TEST(AsyncEmplaceTest, DeferredControlBlockAndQueueAreInitialized)
{
  Async<NonDefault> value;
  DebugScheduler sched;

  auto initial_control = value.value_ptr();
  auto queue = value.queue();
  EXPECT_FALSE(queue.has_pending_writers());
  EXPECT_EQ(initial_control.get(), nullptr);

  sched.schedule([](WriteBuffer<NonDefault> buffer, Async<NonDefault> & target) static->AsyncTask {
    auto& ref = co_await buffer.emplace(123);
    EXPECT_EQ(ref.v, 123);

    auto reader = target.read();
    auto& read_ref = co_await reader;
    EXPECT_EQ(read_ref.v, 123);
    co_return;
  }(value.write(), value));

  sched.run_all();

  auto control_after = value.value_ptr();
  auto control_after_second = value.value_ptr();
  EXPECT_NE(control_after.get(), nullptr);
  EXPECT_EQ(control_after.get(), control_after_second.get());
  EXPECT_EQ(control_after.get(), value.storage().get());
}
