#include "async/async.hpp"
#include "async/async_ops.hpp"
#include "async/awaiters.hpp"
#include "gtest/gtest.h"

using namespace uni20::async;

namespace
{

struct OperatorCoAwaitRead
{
    struct Awaiter
    {
        using value_type = int;

        bool await_ready() const noexcept { return true; }

        template <typename T> void await_suspend(T) const noexcept {}

        int await_resume() const noexcept { return 0; }
    };

    Awaiter operator co_await() const noexcept { return {}; }
};

struct OperatorCoAwaitWrite
{
    struct Awaiter
    {
        using value_type = int;

        bool await_ready() const noexcept { return true; }

        template <typename T> void await_suspend(T) const noexcept {}

        int& await_resume() const noexcept
        {
          static int value = 0;
          return value;
        }
    };

    Awaiter operator co_await() const noexcept { return {}; }
};

} // namespace

TEST(ConceptTest, AsyncIntSatisfiesConcepts)
{
  static_assert(async_reader<Async<int>>);
  static_assert(async_writer<Async<int>>);
  static_assert(async_like<Async<int>>);
}

TEST(ConceptTest, ReadBufferSatisfiesConcept) { static_assert(read_buffer_awaitable_of<ReadBuffer<int>, int>); }

TEST(ConceptTest, WriteBufferSatisfiesConcepts)
{
  static_assert(write_buffer_awaitable_of<WriteBuffer<int>, int>);
  static_assert(read_write_buffer_awaitable_of<WriteBuffer<int>, int>);
}

TEST(ConceptTest, AsyncDoubleSatisfiesConcepts)
{
  static_assert(async_reader<Async<double>>);
  static_assert(async_writer<Async<double>>);
  static_assert(async_like<Async<double>>);
}

TEST(ConceptTest, OperatorCoAwaitOnlyAwaitableSatisfiesConcepts)
{
  static_assert(read_buffer_awaitable_of<OperatorCoAwaitRead, int>);
  static_assert(write_buffer_awaitable_of<OperatorCoAwaitWrite, int>);
}
