#include "gtest/gtest.h"
#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/cuda_task.hpp>

#include <concepts>
#include <coroutine>
#include <type_traits>
#include <utility>

using namespace uni20::async;

namespace
{

struct OperatorCoAwaitRead
{
    struct Awaiter
    {
        using value_type = int;

        [[nodiscard]] bool await_ready() const noexcept { return true; }

        template <typename T> void await_suspend(T) const noexcept {}

        [[nodiscard]] int await_resume() const noexcept { return 0; }
    };

    Awaiter operator co_await() const noexcept { return {}; }
};

struct OperatorCoAwaitWrite
{
    struct Awaiter
    {
        using value_type = int;

        [[nodiscard]] bool await_ready() const noexcept { return true; }

        template <typename T> void await_suspend(T) const noexcept {}

        [[nodiscard]] int& await_resume() const noexcept
        {
          static int value = 0;
          return value;
        }
    };

    Awaiter operator co_await() const noexcept { return {}; }
};

struct TransferValueAwaiter
{
    [[nodiscard]] bool await_ready() const noexcept { return false; }

    BasicTask await_suspend(BasicTask task) const noexcept { return task; }

    [[nodiscard]] int await_resume() const noexcept { return 0; }
};

struct ReturningTaskFactoryAwaiter
{
    [[nodiscard]] bool await_ready() const noexcept { return false; }
    [[nodiscard]] int num_awaiters() const noexcept { return 1; }

    BasicTask await_suspend(TaskFactory factory) const { return factory.take_next(); }

    [[nodiscard]] int await_resume() const noexcept { return 0; }
};

template <class... Awaiter>
concept CanAwaitAll = requires(Awaiter&&... awaiter) { all(std::forward<Awaiter>(awaiter)...); };

template <typename Scheduler>
concept PubliclySchedulesAsyncTask =
    requires(Scheduler& scheduler, AsyncTask&& task) { scheduler.schedule(std::move(task)); };

template <typename Scheduler>
concept PubliclyReschedulesBasicTask =
    requires(Scheduler& scheduler, BasicTask&& task) { scheduler.reschedule(std::move(task)); };

} // namespace

TEST(ConceptTest, SchedulerInterfacesSeparateInitialSubmissionFromInternalRescheduling)
{
  static_assert(std::derived_from<AsyncTask, AsyncTask::base_type>);
  static_assert(std::derived_from<CudaTask, CudaTask::base_type>);
  static_assert(!std::same_as<AsyncTask::promise_type, CudaTask::promise_type>);
  static_assert(std::derived_from<AsyncTask::promise_type, TaskPromiseBase>);
  static_assert(std::derived_from<CudaTask::promise_type, TaskPromiseBase>);
  static_assert(!std::is_polymorphic_v<TaskPromiseBase>);
  static_assert(std::constructible_from<BasicTask, AsyncTask&&>);
  static_assert(std::constructible_from<BasicTask, CudaTask&&>);
  static_assert(!std::constructible_from<TaskHandle, std::coroutine_handle<>, TaskPromiseBase*>);
  static_assert(std::derived_from<IAsyncScheduler, IScheduler>);
  static_assert(PubliclySchedulesAsyncTask<IAsyncScheduler>);
  static_assert(!PubliclySchedulesAsyncTask<IScheduler>);
  static_assert(!PubliclyReschedulesBasicTask<IScheduler>);
  static_assert(!PubliclyReschedulesBasicTask<IAsyncScheduler>);
  static_assert(std::same_as<decltype(std::declval<AsyncTask&>().debug_name("task")), AsyncTask&>);
}

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

TEST(ConceptTest, AllAcceptsOnlyRegistrationStyleAwaiters)
{
  using Immediate = ValueAwaiter<int>;
  using Group = decltype(all(std::declval<Immediate>(), std::declval<Immediate>()));

  static_assert(TaskAwaitable<TransferValueAwaiter>);
  static_assert(!TaskFactoryChildAwaitable<TransferValueAwaiter>);
  static_assert(!CanAwaitAll<TransferValueAwaiter, Immediate>);

  static_assert(!TaskFactoryChildAwaitable<AsyncTask>);
  static_assert(!TaskFactoryChildAwaitable<CudaTask>);
  static_assert(!CanAwaitAll<AsyncTask&, Immediate>);
  static_assert(!CanAwaitAll<CudaTask&, Immediate>);

  static_assert(TaskFactoryAwaitable<Group>);
  static_assert(!TaskFactoryChildAwaitable<Group>);
  static_assert(!CanAwaitAll<Group&, Immediate>);

  static_assert(!TaskFactoryAwaitable<ReturningTaskFactoryAwaiter>);
}
