#pragma once

/**
 * \file async_impl.hpp
 * \brief Inline implementation of `Async<T>` helpers and task await transforms.
 */

#include "async_ops.hpp"
#include "async_task_promise.hpp"
#include "buffers.hpp"

namespace uni20::async
{

/// \brief Waits for the latest readable value without retrieving it.
/// \tparam T Async value type.
template <typename T> void Async<T>::wait() const { this->read().wait(); }

/// \brief Waits for the latest readable value using an explicit scheduler without retrieving it.
/// \tparam T Async value type.
/// \param sched Scheduler used to drive pending work.
template <typename T> void Async<T>::wait(IScheduler& sched) const { this->read().wait(sched); }

/// \brief Waits for the latest readable value and returns a const reference.
/// \tparam T Async value type.
/// \return Reference to the materialized value.
template <typename T> T const& Async<T>::get_wait() const { return this->read().get_wait(); }

/// \brief Waits for the latest readable value using an explicit scheduler.
/// \tparam T Async value type.
/// \param sched Scheduler used to drive pending work.
/// \return Reference to the materialized value.
template <typename T> T const& Async<T>::get_wait(IScheduler& sched) const { return this->read().get_wait(sched); }

/// \brief Waits for write access and moves the current value out.
/// \tparam T Async value type.
/// \return Moved value extracted from storage.
template <typename T> T Async<T>::move_from_wait() { return this->write().move_from_wait(); }

/// \brief Transforms lvalue awaitables into task-aware awaiters.
/// \tparam A Awaitable type.
/// \param a Awaitable object.
/// \return Task-aware awaiter bound to this promise.
template <AsyncTaskAwaitable A> inline auto BasicAsyncTaskPromise::await_transform(A& a)
{
  return AsyncTaskAwaiter<A&>(a, *this);
}

/// \brief Transforms rvalue awaitables into task-aware awaiters.
/// \tparam A Awaitable type.
/// \param a Awaitable object.
/// \return Task-aware awaiter bound to this promise.
template <AsyncTaskAwaitable A> inline auto BasicAsyncTaskPromise::await_transform(A&& a)
{
  return AsyncTaskAwaiter<std::remove_reference_t<A>>(std::move(a), *this);
}

/// \brief Transforms lvalue awaitable factories into task-aware awaiters.
/// \tparam A Awaitable-factory type.
/// \param a Awaitable factory object.
/// \return Task-aware awaiter bound to this promise.
template <AsyncTaskFactoryAwaitable A> inline auto BasicAsyncTaskPromise::await_transform(A& a)
{
  return AsyncTaskFactoryAwaiter<A&>(a, *this);
}

/// \brief Transforms rvalue awaitable factories into task-aware awaiters.
/// \tparam A Awaitable-factory type.
/// \param a Awaitable factory object.
/// \return Task-aware awaiter bound to this promise.
template <AsyncTaskFactoryAwaitable A> inline auto BasicAsyncTaskPromise::await_transform(A&& a)
{
  return AsyncTaskFactoryAwaiter<std::remove_reference_t<A>>(std::move(a), *this);
}

} // namespace uni20::async
