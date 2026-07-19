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

template <typename T>
template <typename U>
  requires detail::async_assignment_source<T, U>
Async<T>& Async<T>::operator=(U&& rhs)
{
  if constexpr (!is_async_alias_v<T>)
  {
    if constexpr (detail::async_source<U>)
    {
      Async<T> replacement;
      async_initialize(replacement, std::forward<U>(rhs));
      *this = std::move(replacement);
    }
    else
    {
      *this = Async<T>(std::forward<U>(rhs));
    }
  }
  else
  {
    async_assign(*this, std::forward<U>(rhs));
  }
  return *this;
}

/// \brief Waits for the latest readable value without retrieving it.
/// \tparam T Async value type.
template <typename T> void Async<T>::wait() const { this->read().wait(); }

/// \brief Waits for the latest readable value using an explicit scheduler without retrieving it.
/// \tparam T Async value type.
/// \param sched Scheduler used to drive pending work.
template <typename T> void Async<T>::wait(IAsyncScheduler& sched) const { this->read().wait(sched); }

/// \brief Waits for the latest readable value and returns a const reference.
/// \tparam T Async value type.
/// \return Reference to the materialized value.
template <typename T> T const& Async<T>::get_wait() const { return this->read().get_wait(); }

/// \brief Waits for the latest readable value using an explicit scheduler.
/// \tparam T Async value type.
/// \param sched Scheduler used to drive pending work.
/// \return Reference to the materialized value.
template <typename T> T const& Async<T>::get_wait(IAsyncScheduler& sched) const { return this->read().get_wait(sched); }

/// \brief Waits for write access and moves the current value out.
/// \tparam T Async value type.
/// \return Moved value extracted from storage.
template <typename T>
T Async<T>::move_from_wait()
  requires(!is_async_alias_v<T> && std::constructible_from<T, T &&>)
{
  return this->write().move_from_wait();
}

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
