#pragma once

#include "async_ops.hpp"
#include "async_task_promise.hpp"
#include "buffers.hpp"

namespace uni20::async
{

template <typename T> T const& Async<T>::get_wait() const { return this->read().get_wait(); }

template <typename T> T const& Async<T>::get_wait(IScheduler& sched) const { return this->read().get_wait(sched); }

template <typename T> T Async<T>::move_from_wait() { return this->write().move_from_wait(); }

template <AsyncTaskAwaitable A> inline auto BasicAsyncTaskPromise::await_transform(A& a)
{
  return AsyncTaskAwaiter<A&>(a);
}

template <AsyncTaskAwaitable A> inline auto BasicAsyncTaskPromise::await_transform(A&& a)
{
  return AsyncTaskAwaiter<std::remove_reference_t<A>>(std::move(a));
}

template <AsyncTaskFactoryAwaitable A> inline auto BasicAsyncTaskPromise::await_transform(A& a)
{
  return AsyncTaskFactoryAwaiter<A&>(a);
}

template <AsyncTaskFactoryAwaitable A> inline auto BasicAsyncTaskPromise::await_transform(A&& a)
{
  return AsyncTaskFactoryAwaiter<std::remove_reference_t<A>>(std::move(a));
}

} // namespace uni20::async
