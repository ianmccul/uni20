#pragma once

#include "async_task.hpp"
#include "async_task_promise.hpp"

namespace uni20::async
{

template <IsAsyncTaskPromise T> void BasicAsyncTask<T>::reschedule(BasicAsyncTask<T> task)
{
  TRACE_MODULE(ASYNC, "rescheduling AsyncTask", &task, task.h_);
  task = BasicAsyncTask<T>::make_sole_owner(std::move(task));
  if (task)
  {
    DEBUG_CHECK(task.h_.promise().sched_, "unexpected: task scheduler is not set!");
    TRACE_MODULE(ASYNC, "rescheduling AsyncTask, submitting to queue", &task, task.h_);
    auto sched = task.h_.promise().sched_;
    sched->reschedule(std::move(task));
  }
  else
  {
    DEBUG_TRACE_MODULE(ASYNC, "AsyncTask is not sole-owner");
  }
}

template <IsAsyncTaskPromise T> BasicAsyncTask<T> BasicAsyncTask<T>::make_sole_owner(BasicAsyncTask<T>&& task)
{
  DEBUG_CHECK(task.h_);
  auto& p = task.h_.promise();
  if (p.release_awaiter() == 1)
  {
    // We were the last - reacquire ownership explicitly
    p.add_awaiter();
  }
  else
  {
    // Not the last owner, release ownership
    task.h_ = nullptr;
  }
  return std::move(task);
}

template <IsAsyncTaskPromise T>
bool BasicAsyncTask<T>::can_destroy_coroutine(BasicAsyncTask<T>::handle_type h) const noexcept
{
  if (!h) return true;

  auto const cancelled = cancel_.load(std::memory_order_acquire);
  auto const done = h.done();
  return cancelled || done;
}

template <IsAsyncTaskPromise T> BasicAsyncTask<T>::handle_type BasicAsyncTask<T>::release_handle()
{
  CHECK(h_);
  if (!h_.promise().release_awaiter()) PANIC("Attempt to resume() a non-exclusive AsyncTask");

  bool to_destroy = cancel_.load(std::memory_order_acquire); // h_.promise().is_destroy_on_resume();

  auto handle = h_;
  h_ = nullptr; // Always drop ownership, we are now effectively in a 'moved from' state
  // we need to do this *before* calling h_.resume(), because it is possible that this AsyncTask
  // is on that coroutine frame, and it might get destroyed.

  if (to_destroy)
  {
    CHECK(can_destroy_coroutine(handle), "unexpected destruction of an active AsyncTask without cancellation");
    // if we need to destroy the coroutine, recursively destroy any continuations as well
    while (handle)
    {
      TRACE_MODULE(ASYNC, "Destroying AsyncTask", handle);
      handle = handle.promise().destroy_with_continuation();
    }
  }
  else
  {
    handle.promise().mark_started();
  }
  return handle;
}

template <IsAsyncTaskPromise T> void BasicAsyncTask<T>::resume()
{
  auto handle = this->release_handle();
  TRACE_MODULE(ASYNC, "Resuming AsyncTask", handle);
  if (handle) handle.resume();
  TRACE_MODULE(ASYNC, "returned from coroutine::resume");
}

template <IsAsyncTaskPromise T> void BasicAsyncTask<T>::cancel_on_resume() noexcept
{
  cancel_.store(true, std::memory_order_release);
}

template <IsAsyncTaskPromise T> void BasicAsyncTask<T>::exception_on_resume(std::exception_ptr) noexcept
{}

template <IsAsyncTaskPromise T> BasicAsyncTask<T>& BasicAsyncTask<T>::operator=(BasicAsyncTask<T>&& other) noexcept
{
  // TRACE_MODULE(ASYNC, "AsyncTask move assignment", this, h_, &other, other.h_);
  if (this != &other)
  {
    if (h_ && h_.promise().release_awaiter()) h_.destroy();

    h_ = std::exchange(other.h_, nullptr);
    cancel_.store(other.cancel_.load(std::memory_order_acquire), std::memory_order_release);
  }
  return *this;
}

template <IsAsyncTaskPromise T> void BasicAsyncTask<T>::destroy_owned_coroutine() noexcept
{
  auto handle = h_;
  CHECK(can_destroy_coroutine(handle), "unexpected destruction of an active AsyncTask without cancellation");
  while (handle)
  {
    DEBUG_TRACE_MODULE(ASYNC, "AsyncTask destructor is destroying the coroutine!", this, handle);
    handle = handle.promise().destroy_with_continuation();
  }
  h_ = nullptr;
}

template <IsAsyncTaskPromise T> void BasicAsyncTask<T>::release() noexcept
{
  // TRACE_MODULE(ASYNC, "Destroying AsyncTask", this, h_);
  if (h_ && h_.promise().release_awaiter())
  {
    // Recursively destroy the coroutine and any continuation
    destroy_owned_coroutine();
  }
}

template <IsAsyncTaskPromise T> BasicAsyncTask<T>::~BasicAsyncTask() noexcept
{
  if (h_ && h_.promise().release_awaiter())
  {
    // Recursively destroy the coroutine and any continuation
    destroy_owned_coroutine();
  }
}

template <IsAsyncTaskPromise T> bool BasicAsyncTask<T>::set_scheduler(IScheduler* sched)
{
  if (h_)
  {
    h_.promise().sched_ = sched;
    return true;
  }
  return false;
}

template <IsAsyncTaskPromise T> std::optional<int> BasicAsyncTask<T>::preferred_numa_node() const noexcept
{
  if (!h_) return std::nullopt;
  return h_.promise().preferred_numa_node();
}

template <IsAsyncTaskPromise T> void BasicAsyncTask<T>::set_preferred_numa_node(std::optional<int> node) noexcept
{
  if (h_) h_.promise().set_preferred_numa_node(node);
}

template <IsAsyncTaskPromise T>
BasicAsyncTask<T>::handle_type BasicAsyncTask<T>::await_suspend(BasicAsyncTask<T>::handle_type Outer)
{
  DEBUG_CHECK(h_);
  DEBUG_CHECK(!h_.promise().continuation_);
  // h_.promise().current_awaiter_ = this;
  h_.promise().continuation_ = Outer;
  h_.promise().sched_ = Outer.promise().sched_;
  h_.promise().mark_started();
  auto h_transfer = h_.promise().release_ownership();
  h_ = nullptr; // finish transferring ownership
  CHECK(h_transfer, "error: co_await on an AsyncTask that has shared ownership");
  return h_transfer;
}

template <IsAsyncTaskPromise T> void BasicAsyncTask<T>::await_resume() const noexcept
{
  // if (cancel_)
  // {
  //   if (exception_)
  //     std::rethrow_exception(exception_);
  //   else
  //     throw buffer_cancelled();
  // }
}

// template <typename T> void BasicAsyncTask<T>::set_cancel() noexcept { h_.promise().set_cancel(); }
//
// template <typename T> void BasicAsyncTask<T>::set_exception(std::exception_ptr e) noexcept
// {
//   h_.promise().set_exception(e);
// }
//
} // namespace uni20::async
