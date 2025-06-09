#pragma once

#include "async_task.hpp"
#include "async_task_promise.hpp"

namespace uni20::async
{

template <typename T> void BasicAsyncTask<T>::reschedule(BasicAsyncTask<T> task)
{
  TRACE("rescheduling AsyncTask", &task);
  task = BasicAsyncTask<T>::make_sole_owner(std::move(task));
  if (task)
  {
    DEBUG_CHECK(task.h_.promise().sched_, "unexpected: task scheduler is not set!");
    TRACE("rescheduling AsyncTask, submitting to queue", &task, task.h_);
    auto sched = task.h_.promise().sched_;
    sched->reschedule(std::move(task));
  }
}

template <typename T> BasicAsyncTask<T> BasicAsyncTask<T>::make_sole_owner(BasicAsyncTask<T>&& task)
{
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

template <typename T> void BasicAsyncTask<T>::resume()
{
  CHECK(h_);
  TRACE("Resuming AsyncTask", h_);
  if (!h_.promise().release_awaiter()) PANIC("Attempt to resume() a non-exclusive AsyncTask");

  bool to_destroy = cancel_.load(std::memory_order_acquire); // h_.promise().is_destroy_on_resume();

  auto handle = h_;
  h_ = nullptr; // Always drop ownership, we are now effectively in a 'moved from' state
  // we need to do this *before* calling h_.resume(), because it is possible that this AsyncTask
  // is on that coroutine frame, and it might get destroyed.

  if (to_destroy)
  {
    // if we need to destroy the coroutine, recursively destroy any continuations as well
    while (handle)
    {
      TRACE("Destroying AsyncTask", handle);
      handle = handle.promise().destroy_with_continuation();
    }
  }
  else
  {
    handle.resume();
  }

  TRACE("returned from coroutine::resume");
}

// template <typename T> void BasicAsyncTask<T>::cancel_on_resume() noexcept
// {
//   CHECK(h_);
//   CHECK(current_awaiter_);
//   current_awaiter_->set_cancel();
// }
//
// template <typename T> void BasicAsyncTask<T>::exception_on_resume(std::exception_ptr e) noexcept
// {
//   CHECK(h_);
//   CHECK(current_awaiter_);
//   current_awaiter_->set_exception(e);
// }
//
template <typename T> BasicAsyncTask<T>& BasicAsyncTask<T>::operator=(BasicAsyncTask<T>&& other) noexcept
{
  // TRACE("AsyncTask move assignment", this, h_, &other, other.h_);
  if (this != &other)
  {
    if (h_ && h_.promise().release_awaiter()) h_.destroy();

    h_ = std::exchange(other.h_, nullptr);
    cancel_.store(other.cancel_.load(std::memory_order_acquire), std::memory_order_release);
  }
  return *this;
}

template <typename T> void BasicAsyncTask<T>::release() noexcept
{
  // TRACE("Destroying AsyncTask", this, h_);
  if (h_ && h_.promise().release_awaiter())
  {
    while (h_)
    {
      TRACE("AsyncTask destructor is destroying the coroutine!", this, h_);
      h_ = h_.promise().destroy_with_continuation();
    }
  }
}

template <typename T> BasicAsyncTask<T>::~BasicAsyncTask() noexcept { this->release(); }

template <typename T> bool BasicAsyncTask<T>::set_scheduler(IScheduler* sched)
{
  if (h_)
  {
    h_.promise().sched_ = sched;
    return true;
  }
  return false;
}

template <typename T>
BasicAsyncTask<T>::handle_type BasicAsyncTask<T>::await_suspend(BasicAsyncTask<T>::handle_type Outer)
{
  DEBUG_CHECK(h_);
  DEBUG_CHECK(!h_.promise().continuation_);
  // h_.promise().current_awaiter_ = this;
  h_.promise().continuation_ = Outer;
  h_.promise().sched_ = Outer.promise().sched_;
  auto h_transfer = h_.promise().release_ownership();
  h_ = nullptr; // finish transferring ownership
  CHECK(h_transfer, "error: co_await on an AsyncTask that has shared ownership");
  return h_transfer;
}

template <typename T> void BasicAsyncTask<T>::await_resume() const noexcept
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
