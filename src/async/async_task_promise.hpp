/// \file async_task_promise.hpp
/// \brief Defines AsyncTask::promise_type, the fire-and-forget coroutine handle.
/// \ingroup async_core

#pragma once

#include "scheduler.hpp"
#include <atomic>
#include <coroutine>
#include <exception>

namespace uni20::async
{

/// \brief Promise type for AsyncTask.
struct AsyncTask::promise_type
{
    IScheduler* sched_ = nullptr; ///< Scheduler to notify when ready.

    /// \brief Default-construct the promise.
    constexpr promise_type() noexcept = default;

    /// \brief Obtain the AsyncTask associated with this promise.
    /// \return An AsyncTask wrapping the coroutine.
    AsyncTask get_return_object() noexcept
    {
      return AsyncTask{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    /// \brief Suspend immediately on coroutine entry.
    /// \return always suspend.
    constexpr std::suspend_always initial_suspend() noexcept { return {}; }

    /// \brief Never suspend at final suspend point (fire-and-forget).
    /// \return never suspend.
    constexpr std::suspend_never final_suspend() noexcept { return {}; }

    /// \brief Called when coroutine returns void.
    constexpr void return_void() noexcept {}

    /// \brief Called on unhandled exception escaping the coroutine.
    [[noreturn]] void unhandled_exception() { std::terminate(); }

    /// \brief Called when the task is ready to run again.
    /// \param h The coroutine handle to schedule.
    void notify_ready(std::coroutine_handle<promise_type> h) { sched_->schedule(h); }
};

} // namespace uni20::async
