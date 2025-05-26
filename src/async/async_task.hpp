/// \file async_task.hpp
/// \brief Defines AsyncTask, the fire-and-forget coroutine handle.
/// \ingroup async_core

#pragma once

#include "common/trace.hpp"
#include <atomic>
#include <coroutine>
#include <exception>

namespace uni20::async
{

class IScheduler;

/// \brief A fire-and-forget coroutine handle.
/// \ingroup async_core
struct AsyncTask : public trace::TracingBaseClass<AsyncTask>
{
    /// \brief Promise type for AsyncTask, forward declare
    struct promise_type;

    /// \brief Destroy the coroutine if still present.
    ~AsyncTask();

    // bool done() const noexcept { return !h_ || h_.done(); }

    /// \brief Check if this task refers to a coroutine.
    /// \return true if the handle is non-null.
    explicit operator bool() const noexcept { return static_cast<bool>(h_); }

    /// \brief Resume the coroutine; null the coroutine handle if ownership has been transferred
    void resume();

    /// \brief Attempt to set the coroutine scheduler to use when rescheduling the task.
    /// \return true if the handle is non-null (in which case the scheduler has been set)
    bool set_scheduler(IScheduler* sched);

    /// \brief Resubmit a suspended AsyncTask to its scheduler, if this is the sole remaining owner.
    ///
    /// This transfers ownership to the scheduler only if the task has exclusive ownership of the coroutine.
    /// If other awaiters remain, the task is discarded and not rescheduled.
    ///
    /// \pre The scheduler in the promise must have been set.
    /// \param task The task to be conditionally rescheduled.
    static void reschedule(AsyncTask task);

    /// \brief Retain the task only if it is the sole remaining owner.
    ///
    /// This operation checks the awaiter count. If this is the last owner,
    /// then return the task unchanged. Otherwise, releases the task and leaves
    /// it requivalent to a moved-from state.
    ///
    /// \return The original task if it had exclusive ownership; otherwise a null task.
    static AsyncTask make_sole_owner(AsyncTask&& task);

    AsyncTask() = default;

    AsyncTask(const AsyncTask&) = delete;            ///< non-copyable
    AsyncTask& operator=(const AsyncTask&) = delete; ///< non-copyable

    /// \brief Move-construct.
    AsyncTask(AsyncTask&& other) noexcept : trace::TracingBaseClass<AsyncTask>(std::move(other)), h_{other.h_}
    {
      TRACE("AsyncTask move", this, &other, h_);
      other.h_ = nullptr;
    }

    /// \brief Move-assign.
    AsyncTask& operator=(AsyncTask&& other) noexcept;

    //
    // Awaiter interface
    // If we co_await on an AsyncTask, then it has the semantics of transferring execution
    // to that task, and then resuming the current coroutine afterwards.
    // if coroutine Outer contains co_await Inner, then we end up with
    // Inner.await_suspend(Outer)
    //

    bool await_ready() const noexcept { return !h_ || h_.done(); }

    std::coroutine_handle<AsyncTask::promise_type> await_suspend(std::coroutine_handle<AsyncTask::promise_type> Outer);

    void await_resume() const noexcept;

    //  private:
    std::coroutine_handle<promise_type> h_; ///< Underlying coroutine handle.

    /// \brief Construct from a coroutine handle.
    /// \param h The coroutine handle.
    explicit AsyncTask(std::coroutine_handle<promise_type> h) noexcept : trace::TracingBaseClass<AsyncTask>(h), h_{h} {}

    friend struct AsyncTask::promise_type;
    friend class AsyncTaskFactory;
};

} // namespace uni20::async
