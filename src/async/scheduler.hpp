/// \file scheduler.hpp
/// \brief Defines the IScheduler interface
/// \ingroup async_core

// see https://github.com/dbittman/waitfree-mpsc-queue/blob/master/mpsc.c

#pragma once

#include "async_task.hpp"

namespace uni20::async
{

/// \brief Minimal abstract interface for scheduling coroutine handles.
/// \ingroup async_core
class IScheduler {
  public:
    /// \brief Virtual destructor.
    virtual ~IScheduler() = default;

    /// \brief Schedule a coroutine for its initial execution.
    /// \param h The coroutine handle to schedule.
    virtual void schedule(AsyncTask&& h) = 0;

    /// \brief Pause the scheduler.
    /// Tasks can still be scheduled, but they will not start running until resume() is called
    virtual void pause() = 0;

    /// \brief Resume the scheduler.  Tasks cheduled while paused can start running, as can
    /// newly scheduled tasks.
    virtual void resume() = 0;

  protected:
    // using promise_type = AsyncTask::promise_type;

  private:
    friend AsyncTask;

    /// \brief Schedule a coroutine to be resumed later.
    /// \param h The coroutine handle to schedule.
    virtual void reschedule(AsyncTask&& h) = 0;
};

} // namespace uni20::async
