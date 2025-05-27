/// \file scheduler.hpp
/// \brief Defines the IScheduler interface
/// \ingroup async_core

#pragma once

#include "async_task.hpp"
#include <coroutine>

namespace uni20::async
{

/// \brief Minimal abstract interface for scheduling coroutine handles.
/// \ingroup async_core
class IScheduler {
  public:
    /// \brief Virtual destructor.
    virtual ~IScheduler() = default;

  protected:
    using promise_type = AsyncTask::promise_type;

  private:
    friend AsyncTask;

    /// \brief Schedule a coroutine to be resumed later.
    /// \param h The coroutine handle to schedule.
    virtual void reschedule(AsyncTask&& h) = 0;
};

} // namespace uni20::async
