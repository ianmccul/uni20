/// \file scheduler.hpp
/// \brief Defines the IScheduler interface
/// \ingroup async_core

// see https://github.com/dbittman/waitfree-mpsc-queue/blob/master/mpsc.c

#pragma once

#include "async_task.hpp"
#include <functional>
#include <thread>

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

    using WaitPredicate = std::function<bool()>;

    /// \brief Allow a scheduler to advance queued work while a thread is blocking.
    ///
    /// Blocking waits (e.g., `Async<T>::get_wait()`) call this hook to
    /// cooperatively drive progress on the owning scheduler until
    /// `is_ready()` reports completion. The default implementation simply
    /// yields the calling thread until the predicate succeeds, which is
    /// suitable for schedulers that rely on dedicated worker threads.
    virtual void help_while_waiting(const WaitPredicate& is_ready)
    {
      while (!is_ready())
      {
        std::this_thread::yield();
      }
    }

    /// \brief Block the calling thread until \p is_ready returns true.
    ///
    /// Implementations may override this to provide scheduler-specific
    /// waiting semantics (e.g., parking on a condition variable). The
    /// default implementation repeatedly invokes help_while_waiting until
    /// the predicate succeeds.
    virtual void wait_for(const WaitPredicate& is_ready)
    {
      while (!is_ready())
      {
        this->help_while_waiting(is_ready);
      }
    }

  protected:
    // using promise_type = AsyncTask::promise_type;

  private:
    friend AsyncTask;

    /// \brief Schedule a coroutine to be resumed later.
    /// \param h The coroutine handle to schedule.
    virtual void reschedule(AsyncTask&& h) = 0;
};

} // namespace uni20::async
