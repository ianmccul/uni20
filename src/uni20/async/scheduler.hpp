/// \file scheduler.hpp
/// \brief Defines internal task routing and the host-oriented scheduler interface.

// see https://github.com/dbittman/waitfree-mpsc-queue/blob/master/mpsc.c

#pragma once

#include "async_task.hpp"

#include <functional>
#include <thread>

namespace uni20::async
{

class AsyncTask;
class TaskPromiseBase;

/// \brief Internal route for resuming an already-bound coroutine task.
class IScheduler {
  public:
    /// \brief Virtual destructor.
    virtual ~IScheduler() = default;

  private:
    friend class BasicTask;
    friend class TaskPromiseBase;

    /// \brief Report whether this scheduler can resume a task route.
    /// \details This validates declared task domain and any established CUDA
    ///          device affinity. It does not admit or take ownership of a task.
    virtual bool accepts_route(TaskRoute route) const noexcept = 0;

    /// \brief Decide whether execution may transfer directly between two tasks.
    /// \details The caller has already verified that both tasks use this scheduler
    ///          and declare the same task domain.
    virtual bool can_direct_transfer(TaskHandle from, TaskHandle to) const noexcept = 0;

    /// \brief Schedule an already-bound coroutine to be resumed later.
    /// \param task Basic task state whose scheduler route is already fixed.
    virtual void reschedule(BasicTask&& task) = 0;
};

/// \brief Scheduler interface for initial `AsyncTask` submission and host-side progress.
class IAsyncScheduler : public virtual IScheduler {
  public:
    /// \brief Virtual destructor.
    ~IAsyncScheduler() override = default;

    /// \brief Schedule a coroutine for its initial execution.
    /// \param task Task to bind to this scheduler and submit.
    virtual void schedule(AsyncTask&& task) = 0;

    /// \brief Pause the scheduler.
    /// Tasks can still be scheduled, but they will not start running until resume() is called
    virtual void pause() = 0;

    /// \brief Resume the scheduler.  Tasks scheduled while paused can start running, as can
    /// newly scheduled tasks.
    virtual void resume() = 0;

    using WaitPredicate = std::function<bool()>;
    using WaitWakeup = std::function<void()>;
    using WaitWakeupRegistration = std::function<void(WaitWakeup)>;

    /// \brief Readiness predicate and optional targeted wakeup registration for a blocking wait.
    struct WaitRequest
    {
        /// Predicate that reports whether the requested value is available.
        WaitPredicate is_ready;
        /// Optional callback registration used to wake a suspended scheduler stack directly.
        WaitWakeupRegistration notify_when_ready{};
    };

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

    /// \brief Block until a structured wait request becomes ready.
    /// \param request Readiness predicate and optional targeted notification registration.
    ///
    /// The default implementation preserves compatibility with schedulers that
    /// advance work by polling the readiness predicate. Schedulers with a
    /// suspend/resume mechanism can override this overload and register the
    /// supplied wakeup callback.
    virtual void wait_for(WaitRequest const& request) { this->wait_for(request.is_ready); }
};

} // namespace uni20::async
