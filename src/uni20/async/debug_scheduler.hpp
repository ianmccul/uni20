/// \file debug_scheduler.hpp
/// \brief Coroutine scheduler for AsyncTask.

#pragma once

#include "async.hpp"
#include "scheduler.hpp"
#include "task_registry.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace uni20::async
{

/// \brief Simple FIFO scheduler
class DebugScheduler final : public IScheduler {
  public:
    /// \brief Default-construct an empty scheduler.
    DebugScheduler() = default;

    /// \brief Enqueue a task for later run.
    /// \param task An AsyncTask bound to *this* scheduler.
    void schedule(AsyncTask&& task) override
    {
      TRACE_MODULE(ASYNC, "Scheduling a task", &task, task.h_);
      if (task.set_scheduler(this))
      {
        Handles_.push_back(std::move(task));
      }
    }

    /// \brief Destructor.
    ~DebugScheduler()
    {
      TRACE_MODULE(ASYNC, "~DebugScheduler", Handles_.size());

      if (std::uncaught_exceptions() > 0)
      {
        while (!Handles_.empty())
        {
          Handles_.back().abandon_leak();
          Handles_.pop_back();
        }
      }
    }

    /// \brief Reports whether at least one runnable task exists.
    /// \return `true` when the scheduler is not paused and has queued tasks.
    bool can_run() const noexcept { return !Blocked_ && !Handles_.empty(); }

    /// \brief Run one batch of scheduled coroutines (in LIFO order).
    void run();

    /// \brief Run until no pending tasks remain.
    void run_all();

    /// \brief Block the scheduler from running.
    /// \note This turns run() and run_all() into no-operation;
    void pause() override { Blocked_ = true; }

    /// \brief Unblock the scheduler
    void resume() override { Blocked_ = false; }

    /// \brief Drives one runnable task while waiting for readiness.
    /// \param is_ready Predicate that reports target readiness.
    void help_while_waiting(const WaitPredicate& is_ready) override
    {
      if (is_ready())
      {
        return;
      }

      if (Blocked_ || Handles_.empty())
      {
        TaskRegistry::dump();
        PANIC("**DEADLOCK** get_wait object is not available but there are no runnable tasks!");
      }
      this->run();
    }

    /// \brief Check if there are no pending tasks.
    /// \return true if the scheduler queue is empty.
    bool done() const noexcept { return Handles_.empty(); }

  private:
    // Internal resubmission
    void reschedule(AsyncTask&& task) override
    {
      TRACE_MODULE(ASYNC, "Rescheduling a task", &task, task.h_);
      // Assume sched_ is already set
      Handles_.push_back(std::move(task));
    }

    bool Blocked_ = false;

    std::vector<AsyncTask> Handles_;
};

namespace detail
{
/// \brief Process-wide default debug scheduler instance.
inline DebugScheduler DefaultScheduler;
/// \brief Global scheduler pointer used by free `schedule(...)` helpers.
inline IScheduler* global_scheduler = &DefaultScheduler;
} // namespace detail

/// \brief Sets the process-wide scheduler used by async helpers.
/// \param sched Scheduler pointer to install.
inline void set_global_scheduler(IScheduler* sched) { detail::global_scheduler = sched; }

/// \brief Returns the currently-installed process-wide scheduler.
/// \return Pointer to the active scheduler.
inline IScheduler* get_global_scheduler() { return detail::global_scheduler; }

/// \brief Restores the process-wide scheduler to the default debug scheduler.
inline void reset_global_scheduler() { detail::global_scheduler = &detail::DefaultScheduler; }

// ScopedScheduler is useful for testing; set the scheduler for the lifetime of a block
/// \brief RAII helper that temporarily overrides the global scheduler.
class ScopedScheduler {
  public:
    /// \brief Install a temporary global scheduler until destruction.
    /// \param sched Scheduler pointer to activate.
    explicit ScopedScheduler(IScheduler* sched)
    {
      old_ = get_global_scheduler();
      set_global_scheduler(sched);
    }
    /// \brief Restore the previous global scheduler.
    ~ScopedScheduler() { set_global_scheduler(old_); }

  private:
    IScheduler* old_;
};

/// \brief Schedule a task on the currently configured global scheduler.
/// \param task Task to schedule.
inline void schedule(AsyncTask&& task) { get_global_scheduler()->schedule(std::move(task)); }

/// \brief Wait for a reader context using the global scheduler.
/// \tparam T Stored value type.
/// \return Reference to the ready value.
template <typename T> T const& EpochContextReader<T>::get_wait() const
{
  auto* sched = get_global_scheduler();
  if (!this->ready())
  {
    CHECK(sched);
    sched->wait_for([this] { return this->ready(); });
  }
  return this->data();
}

/// \brief Wait for a reader context using an explicit scheduler.
/// \tparam T Stored value type.
/// \param sched Scheduler used to drive progress.
/// \return Reference to the ready value.
template <typename T> T const& EpochContextReader<T>::get_wait(IScheduler& sched) const
{
  if (!this->ready())
  {
    sched.wait_for([this] { return this->ready(); });
  }
  return this->data();
}

/// \brief Wait for a writer context and move out the stored value.
/// \tparam T Stored value type.
/// \return Moved value from writer storage.
template <typename T> T&& EpochContextWriter<T>::move_from_wait()
{
  auto* sched = get_global_scheduler();
  if (!this->ready())
  {
    CHECK(sched);
    sched->wait_for([this] { return this->ready(); });
  }
  return std::move(this->data());
}

//-----------------------------------------------------------------------------
// Inline definitions
//-----------------------------------------------------------------------------

inline void DebugScheduler::run()
{
  DEBUG_TRACE_MODULE(ASYNC, "DebugScheduler::run");
  if (Blocked_)
  {
    DEBUG_TRACE_MODULE(ASYNC, "run() on a blocked DebugQueue: doing nothing");
    return;
  }
  TRACE_MODULE(ASYNC, "Got some coroutines to resume", Handles_.size());

  std::vector<AsyncTask> H;
  std::swap(H, Handles_);
  std::reverse(H.begin(), H.end());
  for (auto&& h : H)
  {
    TRACE_MODULE(ASYNC, "resuming coroutine...", &h, h.h_);
    h.resume();
    CHECK(!h);
    TRACE_MODULE(ASYNC, "here", &h, Handles_.size());
  }
}

inline void DebugScheduler::run_all()
{
  DEBUG_TRACE_MODULE(ASYNC, "DebugScheduler::run_all");
  if (Blocked_)
  {
    DEBUG_TRACE_MODULE(ASYNC, "run() on a blocked DebugQueue: doing nothing");
    return;
  }
  while (!done())
  {
    run();
  }
}

} // namespace uni20::async
