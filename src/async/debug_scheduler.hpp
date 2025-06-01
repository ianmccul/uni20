/// \file debug_scheduler.hpp
/// \brief Coroutine scheduler for AsyncTask.
/// \ingroup async_core

#pragma once

#include "scheduler.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace uni20::async
{

/// \brief Simple FIFO scheduler
/// \ingroup async_core
class DebugScheduler final : public IScheduler {
  public:
    /// \brief Default-construct an empty scheduler.
    DebugScheduler() = default;

    /// \brief Enqueue a task for later run.
    /// \param task An AsyncTask bound to *this* scheduler.
    void schedule(AsyncTask&& task)
    {
      TRACE("Scheduling a task", &task, task.h_);
      if (task.set_scheduler(this))
      {
        Handles_.push_back(std::move(task));
      }
    }

    inline static DebugScheduler* global_scheduler = nullptr;

    bool can_run() const noexcept { return !Blocked_ && !Handles_.empty(); }

    /// \brief Run one batch of scheduled coroutines (in LIFO order).
    void run();

    /// \brief Run until no pending tasks remain.
    void run_all();

    /// \brief Block the scheduler from running.
    /// \note This turns run() and run_all() into no-operation;
    void block() { Blocked_ = true; }

    /// \brief Unblock the scheduler
    void unblock() { Blocked_ = false; }

    /// \brief Check if there are no pending tasks.
    /// \return true if the scheduler queue is empty.
    bool done() const noexcept { return Handles_.empty(); }

  private:
    // Internal resubmission
    virtual void reschedule(AsyncTask&& task)
    {
      TRACE("Rescheduling a task", &task, task.h_);
      // Assume sched_ is already set
      Handles_.push_back(std::move(task));
    }

    bool Blocked_ = false;

    std::vector<AsyncTask> Handles_;
};

inline void set_global_scheduler(DebugScheduler* sched) { DebugScheduler::global_scheduler = sched; }

inline DebugScheduler* get_global_scheduler() { return DebugScheduler::global_scheduler; }

inline void schedule(AsyncTask&& task) { get_global_scheduler()->schedule(std::move(task)); }

template <typename T> T& Async<T>::get_wait()
{
  auto* sched = get_global_scheduler();
  while (impl_->queue_.has_pending_writers())
  {
    auto ds = dynamic_cast<DebugScheduler*>(sched);
    if (ds)
    {
      CHECK(ds->can_run(), "**DEADLOCK** get_wait object is not available but there are no runnable tasks!");
    }
    TRACE("Async::get_wait: has pending writers");
    sched->run();
  }
  return impl_->value_;
}

template <typename T> T const& Async<T>::get_wait() const
{
  auto* sched = get_global_scheduler();
  while (impl_->queue_.has_pending_writers())
  {
    auto ds = dynamic_cast<DebugScheduler*>(sched);
    if (ds)
    {
      CHECK(ds->can_run(), "**DEADLOCK** get_wait object is not available but there are no runnable tasks!");
    }
    TRACE("Async::get_wait: has pending writers");
    sched->run();
  }
  return impl_->value_;
}

//-----------------------------------------------------------------------------
// Inline definitions
//-----------------------------------------------------------------------------

inline void DebugScheduler::run()
{
  if (Blocked_)
  {
    DEBUG_TRACE("run() on a blocked DebugQueue: doing nothing");
    return;
  }
  TRACE("Got some coroutines to resume", Handles_.size());
  // fmt::print("Got some coroutines to resume {}\n", Handles_.size());
  std::vector<AsyncTask> H;
  std::swap(H, Handles_);
  std::reverse(H.begin(), H.end());
  for (auto&& h : H)
  {
    TRACE("resuming coroutine...");
    h.resume();
    CHECK(!h);
    TRACE("here", &h, Handles_.size());
  }
}

inline void DebugScheduler::run_all()
{
  if (Blocked_)
  {
    DEBUG_TRACE("run() on a blocked DebugQueue: doing nothing");
    return;
  }
  while (!done())
  {
    run();
  }
}

} // namespace uni20::async
