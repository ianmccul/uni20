/// \file debug_scheduler.hpp
/// \brief Coroutine scheduler for AsyncTask.
/// \ingroup async_core

#pragma once

#include "async.hpp"
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
      TRACE_MODULE(ASYNC, "Scheduling a task", &task, task.h_);
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
      TRACE_MODULE(ASYNC, "Rescheduling a task", &task, task.h_);
      // Assume sched_ is already set
      Handles_.push_back(std::move(task));
    }

    bool Blocked_ = false;

    std::vector<AsyncTask> Handles_;
};

inline void set_global_scheduler(DebugScheduler* sched) { DebugScheduler::global_scheduler = sched; }

inline DebugScheduler* get_global_scheduler() { return DebugScheduler::global_scheduler; }

inline void schedule(AsyncTask&& task) { get_global_scheduler()->schedule(std::move(task)); }

template <typename T> T const& EpochContextReader<T>::get_wait() const
{
  auto* sched = get_global_scheduler();
  while (!this->ready())
  {
    CHECK(sched);
    if (auto* dbg = dynamic_cast<DebugScheduler*>(sched))
    {
      CHECK(dbg->can_run(), "**DEADLOCK** get_wait object is not available but there are no runnable tasks!");
    }
    sched->run();
  }
  return this->data();
}

template <typename T> T&& EpochContextWriter<T>::move_from_wait() const
{
  auto* sched = get_global_scheduler();
  while (!this->ready())
  {
    CHECK(sched);
    if (auto* dbg = dynamic_cast<DebugScheduler*>(sched))
    {
      CHECK(dbg->can_run(), "**DEADLOCK** get_wait object is not available but there are no runnable tasks!");
    }
    sched->run();
  }
  return std::move(this->data());
}

//-----------------------------------------------------------------------------
// Inline definitions
//-----------------------------------------------------------------------------

inline void DebugScheduler::run()
{
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
    TRACE_MODULE(ASYNC, "resuming coroutine...");
    h.resume();
    CHECK(!h);
    TRACE_MODULE(ASYNC, "here", &h, Handles_.size());
  }
}

inline void DebugScheduler::run_all()
{
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
