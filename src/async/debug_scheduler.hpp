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
      if (task.set_scheduler(this))
      {
        Handles_.push_back(std::move(task));
      }
    }

    /// \brief Run one batch of scheduled coroutines (in LIFO order).
    void run();

    /// \brief Run until no pending tasks remain.
    void run_all();

    /// \brief Check if there are no pending tasks.
    /// \return true if the scheduler queue is empty.
    bool done() const noexcept { return Handles_.empty(); }

  private:
    // Internal resubmission
    virtual void reschedule(AsyncTask&& task)
    {
      // Assume sched_ is already set
      Handles_.push_back(std::move(task));
    }

    std::vector<AsyncTask> Handles_;
};

//-----------------------------------------------------------------------------
// Inline definitions
//-----------------------------------------------------------------------------

inline void DebugScheduler::run()
{
  std::vector<AsyncTask> batch;
  std::swap(batch, Handles_);
  TRACE("Got some coroutines to resume", batch.size());
  std::reverse(batch.begin(), batch.end());
  for (auto&& h : batch)
  {
    TRACE("resuming coroutine...");
    h.resume();
    TRACE("here", &h, h.done());
    if (!h.done()) Handles_.push_back(std::move(h));
    TRACE("here", &h);
  }
}

inline void DebugScheduler::run_all()
{
  while (!done())
  {
    run();
  }
}

} // namespace uni20::async
