/// \file scheduler.hpp
/// \brief Coroutine scheduler for AsyncTask.
/// \ingroup async_core

#pragma once

#include "scheduler.hpp"
#include <algorithm>
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
      auto h = std::exchange(task.h_, nullptr);
      h.promise().sched_ = this;
      Handles_.push_back(h);
    }

    /// \brief Run one batch of scheduled coroutines (in LIFO order).
    void run();

    /// \brief Run until no pending tasks remain.
    void run_all();

    /// \brief Check if there are no pending tasks.
    /// \return true if the scheduler queue is empty.
    bool done() const noexcept { return Handles_.empty(); }

  private:
    void schedule(std::coroutine_handle<promise_type> h_) { Handles_.push_back(h_); }

    std::vector<std::coroutine_handle<promise_type>> Handles_;
};

//-----------------------------------------------------------------------------
// Inline definitions
//-----------------------------------------------------------------------------

inline void DebugScheduler::run()
{
  std::vector<std::coroutine_handle<promise_type>> batch;
  std::swap(batch, Handles_);
  TRACE("Got some coroutines to resume", batch.size());
  std::reverse(batch.begin(), batch.end());
  for (auto h : batch)
  {
    TRACE("running coroutine...");
    h.resume();
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
