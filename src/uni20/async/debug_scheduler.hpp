/// \file debug_scheduler.hpp
/// \brief Coroutine scheduler for AsyncTask.

#pragma once

#include "async.hpp"
#include "scheduler.hpp"
#include "task_registry.hpp"
#include <algorithm>
#include <cstdint>
#include <random>
#include <uni20/common/display.hpp>
#include <utility>
#include <vector>

namespace uni20::async
{

namespace detail
{
inline void service_task_registry_debug_requests()
{
#if UNI20_DEBUG_ASYNC_TASKS
  TaskRegistry::service_debug_requests();
#endif
}

inline void dump_deadlock_graphviz_snapshot()
{
#if UNI20_DEBUG_ASYNC_TASKS
  auto const path = TaskRegistry::default_graphviz_dump_path();
  if (TaskRegistry::dump_graphviz_file_best_effort(path))
  {
    display::info("Async Graphviz DAG snapshot written to {}", path);
  }
#endif
}
} // namespace detail

/// \brief Runnable-batch ordering policy for `DebugScheduler`.
enum class DebugSchedulerOrder
{
  /// Execute each runnable batch in submission order.
  fifo,
  /// Execute each runnable batch in reverse submission order.
  reverse,
  /// Execute each runnable batch in seeded pseudo-random order.
  random,
};

/// \brief Configuration for deterministic debug scheduling.
struct DebugSchedulerOptions
{
    /// Runnable-batch ordering policy.
    DebugSchedulerOrder order = DebugSchedulerOrder::reverse;
    /// Initial state for the reproducible random policy.
    std::uint64_t random_seed = 0;
};

/// \brief Single-threaded scheduler with configurable runnable-batch ordering.
class DebugScheduler : public IAsyncScheduler {
  public:
    /// \brief Default-construct an empty scheduler.
    explicit DebugScheduler(DebugSchedulerOptions options = {}) : options_(options), random_engine_(options.random_seed)
    {}

    /// \brief Enqueue a task for later run.
    /// \param task An AsyncTask bound to *this* scheduler.
    void schedule(AsyncTask&& task) override
    {
      TRACE_MODULE(ASYNC, "Scheduling a task", &task, task.coroutine_handle());
      TaskRegistry::record_task_scheduled(task.coroutine_handle());
      if (task.set_scheduler(this))
      {
        this->enqueue_task(std::move(task));
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
    [[nodiscard]] bool can_run() const noexcept { return !Blocked_ && !Handles_.empty(); }

    /// \brief Run one batch of scheduled coroutines in the configured order.
    void run();

    /// \brief Run until no runnable tasks remain.
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
        detail::dump_deadlock_graphviz_snapshot();
        TaskRegistry::dump();
        PANIC("**DEADLOCK** get_wait object is not available but there are no runnable tasks!");
      }
      this->run();
    }

    /// \brief Check whether the runnable queue is empty.
    /// \return `true` if the scheduler has no queued tasks.
    [[nodiscard]] bool done() const noexcept { return Handles_.empty(); }

    /// \brief Return the scheduler's immutable ordering configuration.
    [[nodiscard]] DebugSchedulerOptions const& options() const noexcept { return options_; }

  private:
    bool can_direct_transfer(TaskHandle, TaskHandle) const noexcept override { return true; }

    // Internal resubmission
    void reschedule(BasicTask&& task) override
    {
      TRACE_MODULE(ASYNC, "Rescheduling a task", &task, task.coroutine_handle());
      // Assume sched_ is already set
      this->enqueue_task(std::move(task));
    }

    bool Blocked_ = false;

    std::vector<BasicTask> Handles_;
    DebugSchedulerOptions options_;
    std::mt19937_64 random_engine_;

    void order_batch(std::vector<BasicTask>& batch)
    {
      switch (options_.order)
      {
        case DebugSchedulerOrder::fifo:
          return;
        case DebugSchedulerOrder::reverse:
          std::reverse(batch.begin(), batch.end());
          return;
        case DebugSchedulerOrder::random:
          // Keep the permutation stable across standard-library implementations.
          for (std::size_t remaining = batch.size(); remaining > 1; --remaining)
          {
            auto const selected = static_cast<std::size_t>(random_engine_() % remaining);
            std::swap(batch[remaining - 1], batch[selected]);
          }
          return;
      }
      PANIC("invalid DebugScheduler ordering policy", static_cast<int>(options_.order));
    }

  protected:
    /// \brief Add one already-bound task to the runnable queue.
    void enqueue_task(BasicTask&& task) { Handles_.push_back(std::move(task)); }

    /// \brief Resume one queued task in the execution context selected by this scheduler.
    virtual void resume_task(BasicTask& task) { task.resume(); }
};

namespace detail
{
/// \brief Process-wide default debug scheduler instance.
inline DebugScheduler DefaultScheduler;
/// \brief Global scheduler pointer used by free `schedule(...)` helpers.
inline IAsyncScheduler* global_scheduler = &DefaultScheduler;
} // namespace detail

/// \brief Sets the process-wide scheduler used by async helpers.
/// \param sched Scheduler pointer to install.
inline void set_global_scheduler(IAsyncScheduler* sched) { detail::global_scheduler = sched; }

/// \brief Returns the currently-installed process-wide scheduler.
/// \return Pointer to the active scheduler.
inline IAsyncScheduler* get_global_scheduler() { return detail::global_scheduler; }

/// \brief Restores the process-wide scheduler to the default debug scheduler.
inline void reset_global_scheduler() { detail::global_scheduler = &detail::DefaultScheduler; }

// ScopedScheduler is useful for testing; set the scheduler for the lifetime of a block
/// \brief RAII helper that temporarily overrides the global scheduler.
class ScopedScheduler {
  public:
    /// \brief Install a temporary global scheduler until destruction.
    /// \param sched Scheduler pointer to activate.
    explicit ScopedScheduler(IAsyncScheduler* sched)
    {
      old_ = get_global_scheduler();
      set_global_scheduler(sched);
    }
    /// \brief Restore the previous global scheduler.
    ~ScopedScheduler() { set_global_scheduler(old_); }

  private:
    IAsyncScheduler* old_;
};

/// \brief Schedule a task on the currently configured global scheduler.
/// \param task Task to schedule.
inline void schedule(AsyncTask&& task) { get_global_scheduler()->schedule(std::move(task)); }

/// \brief Wait for a reader context using the global scheduler.
/// \tparam T Stored value type.
template <typename T> void EpochContextReader<T>::wait() const
{
  auto* sched = get_global_scheduler();
  if (!this->ready())
  {
    CHECK(sched);
    auto epoch = epoch_;
    sched->wait_for(IAsyncScheduler::WaitRequest{
        .is_ready = [epoch] { return epoch->reader_ready(); },
        .notify_when_ready =
            [epoch](IAsyncScheduler::WaitWakeup notify) { epoch->reader_notify_when_ready(std::move(notify)); },
    });
  }
}

/// \brief Wait for a reader context using an explicit scheduler.
/// \tparam T Stored value type.
/// \param sched Scheduler used to drive progress.
template <typename T> void EpochContextReader<T>::wait(IAsyncScheduler& sched) const
{
  if (!this->ready())
  {
    auto epoch = epoch_;
    sched.wait_for(IAsyncScheduler::WaitRequest{
        .is_ready = [epoch] { return epoch->reader_ready(); },
        .notify_when_ready =
            [epoch](IAsyncScheduler::WaitWakeup notify) { epoch->reader_notify_when_ready(std::move(notify)); },
    });
  }
}

/// \brief Wait for a reader context using the global scheduler.
/// \tparam T Stored value type.
/// \return Reference to the ready value.
template <typename T> T const& EpochContextReader<T>::get_wait() const
{
  this->wait();
  return this->data();
}

/// \brief Wait for a reader context using an explicit scheduler.
/// \tparam T Stored value type.
/// \param sched Scheduler used to drive progress.
/// \return Reference to the ready value.
template <typename T> T const& EpochContextReader<T>::get_wait(IAsyncScheduler& sched) const
{
  this->wait(sched);
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
  detail::service_task_registry_debug_requests();
  if (Blocked_)
  {
    DEBUG_TRACE_MODULE(ASYNC, "run() on a blocked DebugQueue: doing nothing");
    return;
  }
  TRACE_MODULE(ASYNC, "Got some coroutines to resume", Handles_.size());

  std::vector<BasicTask> H;
  std::swap(H, Handles_);
  this->order_batch(H);
  for (auto&& h : H)
  {
    TRACE_MODULE(ASYNC, "resuming coroutine...", &h, h.coroutine_handle());
    this->resume_task(h);
    CHECK(!h);
    TRACE_MODULE(ASYNC, "here", &h, Handles_.size());
  }
  detail::service_task_registry_debug_requests();
}

inline void DebugScheduler::run_all()
{
  DEBUG_TRACE_MODULE(ASYNC, "DebugScheduler::run_all");
  detail::service_task_registry_debug_requests();
  if (Blocked_)
  {
    DEBUG_TRACE_MODULE(ASYNC, "run() on a blocked DebugQueue: doing nothing");
    return;
  }
  while (!done())
  {
    run();
  }
  detail::service_task_registry_debug_requests();
}

} // namespace uni20::async
