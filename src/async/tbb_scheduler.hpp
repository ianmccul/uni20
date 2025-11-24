#pragma once
/// \file tbb_scheduler.hpp
/// \brief Scheduler implementation using oneAPI oneTBB task_arena + task_group.
/// \ingroup async_core

#include "scheduler.hpp"
#include <condition_variable>
#include <mutex>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <utility>
#include <vector>

namespace uni20::async
{

/// \brief Scheduler backend that uses Intel oneTBB to resume coroutines.
///
/// Tasks scheduled on this scheduler are enqueued into a TBB task_arena,
/// and executed under a single task_group. Resumption occurs on one of
/// the worker threads managed by the arena.
///
/// \note Each coroutine is pinned to the scheduler it was created on
///       via BasicAsyncTaskPromise::sched_. Resumption always returns
///       to the same scheduler.
/// \note This scheduler does not attempt to provide determinism or
///       deadlock detection; use DebugScheduler for that.
///
class TbbScheduler final : public IScheduler {
  public:
    /// \brief Construct a TBB scheduler with a given number of worker threads.
    /// \param threads Number of threads. Use task_arena::automatic for default.
    explicit TbbScheduler(int threads = oneapi::tbb::task_arena::automatic)
        : arena_(threads, /*reserved_for_masters=*/0), paused_(false)
    {
      arena_.initialize(threads, /*reserved_for_masters=*/0);
    }

    /// \brief Construct a TBB scheduler constrained to a specific NUMA node.
    /// \param constraints Binding constraints applied to the underlying arena.
    /// \param threads Number of threads. Use task_arena::automatic for default.
    explicit TbbScheduler(oneapi::tbb::task_arena::constraints constraints,
                          int threads = oneapi::tbb::task_arena::automatic)
        : arena_(constraints, /*reserved_for_masters=*/0), paused_(false)
    {
      if (threads != oneapi::tbb::task_arena::automatic)
      {
        constraints.set_max_concurrency(threads);
      }
      arena_.initialize(constraints, /*reserved_for_masters=*/0);
    }

    ~TbbScheduler() noexcept override
    {
      // ensure all tasks finish before destruction
      arena_.execute([&] { tg_.wait(); });
    }

    /// \brief Schedule a coroutine for initial execution.
    void schedule(AsyncTask&& t) override
    {
      if (t.set_scheduler(this)) this->enqueue_task(std::move(t));
    }

    /// \brief Block until all tasks scheduled on this scheduler are complete.
    ///
    /// \note This guarantees quiescence with respect to tasks that were
    ///       scheduled on this TbbScheduler. Tasks blocked on external
    ///       events (I/O, MPI, etc.) may still be logically alive and
    ///       will resume later if rescheduled.
    void run_all()
    {
      this->resume();
      arena_.execute([&] { tg_.wait(); });
    }

    /// \brief Pause the scheduler. Don't execute scheduled tasks, but instead add them to a queue.
    void pause() override
    {
      std::scoped_lock lock(pause_mutex_);
      paused_.store(true, std::memory_order_release);
    }

    /// \brief Unpause the scheduler, and execute any tasks that have been queued.
    void resume() override
    {
      std::vector<AsyncTask::handle_type> drained;
      {
        std::scoped_lock lock(pause_mutex_);
        paused_.store(false, std::memory_order_release);
        AsyncTask::handle_type h;
        while (queue_.try_pop(h))
        {
#if UNI20_ASYNC_DEBUG
          drained_on_resume_.fetch_add(1, std::memory_order_relaxed);
#endif
          drained.push_back(h);
        }
      }

      for (auto h : drained)
      {
        TRACE_MODULE(ASYNC, "scheduling coroutine", h);
        this->dispatch_handle(h);
      }
    }

    void help_while_waiting(const WaitPredicate& is_ready) override { this->wait_for(is_ready); }

    void wait_for(const WaitPredicate& is_ready) override
    {
      if (is_ready())
      {
        return;
      }

      if (oneapi::tbb::this_task_arena::current_thread_index() != oneapi::tbb::task_arena::not_initialized)
      {
        while (!is_ready())
        {
          std::this_thread::yield();
        }
        return;
      }

      std::unique_lock<std::mutex> lock(wait_mutex_);
      wait_cv_.wait(lock, [&] { return is_ready(); });
    }

  protected:
    /// \brief Reschedule a previously suspended coroutine.
    void reschedule(AsyncTask&& t) override { this->enqueue_task(std::move(t)); }

  private:
    struct DebugCounters
    {
#if UNI20_ASYNC_DEBUG
      uint64_t enqueued;
      uint64_t paused_enqueues;
      uint64_t dispatches;
      uint64_t drained_on_resume;
#endif
    };

    DebugCounters counters() const noexcept
    {
#if UNI20_ASYNC_DEBUG
      return DebugCounters{.enqueued = enqueued_.load(std::memory_order_relaxed),
                           .paused_enqueues = paused_enqueues_.load(std::memory_order_relaxed),
                           .dispatches = dispatches_.load(std::memory_order_relaxed),
                           .drained_on_resume = drained_on_resume_.load(std::memory_order_relaxed)};
#else
      return DebugCounters{};
#endif
    }

    void trace_counters(char const* label = nullptr) const noexcept
    {
#if UNI20_ASYNC_DEBUG
      auto const snapshot = this->counters();
      TRACE_MODULE(ASYNC, label ? label : "tbb_scheduler", snapshot.enqueued, snapshot.paused_enqueues, snapshot.dispatches,
                   snapshot.drained_on_resume);
#else
      (void)label;
#endif
    }

    void enqueue_task(AsyncTask&& t)
    {
      if (auto h = t.release_handle())
      {
#if UNI20_ASYNC_DEBUG
        enqueued_.fetch_add(1, std::memory_order_relaxed);
#endif
        bool paused = paused_.load(std::memory_order_acquire);
        if (paused)
        {
#if UNI20_ASYNC_DEBUG
          paused_enqueues_.fetch_add(1, std::memory_order_relaxed);
#endif
          queue_.push(h);
        }
        else
        {
          {
            std::scoped_lock lock(pause_mutex_);
            if (paused_.load(std::memory_order_acquire))
            {
#if UNI20_ASYNC_DEBUG
              paused_enqueues_.fetch_add(1, std::memory_order_relaxed);
#endif
              queue_.push(h);
              return;
            }
          }

          this->dispatch_handle(h);
        }
      }
    }

    void dispatch_handle(AsyncTask::handle_type h)
    {
#if UNI20_ASYNC_DEBUG
      dispatches_.fetch_add(1, std::memory_order_relaxed);
#endif
      arena_.execute([this, h]() {
        tg_.run([this, h]() {
          TRACE_MODULE(ASYNC, "resuming coroutine", h);
          try
          {
            h.resume();
          }
          catch (...)
          {
            wait_cv_.notify_all();
            throw;
          }
          wait_cv_.notify_all();
        });
      });
    }

    oneapi::tbb::task_arena arena_;
    oneapi::tbb::task_group tg_;
    std::atomic<bool> paused_;
    std::mutex pause_mutex_;
    oneapi::tbb::concurrent_queue<AsyncTask::handle_type> queue_;
    std::condition_variable wait_cv_;
    std::mutex wait_mutex_;
#if UNI20_ASYNC_DEBUG
    std::atomic<uint64_t> enqueued_{0};
    std::atomic<uint64_t> paused_enqueues_{0};
    std::atomic<uint64_t> dispatches_{0};
    std::atomic<uint64_t> drained_on_resume_{0};
#endif
    // FIXME: the concurrent_queue is overkill here, since we don't need to preserve order of tasks
};

} // namespace uni20::async
