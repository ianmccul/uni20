#pragma once
/// \file tbb_scheduler.hpp
/// \brief Scheduler implementation using oneAPI oneTBB task_arena + task_group.
/// \ingroup async_core

#include "scheduler.hpp"
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <utility>

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
    explicit TbbScheduler(int threads = oneapi::tbb::task_arena::automatic) : arena_(threads), paused_(false)
    {
      arena_.initialize(threads);
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
    void pause() override { paused_.store(true, std::memory_order_relaxed); }

    /// \brief Unpause the scheduler, and execute any tasks that have been queued.
    void resume() override
    {
      paused_.store(false, std::memory_order_relaxed);
      AsyncTask::handle_type h;
      while (queue_.try_pop(h))
      {
        TRACE_MODULE(ASYNC, "scheduling coroutine", h);
        arena_.execute([this, h]() {
          tg_.run([h]() {
            TRACE_MODULE(ASYNC, "resuming coroutine", h);
            h.resume();
          });
        });
      }
    }

    void help_while_waiting(const WaitPredicate& is_ready) override
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

      arena_.execute([&] {
        while (!is_ready())
        {
          tg_.wait();
        }
      });
    }

  protected:
    /// \brief Reschedule a previously suspended coroutine.
    void reschedule(AsyncTask&& t) override { this->enqueue_task(std::move(t)); }

  private:
    void enqueue_task(AsyncTask&& t)
    {
      if (auto h = t.release_handle())
      {
        if (paused_)
        {
          queue_.push(h);
        }
        else
        {
          arena_.execute([this, h]() {
            tg_.run([h]() {
              TRACE_MODULE(ASYNC, "resuming coroutine", h);
              h.resume();
            });
          });
        }
      }
    }

    oneapi::tbb::task_arena arena_;
    oneapi::tbb::task_group tg_;
    std::atomic<bool> paused_;
    oneapi::tbb::concurrent_queue<AsyncTask::handle_type> queue_;
    // FIXME: the concurrent_queue is overkill here, since we don't need to preserve order of tasks
};

} // namespace uni20::async
