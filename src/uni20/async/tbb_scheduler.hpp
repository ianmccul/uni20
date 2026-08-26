#pragma once
/// \file tbb_scheduler.hpp
/// \brief Scheduler implementation using oneAPI oneTBB task_arena + task_group.

#include "async_errors.hpp"
#include "scheduler.hpp"
#include "task_registry.hpp"
#include "tbb_task_submission.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <optional>
#include <uni20/common/display.hpp>
#include <uni20/config.hpp>
#include <utility>
#include <vector>

namespace uni20::async
{

namespace detail
{
inline void service_tbb_task_registry_debug_requests()
{
#if UNI20_DEBUG_ASYNC_TASKS
  TaskRegistry::service_debug_requests();
#endif
}
} // namespace detail

/// \brief Runtime policy for blocking waits driven by a TBB scheduler.
struct TbbSchedulerWaitOptions
{
    /// Idle interval before a scheduler-visible no-progress wait raises `async_wait_timeout`; null disables it.
#if UNI20_ASYNC_DEBUG
    std::optional<std::chrono::milliseconds> watchdog_timeout{std::chrono::seconds{5}};
#else
    std::optional<std::chrono::milliseconds> watchdog_timeout{std::nullopt};
#endif
};

/// \brief Scheduler backend that uses Intel oneTBB to resume coroutines.
///
/// Tasks scheduled on this scheduler are submitted to a task_group inside a
/// TBB task_arena. Resumption occurs on a worker thread or an application
/// thread participating in the arena.
///
/// \note Each coroutine records a non-owning scheduler route. A suspended task
///       may replace that route while externally owned; subsequent activations
///       are submitted only to the replacement scheduler.
/// \note This scheduler provides a configurable no-progress watchdog, but it
///       does not prove that the dependency graph contains a cycle. Use
///       DebugScheduler and TaskRegistry diagnostics for deterministic DAG
///       investigation.
///
class TbbNumaScheduler;

class TbbScheduler : public IAsyncScheduler {
    class SuspendedWait;
    class ReadySignal;
    class ExecutionScope;

  public:
    /// \brief Construct a TBB scheduler with a given arena concurrency.
    /// \param max_concurrency Maximum arena participation, including application threads.
    ///                        Use task_arena::automatic for the oneTBB default.
    /// \param wait_options Blocking-wait watchdog policy.
    explicit TbbScheduler(int max_concurrency = oneapi::tbb::task_arena::automatic,
                          TbbSchedulerWaitOptions wait_options = {})
        : arena_(max_concurrency, /*reserved_for_application_threads=*/1), wait_options_(std::move(wait_options)),
          paused_(false)
    {
      CHECK(!wait_options_.watchdog_timeout || wait_options_.watchdog_timeout->count() > 0);
    }

    /// \brief Construct a TBB scheduler constrained to a specific NUMA node.
    /// \param constraints Binding constraints applied to the underlying arena.
    /// \param wait_options Blocking-wait watchdog policy.
    explicit TbbScheduler(oneapi::tbb::task_arena::constraints constraints, TbbSchedulerWaitOptions wait_options = {})
        : arena_(constraints, /*reserved_for_application_threads=*/1), wait_options_(std::move(wait_options)),
          paused_(false)
    {
      CHECK(!wait_options_.watchdog_timeout || wait_options_.watchdog_timeout->count() > 0);
    }

    ~TbbScheduler() noexcept override
    {
      // ensure all tasks finish before destruction
      this->wait_for_submitted_tasks();
    }

    /// \brief Schedule a coroutine for initial execution.
    void schedule(AsyncTask&& t) override
    {
      TaskRegistry::record_task_scheduled(t.coroutine_handle());
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
      detail::service_tbb_task_registry_debug_requests();
      this->wait_for_submitted_tasks();
      detail::service_tbb_task_registry_debug_requests();
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
      std::vector<TaskHandle> drained;
      {
        std::scoped_lock lock(pause_mutex_);
        paused_.store(false, std::memory_order_release);
        TaskHandle h;
        while (queue_.try_pop(h))
        {
          drained.push_back(h);
        }
      }

      for (auto h : drained)
      {
        TRACE_MODULE(ASYNC, "scheduling coroutine", h.coroutine());
        this->dispatch_handle(h);
      }
    }

    void help_while_waiting(const WaitPredicate& is_ready) override { this->wait_for(is_ready); }

    /// \brief Wait for a predicate using scheduler progress notifications.
    /// \param is_ready Predicate that reports completion.
    void wait_for(const WaitPredicate& is_ready) override { this->wait_for(WaitRequest{.is_ready = is_ready}); }

    /// \brief Wait for targeted readiness while allowing the arena to execute other work.
    /// \param request Readiness predicate and optional direct wakeup registration.
    void wait_for(WaitRequest const& request) override
    {
      if (request.is_ready()) return;

      auto ready_signal = std::make_shared<ReadySignal>(*this);
      if (request.notify_when_ready)
      {
        request.notify_when_ready([weak_signal = std::weak_ptr<ReadySignal>(ready_signal)]() noexcept {
          if (auto signal = weak_signal.lock()) signal->notify_ready();
        });
      }

      arena_.execute([&] { this->wait_in_arena(request, ready_signal); });
    }

  private:
    void execute_batch_impl(LightweightTaskBatch const& batch) override
    {
      if (batch.size() == 0) return;

      arena_.initialize();
      if (arena_.max_concurrency() == 1)
      {
        for (std::size_t index = 0; index < batch.size(); ++index)
          batch(index);
        return;
      }

      arena_.execute(
          [&] { oneapi::tbb::parallel_for(std::size_t{0}, batch.size(), [&](std::size_t index) { batch(index); }); });
    }

    bool accepts_route(TaskRoute route) const noexcept override
    {
      return route.domain == TaskDomain::host && !route.cuda_device;
    }

    bool can_direct_transfer(TaskHandle, TaskHandle) const noexcept override { return true; }

    /// \brief Reschedule a previously suspended coroutine.
    void reschedule(BasicTask&& t) override { this->enqueue_task(std::move(t)); }

    class SuspendedWait {
      public:
        SuspendedWait(bool wake_on_progress, bool nested_wait)
            : wake_on_progress_(wake_on_progress), nested_wait_(nested_wait)
        {}

        void install(oneapi::tbb::task::suspend_point point) noexcept
        {
          oneapi::tbb::task::suspend_point point_to_resume = nullptr;
          {
            std::scoped_lock lock(mutex_);
            point_ = point;
            if (signaled_ && !resume_issued_)
            {
              resume_issued_ = true;
              point_to_resume = point_;
            }
          }
          if (point_to_resume) oneapi::tbb::task::resume(point_to_resume);
        }

        void signal() noexcept
        {
          oneapi::tbb::task::suspend_point point_to_resume = nullptr;
          {
            std::scoped_lock lock(mutex_);
            signaled_ = true;
            if (point_ && !resume_issued_)
            {
              resume_issued_ = true;
              point_to_resume = point_;
            }
          }
          if (point_to_resume) oneapi::tbb::task::resume(point_to_resume);
        }

        [[nodiscard]] bool wake_on_progress() const noexcept { return wake_on_progress_; }
        [[nodiscard]] bool nested_wait() const noexcept { return nested_wait_; }

      private:
        std::mutex mutex_;
        oneapi::tbb::task::suspend_point point_{nullptr};
        bool signaled_{false};
        bool resume_issued_{false};
        bool wake_on_progress_;
        bool nested_wait_;
    };

    class ReadySignal {
      public:
        explicit ReadySignal(TbbScheduler& scheduler) : scheduler_(&scheduler) {}

        void attach(std::shared_ptr<SuspendedWait> const& wait) noexcept
        {
          bool already_ready = false;
          {
            std::scoped_lock lock(mutex_);
            active_wait_ = wait;
            already_ready = ready_;
          }
          if (already_ready) wait->signal();
        }

        void detach(SuspendedWait const* wait) noexcept
        {
          std::scoped_lock lock(mutex_);
          auto active = active_wait_.lock();
          if (active.get() == wait) active_wait_.reset();
        }

        void notify_ready() noexcept
        {
          std::shared_ptr<SuspendedWait> active;
          {
            std::scoped_lock lock(mutex_);
            ready_ = true;
            active = active_wait_.lock();
          }
          if (active) active->signal();
          scheduler_->wait_cv_.notify_all();
        }

      private:
        TbbScheduler* scheduler_;
        std::mutex mutex_;
        std::weak_ptr<SuspendedWait> active_wait_;
        bool ready_{false};
    };

    class ExecutionScope {
      public:
        explicit ExecutionScope(TbbScheduler& scheduler) : scheduler_(&scheduler), previous_(current_)
        {
          current_ = this;
        }

        ExecutionScope(ExecutionScope const&) = delete;
        ExecutionScope& operator=(ExecutionScope const&) = delete;

        ~ExecutionScope()
        {
          CHECK(!blocked_);
          CHECK_EQUAL(current_, this);
          current_ = previous_;
          scheduler_->finish_runnable_quantum();
        }

        static ExecutionScope* block_current() noexcept
        {
          auto* scope = current_;
          if (!scope) return nullptr;

          CHECK(!scope->blocked_);
          CHECK_EQUAL(current_, scope);
          current_ = scope->previous_;
          scope->blocked_ = true;
          scope->scheduler_->block_runnable_quantum();
          return scope;
        }

        static void unblock(ExecutionScope* scope) noexcept
        {
          if (!scope) return;

          CHECK(scope->blocked_);
          CHECK_EQUAL(current_, scope->previous_);
          scope->scheduler_->unblock_runnable_quantum();
          scope->blocked_ = false;
          current_ = scope;
        }

      private:
        inline static thread_local ExecutionScope* current_{nullptr};

        TbbScheduler* scheduler_;
        ExecutionScope* previous_;
        bool blocked_{false};
    };

    class BlockedExecutionScope {
      public:
        BlockedExecutionScope() noexcept : scope_(ExecutionScope::block_current()) {}
        BlockedExecutionScope(BlockedExecutionScope const&) = delete;
        BlockedExecutionScope& operator=(BlockedExecutionScope const&) = delete;
        ~BlockedExecutionScope() { ExecutionScope::unblock(scope_); }

        [[nodiscard]] bool blocks_execution_quantum() const noexcept { return scope_ != nullptr; }

      private:
        ExecutionScope* scope_;
    };

    void wait_in_arena(WaitRequest const& request, std::shared_ptr<ReadySignal> const& ready_signal)
    {
      BlockedExecutionScope blocked_execution;

      while (!request.is_ready())
      {
        detail::service_tbb_task_registry_debug_requests();
        if (runnable_quanta_.load(std::memory_order_acquire) == 0)
        {
          if (!this->wait_for_idle_progress(request)) this->raise_wait_timeout();
          continue;
        }

        auto suspended_wait =
            std::make_shared<SuspendedWait>(!request.notify_when_ready, blocked_execution.blocks_execution_quantum());
        ready_signal->attach(suspended_wait);
        this->register_suspended_wait(suspended_wait);

        if (request.is_ready() || runnable_quanta_.load(std::memory_order_acquire) == 0) suspended_wait->signal();

        oneapi::tbb::task::suspend(
            [suspended_wait](oneapi::tbb::task::suspend_point point) { suspended_wait->install(point); });
        ready_signal->detach(suspended_wait.get());
      }
    }

    [[nodiscard]] bool wait_for_idle_progress(WaitRequest const& request)
    {
      auto const observed_generation = work_generation_.load(std::memory_order_acquire);
      auto made_progress = [&] {
        return request.is_ready() || runnable_quanta_.load(std::memory_order_acquire) != 0 ||
               work_generation_.load(std::memory_order_acquire) != observed_generation;
      };

      std::unique_lock lock(wait_mutex_);
      if (!wait_options_.watchdog_timeout)
      {
        wait_cv_.wait(lock, made_progress);
        return true;
      }
      return wait_cv_.wait_for(lock, *wait_options_.watchdog_timeout, made_progress);
    }

    [[noreturn]] void raise_wait_timeout()
    {
      detail::service_tbb_task_registry_debug_requests();
#if UNI20_DEBUG_ASYNC_TASKS
      auto const path = TaskRegistry::default_graphviz_dump_path();
      if (TaskRegistry::dump_graphviz_file_best_effort(path))
        display::info("Async Graphviz DAG snapshot written to {}", path);
      TaskRegistry::dump();
#endif
      trace::raise(async_wait_timeout(*wait_options_.watchdog_timeout, paused_.load(std::memory_order_acquire)));
    }

    void submit_runnable_quantum()
    {
      runnable_quanta_.fetch_add(1, std::memory_order_release);
      work_generation_.fetch_add(1, std::memory_order_release);
      wait_cv_.notify_all();
    }

    void finish_runnable_quantum() noexcept
    {
      auto const previous = runnable_quanta_.fetch_sub(1, std::memory_order_acq_rel);
      CHECK(previous > 0);
      this->notify_suspended_waiters(previous == 1);
      wait_cv_.notify_all();
    }

    void block_runnable_quantum() noexcept
    {
      auto const previous = runnable_quanta_.fetch_sub(1, std::memory_order_acq_rel);
      CHECK(previous > 0);
      if (previous == 1) this->notify_suspended_waiters(true);
      wait_cv_.notify_all();
    }

    void unblock_runnable_quantum() noexcept
    {
      runnable_quanta_.fetch_add(1, std::memory_order_release);
      wait_cv_.notify_all();
    }

    void register_suspended_wait(std::shared_ptr<SuspendedWait> const& wait)
    {
      {
        std::scoped_lock lock(suspended_waits_mutex_);
        std::erase_if(suspended_waits_, [](auto const& weak_wait) { return weak_wait.expired(); });
        suspended_waits_.push_back(wait);
      }
      if (runnable_quanta_.load(std::memory_order_acquire) == 0) wait->signal();
    }

    void notify_suspended_waiters(bool scheduler_idle) noexcept
    {
      std::vector<std::shared_ptr<SuspendedWait>> waits;
      {
        std::scoped_lock lock(suspended_waits_mutex_);
        auto output = suspended_waits_.begin();
        for (auto input = suspended_waits_.begin(); input != suspended_waits_.end(); ++input)
        {
          if (auto wait = input->lock())
          {
            waits.push_back(wait);
            *output++ = *input;
          }
        }
        suspended_waits_.erase(output, suspended_waits_.end());
      }

      if (scheduler_idle)
      {
        // Resumable task stacks can be nested. Wake only the innermost nested
        // wait; its completion may make a suspended ancestor runnable. When no
        // nested stack exists, independent application-thread waits can all
        // enter their watchdog paths.
        auto nested = std::find_if(waits.rbegin(), waits.rend(), [](auto const& wait) { return wait->nested_wait(); });
        if (nested != waits.rend())
          (*nested)->signal();
        else
          for (auto const& wait : waits)
            wait->signal();
        return;
      }

      for (auto const& wait : waits)
      {
        if (wait->wake_on_progress()) wait->signal();
      }
    }

  protected:
    /// \brief Wait for every activation registered with the shared task group.
    void wait_for_submitted_tasks()
    {
      arena_.execute([&] { tg_.wait(); });
    }

    /// \brief Enqueue one task through the scheduler's domain-routing hook.
    void enqueue_task(BasicTask&& t)
    {
      TRACE_MODULE(ASYNC, "TBB scheduler enqueuing task", t.coroutine_handle());
      if (auto h = t.release_handle())
      {
        bool paused = paused_.load(std::memory_order_acquire);
        if (paused)
        {
          queue_.push(h);
        }
        else
        {
          {
            std::scoped_lock lock(pause_mutex_);
            if (paused_.load(std::memory_order_acquire))
            {
              queue_.push(h);
              return;
            }
          }

          this->dispatch_handle(h);
        }
      }
    }

    /// \brief Route one runnable task activation to its execution arena.
    virtual void dispatch_handle(TaskHandle handle)
    {
      this->dispatch_handle_in_arena(arena_, handle, {.scheduler = "TbbScheduler"}, [] {});
    }

    /// \brief Submit one activation to an arena while sharing scheduler accounting.
    /// \tparam Prepare callable that establishes or verifies the arena execution context.
    /// \param arena Arena that will execute the activation.
    /// \param handle Promise-neutral task identity.
    /// \param context Diagnostic context for admission failure.
    /// \param prepare Callable invoked immediately before coroutine resumption.
    template <typename Prepare>
    void dispatch_handle_in_arena(oneapi::tbb::task_arena& arena, TaskHandle handle,
                                  detail::TbbTaskAdmissionContext context, Prepare&& prepare)
    {
      this->submit_runnable_quantum();
      detail::enqueue_tbb_task(
          arena, tg_,
          [this, handle, prepare = std::forward<Prepare>(prepare)] {
            ExecutionScope execution(*this);
            prepare();
            TRACE_MODULE(ASYNC, "resuming coroutine", handle.coroutine());
            TaskPromiseBase::resume_and_track(handle);
            detail::service_tbb_task_registry_debug_requests();
          },
          context);
    }

  private:
    oneapi::tbb::task_arena arena_;
    oneapi::tbb::task_group tg_;
    TbbSchedulerWaitOptions wait_options_;
    std::atomic<bool> paused_;
    std::mutex pause_mutex_;
    oneapi::tbb::concurrent_queue<TaskHandle> queue_;
    std::atomic<std::size_t> runnable_quanta_{0};
    std::atomic<std::uint64_t> work_generation_{0};
    std::mutex suspended_waits_mutex_;
    std::vector<std::weak_ptr<SuspendedWait>> suspended_waits_;
    std::condition_variable wait_cv_;
    std::mutex wait_mutex_;
    // FIXME: the concurrent_queue is overkill here, since we don't need to preserve order of tasks

    friend class TbbNumaScheduler;
};

} // namespace uni20::async
