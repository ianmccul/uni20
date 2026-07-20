/**
 * \file tbb_cuda_scheduler.hpp
 * \brief oneTBB coroutine scheduler bound to one CUDA device.
 */

#pragma once

#include "cuda_task.hpp"
#include "task_registry.hpp"
#include "tbb_task_submission.hpp"

#include <cuda_runtime_api.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <oneapi/tbb/task_scheduler_observer.h>
#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/config.hpp>

#include <vector>

namespace uni20::async
{

namespace detail
{

[[noreturn]] inline void cuda_arena_observer_failure(cudaError_t status, char const* operation, int device)
{
  PANIC("CUDA arena observer operation failed", operation, device, cudaGetErrorName(status),
        cudaGetErrorString(status));
}

inline void check_cuda_arena_observer(cudaError_t status, char const* operation, int device) noexcept
{
  if (status != cudaSuccess) cuda_arena_observer_failure(status, operation, device);
}

/// \brief Establishes one CUDA device while a thread participates in a TBB arena.
class CudaArenaObserver final : public oneapi::tbb::task_scheduler_observer {
  public:
    CudaArenaObserver(oneapi::tbb::task_arena& arena, int device)
        : oneapi::tbb::task_scheduler_observer(arena), device_(device)
    {}

    CudaArenaObserver(CudaArenaObserver const&) = delete;
    CudaArenaObserver& operator=(CudaArenaObserver const&) = delete;

    ~CudaArenaObserver() override { this->stop(); }

    void start() { this->observe(true); }

    void stop() noexcept
    {
      if (this->is_observing()) this->observe(false);
    }

  private:
    struct Selection
    {
        CudaArenaObserver const* observer;
        int previous_device;
    };

    static std::vector<Selection>& selection_stack()
    {
      static thread_local std::vector<Selection> stack;
      return stack;
    }

    void on_scheduler_entry(bool) noexcept override
    {
      int previous_device = -1;
      check_cuda_arena_observer(cudaGetDevice(&previous_device), "cudaGetDevice on arena entry", device_);
      selection_stack().push_back(Selection{this, previous_device});
      check_cuda_arena_observer(cudaSetDevice(device_), "cudaSetDevice on arena entry", device_);
    }

    void on_scheduler_exit(bool) noexcept override
    {
      auto& stack = selection_stack();
      CHECK(!stack.empty());
      CHECK_EQUAL(stack.back().observer, this);
      int const previous_device = stack.back().previous_device;
      stack.pop_back();
      check_cuda_arena_observer(cudaSetDevice(previous_device), "cudaSetDevice on arena exit", previous_device);
    }

    int device_;
};

inline void service_tbb_cuda_task_registry_debug_requests()
{
#if UNI20_DEBUG_ASYNC_TASKS
  TaskRegistry::service_debug_requests();
#endif
}

inline void verify_tbb_cuda_device(int expected_device)
{
#if UNI20_ASYNC_DEBUG
  int current_device = -1;
  check_cuda_arena_observer(cudaGetDevice(&current_device), "cudaGetDevice before coroutine resume", expected_device);
  DEBUG_CHECK_EQUAL(current_device, expected_device);
#else
  static_cast<void>(expected_device);
#endif
}

} // namespace detail

/// \brief oneTBB coroutine scheduler whose arena is bound to one CUDA device.
/// \details Every worker or application thread participating in the arena has
///          the scheduler device selected by `CudaArenaObserver`. The previous
///          thread-local CUDA device is restored when participation ends. Tasks
///          must not change the current CUDA device directly; cross-device work
///          must enter the scheduler bound to that device.
class TbbCudaScheduler final : public ICudaScheduler {
  public:
    /// \brief Construct a CUDA scheduler with a given arena concurrency.
    /// \param device Device selected while executing scheduler tasks.
    /// \param max_concurrency Maximum arena participation, including application threads.
    explicit TbbCudaScheduler(cuda::Device device, int max_concurrency = oneapi::tbb::task_arena::automatic)
        : device_(device), arena_(max_concurrency, /*reserved_for_application_threads=*/1),
          observer_(arena_, device_.ordinal())
    {
      arena_.initialize();
      {
        cuda::ScopedDevice device_scope(device_.ordinal());
        cuda::check(cudaSetDevice(device_.ordinal()), "cudaSetDevice for TBB CUDA scheduler initialization",
                    device_.ordinal());
      }
      observer_.start();
    }

    /// \brief Validate a CUDA device ordinal and construct its scheduler.
    /// \param device CUDA runtime device ordinal.
    /// \param max_concurrency Maximum arena participation, including application threads.
    explicit TbbCudaScheduler(int device, int max_concurrency = oneapi::tbb::task_arena::automatic)
        : TbbCudaScheduler(cuda::Device::get(device), max_concurrency)
    {}

    TbbCudaScheduler(TbbCudaScheduler const&) = delete;
    TbbCudaScheduler& operator=(TbbCudaScheduler const&) = delete;

    ~TbbCudaScheduler() noexcept override
    {
      arena_.execute([this] { tasks_.wait(); });
      observer_.stop();
    }

    /// \brief Bind and submit a CUDA task for initial execution.
    /// \param task CUDA task to admit.
    void schedule(CudaTask&& task) override
    {
      cuda_promise(task.handle()).bind_device(device_.ordinal());
      TaskRegistry::record_task_scheduled(task.coroutine_handle());
      if (task.set_scheduler(this)) this->enqueue_task(std::move(task));
    }

    /// \brief Wait for all currently runnable activations in this scheduler.
    /// \note Tasks suspended on external dependencies remain alive and may be
    ///       rescheduled later.
    void run_all()
    {
      detail::service_tbb_cuda_task_registry_debug_requests();
      arena_.execute([this] { tasks_.wait(); });
      detail::service_tbb_cuda_task_registry_debug_requests();
    }

    /// \brief Return the CUDA device bound to this scheduler.
    [[nodiscard]] cuda::Device device() const noexcept { return device_; }

  private:
    void reschedule(BasicTask&& task) override { this->enqueue_task(std::move(task)); }

    void enqueue_task(BasicTask&& task)
    {
      TRACE_MODULE(ASYNC, "TBB CUDA scheduler enqueuing task", task.coroutine_handle());
      if (auto handle = task.release_handle())
      {
        detail::enqueue_tbb_task(arena_, tasks_,
                                 [this, handle] {
                                   detail::verify_tbb_cuda_device(device_.ordinal());
                                   TRACE_MODULE(ASYNC, "TBB CUDA scheduler resuming coroutine", handle.coroutine());
                                   TaskPromiseBase::resume_and_track(handle);
                                   detail::service_tbb_cuda_task_registry_debug_requests();
                                 },
                                 {.scheduler = "TbbCudaScheduler", .device = device_.ordinal()});
      }
    }

    cuda::Device device_;
    oneapi::tbb::task_arena arena_;
    detail::CudaArenaObserver observer_;
    oneapi::tbb::task_group tasks_;
};

} // namespace uni20::async
