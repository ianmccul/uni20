/**
 * \file tbb_cuda_scheduler.hpp
 * \brief Unified oneTBB scheduler for host and multi-device CUDA tasks.
 */

#pragma once

#include "cuda_task.hpp"
#include "task_registry.hpp"
#include "tbb_scheduler.hpp"
#include "tbb_task_submission.hpp"

#include <cuda_runtime_api.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <oneapi/tbb/task_scheduler_observer.h>
#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/config.hpp>

#include <algorithm>
#include <memory>
#include <utility>
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

/// \brief Arena configuration for the unified host/CUDA TBB scheduler.
struct TbbCudaSchedulerOptions
{
    /// Maximum participation in the host arena.
    int host_max_concurrency = oneapi::tbb::task_arena::automatic;
    /// Maximum worker participation in each CUDA device arena.
    int cuda_max_concurrency_per_device = oneapi::tbb::task_arena::automatic;
    /// Blocking-wait watchdog policy shared by every task domain.
    TbbSchedulerWaitOptions wait_options{};
};

/// \brief Unified oneTBB scheduler with one host arena and one arena per CUDA device.
/// \details Ordinary and CUDA tasks share scheduler ownership, task-group
///          accounting, pause state, waits, and rescheduling. CUDA activations
///          are routed by the immutable device stored in `CudaTaskPromise`.
class TbbCudaScheduler final : public TbbScheduler, public ICudaScheduler {
  public:
    /// \brief Construct a scheduler enrolling every visible CUDA device.
    /// \param options Host and per-device arena configuration.
    explicit TbbCudaScheduler(TbbCudaSchedulerOptions options = {})
        : TbbCudaScheduler(cuda::Device::enumerate(), std::move(options))
    {}

    /// \brief Construct a scheduler for an explicit set of CUDA devices.
    /// \param devices Validated devices to enroll.
    /// \param options Host and per-device arena configuration.
    explicit TbbCudaScheduler(std::vector<cuda::Device> devices, TbbCudaSchedulerOptions options = {})
        : TbbScheduler(options.host_max_concurrency, std::move(options.wait_options))
    {
      device_arenas_.reserve(devices.size());
      for (auto device : devices)
      {
        CHECK(!this->find_device_arena(device.ordinal()), "CUDA device enrolled more than once", device.ordinal());
        device_arenas_.push_back(std::make_unique<DeviceArena>(device, options.cuda_max_concurrency_per_device));
      }
    }

    TbbCudaScheduler(TbbCudaScheduler const&) = delete;
    TbbCudaScheduler& operator=(TbbCudaScheduler const&) = delete;

    ~TbbCudaScheduler() noexcept override { this->wait_for_submitted_tasks(); }

    using TbbScheduler::schedule;

    /// \brief Bind and submit a CUDA task to an enrolled device arena.
    /// \param task CUDA task to admit.
    /// \param device CUDA runtime device ordinal selected for the task.
    void schedule(CudaTask&& task, int device) override
    {
      auto& device_arena = this->device_arena(device);
      cuda_promise(task.handle()).bind_device(device_arena.device.ordinal());
      TaskRegistry::record_task_scheduled(task.coroutine_handle());
      if (task.set_scheduler(this)) this->enqueue_task(std::move(task));
    }

    /// \brief Report whether a CUDA device has an execution arena in this scheduler.
    [[nodiscard]] bool has_device(int device) const noexcept { return this->find_device_arena(device) != nullptr; }

  private:
    class DeviceArena {
      public:
        DeviceArena(cuda::Device selected_device, int max_concurrency)
            : device(selected_device), arena(max_concurrency, /*reserved_for_application_threads=*/0),
              observer(arena, device.ordinal())
        {
          arena.initialize();
          {
            cuda::ScopedDevice device_scope(device.ordinal());
            cuda::check(cudaSetDevice(device.ordinal()), "cudaSetDevice for TBB CUDA scheduler initialization",
                        device.ordinal());
          }
          observer.start();
        }

        ~DeviceArena() { observer.stop(); }

        cuda::Device device;
        oneapi::tbb::task_arena arena;
        detail::CudaArenaObserver observer;
    };

    bool can_direct_transfer(TaskHandle from, TaskHandle to) const noexcept override
    {
      if (from.domain() != TaskDomain::cuda) return true;
      auto const from_device = cuda_promise(from).device();
      auto const to_device = cuda_promise(to).device();
      return from_device && from_device == to_device && this->has_device(*from_device);
    }

    void dispatch_handle(TaskHandle handle) override
    {
      if (handle.domain() != TaskDomain::cuda)
      {
        TbbScheduler::dispatch_handle(handle);
        return;
      }

      auto const device = cuda_promise(handle).device();
      CHECK(device, "CUDA task has no bound device at activation");
      auto& selected = this->device_arena(*device);
      this->dispatch_handle_in_arena(selected.arena, handle, {.scheduler = "TbbCudaScheduler", .device = *device},
                                     [device = *device] { detail::verify_tbb_cuda_device(device); });
    }

    [[nodiscard]] DeviceArena* find_device_arena(int device) const noexcept
    {
      auto const found = std::find_if(device_arenas_.begin(), device_arenas_.end(), [device](auto const& candidate) {
        return candidate->device.ordinal() == device;
      });
      return found == device_arenas_.end() ? nullptr : found->get();
    }

    DeviceArena& device_arena(int device)
    {
      auto* result = this->find_device_arena(device);
      CHECK(result, "CUDA task submitted to a device not enrolled in this scheduler", device);
      return *result;
    }

    std::vector<std::unique_ptr<DeviceArena>> device_arenas_;
};

} // namespace uni20::async
