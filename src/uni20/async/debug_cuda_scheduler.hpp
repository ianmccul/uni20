/**
 * \file debug_cuda_scheduler.hpp
 * \brief Deterministic CUDA task scheduler bound to one CUDA device.
 */

#pragma once

#include "cuda_task.hpp"
#include "task_registry.hpp"

#include <algorithm>
#include <exception>
#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <utility>
#include <vector>

namespace uni20::async
{

/// \brief FIFO-batch CUDA task scheduler that executes on the calling thread.
/// \details The scheduler selects its bound CUDA device while running tasks and
///          restores the calling thread's previous device before returning.
class DebugCudaScheduler final : public ICudaScheduler {
  public:
    /// \brief Construct a scheduler bound to a validated CUDA device.
    /// \param device Device on which admitted tasks execute.
    explicit DebugCudaScheduler(cuda::Device device) noexcept : device_(device) {}

    /// \brief Validate a CUDA device ordinal and bind the scheduler to it.
    /// \param device CUDA runtime device ordinal.
    explicit DebugCudaScheduler(int device) : DebugCudaScheduler(cuda::Device::get(device)) {}

    /// \brief Release queued tasks during exception unwinding.
    ~DebugCudaScheduler() override
    {
      if (std::uncaught_exceptions() > 0)
      {
        while (!tasks_.empty())
        {
          tasks_.back().abandon_leak();
          tasks_.pop_back();
        }
      }
    }

    /// \brief Bind and enqueue a CUDA task for initial execution.
    /// \param task CUDA task to admit.
    void schedule(CudaTask&& task) override
    {
      cuda_promise(task.handle()).bind_device(device_.ordinal());
      TaskRegistry::record_task_scheduled(task.coroutine_handle());
      if (task.set_scheduler(this)) tasks_.push_back(std::move(task));
    }

    /// \brief Run one batch of queued tasks on the bound CUDA device.
    void run()
    {
      cuda::ScopedDevice device_scope(device_.ordinal());
      this->run_current_device();
    }

    /// \brief Run until no CUDA tasks remain runnable.
    void run_all()
    {
      cuda::ScopedDevice device_scope(device_.ordinal());
      while (!this->done())
        this->run_current_device();
    }

    /// \brief Report whether the runnable queue is empty.
    [[nodiscard]] bool done() const noexcept { return tasks_.empty(); }

    /// \brief Return the CUDA device bound to this scheduler.
    [[nodiscard]] cuda::Device device() const noexcept { return device_; }

  private:
    void reschedule(BasicTask&& task) override { tasks_.push_back(std::move(task)); }

    void run_current_device()
    {
#if UNI20_DEBUG_ASYNC_TASKS
      TaskRegistry::service_debug_requests();
#endif
      std::vector<BasicTask> tasks;
      std::swap(tasks, tasks_);
      std::reverse(tasks.begin(), tasks.end());
      for (auto&& task : tasks)
      {
        task.resume();
        CHECK(!task);
      }
#if UNI20_DEBUG_ASYNC_TASKS
      TaskRegistry::service_debug_requests();
#endif
    }

    cuda::Device device_;
    std::vector<BasicTask> tasks_;
};

} // namespace uni20::async
