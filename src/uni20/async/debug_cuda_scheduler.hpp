/**
 * \file debug_cuda_scheduler.hpp
 * \brief Deterministic scheduler for ordinary and CUDA task domains.
 */

#pragma once

#include "cuda_task.hpp"
#include "debug_scheduler.hpp"

#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/runtime.hpp>

namespace uni20::async
{

/// \brief Configurable debug scheduler that admits ordinary and CUDA tasks.
/// \details All tasks share one runnable queue and the ordering policy inherited
///          from `DebugScheduler`. Each CUDA activation selects the device
///          stored in its promise and restores the calling thread's previous
///          device when it suspends or completes.
class DebugCudaScheduler final : public DebugScheduler, public ICudaScheduler {
  public:
    /// \brief Construct with device zero as the default CUDA activation device.
    explicit DebugCudaScheduler(DebugSchedulerOptions options = {})
        : DebugScheduler(options), default_device_(cuda::Device::get(0).ordinal())
    {}

    /// \brief Construct with an explicit default CUDA activation device.
    /// \param default_device Device used while a CUDA task has no affinity.
    /// \param options Deterministic runnable ordering configuration.
    explicit DebugCudaScheduler(cuda::Device default_device, DebugSchedulerOptions options = {})
        : DebugScheduler(options), default_device_(default_device.ordinal())
    {}

    using DebugScheduler::schedule;

    /// \brief Enqueue a CUDA task without establishing device affinity.
    /// \param task CUDA task to admit through the default device context.
    void schedule(CudaTask&& task) override
    {
      TaskRegistry::record_task_scheduled(task.coroutine_handle());
      if (task.set_scheduler(this)) this->enqueue_task(std::move(task));
    }

    /// \brief Bind and enqueue a CUDA task for initial execution on a device.
    /// \param task CUDA task to admit.
    /// \param device CUDA runtime device ordinal.
    void schedule(CudaTask&& task, int device) override
    {
      auto const validated_device = cuda::Device::get(device);
      cuda_promise(task.handle()).bind_device(validated_device.ordinal());
      TaskRegistry::record_task_scheduled(task.coroutine_handle());
      if (task.set_scheduler(this)) this->enqueue_task(std::move(task));
    }

  private:
    bool can_direct_transfer(TaskHandle from, TaskHandle to) const noexcept override
    {
      if (from.domain() != TaskDomain::cuda) return true;
      return this->activation_device(from) == this->activation_device(to);
    }

    void resume_task(BasicTask& task) override
    {
      if (task.handle().domain() != TaskDomain::cuda)
      {
        DebugScheduler::resume_task(task);
        return;
      }

      int const device = this->activation_device(task.handle());
      cuda::ScopedDevice device_scope(device);
#if UNI20_ASYNC_DEBUG
      DEBUG_CHECK_EQUAL(cuda::Device::current().ordinal(), device);
#endif
      DebugScheduler::resume_task(task);
    }

    [[nodiscard]] int activation_device(TaskHandle task) const noexcept
    {
      return cuda_promise(task).device().value_or(default_device_);
    }

    int default_device_;
};

} // namespace uni20::async
