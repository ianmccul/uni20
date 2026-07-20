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

/// \brief FIFO debug scheduler that admits ordinary and CUDA tasks.
/// \details All tasks share one deterministic runnable queue. Each CUDA
///          activation selects the device stored in its promise and restores
///          the calling thread's previous device when it suspends or completes.
class DebugCudaScheduler final : public DebugScheduler, public ICudaScheduler {
  public:
    using DebugScheduler::schedule;

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
      auto const from_device = cuda_promise(from).device();
      auto const to_device = cuda_promise(to).device();
      return from_device && from_device == to_device;
    }

    void resume_task(BasicTask& task) override
    {
      if (task.handle().domain() != TaskDomain::cuda)
      {
        DebugScheduler::resume_task(task);
        return;
      }

      auto const device = cuda_promise(task.handle()).device();
      CHECK(device, "CUDA task has no bound device at activation");
      cuda::ScopedDevice device_scope(*device);
      DebugScheduler::resume_task(task);
    }
};

} // namespace uni20::async
