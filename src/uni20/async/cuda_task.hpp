/**
 * \file cuda_task.hpp
 * \brief Defines the CUDA task admission type and scheduler interface.
 */

#pragma once

#include "async_task_promise.hpp"

namespace uni20::async
{

/// \brief CUDA-constrained public task type used for typed initial admission.
class CudaTask final : public BasicTask {
  public:
    using base_type = BasicTask;
    using promise_type = CudaTaskPromise;

    CudaTask() = default;
    CudaTask(CudaTask const&) = delete;
    CudaTask& operator=(CudaTask const&) = delete;
    CudaTask(CudaTask&&) noexcept = default;
    CudaTask& operator=(CudaTask&&) noexcept = default;

    /// \brief Assign an optional debug label while preserving the concrete task type.
    /// \param label Human-readable label used only by debug diagnostics.
    /// \return Reference to this task for call chaining.
    CudaTask& debug_name(std::string const& label)
    {
      this->BasicTask::debug_name(label);
      return *this;
    }

  private:
    explicit CudaTask(TaskHandle handle) noexcept : BasicTask(handle, construction_key{}) {}

    friend class CudaTaskPromise;
};

inline CudaTask CudaTaskPromise::get_return_object() noexcept
{
  auto handle = std::coroutine_handle<CudaTaskPromise>::from_promise(*this);
  return CudaTask(this->initialize_task(handle));
}

/// \brief Scheduler interface for initial `CudaTask` admission.
/// \details Virtual inheritance permits one scheduler object to implement both
///          ordinary and CUDA admission with one unambiguous internal route.
class ICudaScheduler : public virtual IScheduler {
  public:
    ~ICudaScheduler() override = default;

    /// \brief Bind and submit a CUDA task for initial execution on a device.
    /// \param task CUDA task to admit.
    /// \param device CUDA runtime device ordinal.
    virtual void schedule(CudaTask&& task, int device) = 0;
};

} // namespace uni20::async
