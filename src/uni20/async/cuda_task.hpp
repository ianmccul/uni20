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

    /// \brief Submit a CUDA task without establishing device affinity.
    /// \details The scheduler routes the task through its default CUDA device
    ///          until the task explicitly selects a device.
    /// \param task CUDA task to admit.
    virtual void schedule(CudaTask&& task) = 0;

    /// \brief Bind and submit a CUDA task for initial execution on a device.
    /// \param task CUDA task to admit.
    /// \param device CUDA runtime device ordinal.
    virtual void schedule(CudaTask&& task, int device) = 0;
};

} // namespace uni20::async

namespace uni20::cuda
{

/// \brief Select the CUDA device for subsequent activations of the current task.
/// \details This operation is valid only in a `CudaTask` coroutine. It always
///          suspends and resubmits the task through its recorded scheduler.
/// \param device CUDA runtime device ordinal.
/// \return Device-selection awaiter.
[[nodiscard]] inline async::CudaDeviceAwaiter set_device(int device) noexcept
{
  return async::CudaDeviceAwaiter(device);
}

/// \brief Select a CUDA device represented by a device-like value.
/// \tparam DeviceLike Type exposing an integer `ordinal()` member.
/// \param device Device identity whose ordinal will be selected.
/// \return Device-selection awaiter.
template <typename DeviceLike>
  requires requires(DeviceLike const& device) {
    { device.ordinal() } -> std::convertible_to<int>;
  }
[[nodiscard]] inline async::CudaDeviceAwaiter set_device(DeviceLike const& device) noexcept
{
  return set_device(static_cast<int>(device.ordinal()));
}

} // namespace uni20::cuda
