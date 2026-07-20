/**
 * \file task_awaiters.hpp
 * \ingroup backend_cuda
 * \brief CUDA-specific awaiters for device and runtime resource scheduling.
 */

#pragma once

#include <uni20/async/cuda_task.hpp>
#include <uni20/config.hpp>

#if !UNI20_BACKEND_CUDA
#error "CUDA task awaiters require the CUDA backend"
#endif

#include "cuda_error.hpp"

#include <cuda_runtime_api.h>

namespace uni20::async
{

/// \brief Awaiter that establishes or changes a CUDA task's device affinity.
/// \details The scheduler-established current CUDA device determines whether
///          selection needs another activation. A different device resubmits
///          through the recorded scheduler; an already-current device resumes
///          immediately after recording affinity.
class CudaDeviceAwaiter : public CudaTaskAwaiterTag {
  public:
    explicit constexpr CudaDeviceAwaiter(int device) noexcept : device_(device) {}

    [[nodiscard]] constexpr bool await_ready() const noexcept { return false; }

    bool await_suspend(std::coroutine_handle<CudaTaskPromise> handle) const
    {
      int current_device = -1;
      cuda::check(cudaGetDevice(&current_device), "cudaGetDevice for CUDA task device selection");

      auto current = TaskHandle::from(handle);
      auto& promise = handle.promise();
      promise.select_device(device_);
      if (current_device == device_) return false;

      TaskPromiseBase::note_suspended(current);
      BasicTask::reschedule(promise.take_ownership());
      return true;
    }

    constexpr void await_resume() const noexcept {}

  private:
    int device_;
};

} // namespace uni20::async

namespace uni20::cuda
{

/// \brief Select the CUDA device for subsequent activations of the current task.
/// \details This operation is valid only in a `CudaTask` coroutine. Selection
///          continues immediately when the scheduler has already established
///          the requested current device; otherwise the task is resubmitted.
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
