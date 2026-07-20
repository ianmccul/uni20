/**
 * \file task_awaiters.hpp
 * \ingroup backend_cuda
 * \brief CUDA-specific awaiters for device and runtime resource scheduling.
 */

#pragma once

#include <uni20/async/cuda_task.hpp>
#include <uni20/backend/cuda/resource_pool.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/config.hpp>

#if !UNI20_BACKEND_CUDA
#error "CUDA task awaiters require the CUDA backend"
#endif

#include "cuda_error.hpp"

#include <cuda_runtime_api.h>

#include <atomic>

namespace uni20::async
{

/// \brief CUDA-task awaiter for one provider resource from a fixed-capacity pool.
template <class Resource>
class CudaResourceAcquireAwaiter : public CudaTaskAwaiterTag, private cuda::detail::ResourceWaiter<Resource> {
  public:
    explicit CudaResourceAcquireAwaiter(cuda::ResourcePool<Resource>& pool) noexcept
        : cuda::detail::ResourceWaiter<Resource>(&CudaResourceAcquireAwaiter::resource_available), pool_(&pool)
    {}

    CudaResourceAcquireAwaiter(CudaResourceAcquireAwaiter const&) = delete;
    CudaResourceAcquireAwaiter& operator=(CudaResourceAcquireAwaiter const&) = delete;
    CudaResourceAcquireAwaiter(CudaResourceAcquireAwaiter&&) = delete;
    CudaResourceAcquireAwaiter& operator=(CudaResourceAcquireAwaiter&&) = delete;

    [[nodiscard]] bool await_ready()
    {
      lease_ = pool_->try_acquire();
      return lease_.has_value();
    }

    bool await_suspend(std::coroutine_handle<CudaTaskPromise> handle) noexcept
    {
      auto current = TaskHandle::from(handle);
      state_.store(registration_state::registering, std::memory_order_relaxed);
      task_ = handle.promise().take_ownership();
      TaskPromiseBase::note_suspended(current);

      auto lease = pool_->acquire_or_enqueue(*this);
      if (lease)
      {
        lease_ = std::move(*lease);
        this->publish_ready();
      }

      auto const prior = state_.exchange(registration_state::suspended, std::memory_order_acq_rel);
      if (prior == registration_state::ready) BasicTask::reschedule(std::move(task_));
      return true;
    }

    [[nodiscard]] cuda::ResourceLease<Resource> await_resume()
    {
      CHECK(lease_.has_value());
      return std::move(*lease_);
    }

  private:
    static void resource_available(cuda::detail::ResourceWaiter<Resource>& waiter,
                                   cuda::ResourceLease<Resource> lease) noexcept
    {
      auto& self = static_cast<CudaResourceAcquireAwaiter&>(waiter);
      self.lease_ = std::move(lease);
      self.publish_ready();
    }

    enum class registration_state
    {
      registering,
      suspended,
      ready
    };

    void publish_ready() noexcept
    {
      auto const prior = state_.exchange(registration_state::ready, std::memory_order_acq_rel);
      if (prior == registration_state::suspended) BasicTask::reschedule(std::move(task_));
    }

    cuda::ResourcePool<Resource>* pool_;
    std::optional<cuda::ResourceLease<Resource>> lease_;
    BasicTask task_;
    std::atomic<registration_state> state_{registration_state::registering};
};

/// \brief CUDA-task awaiter for an actually-idle stream-pool lease.
class CudaStreamAcquireAwaiter : public CudaTaskAwaiterTag, private cuda::detail::StreamWaiter {
  public:
    explicit CudaStreamAcquireAwaiter(cuda::StreamPool& pool) noexcept
        : cuda::detail::StreamWaiter(&CudaStreamAcquireAwaiter::stream_available), pool_(&pool)
    {}

    CudaStreamAcquireAwaiter(CudaStreamAcquireAwaiter const&) = delete;
    CudaStreamAcquireAwaiter& operator=(CudaStreamAcquireAwaiter const&) = delete;
    CudaStreamAcquireAwaiter(CudaStreamAcquireAwaiter&&) = delete;
    CudaStreamAcquireAwaiter& operator=(CudaStreamAcquireAwaiter&&) = delete;

    [[nodiscard]] bool await_ready()
    {
      stream_ = pool_->try_acquire();
      return stream_.has_value();
    }

    bool await_suspend(std::coroutine_handle<CudaTaskPromise> handle) noexcept
    {
      auto current = TaskHandle::from(handle);
      state_.store(registration_state::registering, std::memory_order_relaxed);
      task_ = handle.promise().take_ownership();
      TaskPromiseBase::note_suspended(current);

      auto stream = pool_->acquire_or_enqueue(*this);
      if (stream)
      {
        stream_ = std::move(*stream);
        this->publish_ready();
      }

      auto const prior = state_.exchange(registration_state::suspended, std::memory_order_acq_rel);
      if (prior == registration_state::ready) BasicTask::reschedule(std::move(task_));
      return true;
    }

    [[nodiscard]] cuda::Stream await_resume()
    {
      CHECK(stream_.has_value());
      int current_device = -1;
      cuda::check(cudaGetDevice(&current_device), "cudaGetDevice after stream acquisition");
      CHECK_EQUAL(current_device, stream_->device(), "CUDA task resumed outside the acquired stream's device");
      return std::move(*stream_);
    }

  private:
    static void stream_available(cuda::detail::StreamWaiter& waiter, cuda::Stream stream) noexcept
    {
      auto& self = static_cast<CudaStreamAcquireAwaiter&>(waiter);
      self.stream_ = std::move(stream);
      self.publish_ready();
    }

    enum class registration_state
    {
      registering,
      suspended,
      ready
    };

    void publish_ready() noexcept
    {
      auto const prior = state_.exchange(registration_state::ready, std::memory_order_acq_rel);
      if (prior == registration_state::suspended) BasicTask::reschedule(std::move(task_));
    }

    cuda::StreamPool* pool_;
    std::optional<cuda::Stream> stream_;
    BasicTask task_;
    std::atomic<registration_state> state_{registration_state::registering};
};

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

/// \brief Await one provider resource without blocking the CUDA scheduler participant.
template <class Resource>
[[nodiscard]] inline async::CudaResourceAcquireAwaiter<Resource> acquire_resource(ResourcePool<Resource>& pool)
{
  return async::CudaResourceAcquireAwaiter<Resource>(pool);
}

/// \brief Await one actually-idle stream without blocking the CUDA scheduler participant.
[[nodiscard]] inline async::CudaStreamAcquireAwaiter acquire_stream(StreamPool& pool)
{
  return async::CudaStreamAcquireAwaiter(pool);
}

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
