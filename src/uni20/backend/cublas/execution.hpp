#pragma once

/**
 * \file execution.hpp
 * \ingroup backend_cublas
 * \brief cuBLAS handle slots and dynamically paired stream execution leases.
 */

#include <uni20/backend/cuda/resource_pool.hpp>
#include <uni20/backend/cuda/runtime.hpp>

#include <cublas_v2.h>

#include <cstddef>
#include <optional>

namespace uni20::cuda
{
class DeviceResources;
}

namespace uni20::cublas
{

class ExecutionAcquireAwaiter;

/// \brief One expensive reusable cuBLAS handle and its handle-local state.
class HandleSlot {
  public:
    explicit HandleSlot(int device);
    HandleSlot(HandleSlot const&) = delete;
    HandleSlot& operator=(HandleSlot const&) = delete;
    HandleSlot(HandleSlot&& other) noexcept;
    HandleSlot& operator=(HandleSlot&& other) noexcept;
    ~HandleSlot();

    /// \brief Return the device on which this handle was created.
    [[nodiscard]] int device() const noexcept { return device_; }

    /// \brief Return the raw handle for narrow provider-wrapper use.
    [[nodiscard]] cublasHandle_t native_handle() const noexcept { return handle_; }

    /// \brief Establish deterministic host-pointer mode and bind one operation stream.
    void bind(cuda::Stream const& stream);

  private:
    void reset() noexcept;

    int device_ = -1;
    cublasHandle_t handle_ = nullptr;
};

/// \brief Move-only dynamic pairing of one cuBLAS handle slot and one idle stream.
/// \details The handle returns to its pool at the submitted stream tail. The
///          stream follows its own reference-counted idle-pool lifetime.
class ExecutionLease {
  public:
    using handle_lease_type = cuda::ResourceLease<HandleSlot>;

    ExecutionLease() noexcept = default;
    ExecutionLease(ExecutionLease const&) = delete;
    ExecutionLease& operator=(ExecutionLease const&) = delete;
    ExecutionLease(ExecutionLease&& other) noexcept;
    ExecutionLease& operator=(ExecutionLease&& other) noexcept;
    ~ExecutionLease();

    /// \brief Return whether this lease owns both required execution resources.
    [[nodiscard]] explicit operator bool() const noexcept
    {
      return static_cast<bool>(handle_) && static_cast<bool>(stream_);
    }

    /// \brief Access the exclusively leased cuBLAS handle slot.
    [[nodiscard]] HandleSlot& handle() const { return handle_.get(); }

    /// \brief Access the dynamically selected CUDA stream.
    [[nodiscard]] cuda::Stream const& stream() const noexcept { return stream_; }

    /// \brief Publish deferred handle return at the current stream tail.
    void release() noexcept;

  private:
    ExecutionLease(handle_lease_type handle, cuda::Stream stream);

    handle_lease_type handle_;
    cuda::Stream stream_;

    friend class ExecutionPool;
    friend class ExecutionAcquireAwaiter;
};

/// \brief Device-local cuBLAS handle pool combined with a shared idle-stream pool.
/// \details Handle slots and streams are independent persistent pools. Each
///          execution acquisition reserves a handle first, then obtains an
///          actually-idle stream and returns the temporary pair. The execution
///          pool and its stream pool must outlive all leases and deferred handle
///          returns.
class ExecutionPool {
  public:
    /// \brief Construct `handle_count` cuBLAS handles for the stream pool's device.
    ExecutionPool(cuda::StreamPool& streams, std::size_t handle_count);
    ExecutionPool(ExecutionPool const&) = delete;
    ExecutionPool& operator=(ExecutionPool const&) = delete;
    ExecutionPool(ExecutionPool&&) = delete;
    ExecutionPool& operator=(ExecutionPool&&) = delete;

    /// \brief Try to acquire a handle and stream without waiting.
    [[nodiscard]] std::optional<ExecutionLease> try_acquire();

    /// \brief Block until a handle and then an idle stream are available.
    [[nodiscard]] ExecutionLease acquire();

    /// \brief Return the CUDA device served by this execution pool.
    [[nodiscard]] int device() const noexcept { return streams_->device(); }

    /// \brief Return the fixed number of provider handles.
    [[nodiscard]] std::size_t handle_count() const noexcept { return handles_.size(); }

    /// \brief Return the number of currently idle provider handles.
    [[nodiscard]] std::size_t idle_handle_count() const noexcept { return handles_.idle_count(); }

  private:
    [[nodiscard]] ExecutionLease make_lease(cuda::ResourceLease<HandleSlot> handle, cuda::Stream stream);

    cuda::StreamPool* streams_;
    cuda::ResourcePool<HandleSlot> handles_;

    friend class ExecutionAcquireAwaiter;
};

/// \brief Return the lazily constructed cuBLAS execution pool for one CUDA device.
/// \details The device resources own the pool and retain it until after every
///          Tensor and operation using those resources has finished.
[[nodiscard]] ExecutionPool& execution_pool(cuda::DeviceResources& resources);

} // namespace uni20::cublas
