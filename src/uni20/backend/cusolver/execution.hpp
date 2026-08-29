#pragma once

/**
 * \file execution.hpp
 * \ingroup backend_cusolver
 * \brief cuSOLVER handle slots and dynamically paired stream execution leases.
 */

#include <uni20/backend/cuda/resource_pool.hpp>
#include <uni20/backend/cuda/runtime.hpp>

#include <cusolverDn.h>

#include <cstddef>
#include <optional>

namespace uni20::cusolver
{

/// \brief One reusable cuSOLVER handle and its handle-local state.
class HandleSlot {
  public:
    explicit HandleSlot(int device);
    HandleSlot(HandleSlot const&) = delete;
    HandleSlot& operator=(HandleSlot const&) = delete;
    HandleSlot(HandleSlot&& other) noexcept;
    HandleSlot& operator=(HandleSlot&& other) noexcept;
    ~HandleSlot();

    [[nodiscard]] int device() const noexcept { return device_; }
    [[nodiscard]] cusolverDnHandle_t native_handle() const noexcept { return handle_; }

    /// \brief Bind one operation stream to the exclusively leased handle.
    void bind(cuda::Stream const& stream);

  private:
    void reset() noexcept;

    int device_ = -1;
    cusolverDnHandle_t handle_ = nullptr;
};

/// \brief Move-only pairing of one cuSOLVER handle slot and one idle stream.
class ExecutionLease {
  public:
    using handle_lease_type = cuda::ResourceLease<HandleSlot>;

    ExecutionLease() noexcept = default;
    ExecutionLease(ExecutionLease const&) = delete;
    ExecutionLease& operator=(ExecutionLease const&) = delete;
    ExecutionLease(ExecutionLease&& other) noexcept;
    ExecutionLease& operator=(ExecutionLease&& other) noexcept;
    ~ExecutionLease();

    [[nodiscard]] explicit operator bool() const noexcept
    {
      return static_cast<bool>(handle_) && static_cast<bool>(stream_);
    }
    [[nodiscard]] HandleSlot& handle() const { return handle_.get(); }
    [[nodiscard]] cuda::Stream const& stream() const noexcept { return stream_; }

    /// \brief Publish deferred handle return at the current stream tail.
    void release() noexcept;

  private:
    ExecutionLease(handle_lease_type handle, cuda::Stream stream);

    handle_lease_type handle_;
    cuda::Stream stream_;

    friend class ExecutionPool;
};

/// \brief Device-local cuSOLVER handle pool paired with the shared stream pool.
class ExecutionPool {
  public:
    ExecutionPool(cuda::StreamPool& streams, std::size_t handle_count);
    ExecutionPool(ExecutionPool const&) = delete;
    ExecutionPool& operator=(ExecutionPool const&) = delete;
    ExecutionPool(ExecutionPool&&) = delete;
    ExecutionPool& operator=(ExecutionPool&&) = delete;

    [[nodiscard]] std::optional<ExecutionLease> try_acquire();
    [[nodiscard]] ExecutionLease acquire();
    [[nodiscard]] int device() const noexcept { return streams_->device(); }
    [[nodiscard]] std::size_t handle_count() const noexcept { return handles_.size(); }
    [[nodiscard]] std::size_t idle_handle_count() const noexcept { return handles_.idle_count(); }

  private:
    [[nodiscard]] ExecutionLease make_lease(cuda::ResourceLease<HandleSlot> handle, cuda::Stream stream);

    cuda::StreamPool* streams_;
    cuda::ResourcePool<HandleSlot> handles_;
};

/// \brief Return the lazily constructed cuSOLVER execution pool for one device.
/// \details Unless configured explicitly, the pool contains two handles, capped
///          by the device's stream capacity.
[[nodiscard]] ExecutionPool& execution_pool(cuda::DeviceResources& resources);

/// \brief Configure or return the canonical cuSOLVER execution pool for one device.
/// \details This first-use configuration must precede concurrent provider use.
///          Repeated calls must request the same handle count. The handle count
///          cannot exceed the device's stream capacity.
/// \param resources Device resources that own the canonical pool.
/// \param handle_count Number of reusable cuSOLVER handles.
[[nodiscard]] ExecutionPool& execution_pool(cuda::DeviceResources& resources, std::size_t handle_count);

} // namespace uni20::cusolver
