#include <uni20/backend/cusolver/execution.hpp>

#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/backend/cusolver/cusolver_error.hpp>
#include <uni20/common/trace.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <new>
#include <utility>
#include <vector>

namespace uni20::cusolver
{
namespace
{

std::vector<HandleSlot> make_handle_slots(int device, std::size_t count)
{
  CHECK(count > 0, count);
  std::vector<HandleSlot> result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    result.emplace_back(device);
  return result;
}

struct DeferredHandleReturn
{
    cuda::ResourceLease<HandleSlot> handle;
};

void CUDART_CB return_handle_callback(void* raw_payload) noexcept
{
  delete static_cast<DeferredHandleReturn*>(raw_payload);
}

[[noreturn]] void callback_registration_failure(cudaError_t status, cuda::Stream const& stream,
                                                DeferredHandleReturn* payload) noexcept
{
  try
  {
    stream.synchronize();
  }
  catch (...)
  {
    PANIC("cuSOLVER handle return failed and stream synchronization also failed", stream.device());
  }
  delete payload;
  PANIC("failed to enqueue cuSOLVER handle return callback", stream.device(), cudaGetErrorName(status),
        cudaGetErrorString(status));
}

[[noreturn]] void device_selection_failure(cuda::Stream const& stream, DeferredHandleReturn* payload) noexcept
{
  try
  {
    stream.synchronize();
  }
  catch (...)
  {
    PANIC("CUDA device selection failed while returning a cuSOLVER handle and stream synchronization also failed",
          stream.device());
  }
  delete payload;
  PANIC("CUDA device selection failed while returning a cuSOLVER handle", stream.device());
}

} // namespace

HandleSlot::HandleSlot(int device) : device_(device)
{
  cuda::ScopedDevice guard(device_);
  check(cusolverDnCreate(&handle_), "cusolverDnCreate", device_);
}

HandleSlot::HandleSlot(HandleSlot&& other) noexcept
    : device_(std::exchange(other.device_, -1)), handle_(std::exchange(other.handle_, nullptr))
{}

HandleSlot& HandleSlot::operator=(HandleSlot&& other) noexcept
{
  if (this != &other)
  {
    this->reset();
    device_ = std::exchange(other.device_, -1);
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

HandleSlot::~HandleSlot() { this->reset(); }

void HandleSlot::bind(cuda::Stream const& stream)
{
  CHECK(handle_ != nullptr);
  CHECK_EQUAL(device_, stream.device());
  cuda::ScopedDevice guard(device_);
  check(cusolverDnSetStream(handle_, stream.native_handle()), "cusolverDnSetStream", device_);
}

void HandleSlot::reset() noexcept
{
  if (handle_ == nullptr) return;
  try
  {
    cuda::ScopedDevice guard(device_);
    auto const status = cusolverDnDestroy(handle_);
    if (status != CUSOLVER_STATUS_SUCCESS)
      PANIC("cuSOLVER cleanup operation failed", "cusolverDnDestroy", device_, status_name(status));
  }
  catch (...)
  {
    PANIC("CUDA device selection failed while destroying a cuSOLVER handle", device_);
  }
  handle_ = nullptr;
  device_ = -1;
}

ExecutionLease::ExecutionLease(handle_lease_type handle, cuda::Stream stream)
    : handle_(std::move(handle)), stream_(std::move(stream))
{
  CHECK(handle_ && stream_);
  handle_.get().bind(stream_);
}

ExecutionLease::ExecutionLease(ExecutionLease&& other) noexcept
    : handle_(std::move(other.handle_)), stream_(std::move(other.stream_))
{}

ExecutionLease& ExecutionLease::operator=(ExecutionLease&& other) noexcept
{
  if (this != &other)
  {
    this->release();
    handle_ = std::move(other.handle_);
    stream_ = std::move(other.stream_);
  }
  return *this;
}

ExecutionLease::~ExecutionLease() { this->release(); }

void ExecutionLease::release() noexcept
{
  if (!handle_) return;
  CHECK(stream_);
  auto* payload = new (std::nothrow) DeferredHandleReturn{std::move(handle_)};
  if (payload == nullptr)
  {
    try
    {
      stream_.synchronize();
    }
    catch (...)
    {
      PANIC("failed to allocate cuSOLVER handle return payload and stream synchronization failed", stream_.device());
    }
    PANIC("failed to allocate cuSOLVER handle return callback payload", stream_.device());
  }

  cudaError_t status = cudaSuccess;
  try
  {
    cuda::ScopedDevice guard(stream_.device());
    status = cudaLaunchHostFunc(stream_.native_handle(), &return_handle_callback, payload);
  }
  catch (...)
  {
    device_selection_failure(stream_, payload);
  }
  if (status != cudaSuccess) callback_registration_failure(status, stream_, payload);
  stream_ = {};
}

ExecutionPool::ExecutionPool(cuda::StreamPool& streams, std::size_t handle_count)
    : streams_(&streams), handles_(make_handle_slots(streams.device(), handle_count))
{
  CHECK(handle_count <= streams.size(), "cuSOLVER handle count exceeds CUDA stream capacity", handle_count,
        streams.size());
}

std::optional<ExecutionLease> ExecutionPool::try_acquire()
{
  auto handle = handles_.try_acquire();
  if (!handle) return std::nullopt;
  auto stream = streams_->try_acquire();
  if (!stream) return std::nullopt;
  return this->make_lease(std::move(*handle), std::move(*stream));
}

ExecutionLease ExecutionPool::acquire()
{
  auto handle = handles_.acquire();
  auto stream = streams_->acquire();
  return this->make_lease(std::move(handle), std::move(stream));
}

ExecutionLease ExecutionPool::make_lease(cuda::ResourceLease<HandleSlot> handle, cuda::Stream stream)
{
  return ExecutionLease(std::move(handle), std::move(stream));
}

ExecutionPool& execution_pool(cuda::DeviceResources& resources)
{
  constexpr std::size_t default_handle_count = 2;
  auto const handle_count = std::min(default_handle_count, resources.streams().size());
  return resources.provider_resource<ExecutionPool>(resources.streams(), handle_count);
}

ExecutionPool& execution_pool(cuda::DeviceResources& resources, std::size_t handle_count)
{
  auto& result = resources.provider_resource<ExecutionPool>(resources.streams(), handle_count);
  CHECK_EQUAL(result.handle_count(), handle_count,
              "cuSOLVER execution pool must be configured before its first provider use", resources.device().ordinal());
  return result;
}

} // namespace uni20::cusolver
