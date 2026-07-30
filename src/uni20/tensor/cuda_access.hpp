/**
 * \file cuda_access.hpp
 * \ingroup tensor
 * \brief CUDA data-handle acquisition for buffer descriptors and tensor views.
 */

#pragma once

#include <uni20/storage/cuda_storage.hpp>
#include <uni20/tensor/access.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class Pointer> [[nodiscard]] Pointer offset_cuda_pointer(Pointer pointer, std::size_t offset) noexcept
{
  if (offset == 0) return pointer;
  CHECK(pointer != nullptr, "cannot offset an empty CUDA data handle", offset);
  return pointer + offset;
}

template <class DeviceSpan> void validate_cuda_descriptor_range(DeviceSpan const& span)
{
  auto const& descriptor = span.data_descriptor();
  auto const required = span.mapping().required_span_size();
  CHECK(std::cmp_greater_equal(required, 0), required);
  CHECK(std::cmp_less_equal(required, descriptor.buffer().size()), required, descriptor.buffer().size());
  auto const count = static_cast<std::size_t>(required);
  auto const offset = descriptor.element_offset();
  CHECK(offset <= descriptor.buffer().size() && count <= descriptor.buffer().size() - offset, offset, count,
        descriptor.buffer().size());
}

template <class DeviceSpan, class Pointer>
[[nodiscard]] auto resolve_cuda_mdspan(DeviceSpan const& span, Pointer pointer)
{
  using span_type = std::remove_cvref_t<DeviceSpan>;
  using mdspan_type = stdex::mdspan<typename span_type::element_type, typename span_type::extents_type,
                                    typename span_type::layout_type, typename span_type::accessor_type>;
  return mdspan_type{pointer, span.mapping(), span.accessor()};
}

template <class DeviceSpan, class AccessState>
[[nodiscard]] auto make_cuda_read_mdspan_lease(DeviceSpan device_span, std::size_t element_offset, AccessState state)
{
  auto const pointer = offset_cuda_pointer(state.data(), element_offset);
  auto mdspan = resolve_cuda_mdspan(device_span, pointer);
  return read_mdspan_lease<decltype(mdspan), AccessState>{std::move(state), std::move(mdspan)};
}

template <class DeviceSpan, class AccessState>
[[nodiscard]] auto make_cuda_write_mdspan_lease(DeviceSpan device_span, AccessState state)
{
  auto const offset = static_cast<std::size_t>(device_span.data_descriptor().element_offset());
  auto pointer = offset_cuda_pointer(state.data(), offset);
  auto mdspan = resolve_cuda_mdspan(device_span, pointer);
  return write_mdspan_lease<decltype(mdspan), AccessState>{std::move(state), std::move(mdspan)};
}

template <class StoragePolicy, class DeviceSpan, class AccessState, class BackendSelector>
[[nodiscard]] auto make_cuda_read_tensor_lease(DeviceSpan device_span, std::size_t element_offset, AccessState state,
                                               BackendSelector selector)
{
  auto const pointer = offset_cuda_pointer(state.data(), element_offset);
  auto mdspan = resolve_cuda_mdspan(device_span, pointer);

  using mdspan_type = decltype(mdspan);
  using selector_type = std::remove_cvref_t<BackendSelector>;
  return read_tensor_lease<mdspan_type, AccessState, selector_type, StoragePolicy>{std::move(state), std::move(mdspan),
                                                                                   std::move(selector)};
}

template <class StoragePolicy, class DeviceSpan, class ConstDeviceSpan, class AccessState, class BackendSelector>
[[nodiscard]] auto make_cuda_write_tensor_lease(DeviceSpan device_span, ConstDeviceSpan const_device_span,
                                                AccessState state, BackendSelector selector)
{
  CHECK(device_span.data_descriptor().element_offset() == const_device_span.data_descriptor().element_offset());

  auto const offset = static_cast<std::size_t>(device_span.data_descriptor().element_offset());
  auto pointer = offset_cuda_pointer(state.data(), offset);
  auto mdspan = resolve_cuda_mdspan(device_span, pointer);
  auto const_mdspan = resolve_cuda_mdspan(const_device_span, pointer);

  using mdspan_type = decltype(mdspan);
  using const_mdspan_type = decltype(const_mdspan);
  using selector_type = std::remove_cvref_t<BackendSelector>;
  return write_tensor_lease<mdspan_type, const_mdspan_type, AccessState, selector_type, StoragePolicy>{
      std::move(state), std::move(mdspan), std::move(const_mdspan), std::move(selector)};
}

} // namespace detail

/// \brief Host-wait for prior CUDA work and resolve a borrowed read descriptor.
/// \details The returned policy-free lease retains synchronized access but does
///          not own the buffer identified by `device_span`.
template <detail::CudaBufferDeviceMdspan DeviceSpan>
  requires std::is_const_v<typename std::remove_cvref_t<DeviceSpan>::element_type>
[[nodiscard]] auto acquire_cuda_read_access_sync(DeviceSpan const& device_span)
{
  detail::validate_cuda_descriptor_range(device_span);
  auto const element_offset = static_cast<std::size_t>(device_span.data_descriptor().element_offset());
  auto state = device_span.data_descriptor().buffer().blocking_read_access();
  return detail::make_cuda_read_mdspan_lease(device_span, element_offset, std::move(state));
}

/// \brief Host-wait for prior CUDA work and resolve a borrowed write descriptor.
/// \details The returned policy-free lease retains synchronized access but does
///          not own the buffer identified by `device_span`.
template <class DeviceSpan>
  requires detail::CudaBufferDeviceMdspan<DeviceSpan> && MutableDeviceMdspanLike<DeviceSpan>
[[nodiscard]] auto acquire_cuda_write_access_sync(DeviceSpan& device_span)
{
  detail::validate_cuda_descriptor_range(device_span);
  auto state = device_span.data_descriptor().buffer().blocking_write_access();
  return detail::make_cuda_write_mdspan_lease(device_span, std::move(state));
}

/// \brief Host-wait for prior CUDA work and acquire a CUDA-accessible read lease.
/// \details Synchronization does not copy or otherwise make the data host-accessible.
template <class Tensor>
  requires DeviceTensorView<Tensor> &&
           (!TensorView<Tensor>) && detail::CudaBufferDeviceMdspan<detail::normalized_const_device_mdspan_t<Tensor>>
[[nodiscard]] auto acquire_cuda_read_access_sync(Tensor& tensor)
{
  auto device_span = device_mdspan_of(std::as_const(tensor));
  detail::validate_cuda_descriptor_range(device_span);
  auto const element_offset = static_cast<std::size_t>(device_span.data_descriptor().element_offset());
  auto state = device_span.data_descriptor().buffer().blocking_read_access();
  return detail::make_cuda_read_tensor_lease<detail::tensor_storage_policy_t<Tensor>>(
      std::move(device_span), element_offset, std::move(state), tensor.backend_selector());
}

/// \brief Move an owning CUDA tensor into a synchronized CUDA read lease.
template <class Tensor>
  requires(!std::is_lvalue_reference_v<Tensor>) && DeviceTensorView<std::remove_cvref_t<Tensor>> &&
          (!TensorView<std::remove_cvref_t<Tensor>>) && OwningTensor<std::remove_cvref_t<Tensor>> &&
          detail::CudaBufferDeviceMdspan<detail::normalized_const_device_mdspan_t<std::remove_cvref_t<Tensor>>> &&
          requires(Tensor&& tensor) {
            { std::move(tensor).release_storage() } -> std::same_as<typename std::remove_cvref_t<Tensor>::storage_type>;
          }
[[nodiscard]] auto acquire_cuda_read_access_sync(Tensor&& tensor)
{
  auto device_span = device_mdspan_of(std::as_const(tensor));
  detail::validate_cuda_descriptor_range(device_span);
  auto const element_offset = static_cast<std::size_t>(device_span.data_descriptor().element_offset());
  auto selector = tensor.backend_selector();
  auto storage = std::move(tensor).release_storage();
  auto state = std::move(storage).into_blocking_read_access();
  return detail::make_cuda_read_tensor_lease<detail::tensor_storage_policy_t<Tensor>>(
      std::move(device_span), element_offset, std::move(state), std::move(selector));
}

/// \brief Host-wait for prior CUDA work and acquire a CUDA-accessible write lease.
/// \details Synchronization does not copy or otherwise make the data host-accessible.
template <class Tensor>
  requires MutableDeviceTensorView<Tensor> && (!MutableTensorView<Tensor>) &&
           detail::CudaBufferDeviceMdspan<detail::normalized_const_device_mdspan_t<Tensor>> &&
           detail::CudaBufferDeviceMdspan<detail::normalized_mutable_device_mdspan_t<Tensor>>
[[nodiscard]] auto acquire_cuda_write_access_sync(Tensor& tensor)
{
  auto device_span = device_mdspan_of(tensor);
  auto const_device_span = device_mdspan_of(std::as_const(tensor));
  detail::validate_cuda_descriptor_range(device_span);
  detail::validate_cuda_descriptor_range(const_device_span);
  auto state = device_span.data_descriptor().buffer().blocking_write_access();
  return detail::make_cuda_write_tensor_lease<detail::tensor_storage_policy_t<Tensor>>(
      std::move(device_span), std::move(const_device_span), std::move(state), tensor.backend_selector());
}

/// \brief Resolve borrowed CUDA read access synchronized with an operation stream.
template <detail::CudaBufferDeviceMdspan DeviceSpan>
  requires std::is_const_v<typename std::remove_cvref_t<DeviceSpan>::element_type>
[[nodiscard]] auto acquire_cuda_read_access(DeviceSpan const& device_span, cuda::Stream const& stream)
{
  detail::validate_cuda_descriptor_range(device_span);
  auto const element_offset = static_cast<std::size_t>(device_span.data_descriptor().element_offset());
  auto state = device_span.data_descriptor().buffer().read_synchronized_with(stream);
  return ready_access{detail::make_cuda_read_mdspan_lease(device_span, element_offset, std::move(state))};
}

/// \brief Resolve borrowed CUDA write access synchronized with an operation stream.
template <class DeviceSpan>
  requires detail::CudaBufferDeviceMdspan<DeviceSpan> && MutableDeviceMdspanLike<DeviceSpan>
[[nodiscard]] auto acquire_cuda_write_access(DeviceSpan& device_span, cuda::Stream const& stream)
{
  detail::validate_cuda_descriptor_range(device_span);
  auto state = device_span.data_descriptor().buffer().write_synchronized_with(stream);
  return ready_access{detail::make_cuda_write_mdspan_lease(device_span, std::move(state))};
}

/// \brief Acquire stream-ordered read access as an immediately-ready awaitable.
/// \details Resource admission remains operation-local. Once a stream is
///          available, `CudaBuffer` installs predecessor waits synchronously
///          and the returned awaitable is ready without suspending.
template <class Tensor>
  requires DeviceTensorView<Tensor> &&
           (!TensorView<Tensor>) && detail::CudaBufferDeviceMdspan<detail::normalized_const_device_mdspan_t<Tensor>>
[[nodiscard]] auto acquire_cuda_read_access(Tensor& tensor, cuda::Stream const& stream)
{
  auto device_span = device_mdspan_of(std::as_const(tensor));
  detail::validate_cuda_descriptor_range(device_span);
  auto const element_offset = static_cast<std::size_t>(device_span.data_descriptor().element_offset());
  auto state = device_span.data_descriptor().buffer().read_synchronized_with(stream);
  return ready_access{detail::make_cuda_read_tensor_lease<detail::tensor_storage_policy_t<Tensor>>(
      std::move(device_span), element_offset, std::move(state), tensor.backend_selector())};
}

/// \brief Move an owning CUDA tensor into a stream-ordered read lease.
template <class Tensor>
  requires(!std::is_lvalue_reference_v<Tensor>) && DeviceTensorView<std::remove_cvref_t<Tensor>> &&
          (!TensorView<std::remove_cvref_t<Tensor>>) && OwningTensor<std::remove_cvref_t<Tensor>> &&
          detail::CudaBufferDeviceMdspan<detail::normalized_const_device_mdspan_t<std::remove_cvref_t<Tensor>>> &&
          requires(Tensor&& tensor) {
            { std::move(tensor).release_storage() } -> std::same_as<typename std::remove_cvref_t<Tensor>::storage_type>;
          }
[[nodiscard]] auto acquire_cuda_read_access(Tensor&& tensor, cuda::Stream const& stream)
{
  auto device_span = device_mdspan_of(std::as_const(tensor));
  detail::validate_cuda_descriptor_range(device_span);
  auto const element_offset = static_cast<std::size_t>(device_span.data_descriptor().element_offset());
  auto selector = tensor.backend_selector();
  auto storage = std::move(tensor).release_storage();
  auto state = std::move(storage).into_read_synchronized_with(stream);
  return ready_access{detail::make_cuda_read_tensor_lease<detail::tensor_storage_policy_t<Tensor>>(
      std::move(device_span), element_offset, std::move(state), std::move(selector))};
}

/// \brief Acquire stream-ordered write access as an immediately-ready awaitable.
template <class Tensor>
  requires MutableDeviceTensorView<Tensor> && (!MutableTensorView<Tensor>) &&
           detail::CudaBufferDeviceMdspan<detail::normalized_const_device_mdspan_t<Tensor>> &&
           detail::CudaBufferDeviceMdspan<detail::normalized_mutable_device_mdspan_t<Tensor>>
[[nodiscard]] auto acquire_cuda_write_access(Tensor& tensor, cuda::Stream const& stream)
{
  auto device_span = device_mdspan_of(tensor);
  auto const_device_span = device_mdspan_of(std::as_const(tensor));
  detail::validate_cuda_descriptor_range(device_span);
  detail::validate_cuda_descriptor_range(const_device_span);
  auto state = device_span.data_descriptor().buffer().write_synchronized_with(stream);
  return ready_access{detail::make_cuda_write_tensor_lease<detail::tensor_storage_policy_t<Tensor>>(
      std::move(device_span), std::move(const_device_span), std::move(state), tensor.backend_selector())};
}

} // namespace uni20
