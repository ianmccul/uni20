/**
 * \file cuda_access.hpp
 * \ingroup tensor
 * \brief CUDA data-handle acquisition for descriptor-backed tensor views.
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

template <class Tensor>
concept CudaDeferredTensorView =
    DeviceTensorView<Tensor> && std::same_as<tensor_storage_policy_t<Tensor>, CudaStorage> &&
    requires(std::remove_reference_t<Tensor>& tensor, std::remove_reference_t<Tensor> const& const_tensor) {
      { tensor.device_mdspan() } -> DeviceSpanLike;
      { const_tensor.device_mdspan() } -> DeviceSpanLike;
    } && (!SpanLike<decltype(std::declval<std::remove_reference_t<Tensor> const&>().device_mdspan())>);

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

template <class Tensor, class AccessState>
[[nodiscard]] auto make_cuda_read_tensor_lease(Tensor const& tensor, AccessState state)
{
  auto device_span = tensor.device_mdspan();
  validate_cuda_descriptor_range(device_span);
  auto const pointer =
      offset_cuda_pointer(state.data(), static_cast<std::size_t>(device_span.data_descriptor().element_offset()));
  auto mdspan = resolve_cuda_mdspan(device_span, pointer);

  using tensor_type = std::remove_reference_t<Tensor>;
  using mdspan_type = decltype(mdspan);
  using selector_type = std::remove_cvref_t<decltype(tensor.backend_selector())>;
  using storage_policy = tensor_storage_policy_t<tensor_type>;
  return read_tensor_lease<mdspan_type, AccessState, selector_type, storage_policy>{std::move(state), std::move(mdspan),
                                                                                    tensor.backend_selector()};
}

template <class Tensor, class AccessState>
[[nodiscard]] auto make_cuda_write_tensor_lease(Tensor& tensor, AccessState state)
{
  auto device_span = tensor.device_mdspan();
  auto const_device_span = std::as_const(tensor).device_mdspan();
  validate_cuda_descriptor_range(device_span);
  validate_cuda_descriptor_range(const_device_span);
  CHECK(device_span.data_descriptor().element_offset() == const_device_span.data_descriptor().element_offset());

  auto const offset = static_cast<std::size_t>(device_span.data_descriptor().element_offset());
  auto pointer = offset_cuda_pointer(state.data(), offset);
  auto mdspan = resolve_cuda_mdspan(device_span, pointer);
  auto const_mdspan = resolve_cuda_mdspan(const_device_span, pointer);

  using tensor_type = std::remove_reference_t<Tensor>;
  using mdspan_type = decltype(mdspan);
  using const_mdspan_type = decltype(const_mdspan);
  using selector_type = std::remove_cvref_t<decltype(tensor.backend_selector())>;
  using storage_policy = tensor_storage_policy_t<tensor_type>;
  return write_tensor_lease<mdspan_type, const_mdspan_type, AccessState, selector_type, storage_policy>{
      std::move(state), std::move(mdspan), std::move(const_mdspan), tensor.backend_selector()};
}

} // namespace detail

/// \brief Host-wait for prior CUDA work and acquire a read-only TensorView lease.
template <detail::CudaDeferredTensorView Tensor> [[nodiscard]] auto blocking_read_access(Tensor const& tensor)
{
  auto descriptor = tensor.device_mdspan().data_descriptor();
  return detail::make_cuda_read_tensor_lease(tensor, descriptor.buffer().blocking_read_access());
}

/// \brief Host-wait for prior CUDA work and acquire a mutable TensorView lease.
template <class Tensor>
  requires(detail::CudaDeferredTensorView<Tensor> && MutableDeviceTensorView<Tensor>)
[[nodiscard]] auto blocking_write_access(Tensor& tensor)
{
  auto descriptor = tensor.device_mdspan().data_descriptor();
  return detail::make_cuda_write_tensor_lease(tensor, descriptor.buffer().blocking_write_access());
}

/// \brief Acquire stream-ordered read access as an immediately-ready awaitable.
/// \details Resource admission remains operation-local. Once a stream is
///          available, `CudaBuffer` installs predecessor waits synchronously
///          and the returned awaitable is ready without suspending.
template <detail::CudaDeferredTensorView Tensor>
[[nodiscard]] auto read_access(Tensor const& tensor, cuda::Stream const& stream)
{
  auto descriptor = tensor.device_mdspan().data_descriptor();
  return ready_tensor_access{
      detail::make_cuda_read_tensor_lease(tensor, descriptor.buffer().read_synchronized_with(stream))};
}

/// \brief Acquire stream-ordered write access as an immediately-ready awaitable.
template <class Tensor>
  requires(detail::CudaDeferredTensorView<Tensor> && MutableDeviceTensorView<Tensor>)
[[nodiscard]] auto write_access(Tensor& tensor, cuda::Stream const& stream)
{
  auto descriptor = tensor.device_mdspan().data_descriptor();
  return ready_tensor_access{
      detail::make_cuda_write_tensor_lease(tensor, descriptor.buffer().write_synchronized_with(stream))};
}

} // namespace uni20
