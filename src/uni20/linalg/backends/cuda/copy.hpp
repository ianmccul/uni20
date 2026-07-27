#pragma once

/**
 * \file copy.hpp
 * \ingroup linalg
 * \brief CUDA runtime backend for contiguous Tensor transfers.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/storage/cuda_storage.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::linalg
{
namespace detail::cuda_reference
{

enum class CopyDirection
{
  host_to_device,
  device_to_host,
  device_to_device
};

template <class Accessor> struct IsCudaAccessor : std::false_type
{};

template <class ElementType> struct IsCudaAccessor<uni20::cuda::CudaAccessor<ElementType>> : std::true_type
{};

template <class Accessor> struct IsRawCudaAccessor : std::false_type
{};

template <class ElementType> struct IsRawCudaAccessor<uni20::cuda::CudaAccessor<ElementType>> : std::true_type
{};

template <class ElementType> struct IsRawCudaAccessor<uni20::cuda::CudaPointerAccessor<ElementType>> : std::true_type
{};

template <class Accessor> struct IsConjugatedCudaAccessor : std::false_type
{};

template <class Accessor>
struct IsConjugatedCudaAccessor<uni20::conjugated_accessor<Accessor>> : IsRawCudaAccessor<Accessor>
{};

template <class Accessor> struct IsConjugatedHostAccessor : std::false_type
{};

template <class Accessor>
struct IsConjugatedHostAccessor<uni20::conjugated_accessor<Accessor>>
    : std::bool_constant<uni20::is_default_accessor_v<Accessor>>
{};

template <class Descriptor> struct IsCudaBufferView : std::false_type
{};

template <class ElementType> struct IsCudaBufferView<uni20::cuda::CudaBufferView<ElementType>> : std::true_type
{};

template <class Mdspan, class = void> struct HasCudaBufferDescriptor : std::false_type
{};

template <class Mdspan>
struct HasCudaBufferDescriptor<Mdspan, std::void_t<typename std::remove_cvref_t<Mdspan>::data_descriptor_type>>
    : IsCudaBufferView<typename std::remove_cvref_t<Mdspan>::data_descriptor_type>
{};

template <class Mdspan>
inline constexpr bool is_cuda_mdspan =
    (uni20::SpanLike<Mdspan> && IsCudaAccessor<typename std::remove_cvref_t<Mdspan>::accessor_type>::value) ||
    HasCudaBufferDescriptor<Mdspan>::value;

template <class Mdspan>
inline constexpr bool is_raw_cuda_mdspan =
    is_cuda_mdspan<Mdspan> && IsRawCudaAccessor<typename std::remove_cvref_t<Mdspan>::accessor_type>::value;

template <class Mdspan>
inline constexpr bool is_conjugated_cuda_mdspan =
    is_cuda_mdspan<Mdspan> && IsConjugatedCudaAccessor<typename std::remove_cvref_t<Mdspan>::accessor_type>::value;

template <class Mdspan>
inline constexpr bool is_supported_cuda_mdspan = is_raw_cuda_mdspan<Mdspan> || is_conjugated_cuda_mdspan<Mdspan>;

template <class Mdspan>
inline constexpr bool is_raw_host_mdspan = uni20::DefaultAccessorMdspan<std::remove_cvref_t<Mdspan>>;

template <class Mdspan>
inline constexpr bool is_conjugated_host_mdspan =
    uni20::SpanLike<Mdspan> && IsConjugatedHostAccessor<typename std::remove_cvref_t<Mdspan>::accessor_type>::value;

template <class Mdspan>
inline constexpr bool is_supported_host_mdspan = is_raw_host_mdspan<Mdspan> || is_conjugated_host_mdspan<Mdspan>;

template <class OutputMdspan, class InputMdspan>
concept SupportedCopyMdspans =
    uni20::MutableDeviceSpanLike<std::remove_cvref_t<OutputMdspan>> &&
    uni20::StridedDeviceSpanLike<std::remove_cvref_t<OutputMdspan>> &&
    uni20::StridedDeviceSpanLike<std::remove_cvref_t<InputMdspan>> &&
    (std::remove_cvref_t<OutputMdspan>::rank() == std::remove_cvref_t<InputMdspan>::rank()) &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<OutputMdspan>::element_type>,
                 std::remove_cv_t<typename std::remove_cvref_t<InputMdspan>::element_type>> &&
    ((is_raw_cuda_mdspan<OutputMdspan> &&
      (is_supported_cuda_mdspan<InputMdspan> || is_supported_host_mdspan<InputMdspan>)) ||
     (is_raw_host_mdspan<OutputMdspan> && is_supported_cuda_mdspan<InputMdspan>));

template <class Scalar> struct CopyPlan
{
    KernelAttempt attempt = KernelAttempt::unsupported_instance;
    CopyDirection direction = CopyDirection::device_to_device;
    uni20::cuda::CudaBuffer<Scalar>* output_buffer = nullptr;
    uni20::cuda::CudaBuffer<Scalar> const* input_buffer = nullptr;
    Scalar* output_host = nullptr;
    Scalar const* input_host = nullptr;
    std::size_t output_offset = 0;
    std::size_t input_offset = 0;
    std::size_t element_count = 0;
    std::vector<Scalar> input_staging;
    bool conjugate_host_output = false;
    bool has_work = false;
};

template <class Mapping> [[nodiscard]] bool mapping_is_contiguous(Mapping const& mapping)
{
  return mapping.is_unique() && mapping.is_exhaustive();
}

template <class OutputMdspan, class InputMdspan>
[[nodiscard]] bool physical_orders_match(OutputMdspan const& output, InputMdspan const& input)
{
  for (std::size_t axis = 0; axis < std::remove_cvref_t<OutputMdspan>::rank(); ++axis)
  {
    if (output.extent(axis) > 1 && output.stride(axis) != input.stride(axis)) return false;
  }
  return true;
}

template <class Scalar>
void validate_cuda_range(uni20::cuda::CudaBuffer<Scalar> const& buffer, std::size_t offset, std::size_t count)
{
  CHECK(offset <= buffer.size() && count <= buffer.size() - offset, offset, count, buffer.size());
}

template <class Mdspan> [[nodiscard]] auto cuda_buffer_view(Mdspan& span)
{
  if constexpr (requires { span.data_descriptor(); })
    return span.data_descriptor();
  else
    return span.data_handle();
}

template <class OutputMdspan, class InputMdspan>
  requires SupportedCopyMdspans<OutputMdspan, InputMdspan>
[[nodiscard]] auto prepare_copy(OutputMdspan& output, InputMdspan& input)
{
  using output_type = std::remove_cvref_t<OutputMdspan>;
  using scalar_type = std::remove_cv_t<typename output_type::element_type>;
  CopyPlan<scalar_type> plan;

  for (std::size_t axis = 0; axis < output_type::rank(); ++axis)
  {
    if (output.extent(axis) != input.extent(axis))
    {
      plan.attempt = KernelAttempt::unsupported_shape;
      return plan;
    }
    if (output.extent(axis) > 1 && (output.stride(axis) <= 0 || input.stride(axis) <= 0))
    {
      plan.attempt = KernelAttempt::unsupported_layout;
      return plan;
    }
  }

  if (!mapping_is_contiguous(output.mapping()) || !mapping_is_contiguous(input.mapping()) ||
      !physical_orders_match(output, input))
  {
    plan.attempt = KernelAttempt::unsupported_layout;
    return plan;
  }

  auto const required_span = output.mapping().required_span_size();
  if (required_span != input.mapping().required_span_size() || std::cmp_less(required_span, 0) ||
      std::cmp_greater(required_span, std::numeric_limits<std::size_t>::max()))
  {
    plan.attempt = KernelAttempt::unsupported_shape;
    return plan;
  }

  plan.element_count = static_cast<std::size_t>(required_span);
  CHECK(plan.element_count <= std::numeric_limits<std::size_t>::max() / sizeof(scalar_type), plan.element_count,
        sizeof(scalar_type));

  if constexpr (is_cuda_mdspan<OutputMdspan>)
  {
    auto output_handle = cuda_buffer_view(output);
    plan.output_buffer = std::addressof(output_handle.buffer());
    plan.output_offset = output_handle.element_offset();
    validate_cuda_range(*plan.output_buffer, plan.output_offset, plan.element_count);
  }
  else
  {
    plan.output_host = output.data_handle();
    CHECK(plan.element_count == 0 || plan.output_host != nullptr);
  }

  if constexpr (is_cuda_mdspan<InputMdspan>)
  {
    auto input_handle = cuda_buffer_view(input);
    plan.input_buffer = std::addressof(input_handle.buffer());
    plan.input_offset = input_handle.element_offset();
    validate_cuda_range(*plan.input_buffer, plan.input_offset, plan.element_count);
  }
  else
  {
    plan.input_host = input.data_handle();
    CHECK(plan.element_count == 0 || plan.input_host != nullptr);
  }

  if constexpr (is_cuda_mdspan<OutputMdspan> && is_cuda_mdspan<InputMdspan>)
  {
    plan.direction = CopyDirection::device_to_device;
    if (plan.output_buffer == plan.input_buffer)
    {
      if (plan.output_offset == plan.input_offset)
      {
        plan.attempt = KernelAttempt::success;
        return plan;
      }
      plan.attempt = KernelAttempt::unsupported_instance;
      return plan;
    }
  }
  else if constexpr (is_cuda_mdspan<OutputMdspan>)
  {
    plan.direction = CopyDirection::host_to_device;
  }
  else
  {
    plan.direction = CopyDirection::device_to_host;
  }

  if constexpr (is_conjugated_cuda_mdspan<InputMdspan>)
  {
    if (plan.direction == CopyDirection::device_to_device)
    {
      plan.attempt = KernelAttempt::unsupported_transform;
      return plan;
    }
    plan.conjugate_host_output = true;
  }
  else if constexpr (is_conjugated_host_mdspan<InputMdspan>)
  {
    CHECK(plan.direction == CopyDirection::host_to_device);
    plan.input_staging.resize(plan.element_count);
    for (std::size_t index = 0; index < plan.element_count; ++index)
      plan.input_staging[index] = uni20::conj(plan.input_host[index]);
    plan.input_host = plan.input_staging.data();
  }

  plan.attempt = KernelAttempt::success;
  plan.has_work = plan.element_count != 0;
  return plan;
}

template <class Scalar> void enqueue_device_copy(CopyPlan<Scalar> const& plan, uni20::cuda::Stream const& stream)
{
  CHECK(plan.direction == CopyDirection::device_to_device && plan.output_buffer != nullptr &&
        plan.input_buffer != nullptr);
  CHECK_EQUAL(stream.device(), plan.output_buffer->device().ordinal());

  auto output = plan.output_buffer->write_synchronized_with(stream);
  auto input = plan.input_buffer->read_synchronized_with(stream);
  auto* output_data = output.data() + plan.output_offset;
  auto const* input_data = input.data() + plan.input_offset;
  std::size_t const bytes = plan.element_count * sizeof(Scalar);
  int const output_device = plan.output_buffer->device().ordinal();
  int const input_device = plan.input_buffer->device().ordinal();
  uni20::cuda::ScopedDevice guard(output_device);

  if (output_device == input_device)
  {
    uni20::cuda::check(
        cudaMemcpyAsync(output_data, input_data, bytes, cudaMemcpyDeviceToDevice, stream.native_handle()),
        "cudaMemcpyAsync device-to-device", output_device);
  }
  else
  {
    uni20::cuda::check(
        cudaMemcpyPeerAsync(output_data, output_device, input_data, input_device, bytes, stream.native_handle()),
        "cudaMemcpyPeerAsync", output_device);
  }
}

template <class Scalar> void execute_blocking_copy(CopyPlan<Scalar> const& plan)
{
  CHECK(kernel_attempt_succeeded(plan.attempt));
  if (!plan.has_work) return;

  std::size_t const bytes = plan.element_count * sizeof(Scalar);
  if (plan.direction == CopyDirection::device_to_device)
  {
    auto stream = plan.output_buffer->resources().streams().acquire();
    enqueue_device_copy(plan, stream);
    return;
  }

  if (plan.direction == CopyDirection::host_to_device)
  {
    auto output = plan.output_buffer->blocking_write_access();
    int const device = plan.output_buffer->device().ordinal();
    uni20::cuda::ScopedDevice guard(device);
    uni20::cuda::check(cudaMemcpy(output.data() + plan.output_offset, plan.input_host, bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy host-to-device", device);
    // Pageable H2D cudaMemcpy finishes consuming the host source after staging,
    // but its default-stream DMA may remain outstanding. Publish that stream
    // tail so later non-blocking streams observe the destination dependency.
    try
    {
      output.release_with_completion(uni20::cuda::Completion::record(device, nullptr));
    }
    catch (...)
    {
      uni20::cuda::detail::synchronize_after_failed_publication(nullptr, device,
                                                                "publish pageable host-to-device completion");
      throw;
    }
    return;
  }

  {
    auto input = plan.input_buffer->blocking_read_access();
    int const device = plan.input_buffer->device().ordinal();
    uni20::cuda::ScopedDevice guard(device);
    uni20::cuda::check(cudaMemcpy(plan.output_host, input.data() + plan.input_offset, bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy device-to-host", device);
  }
  if (plan.conjugate_host_output)
  {
    for (std::size_t index = 0; index < plan.element_count; ++index)
      plan.output_host[index] = uni20::conj(plan.output_host[index]);
  }
}

} // namespace detail::cuda_reference

/// \brief Report compile-time eligibility for CUDA Tensor transfer.
template <class OutputMdspan, class InputMdspan>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, copy_op const&, OutputMdspan&, InputMdspan&)
{
  if constexpr (detail::cuda_reference::SupportedCopyMdspans<OutputMdspan, InputMdspan>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Copy compatible contiguous host or CUDA mdspans while preserving supported accessor semantics.
template <class OutputMdspan, class InputMdspan>
KernelAttempt try_kernel(CudaReferenceBackend, copy_op const&, OutputMdspan&& output, InputMdspan&& input)
{
  auto preparation = detail::cuda_reference::prepare_copy(output, input);
  if (!kernel_attempt_succeeded(preparation.attempt)) return preparation.attempt;
  detail::cuda_reference::execute_blocking_copy(preparation);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
