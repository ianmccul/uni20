#pragma once

/**
 * \file copy.hpp
 * \ingroup linalg
 * \brief CUDA runtime backend for contiguous Tensor transfers.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/storage/cuda_storage.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

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

template <class Mdspan>
inline constexpr bool is_cuda_mdspan = IsCudaAccessor<typename std::remove_cvref_t<Mdspan>::accessor_type>::value;

template <class Mdspan>
inline constexpr bool is_host_mdspan = uni20::DefaultAccessorMdspan<std::remove_cvref_t<Mdspan>>;

template <class OutputMdspan, class InputMdspan>
concept SupportedCopyMdspans =
    uni20::MutableStridedMdspan<std::remove_cvref_t<OutputMdspan>> &&
    uni20::StridedMdspan<std::remove_cvref_t<InputMdspan>> &&
    (std::remove_cvref_t<OutputMdspan>::rank() == std::remove_cvref_t<InputMdspan>::rank()) &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<OutputMdspan>::element_type>,
                 std::remove_cv_t<typename std::remove_cvref_t<InputMdspan>::element_type>> &&
    ((is_cuda_mdspan<OutputMdspan> && (is_cuda_mdspan<InputMdspan> || is_host_mdspan<InputMdspan>)) ||
     (is_host_mdspan<OutputMdspan> && is_cuda_mdspan<InputMdspan>));

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
    auto output_handle = output.data_handle();
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
    auto input_handle = input.data_handle();
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
    return;
  }

  auto input = plan.input_buffer->blocking_read_access();
  int const device = plan.input_buffer->device().ordinal();
  uni20::cuda::ScopedDevice guard(device);
  uni20::cuda::check(cudaMemcpy(plan.output_host, input.data() + plan.input_offset, bytes, cudaMemcpyDeviceToHost),
                     "cudaMemcpy device-to-host", device);
}

} // namespace detail::cuda_reference

/// \brief Report compile-time eligibility for raw CUDA Tensor transfer.
template <class OutputMdspan, class InputMdspan>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, copy_op const&, OutputMdspan&, InputMdspan&)
{
  if constexpr (detail::cuda_reference::SupportedCopyMdspans<OutputMdspan, InputMdspan>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Copy compatible contiguous host or CUDA mdspans through the CUDA runtime.
template <class OutputMdspan, class InputMdspan>
KernelAttempt try_kernel(CudaReferenceBackend, copy_op const&, OutputMdspan&& output, InputMdspan&& input)
{
  auto preparation = detail::cuda_reference::prepare_copy(output, input);
  if (!kernel_attempt_succeeded(preparation.attempt)) return preparation.attempt;
  detail::cuda_reference::execute_blocking_copy(preparation);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
