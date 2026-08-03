#pragma once

/**
 * \file copy.hpp
 * \ingroup linalg
 * \brief CUDA runtime backend for contiguous Tensor transfers.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/backends/cuda/elementwise_copy.hpp>
#include <uni20/linalg/backends/cuda/mdspec_traits.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/mdspan/iteration_plan.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/tensor/concepts.hpp>

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

enum class CopyExecution
{
  runtime_copy,
  elementwise_kernel
};

template <class Accessor> struct IsConjugatedCudaAccessor : std::false_type
{};

template <class Accessor>
struct IsConjugatedCudaAccessor<uni20::conjugated_accessor<Accessor>>
    : std::bool_constant<RawCudaAccessorTraits<Accessor>::is_raw>
{};

template <class Accessor> struct IsConjugatedHostAccessor : std::false_type
{};

template <class Accessor>
struct IsConjugatedHostAccessor<uni20::conjugated_accessor<Accessor>>
    : std::bool_constant<uni20::is_default_accessor_v<Accessor>>
{};

template <class Mdspan> inline constexpr bool is_cuda_mdspan = uni20::cuda::BufferMdspec<Mdspan>;

template <class Mdspan>
inline constexpr bool is_conjugated_cuda_mdspan =
    is_cuda_mdspan<Mdspan> && IsConjugatedCudaAccessor<typename std::remove_cvref_t<Mdspan>::accessor_type>::value;

template <class Mdspan>
inline constexpr bool is_supported_cuda_mdspan = is_raw_cuda_mdspec<Mdspan> || is_conjugated_cuda_mdspan<Mdspan>;

template <class Mdspan>
inline constexpr bool is_raw_host_mdspan = uni20::DefaultAccessorMdspanLike<std::remove_cvref_t<Mdspan>>;

template <class Mdspan>
inline constexpr bool is_conjugated_host_mdspan =
    uni20::MdspanLike<Mdspan> && IsConjugatedHostAccessor<typename std::remove_cvref_t<Mdspan>::accessor_type>::value;

template <class Mdspan>
inline constexpr bool is_supported_host_mdspan = is_raw_host_mdspan<Mdspan> || is_conjugated_host_mdspan<Mdspan>;

template <class OutputMdspan, class InputMdspan>
concept SupportedCopyMdspans =
    uni20::MutableMdspecLike<std::remove_cvref_t<OutputMdspan>> &&
    uni20::StridedMdspecLike<std::remove_cvref_t<OutputMdspan>> &&
    uni20::StridedMdspecLike<std::remove_cvref_t<InputMdspan>> &&
    (std::remove_cvref_t<OutputMdspan>::rank() == std::remove_cvref_t<InputMdspan>::rank()) &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<OutputMdspan>::element_type>,
                 std::remove_cv_t<typename std::remove_cvref_t<InputMdspan>::element_type>> &&
    ((is_raw_cuda_mdspec<OutputMdspan> &&
      (is_supported_cuda_mdspan<InputMdspan> || is_supported_host_mdspan<InputMdspan>)) ||
     (is_raw_host_mdspan<OutputMdspan> && is_supported_cuda_mdspan<InputMdspan>));

template <class Scalar> struct CopyPlan
{
    KernelAttempt attempt = KernelAttempt::unsupported_instance;
    CopyDirection direction = CopyDirection::device_to_device;
    CopyExecution execution = CopyExecution::runtime_copy;
    ElementwiseCopyTransform transform = ElementwiseCopyTransform::identity;
    uni20::cuda::CudaBuffer<Scalar>* output_buffer = nullptr;
    uni20::cuda::CudaBuffer<Scalar> const* input_buffer = nullptr;
    Scalar* output_host = nullptr;
    Scalar const* input_host = nullptr;
    std::size_t output_offset = 0;
    std::size_t input_offset = 0;
    std::size_t element_count = 0;
    std::size_t output_span_size = 0;
    std::size_t input_span_size = 0;
    LoweredElementwiseCopyPlan elementwise_plan;
    std::vector<Scalar> input_staging;
    bool conjugate_host_output = false;
    bool has_work = false;
};

template <class Mapping> [[nodiscard]] bool mapping_is_contiguous(Mapping const& mapping)
{
  return mapping.is_unique() && mapping.is_exhaustive();
}

template <class Mdspan> [[nodiscard]] bool active_strides_are_positive(Mdspan const& span)
{
  for (std::size_t axis = 0; axis < std::remove_cvref_t<Mdspan>::rank(); ++axis)
  {
    if (span.extent(axis) == 0) return true;
  }
  for (std::size_t axis = 0; axis < std::remove_cvref_t<Mdspan>::rank(); ++axis)
  {
    if (span.extent(axis) > 1 && span.stride(axis) <= 0) return false;
  }
  return true;
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

template <class OutputMdspan, class InputMdspan>
[[nodiscard]] bool try_make_elementwise_plan(OutputMdspan const& output, InputMdspan const& input,
                                             LoweredElementwiseCopyPlan& plan)
{
  auto host_plan = make_multi_iteration_plan(std::tuple{output.mapping(), input.mapping()});
  return try_lower_strided_elementwise_plan<elementwise_copy_maximum_rank>(host_plan, plan);
}

template <class Scalar>
void validate_cuda_range(uni20::cuda::CudaBuffer<Scalar> const& buffer, std::size_t offset, std::size_t count)
{
  CHECK(offset <= buffer.size() && count <= buffer.size() - offset, offset, count, buffer.size());
}

template <uni20::cuda::BufferMdspec Mdspan> [[nodiscard]] auto cuda_buffer_view(Mdspan& span)
{
  return span.data_descriptor();
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
  }

  if (!active_strides_are_positive(output) || !active_strides_are_positive(input))
  {
    plan.attempt = KernelAttempt::unsupported_layout;
    return plan;
  }

  auto const output_required_span = output.mapping().required_span_size();
  auto const input_required_span = input.mapping().required_span_size();
  if (std::cmp_less(output_required_span, 0) ||
      std::cmp_greater(output_required_span, std::numeric_limits<std::size_t>::max()) ||
      std::cmp_less(input_required_span, 0) ||
      std::cmp_greater(input_required_span, std::numeric_limits<std::size_t>::max()))
  {
    plan.attempt = KernelAttempt::unsupported_shape;
    return plan;
  }
  plan.output_span_size = static_cast<std::size_t>(output_required_span);
  plan.input_span_size = static_cast<std::size_t>(input_required_span);

  if constexpr (is_cuda_mdspan<OutputMdspan> && is_cuda_mdspan<InputMdspan>)
  {
    plan.direction = CopyDirection::device_to_device;
  }
  else if constexpr (is_cuda_mdspan<OutputMdspan>)
  {
    plan.direction = CopyDirection::host_to_device;
  }
  else
  {
    plan.direction = CopyDirection::device_to_host;
  }

  bool const runtime_copy_eligible = mapping_is_contiguous(output.mapping()) &&
                                     mapping_is_contiguous(input.mapping()) && physical_orders_match(output, input);
  bool needs_elementwise_kernel = false;
  if (!runtime_copy_eligible)
  {
    if constexpr (is_cuda_mdspan<OutputMdspan> && is_cuda_mdspan<InputMdspan>)
      needs_elementwise_kernel = true;
    else
    {
      plan.attempt = KernelAttempt::unsupported_layout;
      return plan;
    }
  }

  if constexpr (is_conjugated_cuda_mdspan<InputMdspan>)
  {
    if (plan.direction == CopyDirection::device_to_device)
      needs_elementwise_kernel = true;
    else
      plan.conjugate_host_output = true;
  }

  if (needs_elementwise_kernel)
  {
    if constexpr (!supports_elementwise_copy<scalar_type>)
    {
      plan.attempt = KernelAttempt::unsupported_transform;
      return plan;
    }
    if (!output.mapping().is_unique() || !try_make_elementwise_plan(output, input, plan.elementwise_plan))
    {
      plan.attempt = KernelAttempt::unsupported_layout;
      return plan;
    }
    plan.execution = CopyExecution::elementwise_kernel;
    plan.element_count = plan.elementwise_plan.element_count();
    if constexpr (is_conjugated_cuda_mdspan<InputMdspan>) plan.transform = ElementwiseCopyTransform::conjugate;
  }
  else
  {
    if (plan.output_span_size != plan.input_span_size)
    {
      plan.attempt = KernelAttempt::unsupported_shape;
      return plan;
    }
    plan.element_count = plan.output_span_size;
  }

  CHECK(plan.element_count <= std::numeric_limits<std::size_t>::max() / sizeof(scalar_type), plan.element_count,
        sizeof(scalar_type));

  if constexpr (is_cuda_mdspan<OutputMdspan>)
  {
    auto output_handle = cuda_buffer_view(output);
    plan.output_buffer = std::addressof(output_handle.buffer());
    plan.output_offset = output_handle.element_offset();
    validate_cuda_range(*plan.output_buffer, plan.output_offset, plan.output_span_size);
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
    validate_cuda_range(*plan.input_buffer, plan.input_offset, plan.input_span_size);
  }
  else
  {
    plan.input_host = input.data_handle();
    CHECK(plan.element_count == 0 || plan.input_host != nullptr);
  }

  if (plan.element_count == 0)
  {
    plan.attempt = KernelAttempt::success;
    return plan;
  }

  if constexpr (is_conjugated_host_mdspan<InputMdspan>)
  {
    CHECK(plan.direction == CopyDirection::host_to_device);
    plan.input_staging.resize(plan.element_count);
    for (std::size_t index = 0; index < plan.element_count; ++index)
      plan.input_staging[index] = uni20::conj(plan.input_host[index]);
    plan.input_host = plan.input_staging.data();
  }

  if constexpr (is_cuda_mdspan<OutputMdspan> && is_cuda_mdspan<InputMdspan>)
  {
    if (plan.output_buffer == plan.input_buffer)
    {
      if (plan.output_offset == plan.input_offset && plan.execution == CopyExecution::runtime_copy)
      {
        // Runtime-copy eligibility proves that both accessors observe the same
        // values, so identical storage and offset require no work.
        plan.attempt = KernelAttempt::success;
        return plan;
      }
      if (plan.output_offset == plan.input_offset)
      {
        plan.attempt = KernelAttempt::unsupported_instance;
        return plan;
      }
    }

    if (plan.execution == CopyExecution::elementwise_kernel &&
        plan.output_buffer->device().ordinal() != plan.input_buffer->device().ordinal())
    {
      plan.attempt = KernelAttempt::incompatible_devices;
      return plan;
    }
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

  int const output_device = plan.output_buffer->device().ordinal();
  int const input_device = plan.input_buffer->device().ordinal();
  uni20::cuda::ScopedDevice guard(output_device);

  auto enqueue = [&](Scalar* output_data, Scalar const* input_data) {
    if (plan.execution == CopyExecution::elementwise_kernel)
    {
      CHECK_EQUAL(output_device, input_device);
      if constexpr (supports_elementwise_copy<Scalar>)
      {
        plan.elementwise_plan.visit([&](auto const& elementwise_plan) {
          enqueue_elementwise_copy(output_data, input_data, elementwise_plan, plan.transform, stream.native_handle(),
                                   output_device);
        });
        return;
      }
      else
      {
        PANIC("unsupported scalar reached CUDA reference elementwise copy");
      }
    }

    std::size_t const bytes = plan.element_count * sizeof(Scalar);
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
  };

  if (plan.output_buffer == plan.input_buffer)
  {
    // C++ copy requires non-destructive overlap. One exclusive buffer access
    // retains and orders both pointers when disjoint views share an allocation.
    auto access = plan.output_buffer->write_synchronized_with(stream);
    enqueue(access.data() + plan.output_offset, access.data() + plan.input_offset);
    return;
  }

  auto output = plan.output_buffer->write_synchronized_with(stream);
  auto input = plan.input_buffer->read_synchronized_with(stream);
  enqueue(output.data() + plan.output_offset, input.data() + plan.input_offset);
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

/// \brief Report compile-time eligibility for CUDA mdspan copy lowering.
template <uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan> consteval auto copy_acceptance()
{
  if constexpr (SupportedCopyMdspans<OutputMdspan, InputMdspan>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Copy compatible mdspans after tensor-level CUDA backend selection.
template <class OutputMdspan, class InputMdspan>
  requires SupportedCopyMdspans<OutputMdspan, InputMdspan>
KernelAttempt copy(OutputMdspan&& output, InputMdspan&& input)
{
  auto preparation = prepare_copy(output, input);
  if (!kernel_attempt_succeeded(preparation.attempt)) return preparation.attempt;
  execute_blocking_copy(preparation);
  return KernelAttempt::success;
}
} // namespace detail::cuda_reference

/// \brief Report CUDA copy eligibility for normalized mdspecs.
template <uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, copy_op const&, OutputMdspan&, InputMdspan&)
{
  constexpr auto acceptance = detail::cuda_reference::copy_acceptance<OutputMdspan, InputMdspan>();
  if constexpr (acceptance == KernelTypeAcceptance::maybe)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Perform a CUDA copy from normalized mdspecs.
template <uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan>
KernelAttempt try_kernel(CudaReferenceBackend, copy_op const&, OutputMdspan& output, InputMdspan& input)
{
  return detail::cuda_reference::copy(output, input);
}

} // namespace uni20::linalg
