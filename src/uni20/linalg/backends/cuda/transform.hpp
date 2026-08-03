#pragma once

/**
 * \file transform.hpp
 * \ingroup linalg
 * \brief CUDA reference backend for registered elementwise transforms.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/linalg/backends/cuda/elementwise_negate.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/iteration_plan.hpp>
#include <uni20/storage/cuda_storage.hpp>

#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail::cuda_reference
{

template <class Accessor> struct IsNegateOutputAccessor : std::false_type
{};

template <class ElementType>
struct IsNegateOutputAccessor<uni20::cuda::CudaPointerAccessor<ElementType>>
    : std::bool_constant<!std::is_const_v<ElementType>>
{};

template <class Accessor> struct IsNegateInputAccessor : std::false_type
{};

template <class ElementType>
struct IsNegateInputAccessor<uni20::cuda::CudaPointerAccessor<ElementType>>
    : std::bool_constant<std::is_const_v<ElementType>>
{};

template <class Mdspec>
inline constexpr bool is_negate_output_mdspec =
    uni20::cuda::BufferMdspec<Mdspec> &&
    IsNegateOutputAccessor<typename std::remove_cvref_t<Mdspec>::accessor_type>::value;

template <class Mdspec>
inline constexpr bool is_negate_input_mdspec =
    uni20::cuda::BufferMdspec<Mdspec> &&
    IsNegateInputAccessor<typename std::remove_cvref_t<Mdspec>::accessor_type>::value;

template <class Mdspec> using negate_scalar_t = std::remove_cv_t<typename std::remove_cvref_t<Mdspec>::element_type>;

template <class OutputMdspec, class InputMdspec>
concept SupportedNegateMdspecs =
    uni20::MutableMdspecLike<OutputMdspec> && uni20::MdspecLike<InputMdspec> && is_negate_output_mdspec<OutputMdspec> &&
    is_negate_input_mdspec<InputMdspec> &&
    (std::remove_cvref_t<OutputMdspec>::rank() == std::remove_cvref_t<InputMdspec>::rank()) &&
    std::same_as<negate_scalar_t<OutputMdspec>, negate_scalar_t<InputMdspec>> &&
    (std::remove_cvref_t<OutputMdspec>::rank() == 0 ||
     (uni20::StridedMdspecLike<OutputMdspec> && uni20::StridedMdspecLike<InputMdspec>)) &&
    supports_elementwise_negate<negate_scalar_t<OutputMdspec>>;

template <class Scalar> struct NegatePlan
{
    KernelAttempt attempt = KernelAttempt::unsupported_instance;
    uni20::cuda::CudaBuffer<Scalar>* output_buffer = nullptr;
    uni20::cuda::CudaBuffer<Scalar> const* input_buffer = nullptr;
    std::size_t output_offset = 0;
    std::size_t input_offset = 0;
    std::size_t output_span_size = 0;
    std::size_t input_span_size = 0;
    LoweredElementwiseNegatePlan elementwise_plan;
    bool has_work = false;
};

template <class OutputMdspec, class InputMdspec>
  requires((std::remove_cvref_t<OutputMdspec>::rank() == 0 ||
            (uni20::StridedMdspecLike<OutputMdspec> && uni20::StridedMdspecLike<InputMdspec>)) &&
           (std::remove_cvref_t<OutputMdspec>::rank() == std::remove_cvref_t<InputMdspec>::rank()))
[[nodiscard]] bool try_make_elementwise_negate_plan(OutputMdspec const& output, InputMdspec const& input,
                                                    LoweredElementwiseNegatePlan& plan)
{
  auto host_plan = make_multi_iteration_plan(std::tuple{output.mapping(), input.mapping()});
  return try_lower_strided_elementwise_plan<elementwise_negate_maximum_rank>(host_plan, plan);
}

template <class Mdspec> [[nodiscard]] auto required_span_size(Mdspec const& mdspec) -> std::optional<std::size_t>
{
  auto const required = mdspec.mapping().required_span_size();
  if (std::cmp_less(required, 0) || std::cmp_greater(required, std::numeric_limits<std::size_t>::max()))
    return std::nullopt;
  return static_cast<std::size_t>(required);
}

template <class OutputMdspec, class InputMdspec>
  requires SupportedNegateMdspecs<OutputMdspec, InputMdspec>
[[nodiscard]] auto prepare_negate(OutputMdspec& output, InputMdspec& input)
{
  using output_type = std::remove_cvref_t<OutputMdspec>;
  using scalar_type = negate_scalar_t<OutputMdspec>;
  NegatePlan<scalar_type> plan;

  for (std::size_t axis = 0; axis < output_type::rank(); ++axis)
  {
    if (output.extent(axis) != input.extent(axis))
    {
      plan.attempt = KernelAttempt::unsupported_shape;
      return plan;
    }
  }

  if (!output.mapping().is_unique() || !try_make_elementwise_negate_plan(output, input, plan.elementwise_plan))
  {
    plan.attempt = KernelAttempt::unsupported_layout;
    return plan;
  }

  plan.has_work = plan.elementwise_plan.element_count() != 0;
  if (!plan.has_work)
  {
    plan.attempt = KernelAttempt::success;
    return plan;
  }

  auto const output_span_size = required_span_size(output);
  auto const input_span_size = required_span_size(input);
  if (!output_span_size || !input_span_size)
  {
    plan.attempt = KernelAttempt::unsupported_shape;
    return plan;
  }
  plan.output_span_size = *output_span_size;
  plan.input_span_size = *input_span_size;

  auto output_descriptor = output.data_descriptor();
  auto input_descriptor = input.data_descriptor();
  plan.output_buffer = std::addressof(output_descriptor.buffer());
  plan.input_buffer = std::addressof(input_descriptor.buffer());
  plan.output_offset = output_descriptor.element_offset();
  plan.input_offset = input_descriptor.element_offset();

  CHECK(plan.output_offset <= plan.output_buffer->size() &&
            plan.output_span_size <= plan.output_buffer->size() - plan.output_offset,
        plan.output_offset, plan.output_span_size, plan.output_buffer->size());
  CHECK(plan.input_offset <= plan.input_buffer->size() &&
            plan.input_span_size <= plan.input_buffer->size() - plan.input_offset,
        plan.input_offset, plan.input_span_size, plan.input_buffer->size());

  if (plan.output_buffer->device() != plan.input_buffer->device())
  {
    plan.attempt = KernelAttempt::incompatible_devices;
    return plan;
  }

  if (plan.output_buffer == plan.input_buffer && plan.output_offset == plan.input_offset)
  {
    plan.attempt = KernelAttempt::unsupported_instance;
    return plan;
  }

  plan.attempt = KernelAttempt::success;
  return plan;
}

template <class Scalar> void execute_negate(NegatePlan<Scalar> const& plan)
{
  CHECK(kernel_attempt_succeeded(plan.attempt));
  if (!plan.has_work) return;
  CHECK(plan.output_buffer != nullptr && plan.input_buffer != nullptr);

  auto stream = plan.output_buffer->resources().streams().acquire();
  int const device = plan.output_buffer->device().ordinal();
  CHECK_EQUAL(stream.device(), device);
  uni20::cuda::ScopedDevice guard(device);

  auto launch = [&](Scalar* output, Scalar const* input) {
    if constexpr (supports_elementwise_negate<Scalar>)
    {
      if (plan.elementwise_plan.index_kind == ElementwiseIndexKind::index_32)
        enqueue_elementwise_negate(output, input, plan.elementwise_plan.plan_32, stream.native_handle(), device);
      else
        enqueue_elementwise_negate(output, input, plan.elementwise_plan.plan_64, stream.native_handle(), device);
    }
    else
    {
      PANIC("unsupported scalar reached CUDA reference elementwise negation");
    }
  };

  if (plan.output_buffer == plan.input_buffer)
  {
    // The public overwrite contract requires non-destructive overlap. One
    // exclusive access retains disjoint views into the same allocation.
    auto access = plan.output_buffer->write_synchronized_with(stream);
    launch(access.data() + plan.output_offset, access.data() + plan.input_offset);
    return;
  }

  auto output = plan.output_buffer->write_synchronized_with(stream);
  auto input = plan.input_buffer->read_synchronized_with(stream);
  launch(output.data() + plan.output_offset, input.data() + plan.input_offset);
}

} // namespace detail::cuda_reference

/// \brief Report CUDA eligibility for the registered unary negate transform.
template <uni20::MutableMdspecLike OutputMdspec, uni20::MdspecLike InputMdspec>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, transform_op<negate> const&, OutputMdspec&,
                                    InputMdspec&)
{
  if constexpr (detail::cuda_reference::SupportedNegateMdspecs<OutputMdspec, InputMdspec>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Apply the registered unary negate transform to CUDA mdspec operands.
template <class OutputMdspec, class InputMdspec>
  requires detail::cuda_reference::SupportedNegateMdspecs<OutputMdspec, InputMdspec>
KernelAttempt try_kernel(CudaReferenceBackend, transform_op<negate> const&, OutputMdspec& output, InputMdspec& input)
{
  auto preparation = detail::cuda_reference::prepare_negate(output, input);
  if (!kernel_attempt_succeeded(preparation.attempt)) return preparation.attempt;
  detail::cuda_reference::execute_negate(preparation);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
