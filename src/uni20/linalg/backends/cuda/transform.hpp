#pragma once

/**
 * \file transform.hpp
 * \ingroup linalg
 * \brief CUDA reference backend for registered elementwise transforms.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/linalg/backends/cuda/elementwise_add.hpp>
#include <uni20/linalg/backends/cuda/elementwise_negate.hpp>
#include <uni20/linalg/backends/cuda/elementwise_scale.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/iteration_plan.hpp>
#include <uni20/storage/cuda_storage.hpp>

#include <array>
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

template <class Accessor> struct IsRawCudaOutputAccessor : std::false_type
{};

template <class ElementType>
struct IsRawCudaOutputAccessor<uni20::cuda::CudaPointerAccessor<ElementType>>
    : std::bool_constant<!std::is_const_v<ElementType>>
{};

template <class Accessor> struct IsRawCudaInputAccessor : std::false_type
{};

template <class ElementType>
struct IsRawCudaInputAccessor<uni20::cuda::CudaPointerAccessor<ElementType>>
    : std::bool_constant<std::is_const_v<ElementType>>
{};

template <class Mdspec>
inline constexpr bool is_raw_cuda_output_mdspec =
    uni20::cuda::BufferMdspec<Mdspec> &&
    IsRawCudaOutputAccessor<typename std::remove_cvref_t<Mdspec>::accessor_type>::value;

template <class Mdspec>
inline constexpr bool is_raw_cuda_input_mdspec =
    uni20::cuda::BufferMdspec<Mdspec> &&
    IsRawCudaInputAccessor<typename std::remove_cvref_t<Mdspec>::accessor_type>::value;

template <class Mdspec> using transform_scalar_t = std::remove_cv_t<typename std::remove_cvref_t<Mdspec>::element_type>;

template <class OutputMdspec, class... InputMdspecs>
concept RawCudaOverwriteMdspecs =
    (sizeof...(InputMdspecs) > 0) && uni20::MutableMdspecLike<OutputMdspec> &&
    (uni20::MdspecLike<InputMdspecs> && ...) && is_raw_cuda_output_mdspec<OutputMdspec> &&
    (is_raw_cuda_input_mdspec<InputMdspecs> && ...) &&
    ((std::remove_cvref_t<OutputMdspec>::rank() == std::remove_cvref_t<InputMdspecs>::rank()) && ...) &&
    (std::same_as<transform_scalar_t<OutputMdspec>, transform_scalar_t<InputMdspecs>> && ...) &&
    (std::remove_cvref_t<OutputMdspec>::rank() == 0 ||
     (uni20::StridedMdspecLike<OutputMdspec> && (uni20::StridedMdspecLike<InputMdspecs> && ...)));

template <class OutputMdspec, class InputMdspec>
concept SupportedNegateMdspecs =
    RawCudaOverwriteMdspecs<OutputMdspec, InputMdspec> && supports_elementwise_negate<transform_scalar_t<OutputMdspec>>;

template <class OutputMdspec, class InputMdspec, class Factor>
concept SupportedScaleMdspecs =
    RawCudaOverwriteMdspecs<OutputMdspec, InputMdspec> &&
    supports_elementwise_scale<transform_scalar_t<OutputMdspec>, std::remove_cvref_t<Factor>>;

template <class OutputMdspec, class LhsMdspec, class RhsMdspec>
concept SupportedAddMdspecs = RawCudaOverwriteMdspecs<OutputMdspec, LhsMdspec, RhsMdspec> &&
                              supports_elementwise_add<transform_scalar_t<OutputMdspec>>;

template <class Scalar, std::size_t InputCount, std::size_t MaximumRank> struct OverwriteTransformPlan
{
    using scalar_type = Scalar;
    static constexpr std::size_t input_count = InputCount;
    static constexpr std::size_t maximum_rank = MaximumRank;

    KernelAttempt attempt = KernelAttempt::unsupported_instance;
    uni20::cuda::CudaBuffer<Scalar>* output_buffer = nullptr;
    std::array<uni20::cuda::CudaBuffer<Scalar> const*, InputCount> input_buffers{};
    std::size_t output_offset = 0;
    std::array<std::size_t, InputCount> input_offsets{};
    std::size_t output_span_size = 0;
    std::array<std::size_t, InputCount> input_span_sizes{};
    LoweredStridedElementwisePlan<InputCount + 1, MaximumRank> elementwise_plan;
    bool has_work = false;
};

template <class Scalar> using NegatePlan = OverwriteTransformPlan<Scalar, 1, elementwise_negate_maximum_rank>;

template <class Scalar, class Factor>
struct ScalePlan : OverwriteTransformPlan<Scalar, 1, elementwise_scale_maximum_rank>
{
    Factor factor{};
};

template <class Scalar> using AddPlan = OverwriteTransformPlan<Scalar, 2, elementwise_add_maximum_rank>;

template <std::size_t MaximumRank, class OutputMdspec, class... InputMdspecs>
  requires uni20::MutableMdspecLike<OutputMdspec> && (uni20::MdspecLike<InputMdspecs> && ...) &&
           ((std::remove_cvref_t<OutputMdspec>::rank() == std::remove_cvref_t<InputMdspecs>::rank()) && ...) &&
           (std::remove_cvref_t<OutputMdspec>::rank() == 0 ||
            (uni20::StridedMdspecLike<OutputMdspec> && (uni20::StridedMdspecLike<InputMdspecs> && ...)))
[[nodiscard]] bool
try_make_elementwise_transform_plan(OutputMdspec const& output,
                                    LoweredStridedElementwisePlan<sizeof...(InputMdspecs) + 1, MaximumRank>& plan,
                                    InputMdspecs const&... inputs)
{
  auto host_plan = make_multi_iteration_plan(std::tuple{output.mapping(), inputs.mapping()...});
  return try_lower_strided_elementwise_plan<MaximumRank>(host_plan, plan);
}

template <class OutputMdspec, class InputMdspec>
  requires uni20::MutableMdspecLike<OutputMdspec> && uni20::MdspecLike<InputMdspec> &&
           (std::remove_cvref_t<OutputMdspec>::rank() == std::remove_cvref_t<InputMdspec>::rank()) &&
           (std::remove_cvref_t<OutputMdspec>::rank() == 0 ||
            (uni20::StridedMdspecLike<OutputMdspec> && uni20::StridedMdspecLike<InputMdspec>))
[[nodiscard]] bool try_make_elementwise_negate_plan(OutputMdspec const& output, InputMdspec const& input,
                                                    LoweredElementwiseNegatePlan& plan)
{
  return try_make_elementwise_transform_plan<elementwise_negate_maximum_rank>(output, plan, input);
}

template <class Mdspec> [[nodiscard]] auto required_span_size(Mdspec const& mdspec) -> std::optional<std::size_t>
{
  auto const required = mdspec.mapping().required_span_size();
  if (std::cmp_less(required, 0) || std::cmp_greater(required, std::numeric_limits<std::size_t>::max()))
    return std::nullopt;
  return static_cast<std::size_t>(required);
}

template <class Plan, class OutputMdspec, class... InputMdspecs>
  requires RawCudaOverwriteMdspecs<OutputMdspec, InputMdspecs...> && (Plan::input_count == sizeof...(InputMdspecs))
[[nodiscard]] auto prepare_overwrite_transform(Plan plan, OutputMdspec& output, InputMdspecs&... inputs)
{
  using output_type = std::remove_cvref_t<OutputMdspec>;

  bool extents_match = true;
  auto check_extents = [&](auto const& input) {
    for (std::size_t axis = 0; axis < output_type::rank(); ++axis)
    {
      if (output.extent(axis) != input.extent(axis)) extents_match = false;
    }
  };
  (check_extents(inputs), ...);
  if (!extents_match)
  {
    plan.attempt = KernelAttempt::unsupported_shape;
    return plan;
  }

  if (!output.mapping().is_unique() ||
      !try_make_elementwise_transform_plan<Plan::maximum_rank>(output, plan.elementwise_plan, inputs...))
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
  if (!output_span_size)
  {
    plan.attempt = KernelAttempt::unsupported_shape;
    return plan;
  }
  plan.output_span_size = *output_span_size;

  auto output_descriptor = output.data_descriptor();
  plan.output_buffer = std::addressof(output_descriptor.buffer());
  plan.output_offset = output_descriptor.element_offset();
  CHECK(plan.output_offset <= plan.output_buffer->size() &&
            plan.output_span_size <= plan.output_buffer->size() - plan.output_offset,
        plan.output_offset, plan.output_span_size, plan.output_buffer->size());

  std::size_t input_index = 0;
  auto capture_input = [&](auto& input) {
    auto const input_span_size = required_span_size(input);
    if (!input_span_size) return false;

    auto input_descriptor = input.data_descriptor();
    auto const* input_buffer = std::addressof(input_descriptor.buffer());
    auto const input_offset = input_descriptor.element_offset();
    CHECK(input_offset <= input_buffer->size() && *input_span_size <= input_buffer->size() - input_offset, input_offset,
          *input_span_size, input_buffer->size());

    plan.input_buffers[input_index] = input_buffer;
    plan.input_offsets[input_index] = input_offset;
    plan.input_span_sizes[input_index] = *input_span_size;
    ++input_index;
    return true;
  };
  if (!(capture_input(inputs) && ...))
  {
    plan.attempt = KernelAttempt::unsupported_shape;
    return plan;
  }

  for (std::size_t input = 0; input < Plan::input_count; ++input)
  {
    if (plan.output_buffer->device() != plan.input_buffers[input]->device())
    {
      plan.attempt = KernelAttempt::incompatible_devices;
      return plan;
    }
    if (plan.output_buffer == plan.input_buffers[input] && plan.output_offset == plan.input_offsets[input])
    {
      plan.attempt = KernelAttempt::unsupported_instance;
      return plan;
    }
  }

  plan.attempt = KernelAttempt::success;
  return plan;
}

template <class OutputMdspec, class InputMdspec>
  requires SupportedNegateMdspecs<OutputMdspec, InputMdspec>
[[nodiscard]] auto prepare_negate(OutputMdspec& output, InputMdspec& input)
{
  using scalar_type = transform_scalar_t<OutputMdspec>;
  return prepare_overwrite_transform(NegatePlan<scalar_type>{}, output, input);
}

template <class OutputMdspec, class InputMdspec, class Factor>
  requires SupportedScaleMdspecs<OutputMdspec, InputMdspec, Factor>
[[nodiscard]] auto prepare_scale(OutputMdspec& output, InputMdspec& input, Factor factor)
{
  using scalar_type = transform_scalar_t<OutputMdspec>;
  using factor_type = std::remove_cvref_t<Factor>;
  ScalePlan<scalar_type, factor_type> plan;
  plan.factor = std::move(factor);
  return prepare_overwrite_transform(std::move(plan), output, input);
}

template <class OutputMdspec, class LhsMdspec, class RhsMdspec>
  requires SupportedAddMdspecs<OutputMdspec, LhsMdspec, RhsMdspec>
[[nodiscard]] auto prepare_add(OutputMdspec& output, LhsMdspec& lhs, RhsMdspec& rhs)
{
  using scalar_type = transform_scalar_t<OutputMdspec>;
  return prepare_overwrite_transform(AddPlan<scalar_type>{}, output, lhs, rhs);
}

template <class Plan, class Launch> void execute_overwrite_transform(Plan const& plan, Launch&& launch)
{
  using scalar_type = typename Plan::scalar_type;
  CHECK(kernel_attempt_succeeded(plan.attempt));
  if (!plan.has_work) return;
  CHECK(plan.output_buffer != nullptr);
  for (auto const* input_buffer : plan.input_buffers)
    CHECK(input_buffer != nullptr);

  auto stream = plan.output_buffer->resources().streams().acquire();
  int const device = plan.output_buffer->device().ordinal();
  CHECK_EQUAL(stream.device(), device);
  uni20::cuda::ScopedDevice guard(device);

  auto output_access = plan.output_buffer->write_synchronized_with(stream);
  auto* output = output_access.data() + plan.output_offset;
  std::array<std::optional<uni20::cuda::ReadAccess<scalar_type>>, Plan::input_count> input_accesses;
  std::array<scalar_type const*, Plan::input_count> input_data{};

  for (std::size_t input = 0; input < Plan::input_count; ++input)
  {
    auto const* input_buffer = plan.input_buffers[input];
    if (input_buffer == plan.output_buffer)
    {
      input_data[input] = output_access.data() + plan.input_offsets[input];
      continue;
    }

    std::size_t prior = 0;
    for (; prior < input; ++prior)
      if (plan.input_buffers[prior] == input_buffer) break;

    if (prior == input)
    {
      input_accesses[input].emplace(input_buffer->read_synchronized_with(stream));
      input_data[input] = input_accesses[input]->data() + plan.input_offsets[input];
    }
    else
    {
      CHECK(input_accesses[prior].has_value());
      input_data[input] = input_accesses[prior]->data() + plan.input_offsets[input];
    }
  }

  std::forward<Launch>(launch)(output, input_data, stream.native_handle(), device);
}

template <class Scalar> void execute_negate(NegatePlan<Scalar> const& plan)
{
  execute_overwrite_transform(plan, [&](Scalar* output, auto const& inputs, cudaStream_t stream, int device) {
    if constexpr (supports_elementwise_negate<Scalar>)
    {
      if (plan.elementwise_plan.index_kind == ElementwiseIndexKind::index_32)
        enqueue_elementwise_negate(output, inputs[0], plan.elementwise_plan.plan_32, stream, device);
      else
        enqueue_elementwise_negate(output, inputs[0], plan.elementwise_plan.plan_64, stream, device);
    }
    else
    {
      PANIC("unsupported scalar reached CUDA reference elementwise negation");
    }
  });
}

template <class Scalar, class Factor> void execute_scale(ScalePlan<Scalar, Factor> const& plan)
{
  execute_overwrite_transform(plan, [&](Scalar* output, auto const& inputs, cudaStream_t stream, int device) {
    if constexpr (supports_elementwise_scale<Scalar, Factor>)
    {
      if (plan.elementwise_plan.index_kind == ElementwiseIndexKind::index_32)
        enqueue_elementwise_scale(output, inputs[0], plan.factor, plan.elementwise_plan.plan_32, stream, device);
      else
        enqueue_elementwise_scale(output, inputs[0], plan.factor, plan.elementwise_plan.plan_64, stream, device);
    }
    else
    {
      PANIC("unsupported scalar reached CUDA reference elementwise scaling");
    }
  });
}

template <class Scalar> void execute_add(AddPlan<Scalar> const& plan)
{
  execute_overwrite_transform(plan, [&](Scalar* output, auto const& inputs, cudaStream_t stream, int device) {
    if constexpr (supports_elementwise_add<Scalar>)
    {
      if (plan.elementwise_plan.index_kind == ElementwiseIndexKind::index_32)
        enqueue_elementwise_add(output, inputs[0], inputs[1], plan.elementwise_plan.plan_32, stream, device);
      else
        enqueue_elementwise_add(output, inputs[0], inputs[1], plan.elementwise_plan.plan_64, stream, device);
    }
    else
    {
      PANIC("unsupported scalar reached CUDA reference elementwise addition");
    }
  });
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

/// \brief Report CUDA eligibility for a registered unary scale transform.
template <class Factor, uni20::MutableMdspecLike OutputMdspec, uni20::MdspecLike InputMdspec>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, transform_op<scale<Factor>> const&, OutputMdspec&,
                                    InputMdspec&)
{
  if constexpr (detail::cuda_reference::SupportedScaleMdspecs<OutputMdspec, InputMdspec, Factor>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Apply a registered unary scale transform to CUDA mdspec operands.
template <class Factor, class OutputMdspec, class InputMdspec>
  requires detail::cuda_reference::SupportedScaleMdspecs<OutputMdspec, InputMdspec, Factor>
KernelAttempt try_kernel(CudaReferenceBackend, transform_op<scale<Factor>> const& operation, OutputMdspec& output,
                         InputMdspec& input)
{
  auto preparation = detail::cuda_reference::prepare_scale(output, input, operation.function.factor);
  if (!kernel_attempt_succeeded(preparation.attempt)) return preparation.attempt;
  detail::cuda_reference::execute_scale(preparation);
  return KernelAttempt::success;
}

/// \brief Report CUDA eligibility for the registered binary add transform.
template <uni20::MutableMdspecLike OutputMdspec, uni20::MdspecLike LhsMdspec, uni20::MdspecLike RhsMdspec>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, transform_op<add> const&, OutputMdspec&, LhsMdspec&,
                                    RhsMdspec&)
{
  if constexpr (detail::cuda_reference::SupportedAddMdspecs<OutputMdspec, LhsMdspec, RhsMdspec>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Apply the registered binary add transform to CUDA mdspec operands.
template <class OutputMdspec, class LhsMdspec, class RhsMdspec>
  requires detail::cuda_reference::SupportedAddMdspecs<OutputMdspec, LhsMdspec, RhsMdspec>
KernelAttempt try_kernel(CudaReferenceBackend, transform_op<add> const&, OutputMdspec& output, LhsMdspec& lhs,
                         RhsMdspec& rhs)
{
  auto preparation = detail::cuda_reference::prepare_add(output, lhs, rhs);
  if (!kernel_attempt_succeeded(preparation.attempt)) return preparation.attempt;
  detail::cuda_reference::execute_add(preparation);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
