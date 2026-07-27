#pragma once

/**
 * \file reductions.hpp
 * \ingroup linalg
 * \brief Reference CPU tensor reduction kernels.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/backends/cpu/detail/compensated_sum.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{

template <class Span> consteval bool reduction_value_is_readable()
{
  using span_type = std::remove_cvref_t<Span>;
  return requires(typename span_type::reference element) { static_cast<typename span_type::value_type>(element); };
}

template <class Output, class Scalar, std::size_t Rank> consteval bool reduction_output_is_supported()
{
  using output_type = std::remove_cvref_t<Output>;
  if constexpr (uni20::MutableRankedMdspanLike<output_type, Rank>)
  {
    return std::same_as<typename output_type::value_type, Scalar>;
  }
  else if constexpr (Rank == 0)
  {
    return std::same_as<output_type, Scalar> && requires(Output& output, Scalar value) { output = value; };
  }
  else
  {
    return false;
  }
}

template <class Output> consteval bool reduction_output_is_column_major()
{
  using output_type = std::remove_cvref_t<Output>;
  if constexpr (uni20::MdspanLike<output_type>)
    return std::same_as<typename output_type::layout_type, stdex::layout_left>;
  else
    return false;
}

template <class Output, class Index, class Scalar, std::size_t... Axis>
void write_reduction_output(Output& output, Index const& index, Scalar value, std::index_sequence<Axis...>)
{
  if constexpr (uni20::MutableRankedMdspanLike<std::remove_cvref_t<Output>, sizeof...(Axis)>)
  {
    using output_type = std::remove_cvref_t<Output>;
    using index_type = typename output_type::index_type;
    output.operator[](static_cast<index_type>(index[Axis])...) = value;
  }
  else
  {
    static_assert(sizeof...(Axis) == 0);
    output = value;
  }
}

template <class Reference, class... Others>
void check_reduction_extents(Reference const& reference, Others const&... others)
{
  if constexpr (sizeof...(Others) > 0)
  {
    constexpr std::size_t rank = std::remove_cvref_t<Reference>::rank();
    if constexpr (rank > 0)
    {
      auto check_one = [&](auto const& other) {
        for (std::size_t axis = 0; axis < rank; ++axis)
          CHECK_EQUAL(reference.extent(axis), other.extent(axis));
      };
      (check_one(others), ...);
    }
  }
}

template <class Span, class Index, std::size_t... Axis>
decltype(auto) reduction_element(Span& span, Index const& index, std::index_sequence<Axis...>)
{
  using span_type = std::remove_cvref_t<Span>;
  using index_type = typename span_type::index_type;
  return span.operator[](static_cast<index_type>(index[Axis])...);
}

template <std::size_t Depth, bool ColumnMajor, class Axes, class Accumulate, class State, class FirstSpan,
          class InputIndex, class... RestSpans>
void accumulate_reduced_elements(Axes const& axes, Accumulate& accumulate, State& state, FirstSpan& first,
                                 InputIndex& input_index, RestSpans&... rest)
{
  if constexpr (Depth == Axes::reduced_rank)
  {
    constexpr auto input_axes = std::make_index_sequence<Axes::input_rank>{};
    std::invoke(accumulate, state, reduction_element(first, input_index, input_axes),
                reduction_element(rest, input_index, input_axes)...);
  }
  else
  {
    constexpr std::size_t position = ColumnMajor ? Axes::reduced_rank - Depth - 1 : Depth;
    std::size_t const axis = axes.reduced[position];
    using index_type = typename std::remove_cvref_t<FirstSpan>::index_type;
    for (index_type i = 0; i < static_cast<index_type>(first.extent(axis)); ++i)
    {
      input_index[axis] = i;
      accumulate_reduced_elements<Depth + 1, ColumnMajor>(axes, accumulate, state, first, input_index, rest...);
    }
  }
}

template <std::size_t Depth, bool OutputColumnMajor, bool InputColumnMajor, class Output, class Axes, class Initialize,
          class Accumulate, class Finalize, class OutputIndex, class InputIndex, class FirstSpan, class... RestSpans>
void emit_reduction_outputs(Output& output, Axes const& axes, Initialize& initialize, Accumulate& accumulate,
                            Finalize& finalize, OutputIndex& output_index, InputIndex& input_index, FirstSpan& first,
                            RestSpans&... rest)
{
  if constexpr (Depth == Axes::output_rank)
  {
    auto state = std::invoke(initialize);
    accumulate_reduced_elements<0, InputColumnMajor>(axes, accumulate, state, first, input_index, rest...);
    auto value = std::invoke(finalize, state);
    write_reduction_output(output, output_index, std::move(value), std::make_index_sequence<Axes::output_rank>{});
  }
  else
  {
    constexpr std::size_t output_axis = OutputColumnMajor ? Axes::output_rank - Depth - 1 : Depth;
    std::size_t const input_axis = axes.surviving[output_axis];
    using index_type = typename std::remove_cvref_t<FirstSpan>::index_type;
    for (index_type i = 0; i < static_cast<index_type>(first.extent(input_axis)); ++i)
    {
      output_index[output_axis] = i;
      input_index[input_axis] = i;
      emit_reduction_outputs<Depth + 1, OutputColumnMajor, InputColumnMajor>(
          output, axes, initialize, accumulate, finalize, output_index, input_index, first, rest...);
    }
  }
}

template <class Output, std::size_t InputRank, std::size_t ReducedRank, class Initialize, class Accumulate,
          class Finalize, class FirstSpan, class... RestSpans>
void reference_reduce_axes(Output& output, ReductionAxes<InputRank, ReducedRank> const& axes, Initialize&& initialize,
                           Accumulate&& accumulate, Finalize&& finalize, FirstSpan& first, RestSpans&... rest)
{
  using first_type = std::remove_cvref_t<FirstSpan>;
  static_assert(first_type::rank() == InputRank);
  static_assert(((std::remove_cvref_t<RestSpans>::rank() == InputRank) && ...));
  if constexpr (uni20::MdspanLike<std::remove_cvref_t<Output>>)
    static_assert(std::remove_cvref_t<Output>::rank() == InputRank - ReducedRank);

  CHECK(reduction_axes_are_valid(axes));
  check_reduction_extents(first, rest...);

  if constexpr (uni20::MdspanLike<std::remove_cvref_t<Output>>)
  {
    for (std::size_t output_axis = 0; output_axis < InputRank - ReducedRank; ++output_axis)
      CHECK_EQUAL(output.extent(output_axis), first.extent(axes.surviving[output_axis]));
  }

  std::array<typename first_type::index_type, InputRank - ReducedRank> output_index{};
  std::array<typename first_type::index_type, InputRank> input_index{};
  constexpr bool output_column_major = reduction_output_is_column_major<Output>();
  constexpr bool input_column_major = std::same_as<typename first_type::layout_type, stdex::layout_left>;
  auto initialize_operation = std::forward<Initialize>(initialize);
  auto accumulate_operation = std::forward<Accumulate>(accumulate);
  auto finalize_operation = std::forward<Finalize>(finalize);
  emit_reduction_outputs<0, output_column_major, input_column_major>(output, axes, initialize_operation,
                                                                     accumulate_operation, finalize_operation,
                                                                     output_index, input_index, first, rest...);
}

template <class LhsSpan, class RhsSpan> consteval bool inner_product_inputs_are_supported()
{
  using lhs_type = std::remove_cvref_t<LhsSpan>;
  using rhs_type = std::remove_cvref_t<RhsSpan>;
  using scalar_type = typename lhs_type::value_type;
  if constexpr (lhs_type::rank() != rhs_type::rank() || !std::same_as<scalar_type, typename rhs_type::value_type> ||
                !uni20::RealOrComplex<scalar_type> || !reduction_value_is_readable<LhsSpan>() ||
                !reduction_value_is_readable<RhsSpan>())
  {
    return false;
  }
  else
  {
    return requires(scalar_type lhs, scalar_type rhs) {
      uni20::conj(lhs) * rhs;
      backends::cpu::detail::CompensatedSum<scalar_type>{}.add(uni20::conj(lhs) * rhs);
    };
  }
}

template <class Output, class LhsSpan, class RhsSpan>
void reference_inner_product(Output& output, LhsSpan& lhs, RhsSpan& rhs)
{
  using lhs_type = std::remove_cvref_t<LhsSpan>;
  using scalar_type = typename lhs_type::value_type;
  using state_type = backends::cpu::detail::CompensatedSum<scalar_type>;

  auto initialize = [] { return state_type{}; };
  auto accumulate = [](state_type& state, auto&& lhs_element, auto&& rhs_element) {
    auto const lhs_value = static_cast<scalar_type>(lhs_element);
    auto const rhs_value = static_cast<scalar_type>(rhs_element);
    state.add(uni20::conj(lhs_value) * rhs_value);
  };
  auto finalize = [](state_type const& state) { return state.value(); };
  reference_reduce_axes(output, all_reduction_axes<lhs_type::rank()>(), initialize, accumulate, finalize, lhs, rhs);
}

template <class Real> void update_scaled_sum_of_squares(Real component, Real& scale, Real& sum_of_squares)
{
  using std::abs;
  component = abs(component);
  if (component == Real{} || !(sum_of_squares == sum_of_squares)) return;

  if (!(component == component))
  {
    sum_of_squares = component;
  }
  else if (component > scale)
  {
    Real const ratio = scale / component;
    sum_of_squares = Real{1} + sum_of_squares * ratio * ratio;
    scale = component;
  }
  else if (component == scale)
  {
    sum_of_squares += Real{1};
  }
  else
  {
    Real const ratio = component / scale;
    sum_of_squares += ratio * ratio;
  }
}

template <class InputSpan> consteval bool norm_input_is_supported()
{
  using input_type = std::remove_cvref_t<InputSpan>;
  using scalar_type = typename input_type::value_type;
  if constexpr (!uni20::RealOrComplex<scalar_type> || !reduction_value_is_readable<InputSpan>())
  {
    return false;
  }
  else
  {
    using real_type = uni20::make_real_t<scalar_type>;
    return requires(real_type value) { value += value * value; };
  }
}

template <class Real> struct ScaledSumOfSquaresState
{
    Real scale{};
    Real sum_of_squares{};
};

template <class Output, class InputSpan> void reference_norm(Output& output, InputSpan& input)
{
  using input_type = std::remove_cvref_t<InputSpan>;
  using scalar_type = typename input_type::value_type;
  using result_type = uni20::make_real_t<scalar_type>;
  using state_type = ScaledSumOfSquaresState<result_type>;

  auto initialize = [] { return state_type{}; };
  auto accumulate = [](state_type& state, auto&& element) {
    auto const value = static_cast<scalar_type>(element);
    if constexpr (uni20::Complex<scalar_type>)
    {
      update_scaled_sum_of_squares(value.real(), state.scale, state.sum_of_squares);
      update_scaled_sum_of_squares(value.imag(), state.scale, state.sum_of_squares);
    }
    else
    {
      update_scaled_sum_of_squares(value, state.scale, state.sum_of_squares);
    }
  };
  auto finalize = [](state_type const& state) {
    if (!(state.sum_of_squares == state.sum_of_squares)) return state.sum_of_squares;
    if (state.scale == result_type{}) return result_type{};
    using std::sqrt;
    return state.scale * sqrt(state.sum_of_squares);
  };
  reference_reduce_axes(output, all_reduction_axes<input_type::rank()>(), initialize, accumulate, finalize, input);
}

template <class InputSpan> consteval bool sum_input_is_supported()
{
  using input_type = std::remove_cvref_t<InputSpan>;
  using scalar_type = typename input_type::value_type;
  if constexpr (!uni20::RealOrComplex<scalar_type> || !reduction_value_is_readable<InputSpan>())
  {
    return false;
  }
  else
  {
    return requires(scalar_type value) { backends::cpu::detail::CompensatedSum<scalar_type>{}.add(value); };
  }
}

template <class Output, std::size_t InputRank, std::size_t ReducedRank, class InputSpan>
void reference_sum(Output& output, sum_reduction_op<InputRank, ReducedRank> const& operation, InputSpan& input)
{
  using scalar_type = typename std::remove_cvref_t<InputSpan>::value_type;
  using state_type = backends::cpu::detail::CompensatedSum<scalar_type>;

  auto initialize = [] { return state_type{}; };
  auto accumulate = [](state_type& state, auto&& element) { state.add(static_cast<scalar_type>(element)); };
  auto finalize = [](state_type const& state) { return state.value(); };
  reference_reduce_axes(output, operation.axes, initialize, accumulate, finalize, input);
}

} // namespace detail

/// \brief Report compile-time eligibility for reference CPU inner products.
template <class Output, uni20::MdspanLike LhsSpan, uni20::MdspanLike RhsSpan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, inner_product_op const&, Output&, LhsSpan&, RhsSpan&)
{
  using scalar_type = typename LhsSpan::value_type;
  if constexpr (detail::inner_product_inputs_are_supported<LhsSpan, RhsSpan>() &&
                detail::reduction_output_is_supported<Output, scalar_type, 0>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Compute a conjugate-linear-left inner product through accessors.
template <class Output, class LhsSpan, class RhsSpan>
KernelAttempt try_kernel(CpuReferenceBackend, inner_product_op const&, Output&& output, LhsSpan&& lhs, RhsSpan&& rhs)
{
  detail::reference_inner_product(output, lhs, rhs);
  return KernelAttempt::success;
}

/// \brief Report compile-time eligibility for reference CPU Euclidean norms.
template <class Output, uni20::MdspanLike InputSpan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, norm_op const&, Output&, InputSpan&)
{
  using scalar_type = typename InputSpan::value_type;
  using result_type = uni20::make_real_t<scalar_type>;
  if constexpr (detail::norm_input_is_supported<InputSpan>() &&
                detail::reduction_output_is_supported<Output, result_type, 0>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Compute a stable Euclidean norm through the input accessor.
template <class Output, class InputSpan>
KernelAttempt try_kernel(CpuReferenceBackend, norm_op const&, Output&& output, InputSpan&& input)
{
  detail::reference_norm(output, input);
  return KernelAttempt::success;
}

/// \brief Report compile-time eligibility for reference CPU sum reductions.
template <class Output, std::size_t InputRank, std::size_t ReducedRank, uni20::MdspanLike InputSpan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, sum_reduction_op<InputRank, ReducedRank> const&,
                                    Output&, InputSpan&)
{
  using input_type = std::remove_cvref_t<InputSpan>;
  using scalar_type = typename input_type::value_type;
  if constexpr (input_type::rank() == InputRank && detail::sum_input_is_supported<InputSpan>() &&
                detail::reduction_output_is_supported<Output, scalar_type, InputRank - ReducedRank>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Sum selected input axes through the input accessor.
template <class Output, std::size_t InputRank, std::size_t ReducedRank, class InputSpan>
KernelAttempt try_kernel(CpuReferenceBackend, sum_reduction_op<InputRank, ReducedRank> const& operation,
                         Output&& output, InputSpan&& input)
{
  CHECK(reduction_axes_are_valid(operation.axes));
  detail::reference_sum(output, operation, input);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
