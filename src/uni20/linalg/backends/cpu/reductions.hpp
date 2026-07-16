#pragma once

/**
 * \file reductions.hpp
 * \ingroup linalg
 * \brief Reference CPU inner-product and stable-norm reduction kernels.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>
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

template <class Output, class Scalar> consteval bool scalar_reduction_output_is_supported()
{
  using output_type = std::remove_cvref_t<Output>;
  if constexpr (uni20::MutableRankedSpanLike<output_type, 0>)
  {
    return std::same_as<typename output_type::value_type, Scalar> &&
           requires(output_type& output, Scalar value) { output.operator[]() = value; };
  }
  else
  {
    return std::same_as<output_type, Scalar> && requires(Output& output, Scalar value) { output = value; };
  }
}

template <class Output, class Scalar> void write_scalar_reduction_output(Output& output, Scalar value)
{
  if constexpr (uni20::MutableRankedSpanLike<std::remove_cvref_t<Output>, 0>)
    output.operator[]() = value;
  else
    output = value;
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

template <std::size_t Depth, bool ColumnMajor, class Function, class FirstSpan, class Index, class... RestSpans>
void for_each_reduction_element(Function& function, FirstSpan& first, Index& index, RestSpans&... rest)
{
  using first_type = std::remove_cvref_t<FirstSpan>;
  constexpr std::size_t rank = first_type::rank();
  if constexpr (Depth == rank)
  {
    constexpr auto axes = std::make_index_sequence<rank>{};
    std::invoke(function, reduction_element(first, index, axes), reduction_element(rest, index, axes)...);
  }
  else
  {
    constexpr std::size_t axis = ColumnMajor ? rank - Depth - 1 : Depth;
    using index_type = typename first_type::index_type;
    for (index_type i = 0; i < static_cast<index_type>(first.extent(axis)); ++i)
    {
      index[axis] = i;
      for_each_reduction_element<Depth + 1, ColumnMajor>(function, first, index, rest...);
    }
  }
}

template <class Function, class FirstSpan, class... RestSpans>
void reference_reduce(Function&& function, FirstSpan& first, RestSpans&... rest)
{
  using first_type = std::remove_cvref_t<FirstSpan>;
  check_reduction_extents(first, rest...);
  std::array<typename first_type::index_type, first_type::rank()> index{};
  constexpr bool column_major = std::same_as<typename first_type::layout_type, stdex::layout_left>;
  auto operation = std::forward<Function>(function);
  for_each_reduction_element<0, column_major>(operation, first, index, rest...);
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
    using accumulation_type = uni20::accumulation_scalar_t<scalar_type>;
    return requires(accumulation_type accumulation, scalar_type lhs, scalar_type rhs) {
      accumulation += static_cast<accumulation_type>(uni20::conj(lhs)) * static_cast<accumulation_type>(rhs);
    };
  }
}

template <class LhsSpan, class RhsSpan>
auto reference_inner_product(LhsSpan& lhs, RhsSpan& rhs) -> typename std::remove_cvref_t<LhsSpan>::value_type
{
  using scalar_type = typename std::remove_cvref_t<LhsSpan>::value_type;
  using accumulation_type = uni20::accumulation_scalar_t<scalar_type>;
  accumulation_type accumulation{};
  auto accumulate = [&](auto&& lhs_element, auto&& rhs_element) {
    auto const lhs_value = static_cast<scalar_type>(lhs_element);
    auto const rhs_value = static_cast<scalar_type>(rhs_element);
    accumulation += static_cast<accumulation_type>(uni20::conj(lhs_value)) * static_cast<accumulation_type>(rhs_value);
  };
  reference_reduce(accumulate, lhs, rhs);
  return static_cast<scalar_type>(accumulation);
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
    using accumulation_type = uni20::accumulation_real_t<scalar_type>;
    return requires(accumulation_type value) {
      value = static_cast<accumulation_type>(std::declval<uni20::make_real_t<scalar_type>>());
      value += value * value;
    };
  }
}

template <class InputSpan>
auto reference_norm(InputSpan& input) -> uni20::make_real_t<typename std::remove_cvref_t<InputSpan>::value_type>
{
  using scalar_type = typename std::remove_cvref_t<InputSpan>::value_type;
  using result_type = uni20::make_real_t<scalar_type>;
  using accumulation_type = uni20::accumulation_real_t<scalar_type>;

  accumulation_type scale{};
  accumulation_type sum_of_squares{};
  auto accumulate = [&](auto&& element) {
    auto const value = static_cast<scalar_type>(element);
    if constexpr (uni20::Complex<scalar_type>)
    {
      update_scaled_sum_of_squares(static_cast<accumulation_type>(value.real()), scale, sum_of_squares);
      update_scaled_sum_of_squares(static_cast<accumulation_type>(value.imag()), scale, sum_of_squares);
    }
    else
    {
      update_scaled_sum_of_squares(static_cast<accumulation_type>(value), scale, sum_of_squares);
    }
  };
  reference_reduce(accumulate, input);

  if (!(sum_of_squares == sum_of_squares)) return static_cast<result_type>(sum_of_squares);
  if (scale == accumulation_type{}) return result_type{};
  using std::sqrt;
  return static_cast<result_type>(scale * sqrt(sum_of_squares));
}

} // namespace detail

/// \brief Report compile-time eligibility for reference CPU inner products.
template <class Output, uni20::SpanLike LhsSpan, uni20::SpanLike RhsSpan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, inner_product_op const&, Output&, LhsSpan&, RhsSpan&)
{
  using scalar_type = typename LhsSpan::value_type;
  if constexpr (detail::inner_product_inputs_are_supported<LhsSpan, RhsSpan>() &&
                detail::scalar_reduction_output_is_supported<Output, scalar_type>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Compute a conjugate-linear-left inner product through accessors.
template <class Output, class LhsSpan, class RhsSpan>
KernelAttempt try_kernel(CpuReferenceBackend, inner_product_op const&, Output&& output, LhsSpan&& lhs, RhsSpan&& rhs)
{
  auto const result = detail::reference_inner_product(lhs, rhs);
  detail::write_scalar_reduction_output(output, result);
  return KernelAttempt::success;
}

/// \brief Report compile-time eligibility for reference CPU Euclidean norms.
template <class Output, uni20::SpanLike InputSpan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, norm_op const&, Output&, InputSpan&)
{
  using scalar_type = typename InputSpan::value_type;
  using result_type = uni20::make_real_t<scalar_type>;
  if constexpr (detail::norm_input_is_supported<InputSpan>() &&
                detail::scalar_reduction_output_is_supported<Output, result_type>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Compute a stable Euclidean norm through the input accessor.
template <class Output, class InputSpan>
KernelAttempt try_kernel(CpuReferenceBackend, norm_op const&, Output&& output, InputSpan&& input)
{
  auto const result = detail::reference_norm(input);
  detail::write_scalar_reduction_output(output, result);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
