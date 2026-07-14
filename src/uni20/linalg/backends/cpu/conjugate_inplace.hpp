#pragma once

/**
 * \file conjugate_inplace.hpp
 * \ingroup linalg
 * \brief Reference CPU backend for in-place scalar conjugation.
 */

#include <uni20/core/math.hpp>
#include <uni20/level1/apply_unary.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class Mdspan, std::size_t... Axis>
consteval bool conjugate_element_is_assignable(std::index_sequence<Axis...>)
{
  using span_type = std::remove_cvref_t<Mdspan>;
  using index_type = typename span_type::index_type;
  return requires(span_type& span, index_type index) {
    span.operator[](((void)Axis, index)...) =
        uni20::conj(static_cast<typename span_type::value_type>(span.operator[](((void)Axis, index)...)));
  };
}

template <class Mdspan, std::size_t... Axis>
void conjugate_element(Mdspan& span, std::array<typename Mdspan::index_type, Mdspan::rank()> const& index,
                       std::index_sequence<Axis...>)
{
  using value_type = typename Mdspan::value_type;
  auto&& element = span.operator[](index[Axis]...);
  element = uni20::conj(static_cast<value_type>(element));
}

template <std::size_t Axis, class Mdspan>
void conjugate_elements(Mdspan& span, std::array<typename Mdspan::index_type, Mdspan::rank()>& index)
{
  if constexpr (Axis == Mdspan::rank())
  {
    conjugate_element(span, index, std::make_index_sequence<Mdspan::rank()>{});
  }
  else
  {
    using index_type = typename Mdspan::index_type;
    for (index_type i = 0; i < static_cast<index_type>(span.extent(Axis)); ++i)
    {
      index[Axis] = i;
      conjugate_elements<Axis + 1>(span, index);
    }
  }
}
} // namespace detail

/// \brief Report compile-time eligibility for reference CPU in-place conjugation.
template <uni20::MutableSpanLike Mdspan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, conjugate_inplace_op const&, Mdspan&)
{
  using span_type = std::remove_cvref_t<Mdspan>;
  if constexpr (uni20::Scalar<typename span_type::value_type> &&
                detail::conjugate_element_is_assignable<span_type>(std::make_index_sequence<span_type::rank()>{}))
  {
    return kernel_types_yes;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Conjugate every element in-place through the mdspan accessor.
template <class Mdspan> KernelAttempt try_kernel(CpuReferenceBackend, conjugate_inplace_op const&, Mdspan&& span)
{
  using span_type = std::remove_cvref_t<Mdspan>;
  if constexpr (uni20::MutableStridedMdspan<span_type>)
  {
    using value_type = typename span_type::value_type;
    uni20::apply_unary_inplace(span, [](auto const& value) { return uni20::conj(static_cast<value_type>(value)); });
  }
  else
  {
    std::array<typename span_type::index_type, span_type::rank()> index{};
    detail::conjugate_elements<0>(span, index);
  }
  return KernelAttempt::success;
}

} // namespace uni20::linalg
