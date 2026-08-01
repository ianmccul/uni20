#pragma once

/**
 * \file conjugate_inplace.hpp
 * \ingroup linalg
 * \brief Reference CPU backend for in-place scalar conjugation.
 */

#include <uni20/core/math.hpp>
#include <uni20/linalg/backends/cpu/strided_transform.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail::cpu_reference
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

/// \brief Report compile-time eligibility for resolved CPU in-place conjugation.
template <uni20::MutableMdspanLike Mdspan> consteval auto conjugate_acceptance()
{
  using span_type = std::remove_cvref_t<Mdspan>;
  if constexpr (uni20::Scalar<typename span_type::value_type> &&
                conjugate_element_is_assignable<span_type>(std::make_index_sequence<span_type::rank()>{}))
  {
    return kernel_types_yes;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Conjugate every element in-place through a resolved mdspan accessor.
template <class Mdspan> KernelAttempt conjugate(Mdspan& span)
{
  using span_type = std::remove_cvref_t<Mdspan>;
  if constexpr (uni20::MutableStridedMdspanLike<span_type>)
  {
    using value_type = typename span_type::value_type;
    cpu::detail::transform_strided<true>(span,
                                         [](auto const& value) { return uni20::conj(static_cast<value_type>(value)); });
  }
  else
  {
    std::array<typename span_type::index_type, span_type::rank()> index{};
    conjugate_elements<0>(span, index);
  }
  return KernelAttempt::success;
}
} // namespace detail::cpu_reference

/// \brief Report eligibility for host-accessible mdspec in-place conjugation.
template <uni20::MutableMdspecLike Mdspan>
  requires uni20::HostWritableMdspec<Mdspan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, conjugate_inplace_op const&, Mdspan&)
{
  using span_type = uni20::host_write_mdspan_t<Mdspan>;
  constexpr auto acceptance = detail::cpu_reference::conjugate_acceptance<span_type>();
  if constexpr (acceptance == KernelTypeAcceptance::yes)
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host access and conjugate every element.
template <uni20::MutableMdspecLike Mdspan>
  requires uni20::HostWritableMdspec<Mdspan>
KernelAttempt try_kernel(CpuReferenceBackend, conjugate_inplace_op const&, Mdspan& mdspan)
{
  auto access = acquire_host_write_access_sync(mdspan);
  auto span = access.mdspan();
  return detail::cpu_reference::conjugate(span);
}

} // namespace uni20::linalg
