#pragma once

/**
 * \file mdspan_access.hpp
 * \ingroup linalg
 * \brief Direct BLAS accessor eligibility for ranked mdspan operands.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>

#include <cstddef>
#include <type_traits>

namespace uni20::linalg::blas::detail
{

template <class Accessor>
struct is_blas_direct_read_accessor : std::bool_constant<uni20::is_default_accessor_v<Accessor>>
{};

template <uni20::AccessorPolicy Accessor>
struct is_blas_direct_read_accessor<uni20::conjugated_accessor<Accessor>>
    : std::bool_constant<uni20::is_default_accessor_v<Accessor>>
{};

template <class Accessor>
inline constexpr bool is_blas_direct_read_accessor_v =
    is_blas_direct_read_accessor<std::remove_cvref_t<Accessor>>::value;

template <class Mdspan, std::size_t Rank>
concept blas_readable_mdspan =
    uni20::RankedStridedMdspan<Mdspan, Rank> &&
    uni20::BlasScalar<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>> &&
    is_blas_direct_read_accessor_v<typename std::remove_cvref_t<Mdspan>::accessor_type>;

template <class Mdspan, std::size_t Rank>
concept blas_writable_mdspan =
    uni20::MutableRankedStridedMdspan<Mdspan, Rank> &&
    uni20::BlasScalar<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>> &&
    uni20::DefaultAccessorMdspan<Mdspan>;

} // namespace uni20::linalg::blas::detail
