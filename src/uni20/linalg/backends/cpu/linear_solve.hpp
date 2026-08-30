#pragma once

/**
 * \file linear_solve.hpp
 * \ingroup linalg
 * \brief Reference CPU backend for dense general linear systems.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail::cpu_reference
{

template <class CoefficientMdspan, class RhsMdspan> consteval auto linear_solve_acceptance()
{
  using coefficient_type = std::remove_cvref_t<CoefficientMdspan>;
  using rhs_type = std::remove_cvref_t<RhsMdspan>;
  using scalar_type = typename coefficient_type::value_type;
  using coefficient_index = typename coefficient_type::index_type;
  using rhs_index = typename rhs_type::index_type;

  if constexpr (uni20::RealOrComplex<scalar_type> && std::same_as<scalar_type, typename rhs_type::value_type> &&
                requires(CoefficientMdspan& coefficients, RhsMdspan& rhs, coefficient_index coefficient_i,
                         rhs_index rhs_i, scalar_type value) {
                  static_cast<scalar_type>(coefficients[coefficient_i, coefficient_i]);
                  static_cast<scalar_type>(rhs[rhs_i, rhs_i]);
                  coefficients[coefficient_i, coefficient_i] = value;
                  rhs[rhs_i, rhs_i] = value;
                  value / value;
                  value -= value * value;
                })
    return kernel_types_yes;
  else
    return kernel_types_no;
}

template <class MatrixMdspan>
void swap_rows(MatrixMdspan& matrix, std::size_t lhs, std::size_t rhs, std::size_t columns)
{
  if (lhs == rhs) return;
  using index_type = typename std::remove_cvref_t<MatrixMdspan>::index_type;
  for (std::size_t col = 0; col < columns; ++col)
  {
    auto const left_index = static_cast<index_type>(lhs);
    auto const right_index = static_cast<index_type>(rhs);
    auto const column = static_cast<index_type>(col);
    auto temporary = static_cast<typename std::remove_cvref_t<MatrixMdspan>::value_type>(matrix[left_index, column]);
    auto right_value = static_cast<typename std::remove_cvref_t<MatrixMdspan>::value_type>(matrix[right_index, column]);
    matrix[left_index, column] = right_value;
    matrix[right_index, column] = temporary;
  }
}

template <class CoefficientMdspan, class RhsMdspan>
KernelAttempt linear_solve(CoefficientMdspan& coefficients, RhsMdspan& right_hand_sides)
{
  using coefficient_type = std::remove_cvref_t<CoefficientMdspan>;
  using rhs_type = std::remove_cvref_t<RhsMdspan>;
  using scalar_type = typename coefficient_type::value_type;
  using real_type = uni20::make_real_t<scalar_type>;
  using coefficient_index = typename coefficient_type::index_type;
  using rhs_index = typename rhs_type::index_type;

  CHECK_EQUAL(coefficients.extent(0), coefficients.extent(1));
  CHECK_EQUAL(coefficients.extent(0), right_hand_sides.extent(0));
  std::size_t const order = static_cast<std::size_t>(coefficients.extent(0));
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.extent(1));
  if (rhs_count == 0) return KernelAttempt::success;

  using std::abs;
  for (std::size_t k = 0; k < order; ++k)
  {
    auto const coefficient_k = static_cast<coefficient_index>(k);
    std::size_t pivot_row = k;
    real_type pivot_value = abs(static_cast<scalar_type>(coefficients[coefficient_k, coefficient_k]));
    for (std::size_t row = k + 1; row < order; ++row)
    {
      auto const coefficient_row = static_cast<coefficient_index>(row);
      real_type const candidate = abs(static_cast<scalar_type>(coefficients[coefficient_row, coefficient_k]));
      if (candidate > pivot_value)
      {
        pivot_value = candidate;
        pivot_row = row;
      }
    }

    ERROR_IF(pivot_value == real_type{}, "singular matrix in solve");
    swap_rows(coefficients, k, pivot_row, order);
    swap_rows(right_hand_sides, k, pivot_row, rhs_count);

    scalar_type const pivot = static_cast<scalar_type>(coefficients[coefficient_k, coefficient_k]);
    for (std::size_t row = k + 1; row < order; ++row)
    {
      auto const coefficient_row = static_cast<coefficient_index>(row);
      scalar_type const factor = static_cast<scalar_type>(coefficients[coefficient_row, coefficient_k]) / pivot;
      if (factor == scalar_type{}) continue;

      coefficients[coefficient_row, coefficient_k] = scalar_type{};
      for (std::size_t col = k + 1; col < order; ++col)
      {
        auto const coefficient_col = static_cast<coefficient_index>(col);
        scalar_type value = static_cast<scalar_type>(coefficients[coefficient_row, coefficient_col]);
        value -= factor * static_cast<scalar_type>(coefficients[coefficient_k, coefficient_col]);
        coefficients[coefficient_row, coefficient_col] = value;
      }
      for (std::size_t col = 0; col < rhs_count; ++col)
      {
        auto const rhs_row = static_cast<rhs_index>(row);
        auto const rhs_k = static_cast<rhs_index>(k);
        auto const rhs_col = static_cast<rhs_index>(col);
        scalar_type value = static_cast<scalar_type>(right_hand_sides[rhs_row, rhs_col]);
        value -= factor * static_cast<scalar_type>(right_hand_sides[rhs_k, rhs_col]);
        right_hand_sides[rhs_row, rhs_col] = value;
      }
    }
  }

  for (std::size_t row = order; row-- > 0;)
  {
    auto const coefficient_row = static_cast<coefficient_index>(row);
    scalar_type const pivot = static_cast<scalar_type>(coefficients[coefficient_row, coefficient_row]);
    for (std::size_t col = 0; col < rhs_count; ++col)
    {
      auto const rhs_row = static_cast<rhs_index>(row);
      auto const rhs_col = static_cast<rhs_index>(col);
      scalar_type value = static_cast<scalar_type>(right_hand_sides[rhs_row, rhs_col]);
      for (std::size_t k = row + 1; k < order; ++k)
      {
        auto const coefficient_k = static_cast<coefficient_index>(k);
        auto const rhs_k = static_cast<rhs_index>(k);
        value -= static_cast<scalar_type>(coefficients[coefficient_row, coefficient_k]) *
                 static_cast<scalar_type>(right_hand_sides[rhs_k, rhs_col]);
      }
      right_hand_sides[rhs_row, rhs_col] = value / pivot;
    }
  }

  return KernelAttempt::success;
}

} // namespace detail::cpu_reference

/// \brief Report eligibility for a host-accessible dense general solve.
template <uni20::MutableRankedMdspecLike<2> CoefficientMdspec, uni20::MutableRankedMdspecLike<2> RhsMdspec>
  requires uni20::HostWritableMdspec<CoefficientMdspec> && uni20::HostWritableMdspec<RhsMdspec>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, linear_solve_op const&, CoefficientMdspec&, RhsMdspec&)
{
  using coefficient_span = uni20::host_write_mdspan_t<CoefficientMdspec>;
  using rhs_span = uni20::host_write_mdspan_t<RhsMdspec>;
  constexpr auto acceptance = detail::cpu_reference::linear_solve_acceptance<coefficient_span, rhs_span>();
  if constexpr (acceptance == KernelTypeAcceptance::yes)
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host access and solve a dense general linear system.
template <uni20::MutableRankedMdspecLike<2> CoefficientMdspec, uni20::MutableRankedMdspecLike<2> RhsMdspec>
  requires uni20::HostWritableMdspec<CoefficientMdspec> && uni20::HostWritableMdspec<RhsMdspec>
KernelAttempt try_kernel(CpuReferenceBackend, linear_solve_op const&, CoefficientMdspec& coefficients,
                         RhsMdspec& right_hand_sides)
{
  auto coefficient_access = acquire_host_write_access_sync(coefficients);
  auto rhs_access = acquire_host_write_access_sync(right_hand_sides);
  auto coefficient_span = coefficient_access.mdspan();
  auto rhs_span = rhs_access.mdspan();
  return detail::cpu_reference::linear_solve(coefficient_span, rhs_span);
}

} // namespace uni20::linalg
