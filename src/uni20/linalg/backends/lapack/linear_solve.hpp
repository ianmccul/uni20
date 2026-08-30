#pragma once

/**
 * \file linear_solve.hpp
 * \ingroup linalg
 * \brief LAPACK backend for dense general linear systems.
 */

#include "common.hpp"

#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace uni20::linalg
{
namespace lapack_detail
{

template <class CoefficientMdspan, class RhsMdspan> consteval auto linear_solve_acceptance()
{
  using coefficient_scalar = std::remove_cv_t<typename CoefficientMdspan::element_type>;
  using rhs_scalar = std::remove_cv_t<typename RhsMdspan::element_type>;
  if constexpr (uni20::LapackScalar<coefficient_scalar> && std::same_as<coefficient_scalar, rhs_scalar> &&
                uni20::DefaultAccessorMdspanLike<CoefficientMdspan> && uni20::DefaultAccessorMdspanLike<RhsMdspan>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

template <uni20::MutableRankedStridedMdspanLike<2> CoefficientMdspan,
          uni20::MutableRankedStridedMdspanLike<2> RhsMdspan>
KernelAttempt try_linear_solve(CoefficientMdspan& coefficients, RhsMdspan& right_hand_sides)
{
  CHECK_EQUAL(coefficients.extent(0), coefficients.extent(1));
  CHECK_EQUAL(coefficients.extent(0), right_hand_sides.extent(0));
  std::size_t const order_size = static_cast<std::size_t>(coefficients.extent(0));
  std::size_t const rhs_count_size = static_cast<std::size_t>(right_hand_sides.extent(1));
  if (order_size == 0 || rhs_count_size == 0) return KernelAttempt::success;

  blas_int const order = uni20::blas::try_blas_int(order_size);
  blas_int const rhs_count = uni20::blas::try_blas_int(rhs_count_size);
  if (!uni20::blas::is_valid_blas_int(order) || !uni20::blas::is_valid_blas_int(rhs_count))
    return KernelAttempt::unsupported_shape;

  auto coefficient_matrix = uni20::linalg::blas::try_lapack_writable_matrix(coefficients);
  auto rhs_matrix = uni20::linalg::blas::try_lapack_writable_matrix(right_hand_sides);
  if (!coefficient_matrix || !rhs_matrix) return KernelAttempt::unsupported_layout;

  CHECK_EQUAL(coefficient_matrix->rows, order);
  CHECK_EQUAL(coefficient_matrix->cols, order);
  CHECK_EQUAL(rhs_matrix->rows, order);
  CHECK_EQUAL(rhs_matrix->cols, rhs_count);

  std::vector<blas_int> pivots(order_size);
  uni20::lapack::gesv(order, rhs_count, coefficient_matrix->data, coefficient_matrix->leading_dimension, pivots.data(),
                      rhs_matrix->data, rhs_matrix->leading_dimension);
  return KernelAttempt::success;
}

} // namespace lapack_detail

/// \brief Report eligibility for a host-accessible LAPACK general solve.
template <uni20::MutableRankedStridedMdspecLike<2> CoefficientMdspec,
          uni20::MutableRankedStridedMdspecLike<2> RhsMdspec>
  requires uni20::HostWritableMdspec<CoefficientMdspec> && uni20::HostWritableMdspec<RhsMdspec>
consteval auto kernel_accepts_types(LapackBackend const&, linear_solve_op const&, CoefficientMdspec&, RhsMdspec&)
{
  using coefficient_span = uni20::host_write_mdspan_t<CoefficientMdspec>;
  using rhs_span = uni20::host_write_mdspan_t<RhsMdspec>;
  constexpr auto acceptance = lapack_detail::linear_solve_acceptance<coefficient_span, rhs_span>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve host access and solve a general system through LAPACK `gesv`.
template <uni20::MutableRankedStridedMdspecLike<2> CoefficientMdspec,
          uni20::MutableRankedStridedMdspecLike<2> RhsMdspec>
  requires uni20::HostWritableMdspec<CoefficientMdspec> && uni20::HostWritableMdspec<RhsMdspec>
KernelAttempt try_kernel(LapackBackend, linear_solve_op const&, CoefficientMdspec& coefficients,
                         RhsMdspec& right_hand_sides)
{
  return lapack_detail::with_host_write_mdspans(
      [](auto& coefficient_span, auto& rhs_span) {
        return lapack_detail::try_linear_solve(coefficient_span, rhs_span);
      },
      coefficients, right_hand_sides);
}

} // namespace uni20::linalg
