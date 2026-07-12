#pragma once

/**
 * \file self_adjoint_eigh.hpp
 * \ingroup linalg
 * \brief LAPACK backend for dense symmetric and Hermitian eigensystems.
 */

#include "common.hpp"

#include <uni20/backend/blas/blas_int.hpp>
#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/blas/mdspan_vector.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::linalg
{
namespace lapack_detail
{
inline char lapack_triangle(MatrixTriangle triangle)
{
  switch (triangle)
  {
    case MatrixTriangle::Upper:
      return 'U';
    case MatrixTriangle::Lower:
      return 'L';
  }
  PANIC("invalid MatrixTriangle", std::to_underlying(triangle));
}
} // namespace lapack_detail

/// \brief Report compile-time eligibility for LAPACK self-adjoint eigenanalysis.
template <uni20::MutableRankedStridedMdspan<1> EigenvalueMdspan, uni20::MutableRankedStridedMdspan<2> MatrixMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, self_adjoint_eigh_op const&, EigenvalueMdspan&, MatrixMdspan&)
{
  using matrix_scalar = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using eigenvalue_scalar = std::remove_cv_t<typename EigenvalueMdspan::element_type>;
  if constexpr (uni20::LapackScalar<matrix_scalar> && uni20::LapackReal<eigenvalue_scalar> &&
                std::same_as<eigenvalue_scalar, uni20::make_real_t<matrix_scalar>> &&
                uni20::DefaultAccessorMdspan<EigenvalueMdspan> && uni20::DefaultAccessorMdspan<MatrixMdspan>)
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute an in-place symmetric or Hermitian eigensystem through `syev` or `heev`.
/// \details LAPACK overwrites `matrix_work`. When eigenvectors are requested,
///          its columns contain normalized eigenvectors on return.
template <class EigenvalueMdspan, class MatrixMdspan>
KernelAttempt try_kernel(LapackBackend, self_adjoint_eigh_op const& op, EigenvalueMdspan&& eigenvalues,
                         MatrixMdspan&& matrix_work)
{
  using matrix_type = std::remove_cvref_t<MatrixMdspan>;
  using matrix_scalar = std::remove_cv_t<typename matrix_type::element_type>;
  using real_type = uni20::make_real_t<matrix_scalar>;

  CHECK_EQUAL(matrix_work.extent(0), matrix_work.extent(1));
  std::size_t const n = static_cast<std::size_t>(matrix_work.extent(0));
  CHECK_EQUAL(static_cast<std::size_t>(eigenvalues.extent(0)), n);

  blas_int const order = uni20::blas::try_blas_int(n);
  if (!uni20::blas::is_valid_blas_int(order))
  {
    return KernelAttempt::unsupported_shape;
  }
  if (order == 0)
  {
    return KernelAttempt::success;
  }

  auto matrix = uni20::linalg::blas::try_lapack_writable_matrix(matrix_work);
  auto values = uni20::linalg::blas::try_lapack_writable_vector(eigenvalues);
  if (!matrix || !values)
  {
    return KernelAttempt::unsupported_layout;
  }
  CHECK_EQUAL(matrix->rows, order);
  CHECK_EQUAL(matrix->cols, order);
  CHECK_EQUAL(values->size, order);

  char const jobz = op.compute_vectors ? 'V' : 'N';
  char const uplo = lapack_detail::lapack_triangle(op.triangle);
  if constexpr (uni20::LapackReal<matrix_scalar>)
  {
    matrix_scalar work_query{};
    uni20::lapack::syev(jobz, uplo, order, matrix->data, matrix->leading_dimension, values->data, &work_query, -1);
    blas_int const lwork = lapack_detail::workspace_size(work_query);
    std::vector<matrix_scalar> work(static_cast<std::size_t>(lwork), matrix_scalar{});
    uni20::lapack::syev(jobz, uplo, order, matrix->data, matrix->leading_dimension, values->data, work.data(), lwork);
  }
  else
  {
    auto const three_order = lapack_detail::try_size_product(3, n);
    if (!three_order)
    {
      return KernelAttempt::unsupported_shape;
    }
    std::size_t const rwork_size = std::max<std::size_t>(1, *three_order - 2);
    std::vector<real_type> rwork(rwork_size, real_type{});

    matrix_scalar work_query{};
    uni20::lapack::heev(jobz, uplo, order, matrix->data, matrix->leading_dimension, values->data, &work_query, -1,
                        rwork.data());
    blas_int const lwork = lapack_detail::workspace_size(work_query);
    std::vector<matrix_scalar> work(static_cast<std::size_t>(lwork), matrix_scalar{});
    uni20::lapack::heev(jobz, uplo, order, matrix->data, matrix->leading_dimension, values->data, work.data(), lwork,
                        rwork.data());
  }

  return KernelAttempt::success;
}

} // namespace uni20::linalg
