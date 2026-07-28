#pragma once

/**
 * \file svd.hpp
 * \ingroup linalg
 * \brief LAPACK backend for exact dense singular value decompositions.
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
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

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
template <class Operation>
concept SvdOperation =
    std::same_as<std::remove_cvref_t<Operation>, singular_values_op> ||
    std::same_as<std::remove_cvref_t<Operation>, svd_left_op> ||
    std::same_as<std::remove_cvref_t<Operation>, svd_right_op> || std::same_as<std::remove_cvref_t<Operation>, svd_op>;

inline char svd_job(SvdVectorExtent extent)
{
  switch (extent)
  {
    case SvdVectorExtent::Reduced:
      return 'S';
    case SvdVectorExtent::Full:
      return 'A';
  }
  PANIC("invalid SvdVectorExtent", std::to_underlying(extent));
}

template <class MatrixMdspan> void set_identity(MatrixMdspan&& matrix)
{
  using scalar_type = std::remove_cv_t<typename std::remove_cvref_t<MatrixMdspan>::element_type>;
  for (std::size_t row = 0; row < static_cast<std::size_t>(matrix.extent(0)); ++row)
  {
    for (std::size_t column = 0; column < static_cast<std::size_t>(matrix.extent(1)); ++column)
    {
      matrix[row, column] = row == column ? scalar_type{1} : scalar_type{};
    }
  }
}

template <class SingularValueMdspan, class MatrixMdspan> consteval bool svd_base_types_supported()
{
  using matrix_scalar = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using singular_value_scalar = std::remove_cv_t<typename SingularValueMdspan::element_type>;
  return uni20::LapackScalar<matrix_scalar> && uni20::LapackReal<singular_value_scalar> &&
         std::same_as<singular_value_scalar, uni20::make_real_t<matrix_scalar>> &&
         uni20::DefaultAccessorMdspanLike<SingularValueMdspan> && uni20::DefaultAccessorMdspanLike<MatrixMdspan>;
}

template <class FactorMdspan, class MatrixMdspan> consteval bool svd_factor_types_supported()
{
  using matrix_scalar = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using factor_scalar = std::remove_cv_t<typename FactorMdspan::element_type>;
  return std::same_as<factor_scalar, matrix_scalar> && uni20::DefaultAccessorMdspanLike<FactorMdspan>;
}

template <uni20::LapackScalar Scalar, class MatrixHandle, class ValueHandle>
KernelAttempt run_gesvd(char jobu, char jobvt, std::size_t rank,
                        uni20::linalg::blas::BlasWritableMatrix<Scalar, MatrixHandle> const& matrix,
                        uni20::linalg::blas::BlasWritableVector<uni20::make_real_t<Scalar>, ValueHandle> const& values,
                        Scalar* left, blas_int left_leading_dimension, Scalar* right, blas_int right_leading_dimension)
{
  using real_type = uni20::make_real_t<Scalar>;
  Scalar work_query{};
  if constexpr (uni20::LapackReal<Scalar>)
  {
    uni20::lapack::gesvd(jobu, jobvt, matrix.rows, matrix.cols, matrix.data, matrix.leading_dimension, values.data,
                         left, left_leading_dimension, right, right_leading_dimension, &work_query, -1);
    blas_int const lwork = workspace_size(work_query);
    std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
    uni20::lapack::gesvd(jobu, jobvt, matrix.rows, matrix.cols, matrix.data, matrix.leading_dimension, values.data,
                         left, left_leading_dimension, right, right_leading_dimension, work.data(), lwork);
  }
  else
  {
    auto const rwork_size = try_size_product(5, rank);
    if (!rwork_size)
    {
      return KernelAttempt::unsupported_shape;
    }
    std::vector<real_type> rwork(std::max<std::size_t>(1, *rwork_size), real_type{});
    uni20::lapack::gesvd(jobu, jobvt, matrix.rows, matrix.cols, matrix.data, matrix.leading_dimension, values.data,
                         left, left_leading_dimension, right, right_leading_dimension, &work_query, -1, rwork.data());
    blas_int const lwork = workspace_size(work_query);
    std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
    uni20::lapack::gesvd(jobu, jobvt, matrix.rows, matrix.cols, matrix.data, matrix.leading_dimension, values.data,
                         left, left_leading_dimension, right, right_leading_dimension, work.data(), lwork,
                         rwork.data());
  }
  return KernelAttempt::success;
}

template <class SingularValueMdspan, class MatrixMdspan>
KernelAttempt try_singular_values_kernel(char jobu, char jobvt, SingularValueMdspan&& singular_values,
                                         MatrixMdspan&& matrix_work)
{
  using matrix_scalar = std::remove_cv_t<typename std::remove_cvref_t<MatrixMdspan>::element_type>;
  std::size_t const rows = static_cast<std::size_t>(matrix_work.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix_work.extent(1));
  std::size_t const rank = std::min(rows, cols);
  CHECK_EQUAL(static_cast<std::size_t>(singular_values.extent(0)), rank);
  if (rank == 0)
  {
    return KernelAttempt::success;
  }

  auto matrix = uni20::linalg::blas::try_lapack_writable_matrix(matrix_work);
  auto values = uni20::linalg::blas::try_lapack_writable_vector(singular_values);
  if (!matrix || !values)
  {
    return KernelAttempt::unsupported_layout;
  }

  blas_int const k = uni20::blas::try_blas_int(rank);
  if (!uni20::blas::is_valid_blas_int(k))
  {
    return KernelAttempt::unsupported_shape;
  }
  CHECK_EQUAL(values->size, k);
  return run_gesvd(jobu, jobvt, rank, *matrix, *values, static_cast<matrix_scalar*>(matrix->data), 1,
                   static_cast<matrix_scalar*>(matrix->data), 1);
}
} // namespace lapack_detail

/// \brief Report compile-time eligibility for LAPACK singular values.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, singular_values_op const&, SingularValueMdspan&,
                                    MatrixMdspan&)
{
  if constexpr (lapack_detail::svd_base_types_supported<SingularValueMdspan, MatrixMdspan>())
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute exact singular values through LAPACK `gesvd`.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
KernelAttempt try_kernel(LapackBackend, singular_values_op const&, SingularValueMdspan&& singular_values,
                         MatrixMdspan&& matrix_work)
{
  return lapack_detail::try_singular_values_kernel('N', 'N', singular_values, matrix_work);
}

/// \brief Report compile-time eligibility for LAPACK left singular vectors.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> LeftMdspan, uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, svd_left_op const&, SingularValueMdspan&, LeftMdspan&,
                                    MatrixMdspan&)
{
  if constexpr (lapack_detail::svd_base_types_supported<SingularValueMdspan, MatrixMdspan>() &&
                lapack_detail::svd_factor_types_supported<LeftMdspan, MatrixMdspan>())
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Report compile-time eligibility for input-overwriting LAPACK left singular vectors.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, svd_left_op const&, SingularValueMdspan&, MatrixMdspan&)
{
  if constexpr (lapack_detail::svd_base_types_supported<SingularValueMdspan, MatrixMdspan>())
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute singular values and left singular vectors through LAPACK `gesvd`.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> LeftMdspan, uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
KernelAttempt try_kernel(LapackBackend, svd_left_op const& op, SingularValueMdspan&& singular_values,
                         LeftMdspan&& left_singular_vectors, MatrixMdspan&& matrix_work)
{
  CHECK(!op.overwrite_input);
  using matrix_scalar = std::remove_cv_t<typename std::remove_cvref_t<MatrixMdspan>::element_type>;
  std::size_t const rows = static_cast<std::size_t>(matrix_work.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix_work.extent(1));
  std::size_t const rank = std::min(rows, cols);
  std::size_t const left_cols = op.left == SvdVectorExtent::Full ? rows : rank;
  CHECK_EQUAL(static_cast<std::size_t>(singular_values.extent(0)), rank);
  CHECK_EQUAL(static_cast<std::size_t>(left_singular_vectors.extent(0)), rows);
  CHECK_EQUAL(static_cast<std::size_t>(left_singular_vectors.extent(1)), left_cols);
  if (rank == 0)
  {
    lapack_detail::set_identity(left_singular_vectors);
    return KernelAttempt::success;
  }

  auto matrix = uni20::linalg::blas::try_lapack_writable_matrix(matrix_work);
  auto values = uni20::linalg::blas::try_lapack_writable_vector(singular_values);
  auto left = uni20::linalg::blas::try_lapack_writable_matrix(left_singular_vectors);
  if (!matrix || !values || !left)
  {
    return KernelAttempt::unsupported_layout;
  }

  blas_int const k = uni20::blas::try_blas_int(rank);
  blas_int const left_columns = uni20::blas::try_blas_int(left_cols);
  if (!uni20::blas::is_valid_blas_int(k) || !uni20::blas::is_valid_blas_int(left_columns))
  {
    return KernelAttempt::unsupported_shape;
  }
  CHECK_EQUAL(values->size, k);
  CHECK_EQUAL(left->rows, matrix->rows);
  CHECK_EQUAL(left->cols, left_columns);
  return lapack_detail::run_gesvd(lapack_detail::svd_job(op.left), 'N', rank, *matrix, *values, left->data,
                                  left->leading_dimension, static_cast<matrix_scalar*>(matrix->data), 1);
}

/// \brief Compute reduced left singular vectors in the matrix input allocation.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
KernelAttempt try_kernel(LapackBackend, svd_left_op const& op, SingularValueMdspan&& singular_values,
                         MatrixMdspan&& matrix_work)
{
  CHECK(op.overwrite_input);
  CHECK(op.left == SvdVectorExtent::Reduced);
  return lapack_detail::try_singular_values_kernel('O', 'N', singular_values, matrix_work);
}

/// \brief Report compile-time eligibility for LAPACK right singular vectors.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> RightAdjointMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, svd_right_op const&, SingularValueMdspan&,
                                    RightAdjointMdspan&, MatrixMdspan&)
{
  if constexpr (lapack_detail::svd_base_types_supported<SingularValueMdspan, MatrixMdspan>() &&
                lapack_detail::svd_factor_types_supported<RightAdjointMdspan, MatrixMdspan>())
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Report compile-time eligibility for input-overwriting LAPACK right singular vectors.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, svd_right_op const&, SingularValueMdspan&, MatrixMdspan&)
{
  if constexpr (lapack_detail::svd_base_types_supported<SingularValueMdspan, MatrixMdspan>())
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute singular values and right singular vectors through LAPACK `gesvd`.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> RightAdjointMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
KernelAttempt try_kernel(LapackBackend, svd_right_op const& op, SingularValueMdspan&& singular_values,
                         RightAdjointMdspan&& right_singular_vectors_adjoint, MatrixMdspan&& matrix_work)
{
  CHECK(!op.overwrite_input);
  using matrix_scalar = std::remove_cv_t<typename std::remove_cvref_t<MatrixMdspan>::element_type>;
  std::size_t const rows = static_cast<std::size_t>(matrix_work.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix_work.extent(1));
  std::size_t const rank = std::min(rows, cols);
  std::size_t const right_rows = op.right == SvdVectorExtent::Full ? cols : rank;
  CHECK_EQUAL(static_cast<std::size_t>(singular_values.extent(0)), rank);
  CHECK_EQUAL(static_cast<std::size_t>(right_singular_vectors_adjoint.extent(0)), right_rows);
  CHECK_EQUAL(static_cast<std::size_t>(right_singular_vectors_adjoint.extent(1)), cols);
  if (rank == 0)
  {
    lapack_detail::set_identity(right_singular_vectors_adjoint);
    return KernelAttempt::success;
  }

  auto matrix = uni20::linalg::blas::try_lapack_writable_matrix(matrix_work);
  auto values = uni20::linalg::blas::try_lapack_writable_vector(singular_values);
  auto right = uni20::linalg::blas::try_lapack_writable_matrix(right_singular_vectors_adjoint);
  if (!matrix || !values || !right)
  {
    return KernelAttempt::unsupported_layout;
  }

  blas_int const k = uni20::blas::try_blas_int(rank);
  blas_int const right_rows_blas = uni20::blas::try_blas_int(right_rows);
  if (!uni20::blas::is_valid_blas_int(k) || !uni20::blas::is_valid_blas_int(right_rows_blas))
  {
    return KernelAttempt::unsupported_shape;
  }
  CHECK_EQUAL(values->size, k);
  CHECK_EQUAL(right->rows, right_rows_blas);
  CHECK_EQUAL(right->cols, matrix->cols);
  return lapack_detail::run_gesvd('N', lapack_detail::svd_job(op.right), rank, *matrix, *values,
                                  static_cast<matrix_scalar*>(matrix->data), 1, right->data, right->leading_dimension);
}

/// \brief Compute reduced right singular vectors in the matrix input allocation.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
KernelAttempt try_kernel(LapackBackend, svd_right_op const& op, SingularValueMdspan&& singular_values,
                         MatrixMdspan&& matrix_work)
{
  CHECK(op.overwrite_input);
  CHECK(op.right == SvdVectorExtent::Reduced);
  return lapack_detail::try_singular_values_kernel('N', 'O', singular_values, matrix_work);
}

/// \brief Report compile-time eligibility for LAPACK exact SVD.
template <
    uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan, uni20::MutableRankedStridedMdspanLike<2> LeftMdspan,
    uni20::MutableRankedStridedMdspanLike<2> RightAdjointMdspan, uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, svd_op const&, SingularValueMdspan&, LeftMdspan&,
                                    RightAdjointMdspan&, MatrixMdspan&)
{
  if constexpr (lapack_detail::svd_base_types_supported<SingularValueMdspan, MatrixMdspan>() &&
                lapack_detail::svd_factor_types_supported<LeftMdspan, MatrixMdspan>() &&
                lapack_detail::svd_factor_types_supported<RightAdjointMdspan, MatrixMdspan>())
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Report compile-time eligibility for an input-overwriting LAPACK exact SVD.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> OtherFactorMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, svd_op const&, SingularValueMdspan&, OtherFactorMdspan&,
                                    MatrixMdspan&)
{
  if constexpr (lapack_detail::svd_base_types_supported<SingularValueMdspan, MatrixMdspan>() &&
                lapack_detail::svd_factor_types_supported<OtherFactorMdspan, MatrixMdspan>())
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute an exact dense SVD through LAPACK `gesvd`.
/// \details `matrix_work` is destroyed. The returned right factor is the
///          transpose for real scalars and the conjugate transpose for complex
///          scalars, matching LAPACK's `VT` output.
template <
    uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan, uni20::MutableRankedStridedMdspanLike<2> LeftMdspan,
    uni20::MutableRankedStridedMdspanLike<2> RightAdjointMdspan, uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
KernelAttempt try_kernel(LapackBackend, svd_op const& op, SingularValueMdspan&& singular_values,
                         LeftMdspan&& left_singular_vectors, RightAdjointMdspan&& right_singular_vectors_adjoint,
                         MatrixMdspan&& matrix_work)
{
  CHECK(op.overwrite == SvdOverwrite::None);
  std::size_t const rows = static_cast<std::size_t>(matrix_work.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix_work.extent(1));
  std::size_t const rank = std::min(rows, cols);
  std::size_t const left_cols = op.left == SvdVectorExtent::Full ? rows : rank;
  std::size_t const right_rows = op.right == SvdVectorExtent::Full ? cols : rank;

  CHECK_EQUAL(static_cast<std::size_t>(singular_values.extent(0)), rank);
  CHECK_EQUAL(static_cast<std::size_t>(left_singular_vectors.extent(0)), rows);
  CHECK_EQUAL(static_cast<std::size_t>(left_singular_vectors.extent(1)), left_cols);
  CHECK_EQUAL(static_cast<std::size_t>(right_singular_vectors_adjoint.extent(0)), right_rows);
  CHECK_EQUAL(static_cast<std::size_t>(right_singular_vectors_adjoint.extent(1)), cols);
  if (rank == 0)
  {
    lapack_detail::set_identity(left_singular_vectors);
    lapack_detail::set_identity(right_singular_vectors_adjoint);
    return KernelAttempt::success;
  }

  auto matrix = uni20::linalg::blas::try_lapack_writable_matrix(matrix_work);
  auto values = uni20::linalg::blas::try_lapack_writable_vector(singular_values);
  auto left = uni20::linalg::blas::try_lapack_writable_matrix(left_singular_vectors);
  auto right = uni20::linalg::blas::try_lapack_writable_matrix(right_singular_vectors_adjoint);
  if (!matrix || !values || !left || !right)
  {
    return KernelAttempt::unsupported_layout;
  }

  blas_int const k = uni20::blas::try_blas_int(rank);
  blas_int const left_columns = uni20::blas::try_blas_int(left_cols);
  blas_int const right_rows_blas = uni20::blas::try_blas_int(right_rows);
  if (!uni20::blas::is_valid_blas_int(k) || !uni20::blas::is_valid_blas_int(left_columns) ||
      !uni20::blas::is_valid_blas_int(right_rows_blas))
  {
    return KernelAttempt::unsupported_shape;
  }

  CHECK_EQUAL(values->size, k);
  CHECK_EQUAL(left->rows, matrix->rows);
  CHECK_EQUAL(left->cols, left_columns);
  CHECK_EQUAL(right->rows, right_rows_blas);
  CHECK_EQUAL(right->cols, matrix->cols);
  return lapack_detail::run_gesvd(lapack_detail::svd_job(op.left), lapack_detail::svd_job(op.right), rank, *matrix,
                                  *values, left->data, left->leading_dimension, right->data, right->leading_dimension);
}

/// \brief Compute an exact dense SVD with one reduced factor overwriting the input allocation.
template <uni20::MutableRankedStridedMdspanLike<1> SingularValueMdspan,
          uni20::MutableRankedStridedMdspanLike<2> OtherFactorMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
KernelAttempt try_kernel(LapackBackend, svd_op const& op, SingularValueMdspan&& singular_values,
                         OtherFactorMdspan&& other_factor, MatrixMdspan&& matrix_work)
{
  using matrix_scalar = std::remove_cv_t<typename std::remove_cvref_t<MatrixMdspan>::element_type>;
  std::size_t const rows = static_cast<std::size_t>(matrix_work.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix_work.extent(1));
  std::size_t const rank = std::min(rows, cols);
  CHECK_EQUAL(static_cast<std::size_t>(singular_values.extent(0)), rank);

  char jobu = 'N';
  char jobvt = 'N';
  bool const overwrite_left = op.overwrite == SvdOverwrite::Left;
  if (overwrite_left)
  {
    CHECK(op.left == SvdVectorExtent::Reduced);
    std::size_t const right_rows = op.right == SvdVectorExtent::Full ? cols : rank;
    CHECK_EQUAL(static_cast<std::size_t>(other_factor.extent(0)), right_rows);
    CHECK_EQUAL(static_cast<std::size_t>(other_factor.extent(1)), cols);
    jobu = 'O';
    jobvt = lapack_detail::svd_job(op.right);
  }
  else
  {
    CHECK(op.overwrite == SvdOverwrite::Right);
    CHECK(op.right == SvdVectorExtent::Reduced);
    std::size_t const left_cols = op.left == SvdVectorExtent::Full ? rows : rank;
    CHECK_EQUAL(static_cast<std::size_t>(other_factor.extent(0)), rows);
    CHECK_EQUAL(static_cast<std::size_t>(other_factor.extent(1)), left_cols);
    jobu = lapack_detail::svd_job(op.left);
    jobvt = 'O';
  }

  if (rank == 0)
  {
    lapack_detail::set_identity(other_factor);
    return KernelAttempt::success;
  }

  auto matrix = uni20::linalg::blas::try_lapack_writable_matrix(matrix_work);
  auto values = uni20::linalg::blas::try_lapack_writable_vector(singular_values);
  auto other = uni20::linalg::blas::try_lapack_writable_matrix(other_factor);
  if (!matrix || !values || !other)
  {
    return KernelAttempt::unsupported_layout;
  }

  blas_int const k = uni20::blas::try_blas_int(rank);
  if (!uni20::blas::is_valid_blas_int(k))
  {
    return KernelAttempt::unsupported_shape;
  }
  CHECK_EQUAL(values->size, k);

  matrix_scalar* left = static_cast<matrix_scalar*>(matrix->data);
  blas_int left_leading_dimension = 1;
  matrix_scalar* right = static_cast<matrix_scalar*>(matrix->data);
  blas_int right_leading_dimension = 1;
  if (overwrite_left)
  {
    CHECK_EQUAL(other->cols, matrix->cols);
    right = other->data;
    right_leading_dimension = other->leading_dimension;
  }
  else
  {
    CHECK_EQUAL(other->rows, matrix->rows);
    left = other->data;
    left_leading_dimension = other->leading_dimension;
  }
  return lapack_detail::run_gesvd(jobu, jobvt, rank, *matrix, *values, left, left_leading_dimension, right,
                                  right_leading_dimension);
}

/// \brief Report eligibility for blocking DeviceTensorView SVD operands.
template <lapack_detail::SvdOperation Operation, uni20::MutableDeviceTensorView... Tensors>
  requires(uni20::detail::BlockingWritableTensor<Tensors> && ...)
consteval auto kernel_accepts_types(LapackBackend const&, Operation const&, Tensors&...)
{
  constexpr auto acceptance =
      detail::backend_type_acceptance<LapackBackend, Operation,
                                      uni20::detail::blocking_write_tensor_mdspan_t<Tensors>&...>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve blocking tensor access and run a LAPACK SVD operation.
template <lapack_detail::SvdOperation Operation, uni20::MutableDeviceTensorView... Tensors>
  requires(uni20::detail::BlockingWritableTensor<Tensors> && ...)
KernelAttempt try_kernel(LapackBackend backend, Operation const& operation, Tensors&... tensors)
{
  return uni20::detail::with_blocking_write_tensor_mdspans(
      [&](auto&... spans) { return try_kernel(backend, operation, spans...); }, tensors...);
}

} // namespace uni20::linalg
