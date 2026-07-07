#pragma once

/**
 * \file matrix_operand.hpp
 * \ingroup linalg
 * \brief Provider-ready BLAS matrix operands.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/linalg/blas/matrix_transform.hpp>

namespace uni20::linalg::blas
{

/// \brief Writable matrix shape as seen by a BLAS/LAPACK provider.
/// \details The dimensions describe the untransformed provider matrix, not
///          necessarily the logical mdspan matrix.
template <class Scalar, class Handle = Scalar*> struct BlasWritableMatrix
{
    Handle data{};
    blas_int rows = 0;
    blas_int cols = 0;
    blas_int leading_dimension = 0;
};

/// \brief Readable matrix shape plus transform as seen by a BLAS provider.
/// \details The dimensions describe the untransformed provider matrix. The
///          transform describes how BLAS should read that provider matrix.
template <class Scalar, class Handle = Scalar const*> struct BlasReadableMatrix
{
    Handle data{};
    blas_int rows = 0;
    blas_int cols = 0;
    blas_int leading_dimension = 0;
    MatrixTransform transform = MatrixTransform::normal;
};

/// \brief Rows produced by applying \p transform to an untransformed matrix.
constexpr blas_int transformed_rows(blas_int rows, blas_int cols, MatrixTransform transform)
{
  return swaps_axes(transform) ? cols : rows;
}

/// \brief Columns produced by applying \p transform to an untransformed matrix.
constexpr blas_int transformed_cols(blas_int rows, blas_int cols, MatrixTransform transform)
{
  return swaps_axes(transform) ? rows : cols;
}

} // namespace uni20::linalg::blas
