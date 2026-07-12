#pragma once

/**
 * \file blas_matrix.hpp
 * \ingroup linalg
 * \brief Provider-ready BLAS matrix operands and transform helpers.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>

#include <utility>

namespace uni20::linalg::blas
{

/// \brief Logical matrix transform independent of a provider-specific opcode,
///        although it happens to coincide with the OpenBLAS cblas ordering.
/// \details The underlying values are the two-bit transpose/conjugate mask in
///          `N`, `T`, `R`, `C` order.
enum class MatrixTransform : unsigned
{
  normal = 0U,
  transpose = 1U,
  conjugate = 2U,
  conjugate_transpose = 3U
};

namespace detail
{
constexpr unsigned transform_mask = std::to_underlying(MatrixTransform::conjugate_transpose);
constexpr unsigned transpose_bit = std::to_underlying(MatrixTransform::transpose);
constexpr unsigned conjugate_bit = std::to_underlying(MatrixTransform::conjugate);

constexpr void require_valid_transform(MatrixTransform transform)
{
  if (std::to_underlying(transform) > transform_mask)
  {
    PANIC("invalid MatrixTransform", std::to_underlying(transform));
  }
}
} // namespace detail

/// \brief Return whether a matrix transform swaps the logical axes.
constexpr bool swaps_axes(MatrixTransform transform)
{
  return (std::to_underlying(transform) & detail::transpose_bit) != 0U;
}

/// \brief Return whether a matrix transform applies element-wise conjugation.
constexpr bool conjugates_values(MatrixTransform transform)
{
  return (std::to_underlying(transform) & detail::conjugate_bit) != 0U;
}

/// \brief Compose two matrix transforms as `outer(inner(A))`.
constexpr MatrixTransform compose(MatrixTransform outer, MatrixTransform inner)
{
  return static_cast<MatrixTransform>((std::to_underlying(outer) ^ std::to_underlying(inner)) & detail::transform_mask);
}

/// \brief Transform induced on each GEMM operand by transposing the result.
constexpr MatrixTransform transpose_result_transform(MatrixTransform transform)
{
  return compose(MatrixTransform::transpose, transform);
}

/// \brief Collapse conjugation-only distinctions for real scalar types.
template <uni20::BlasScalar Scalar> constexpr MatrixTransform canonical_transform_for_scalar(MatrixTransform transform)
{
  detail::require_valid_transform(transform);
  if constexpr (uni20::is_complex_v<Scalar>)
  {
    return transform;
  }
  else
  {
    return static_cast<MatrixTransform>(std::to_underlying(transform) & detail::transpose_bit);
  }
}

/// \brief Return whether the current baseline provider path accepts a transform directly.
template <uni20::BlasScalar Scalar> constexpr bool blas_trans_char_is_supported(MatrixTransform transform)
{
  switch (canonical_transform_for_scalar<Scalar>(transform))
  {
    case MatrixTransform::normal:
    case MatrixTransform::transpose:
    case MatrixTransform::conjugate_transpose:
      return true;
    case MatrixTransform::conjugate:
      return false;
  }
  PANIC("invalid canonical MatrixTransform", std::to_underlying(transform));
}

/// \brief Lower a transform to the BLAS transpose character spelling used by selected providers.
template <uni20::BlasScalar Scalar> constexpr char blas_trans_char(MatrixTransform transform)
{
  switch (canonical_transform_for_scalar<Scalar>(transform))
  {
    case MatrixTransform::normal:
      return 'N';
    case MatrixTransform::transpose:
      return 'T';
    case MatrixTransform::conjugate:
      return 'R';
    case MatrixTransform::conjugate_transpose:
      return 'C';
  }
  PANIC("invalid canonical MatrixTransform", std::to_underlying(transform));
}

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
