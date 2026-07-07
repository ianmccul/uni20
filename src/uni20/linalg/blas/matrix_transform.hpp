#pragma once

/**
 * \file matrix_transform.hpp
 * \ingroup linalg
 * \brief Matrix transform algebra used by BLAS-compatible linalg adapters.
 */

#include <uni20/core/scalar_traits.hpp>

#include <optional>

namespace uni20::linalg::blas
{

/// \brief Logical matrix transform independent of a provider-specific opcode.
enum class MatrixTransform
{
  normal,
  transpose,
  conjugate_transpose,
  conjugate
};

namespace detail
{
struct transform_bits
{
    bool transpose = false;
    bool conjugate = false;
};

constexpr transform_bits matrix_transform_bits(MatrixTransform transform)
{
  switch (transform)
  {
    case MatrixTransform::normal:
      return {.transpose = false, .conjugate = false};
    case MatrixTransform::transpose:
      return {.transpose = true, .conjugate = false};
    case MatrixTransform::conjugate_transpose:
      return {.transpose = true, .conjugate = true};
    case MatrixTransform::conjugate:
      return {.transpose = false, .conjugate = true};
  }
  return {};
}

constexpr MatrixTransform matrix_transform_from_bits(transform_bits bits)
{
  if (bits.transpose && bits.conjugate)
  {
    return MatrixTransform::conjugate_transpose;
  }
  if (bits.transpose)
  {
    return MatrixTransform::transpose;
  }
  if (bits.conjugate)
  {
    return MatrixTransform::conjugate;
  }
  return MatrixTransform::normal;
}
} // namespace detail

/// \brief Return whether a matrix transform swaps the logical axes.
constexpr bool swaps_axes(MatrixTransform transform) { return detail::matrix_transform_bits(transform).transpose; }

/// \brief Return whether a matrix transform applies element-wise conjugation.
constexpr bool conjugates_values(MatrixTransform transform)
{
  return detail::matrix_transform_bits(transform).conjugate;
}

/// \brief Compose two matrix transforms as `outer(inner(A))`.
constexpr MatrixTransform compose(MatrixTransform outer, MatrixTransform inner)
{
  auto const outer_bits = detail::matrix_transform_bits(outer);
  auto const inner_bits = detail::matrix_transform_bits(inner);
  return detail::matrix_transform_from_bits({.transpose = outer_bits.transpose != inner_bits.transpose,
                                             .conjugate = outer_bits.conjugate != inner_bits.conjugate});
}

/// \brief Transform induced on each GEMM operand by transposing the result.
constexpr MatrixTransform transpose_result_transform(MatrixTransform transform)
{
  return compose(MatrixTransform::transpose, transform);
}

/// \brief Collapse conjugation-only distinctions for real scalar types.
template <typename Scalar> constexpr MatrixTransform canonical_transform_for_scalar(MatrixTransform transform)
{
  if constexpr (uni20::is_complex_v<Scalar>)
  {
    return transform;
  }
  else
  {
    auto bits = detail::matrix_transform_bits(transform);
    bits.conjugate = false;
    return detail::matrix_transform_from_bits(bits);
  }
}

/// \brief Lower a transform to a standard Fortran BLAS transpose character.
template <typename Scalar> constexpr std::optional<char> standard_blas_trans_char(MatrixTransform transform)
{
  switch (canonical_transform_for_scalar<Scalar>(transform))
  {
    case MatrixTransform::normal:
      return 'N';
    case MatrixTransform::transpose:
      return 'T';
    case MatrixTransform::conjugate_transpose:
      return 'C';
    case MatrixTransform::conjugate:
      return std::nullopt;
  }
  return std::nullopt;
}

} // namespace uni20::linalg::blas
