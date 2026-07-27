#pragma once

/**
 * \file matrix_set.hpp
 * \ingroup linalg
 * \brief Reference CPU backend for structured dense matrix initialization.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for reference CPU matrix initialization.
template <uni20::MutableRankedMdspanLike<2> MatrixMdspan, uni20::Scalar Scalar>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, matrix_set_op const&, MatrixMdspan&, Scalar const&,
                                    Scalar const&)
{
  using matrix_scalar = std::remove_cv_t<typename MatrixMdspan::element_type>;
  if constexpr (std::same_as<matrix_scalar, Scalar> &&
                requires(MatrixMdspan& matrix, typename MatrixMdspan::index_type index, Scalar value) {
                  matrix.operator[](index, index) = value;
                })
  {
    return kernel_types_yes;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Initialize a selected matrix region through its accessor semantics.
template <class MatrixMdspan, class Scalar>
KernelAttempt try_kernel(CpuReferenceBackend, matrix_set_op const& op, MatrixMdspan&& matrix, Scalar diagonal,
                         Scalar off_diagonal)
{
  using matrix_type = std::remove_cvref_t<MatrixMdspan>;
  using index_type = typename matrix_type::index_type;

  auto selected = [&](index_type row, index_type col) {
    switch (op.region)
    {
      case MatrixRegion::All:
        return true;
      case MatrixRegion::Upper:
        return row <= col;
      case MatrixRegion::Lower:
        return row >= col;
    }
    PANIC("invalid MatrixRegion", std::to_underlying(op.region));
  };

  for (index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (index_type col = 0; col < matrix.extent(1); ++col)
    {
      if (selected(row, col))
      {
        matrix[row, col] = row == col ? diagonal : off_diagonal;
      }
    }
  }
  return KernelAttempt::success;
}

} // namespace uni20::linalg
