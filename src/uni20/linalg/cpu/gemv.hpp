#pragma once

/**
 * \file gemv.hpp
 * \ingroup linalg
 * \brief Accessor-respecting reference CPU GEMV for resolved mdspans.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <concepts>
#include <type_traits>

namespace uni20::linalg::cpu
{

/// \brief Report whether resolved mdspans support reference CPU GEMV expressions.
template <class OutputMdspan, class Scalar, class MatrixMdspan, class InputMdspan>
concept GemvCompatible =
    uni20::MutableRankedMdspanLike<OutputMdspan, 1> && uni20::Scalar<Scalar> &&
    uni20::RankedMdspanLike<MatrixMdspan, 2> && uni20::RankedMdspanLike<InputMdspan, 1> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<OutputMdspan>::element_type>, Scalar> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<MatrixMdspan>::element_type>, Scalar> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<InputMdspan>::element_type>, Scalar> &&
    requires(OutputMdspan& output, MatrixMdspan& matrix, InputMdspan& input,
             typename std::remove_cvref_t<OutputMdspan>::index_type index, Scalar value) {
      static_cast<Scalar>(output.operator[](index));
      static_cast<Scalar>(matrix.operator[](index, index));
      static_cast<Scalar>(input.operator[](index));
      output.operator[](index) = value;
      value += value * value;
      { value == Scalar{} } -> std::convertible_to<bool>;
    };

/// \brief Execute reference CPU GEMV on resolved mdspans.
template <class OutputMdspan, uni20::Scalar Scalar, class MatrixMdspan, class InputMdspan>
  requires GemvCompatible<OutputMdspan, Scalar, MatrixMdspan, InputMdspan>
void gemv(OutputMdspan&& output, Scalar alpha, MatrixMdspan&& matrix, InputMdspan&& input, Scalar beta)
{
  using output_type = std::remove_cvref_t<OutputMdspan>;
  using index_type = typename output_type::index_type;

  CHECK_EQUAL(matrix.extent(1), input.extent(0));
  CHECK_EQUAL(output.extent(0), matrix.extent(0));

  index_type const rows = static_cast<index_type>(output.extent(0));
  index_type const inner = static_cast<index_type>(input.extent(0));

  for (index_type row = 0; row < rows; ++row)
  {
    Scalar product{};
    if (alpha != Scalar{})
    {
      for (index_type col = 0; col < inner; ++col)
      {
        product += static_cast<Scalar>(matrix[row, col]) * static_cast<Scalar>(input[col]);
      }
    }

    if (beta == Scalar{})
    {
      output[row] = alpha * product;
    }
    else
    {
      output[row] = beta * static_cast<Scalar>(output[row]) + alpha * product;
    }
  }
}

} // namespace uni20::linalg::cpu
