#pragma once

/**
 * \file gemv.hpp
 * \ingroup linalg
 * \brief Reference CPU GEMV backend for mdspan-like dense operands.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <concepts>
#include <type_traits>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for reference CPU GEMV dispatch.
template <uni20::MutableRankedMdspanLike<1> OutputMdspan, uni20::Scalar Scalar, uni20::RankedMdspanLike<2> MatrixMdspan,
          uni20::RankedMdspanLike<1> InputMdspan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, gemv_op const&, OutputMdspan&, Scalar const&,
                                    MatrixMdspan&, InputMdspan&, Scalar const&)
{
  using output_scalar = std::remove_cv_t<typename OutputMdspan::element_type>;
  using matrix_scalar = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using input_scalar = std::remove_cv_t<typename InputMdspan::element_type>;
  if constexpr (std::same_as<output_scalar, Scalar> && std::same_as<matrix_scalar, Scalar> &&
                std::same_as<input_scalar, Scalar> &&
                requires(OutputMdspan& output, MatrixMdspan& matrix, InputMdspan& input,
                         typename OutputMdspan::index_type index, Scalar value) {
                  static_cast<Scalar>(output.operator[](index));
                  static_cast<Scalar>(matrix.operator[](index, index));
                  static_cast<Scalar>(input.operator[](index));
                  output.operator[](index) = value;
                  value += value * value;
                  {
                    value == Scalar {}
                  } -> std::convertible_to<bool>;
                })
  {
    return kernel_types_yes;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Reference accessor-respecting GEMV fallback.
template <class OutputMdspan, class Scalar, class MatrixMdspan, class InputMdspan>
KernelAttempt try_kernel(CpuReferenceBackend, gemv_op const&, OutputMdspan&& output, Scalar alpha,
                         MatrixMdspan&& matrix, InputMdspan&& input, Scalar beta)
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

  return KernelAttempt::success;
}

} // namespace uni20::linalg
