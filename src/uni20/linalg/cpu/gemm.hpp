#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Accessor-respecting reference CPU GEMM for resolved mdspans.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <concepts>
#include <type_traits>

namespace uni20::linalg::cpu
{

/// \brief Report whether resolved mdspans support reference CPU GEMM expressions.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
concept GemmCompatible =
    uni20::MutableRankedMdspanLike<OutputMdspan, 2> && uni20::Scalar<Scalar> && uni20::RankedMdspanLike<LhsMdspan, 2> &&
    uni20::RankedMdspanLike<RhsMdspan, 2> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<OutputMdspan>::element_type>, Scalar> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<LhsMdspan>::element_type>, Scalar> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<RhsMdspan>::element_type>, Scalar> &&
    requires(OutputMdspan& output, LhsMdspan& lhs, RhsMdspan& rhs,
             typename std::remove_cvref_t<OutputMdspan>::index_type index, Scalar value) {
      static_cast<Scalar>(output.operator[](index, index));
      static_cast<Scalar>(lhs.operator[](index, index));
      static_cast<Scalar>(rhs.operator[](index, index));
      output.operator[](index, index) = value;
      value += value * value;
      { value == Scalar{} } -> std::convertible_to<bool>;
    };

/// \brief Execute reference CPU GEMM on resolved mdspans.
template <class OutputMdspan, uni20::Scalar Scalar, class LhsMdspan, class RhsMdspan>
  requires GemmCompatible<OutputMdspan, Scalar, LhsMdspan, RhsMdspan>
void gemm(OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs, Scalar beta)
{
  using output_type = std::remove_cvref_t<OutputMdspan>;
  using index_type = typename output_type::index_type;

  CHECK_EQUAL(lhs.extent(1), rhs.extent(0));
  CHECK_EQUAL(output.extent(0), lhs.extent(0));
  CHECK_EQUAL(output.extent(1), rhs.extent(1));

  index_type const rows = static_cast<index_type>(output.extent(0));
  index_type const cols = static_cast<index_type>(output.extent(1));
  index_type const inner = static_cast<index_type>(lhs.extent(1));

  if (alpha == Scalar{} || inner == 0)
  {
    if (beta == Scalar{1}) return;

    for (index_type row = 0; row < rows; ++row)
    {
      for (index_type col = 0; col < cols; ++col)
      {
        if (beta == Scalar{})
          output[row, col] = Scalar{};
        else
          output[row, col] = beta * static_cast<Scalar>(output[row, col]);
      }
    }
    return;
  }

  for (index_type row = 0; row < rows; ++row)
  {
    for (index_type col = 0; col < cols; ++col)
    {
      Scalar product{};
      for (index_type k = 0; k < inner; ++k)
      {
        product += static_cast<Scalar>(lhs[row, k]) * static_cast<Scalar>(rhs[k, col]);
      }

      if (beta == Scalar{})
      {
        output[row, col] = alpha * product;
      }
      else
      {
        output[row, col] = beta * static_cast<Scalar>(output[row, col]) + alpha * product;
      }
    }
  }
}

} // namespace uni20::linalg::cpu
