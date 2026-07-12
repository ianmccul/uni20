#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Generic CPU GEMM backend for mdspan-like dense matrices.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <concepts>
#include <type_traits>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for generic CPU GEMM dispatch.
template <uni20::MutableRankedSpanLike<2> OutputMdspan, uni20::Scalar Scalar, uni20::RankedSpanLike<2> LhsMdspan,
          uni20::RankedSpanLike<2> RhsMdspan>
consteval auto kernel_accepts_types(CpuGenericBackend const&, struct gemm_op const&, OutputMdspan&, Scalar const&,
                                    LhsMdspan&, RhsMdspan&, Scalar const&)
{
  using output_scalar = std::remove_cv_t<typename OutputMdspan::element_type>;
  using lhs_scalar = std::remove_cv_t<typename LhsMdspan::element_type>;
  using rhs_scalar = std::remove_cv_t<typename RhsMdspan::element_type>;
  if constexpr (std::same_as<output_scalar, Scalar> && std::same_as<lhs_scalar, Scalar> &&
                std::same_as<rhs_scalar, Scalar>)
  {
    return kernel_types_yes;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Generic accessor-respecting GEMM fallback.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
bool try_kernel(CpuGenericBackend, struct gemm_op const&, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs,
                RhsMdspan&& rhs, Scalar beta)
{
  using output_type = std::remove_cvref_t<OutputMdspan>;
  using index_type = typename output_type::index_type;

  CHECK_EQUAL(lhs.extent(1), rhs.extent(0));
  CHECK_EQUAL(output.extent(0), lhs.extent(0));
  CHECK_EQUAL(output.extent(1), rhs.extent(1));

  index_type const rows = static_cast<index_type>(output.extent(0));
  index_type const cols = static_cast<index_type>(output.extent(1));
  index_type const inner = static_cast<index_type>(lhs.extent(1));

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

  return true;
}

} // namespace uni20::linalg
