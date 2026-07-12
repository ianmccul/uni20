#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief BLAS backend adapter for operation-tag GEMM dispatch.
 */

#include <uni20/linalg/blas/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>

#include <concepts>
#include <utility>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for direct BLAS GEMM dispatch.
template <uni20::MutableRankedStridedMdspan<2> OutputMdspan, class Scalar, uni20::RankedStridedMdspan<2> LhsMdspan,
          uni20::RankedStridedMdspan<2> RhsMdspan>
consteval auto kernel_accepts_types(BlasBackend const&, struct gemm_op const&, OutputMdspan&, Scalar const&, LhsMdspan&,
                                    RhsMdspan&, Scalar const&)
{
  if constexpr (requires(OutputMdspan& output, Scalar alpha, LhsMdspan& lhs, RhsMdspan& rhs) {
                  { uni20::linalg::blas::try_gemm(output, alpha, lhs, rhs, alpha) } -> std::same_as<bool>;
                })
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Try GEMM through the direct mdspan BLAS wrapper.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
bool try_kernel(BlasBackend, struct gemm_op const&, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs,
                RhsMdspan&& rhs, Scalar beta)
{
  return uni20::linalg::blas::try_gemm(std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs),
                                       std::forward<RhsMdspan>(rhs), beta);
}

} // namespace uni20::linalg
