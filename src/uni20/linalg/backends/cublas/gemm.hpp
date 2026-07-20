#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief cuBLAS backend adapter for provider-ready GEMM dispatch.
 */

#include <uni20/linalg/cublas/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>

#include <concepts>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for CUDA mdspan GEMM lowering.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
consteval auto kernel_accepts_types(CublasBackend const&, gemm_op const&, OutputMdspan&, Scalar const&, LhsMdspan&,
                                    RhsMdspan&, Scalar const&)
{
  if constexpr (requires(OutputMdspan& output, LhsMdspan& lhs, RhsMdspan& rhs, Scalar scalar) {
                  { uni20::linalg::cublas::try_gemm(output, scalar, lhs, rhs, scalar) } -> std::same_as<KernelAttempt>;
                })
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Lower CUDA mdspans, acquire execution resources, and enqueue cuBLAS GEMM.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
KernelAttempt try_kernel(CublasBackend, gemm_op const&, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs,
                         RhsMdspan&& rhs, Scalar beta)
{
  return uni20::linalg::cublas::try_gemm(std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs),
                                         std::forward<RhsMdspan>(rhs), beta);
}

} // namespace uni20::linalg
