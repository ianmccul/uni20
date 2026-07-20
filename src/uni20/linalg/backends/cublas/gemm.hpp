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

/// \brief Report compile-time eligibility for provider-ready cuBLAS GEMM.
template <uni20::cublas::CublasScalar Scalar, class OutputHandle, class LhsHandle, class RhsHandle>
consteval auto kernel_accepts_types(CublasBackend const&, gemm_op const&, uni20::cublas::ExecutionLease&,
                                    blas::BlasWritableMatrix<Scalar, OutputHandle>&, Scalar const&,
                                    blas::BlasReadableMatrix<Scalar, LhsHandle>&,
                                    blas::BlasReadableMatrix<Scalar, RhsHandle>&, Scalar const&)
{
  if constexpr (std::convertible_to<OutputHandle, Scalar*> && std::convertible_to<LhsHandle, Scalar const*> &&
                std::convertible_to<RhsHandle, Scalar const*>)
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Try provider-ready GEMM through an already-acquired cuBLAS execution lease.
template <class OutputMatrix, class LhsMatrix, class RhsMatrix, class Scalar>
KernelAttempt try_kernel(CublasBackend, gemm_op const&, uni20::cublas::ExecutionLease& execution, OutputMatrix&& output,
                         Scalar alpha, LhsMatrix&& lhs, RhsMatrix&& rhs, Scalar beta)
{
  return uni20::linalg::cublas::try_gemm(execution, output, lhs, rhs, alpha, beta);
}

} // namespace uni20::linalg
