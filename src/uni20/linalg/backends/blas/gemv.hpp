#pragma once

/**
 * \file gemv.hpp
 * \ingroup linalg
 * \brief BLAS backend adapter for operation-tag GEMV dispatch.
 */

#include <uni20/linalg/blas/gemv.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>

#include <concepts>
#include <utility>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for direct BLAS GEMV dispatch.
template <uni20::MutableRankedStridedMdspan<1> OutputMdspan, class Scalar, uni20::RankedStridedMdspan<2> MatrixMdspan,
          uni20::RankedStridedMdspan<1> InputMdspan>
consteval auto kernel_accepts_types(BlasBackend const&, gemv_op const&, OutputMdspan&, Scalar const&, MatrixMdspan&,
                                    InputMdspan&, Scalar const&)
{
  if constexpr (requires(OutputMdspan& output, Scalar alpha, MatrixMdspan& matrix, InputMdspan& input) {
                  { uni20::linalg::blas::try_gemv(output, alpha, matrix, input, alpha) } -> std::same_as<KernelAttempt>;
                })
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Try GEMV through the direct mdspan BLAS wrapper.
template <class OutputMdspan, class Scalar, class MatrixMdspan, class InputMdspan>
KernelAttempt try_kernel(BlasBackend, gemv_op const&, OutputMdspan&& output, Scalar alpha, MatrixMdspan&& matrix,
                         InputMdspan&& input, Scalar beta)
{
  return uni20::linalg::blas::try_gemv(std::forward<OutputMdspan>(output), alpha, std::forward<MatrixMdspan>(matrix),
                                       std::forward<InputMdspan>(input), beta);
}

} // namespace uni20::linalg
