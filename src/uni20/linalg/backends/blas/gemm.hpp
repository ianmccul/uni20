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

/// \brief Backend value for BLAS dense linalg kernels.
struct BlasBackend
{};

namespace detail
{
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
concept blas_gemm_types_supported = requires(OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs) {
  {
    uni20::linalg::blas::try_gemm(std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs),
                                  std::forward<RhsMdspan>(rhs), alpha)
  } -> std::same_as<bool>;
};
} // namespace detail

/// \brief Report compile-time eligibility for direct BLAS GEMM dispatch.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
consteval KernelTypeAcceptance kernel_accepts_types(BlasBackend const&, struct gemm_op const&, OutputMdspan&,
                                                    Scalar const&, LhsMdspan&, RhsMdspan&, Scalar const&)
{
  if constexpr (detail::blas_gemm_types_supported<OutputMdspan&, Scalar, LhsMdspan&, RhsMdspan&>)
  {
    return KernelTypeAcceptance::maybe;
  }
  else
  {
    return KernelTypeAcceptance::no;
  }
}

/// \brief Try GEMM through the direct mdspan BLAS wrapper.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
bool try_kernel(BlasBackend, struct gemm_op const&, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs,
                RhsMdspan&& rhs, Scalar beta)
{
  static_assert(detail::blas_gemm_types_supported<OutputMdspan&&, Scalar, LhsMdspan&&, RhsMdspan&&>,
                "BLAS GEMM try_kernel called for types rejected by kernel_accepts_types");
  return uni20::linalg::blas::try_gemm(std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs),
                                       std::forward<RhsMdspan>(rhs), beta);
}

} // namespace uni20::linalg
