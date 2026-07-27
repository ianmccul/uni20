#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief BLAS backend adapter for operation-tag GEMM dispatch.
 */

#include <uni20/linalg/blas/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <utility>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for direct BLAS GEMM dispatch.
template <uni20::MutableRankedStridedMdspan<2> OutputMdspan, class Scalar, uni20::RankedStridedMdspan<2> LhsMdspan,
          uni20::RankedStridedMdspan<2> RhsMdspan>
consteval auto kernel_accepts_types(BlasBackend const&, gemm_op const&, OutputMdspan&, Scalar const&, LhsMdspan&,
                                    RhsMdspan&, Scalar const&)
{
  if constexpr (requires(OutputMdspan& output, Scalar alpha, LhsMdspan& lhs, RhsMdspan& rhs) {
                  { uni20::linalg::blas::try_gemm(output, alpha, lhs, rhs, alpha) } -> std::same_as<KernelAttempt>;
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
template <uni20::MutableRankedStridedMdspan<2> OutputMdspan, class Scalar, uni20::RankedStridedMdspan<2> LhsMdspan,
          uni20::RankedStridedMdspan<2> RhsMdspan>
KernelAttempt try_kernel(BlasBackend, gemm_op const&, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs,
                         RhsMdspan&& rhs, Scalar beta)
{
  return uni20::linalg::blas::try_gemm(std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs),
                                       std::forward<RhsMdspan>(rhs), beta);
}

/// \brief Report BLAS eligibility for DeviceTensorView GEMM operands.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
consteval auto kernel_accepts_types(BlasBackend const&, gemm_op const&, OutputTensor&, Scalar const&, LhsTensor&,
                                    RhsTensor&, Scalar const&)
{
  using output_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<OutputTensor&>()))>;
  using lhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<LhsTensor const&>()))>;
  using rhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<RhsTensor const&>()))>;
  constexpr auto acceptance = detail::backend_type_acceptance<BlasBackend, gemm_op, output_span&, Scalar const&,
                                                              lhs_span&, rhs_span&, Scalar const&>();
  if constexpr (acceptance == KernelTypeAcceptance::yes)
    return kernel_types_yes;
  else if constexpr (acceptance == KernelTypeAcceptance::maybe)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Lower DeviceTensorView operands and invoke the BLAS GEMM adapter.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
KernelAttempt try_kernel(BlasBackend backend, gemm_op const& op, OutputTensor& output, Scalar alpha,
                         LhsTensor const& lhs, RhsTensor const& rhs, Scalar beta)
{
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  return try_kernel(backend, op, output_span, alpha, lhs_span, rhs_span, beta);
}

} // namespace uni20::linalg
