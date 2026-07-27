#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief cuBLAS backend adapter for provider-ready GEMM dispatch.
 */

#include <uni20/linalg/backends/cublas/detail/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for CUDA mdspan GEMM lowering.
template <class OutputMdspan, class Scalar, uni20::RankedStridedDeviceSpanLike<2> LhsMdspan,
          uni20::RankedStridedDeviceSpanLike<2> RhsMdspan>
  requires(uni20::MutableDeviceSpanLike<OutputMdspan> && uni20::RankedStridedDeviceSpanLike<OutputMdspan, 2>)
consteval auto kernel_accepts_types(CublasBackend const&, gemm_op const&, OutputMdspan&, Scalar const&, LhsMdspan&,
                                    RhsMdspan&, Scalar const&)
{
  if constexpr (requires(OutputMdspan& output, LhsMdspan& lhs, RhsMdspan& rhs, Scalar scalar) {
                  { detail::cublas_backend::try_gemm(output, scalar, lhs, rhs, scalar) } -> std::same_as<KernelAttempt>;
                })
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Lower CUDA mdspans, block for execution resources, and enqueue cuBLAS GEMM.
template <class OutputMdspan, class Scalar, uni20::RankedStridedDeviceSpanLike<2> LhsMdspan,
          uni20::RankedStridedDeviceSpanLike<2> RhsMdspan>
  requires(uni20::MutableDeviceSpanLike<OutputMdspan> && uni20::RankedStridedDeviceSpanLike<OutputMdspan, 2>)
KernelAttempt try_kernel(CublasBackend, gemm_op const&, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs,
                         RhsMdspan&& rhs, Scalar beta)
{
  return detail::cublas_backend::try_gemm(std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs),
                                          std::forward<RhsMdspan>(rhs), beta);
}

/// \brief Report cuBLAS eligibility for DeviceTensorView GEMM operands.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
consteval auto kernel_accepts_types(CublasBackend const&, gemm_op const&, OutputTensor&, Scalar const&, LhsTensor&,
                                    RhsTensor&, Scalar const&)
{
  using output_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<OutputTensor&>()))>;
  using lhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<LhsTensor const&>()))>;
  using rhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<RhsTensor const&>()))>;
  constexpr auto acceptance = detail::backend_type_acceptance<CublasBackend, gemm_op, output_span&, Scalar const&,
                                                              lhs_span&, rhs_span&, Scalar const&>();
  if constexpr (acceptance == KernelTypeAcceptance::yes)
    return kernel_types_yes;
  else if constexpr (acceptance == KernelTypeAcceptance::maybe)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Lower DeviceTensorView operands and invoke the cuBLAS GEMM adapter.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
KernelAttempt try_kernel(CublasBackend backend, gemm_op const& op, OutputTensor& output, Scalar alpha,
                         LhsTensor const& lhs, RhsTensor const& rhs, Scalar beta)
{
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  return try_kernel(backend, op, output_span, alpha, lhs_span, rhs_span, beta);
}

} // namespace uni20::linalg
