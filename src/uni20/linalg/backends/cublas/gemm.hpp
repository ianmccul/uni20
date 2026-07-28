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

#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Report cuBLAS eligibility for DeviceTensorView GEMM operands.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
consteval auto kernel_accepts_types(CublasBackend const&, gemm_op const&, OutputTensor&, Scalar const&, LhsTensor&,
                                    RhsTensor&, Scalar const&)
{
  using output_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<OutputTensor&>()))>;
  using lhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<LhsTensor const&>()))>;
  using rhs_span = std::remove_cvref_t<decltype(uni20::detail::tensor_device_mdspan(std::declval<RhsTensor const&>()))>;
  if constexpr (detail::cublas_backend::accepts_gemm_types<Scalar, output_span, lhs_span, rhs_span>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Lower DeviceTensorView operands and invoke the cuBLAS GEMM adapter.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
KernelAttempt try_kernel(CublasBackend, gemm_op const&, OutputTensor& output, Scalar alpha, LhsTensor const& lhs,
                         RhsTensor const& rhs, Scalar beta)
{
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  return detail::cublas_backend::try_gemm(output_span, alpha, lhs_span, rhs_span, beta);
}

} // namespace uni20::linalg
