#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief BLAS backend adapter for operation-tag GEMM dispatch.
 */

#include <uni20/linalg/blas/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <utility>

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for direct BLAS GEMM dispatch.
template <uni20::MutableRankedStridedMdspanLike<2> OutputMdspan, class Scalar,
          uni20::RankedStridedMdspanLike<2> LhsMdspan, uni20::RankedStridedMdspanLike<2> RhsMdspan>
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
template <uni20::MutableRankedStridedMdspanLike<2> OutputMdspan, class Scalar,
          uni20::RankedStridedMdspanLike<2> LhsMdspan, uni20::RankedStridedMdspanLike<2> RhsMdspan>
KernelAttempt try_kernel(BlasBackend, gemm_op const&, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs,
                         RhsMdspan&& rhs, Scalar beta)
{
  return uni20::linalg::blas::try_gemm(std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs),
                                       std::forward<RhsMdspan>(rhs), beta);
}

/// \brief Report BLAS eligibility for DeviceTensorView GEMM operands.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<LhsTensor> &&
           uni20::detail::HostReadableTensor<RhsTensor>
consteval auto kernel_accepts_types(BlasBackend const&, gemm_op const&, OutputTensor&, Scalar const&, LhsTensor&,
                                    RhsTensor&, Scalar const&)
{
  using output_span = uni20::detail::host_write_tensor_mdspan_t<OutputTensor>;
  using lhs_span = uni20::detail::host_read_tensor_mdspan_t<LhsTensor>;
  using rhs_span = uni20::detail::host_read_tensor_mdspan_t<RhsTensor>;
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
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<LhsTensor> &&
           uni20::detail::HostReadableTensor<RhsTensor>
KernelAttempt try_kernel(BlasBackend backend, gemm_op const& op, OutputTensor& output, Scalar alpha,
                         LhsTensor const& lhs, RhsTensor const& rhs, Scalar beta)
{
  auto output_access = acquire_host_write_access(output);
  auto lhs_access = acquire_host_read_access(lhs);
  auto rhs_access = acquire_host_read_access(rhs);
  auto output_span = output_access.mdspan();
  auto lhs_span = lhs_access.mdspan();
  auto rhs_span = rhs_access.mdspan();
  return try_kernel(backend, op, output_span, alpha, lhs_span, rhs_span, beta);
}

} // namespace uni20::linalg
