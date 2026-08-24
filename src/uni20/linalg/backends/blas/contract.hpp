#pragma once

/**
 * \file contract.hpp
 * \ingroup linalg
 * \brief BLAS backend adapter for direct pairwise tensor contraction.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/contract.hpp>
#include <uni20/linalg/contraction_strides.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>

#include <concepts>

namespace uni20::linalg
{
namespace detail::blas_contract_backend
{

template <class OutputMdspec, class LhsMdspec, class RhsMdspec>
concept HostContractionAccess = uni20::HostWritableMdspec<OutputMdspec> && uni20::HostReadableMdspec<LhsMdspec> &&
                                uni20::HostReadableMdspec<RhsMdspec>;

template <class OutputMdspec, class Scalar, class LhsMdspec, class RhsMdspec>
consteval bool contraction_types_compatible()
{
  using output_span = uni20::host_write_mdspan_t<OutputMdspec>;
  using lhs_span = uni20::host_read_mdspan_t<LhsMdspec>;
  using rhs_span = uni20::host_read_mdspan_t<RhsMdspec>;
  return requires(DirectContractionGemmPlan const& plan, output_span& output, Scalar alpha, lhs_span& lhs,
                  rhs_span& rhs) {
    { uni20::linalg::blas::try_contract(plan, output, alpha, lhs, rhs, alpha) } -> std::same_as<KernelAttempt>;
  };
}

} // namespace detail::blas_contract_backend

/// \brief Report BLAS eligibility for host-accessible strided contraction operands.
template <
    std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
    MutableRankedContractionMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
    uni20::BlasScalar Scalar, RankedContractionMdspecLike<LhsRank> LhsMdspec,
    RankedContractionMdspecLike<RhsRank> RhsMdspec>
  requires detail::blas_contract_backend::HostContractionAccess<OutputMdspec, LhsMdspec, RhsMdspec>
consteval auto kernel_accepts_types(BlasBackend const&, contract_op<LhsRank, RhsRank, ContractedRank> const&,
                                    OutputMdspec&, Scalar const&, LhsMdspec&, RhsMdspec&, Scalar const&)
{
  if constexpr (detail::blas_contract_backend::contraction_types_compatible<OutputMdspec, Scalar, LhsMdspec,
                                                                            RhsMdspec>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Lower a directly representable contraction to one BLAS GEMM call.
/// \details Grouping and unit-stride projection rejection occur before host
///          access acquisition. Once access is resolved, the ordinary BLAS
///          GEMM implementation performs its final provider-specific layout,
///          dimension, and transform probe.
template <
    std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
    MutableRankedContractionMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
    uni20::BlasScalar Scalar, RankedContractionMdspecLike<LhsRank> LhsMdspec,
    RankedContractionMdspecLike<RhsRank> RhsMdspec>
  requires detail::blas_contract_backend::HostContractionAccess<OutputMdspec, LhsMdspec, RhsMdspec>
KernelAttempt try_kernel(BlasBackend, contract_op<LhsRank, RhsRank, ContractedRank> const& operation,
                         OutputMdspec& output, Scalar alpha, LhsMdspec& lhs, RhsMdspec& rhs, Scalar beta)
{
  auto plan = try_make_direct_contraction_gemm_plan(output, lhs, rhs, operation.axes);
  if (!plan) return KernelAttempt::unsupported_layout;

  auto output_access = acquire_host_write_access_sync(output);
  auto lhs_access = acquire_host_read_access_sync(lhs);
  auto rhs_access = acquire_host_read_access_sync(rhs);
  auto& output_span = output_access.mdspan();
  auto& lhs_span = lhs_access.mdspan();
  auto& rhs_span = rhs_access.mdspan();
  return uni20::linalg::blas::try_contract(*plan, output_span, alpha, lhs_span, rhs_span, beta);
}

} // namespace uni20::linalg
