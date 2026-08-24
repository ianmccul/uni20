#pragma once

/**
 * \file contract.hpp
 * \ingroup linalg
 * \brief Reference CPU backend for pairwise tensor contraction dispatch.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/cpu/contract.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>

#include <type_traits>

namespace uni20::linalg
{
namespace detail::cpu_contract_backend
{

template <class OutputMdspec, class LhsMdspec, class RhsMdspec>
concept HostContractionAccess = uni20::HostWritableMdspec<OutputMdspec> && uni20::HostReadableMdspec<LhsMdspec> &&
                                uni20::HostReadableMdspec<RhsMdspec>;

template <class OutputMdspec, class Scalar, class LhsMdspec, class RhsMdspec, std::size_t LhsRank, std::size_t RhsRank,
          std::size_t ContractedRank>
consteval bool contraction_types_compatible()
{
  using output_span = uni20::host_write_mdspan_t<OutputMdspec>;
  using lhs_span = uni20::host_read_mdspan_t<LhsMdspec>;
  using rhs_span = uni20::host_read_mdspan_t<RhsMdspec>;
  return uni20::linalg::cpu::ContractionCompatible<output_span, Scalar, lhs_span, rhs_span, LhsRank, RhsRank,
                                                   ContractedRank>;
}

} // namespace detail::cpu_contract_backend

/// \brief Report eligibility for host-accessible pairwise contraction operands.
template <std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
          uni20::MutableRankedMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
          uni20::Scalar Scalar, uni20::RankedMdspecLike<LhsRank> LhsMdspec, uni20::RankedMdspecLike<RhsRank> RhsMdspec>
  requires detail::cpu_contract_backend::HostContractionAccess<OutputMdspec, LhsMdspec, RhsMdspec>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, contract_op<LhsRank, RhsRank, ContractedRank> const&,
                                    OutputMdspec&, Scalar const&, LhsMdspec&, RhsMdspec&, Scalar const&)
{
  if constexpr (detail::cpu_contract_backend::contraction_types_compatible<OutputMdspec, Scalar, LhsMdspec, RhsMdspec,
                                                                           LhsRank, RhsRank, ContractedRank>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Acquire host access and execute the reference contraction kernel.
template <std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
          uni20::MutableRankedMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
          uni20::Scalar Scalar, uni20::RankedMdspecLike<LhsRank> LhsMdspec, uni20::RankedMdspecLike<RhsRank> RhsMdspec>
  requires detail::cpu_contract_backend::HostContractionAccess<OutputMdspec, LhsMdspec, RhsMdspec>
KernelAttempt try_kernel(CpuReferenceBackend, contract_op<LhsRank, RhsRank, ContractedRank> const& operation,
                         OutputMdspec& output, Scalar alpha, LhsMdspec& lhs, RhsMdspec& rhs, Scalar beta)
{
  auto output_access = acquire_host_write_access_sync(output);
  auto lhs_access = acquire_host_read_access_sync(lhs);
  auto rhs_access = acquire_host_read_access_sync(rhs);
  auto& output_span = output_access.mdspan();
  auto& lhs_span = lhs_access.mdspan();
  auto& rhs_span = rhs_access.mdspan();
  uni20::linalg::cpu::contract(output_span, alpha, lhs_span, rhs_span, beta, operation.axes);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
