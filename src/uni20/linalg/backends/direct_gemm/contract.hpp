#pragma once

/**
 * \file contract.hpp
 * \ingroup linalg
 * \brief Direct single-GEMM backend for pairwise tensor contraction.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/linalg/contraction_strides.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/linalg/ops/gemm.hpp>

#include <concepts>
#include <string_view>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Contract directly through one GEMM selected from a retained execution selector.
/// \details This backend implements only `contract_op`. It performs
///          handle-independent M/N/K projection, then delegates the resulting
///          rank-two mdspecs to `gemm_op`. The selected GEMM backend owns
///          execution-domain acquisition and provider lowering.
template <KernelBackendSelector GemmSelector> struct DirectGemmContractionBackend
{
    static constexpr std::string_view name = "direct_gemm_contraction";

    GemmSelector gemm_selector;
};

template <class GemmSelector>
DirectGemmContractionBackend(GemmSelector) -> DirectGemmContractionBackend<std::remove_cvref_t<GemmSelector>>;

namespace detail::direct_gemm_contraction_backend
{

template <class GemmSelector, class OutputMdspec, class Scalar, class LhsMdspec, class RhsMdspec>
consteval bool gemm_dispatch_types_compatible()
{
  using output_matrix = contraction_matrix_mdspec_t<OutputMdspec>;
  using lhs_matrix = contraction_matrix_mdspec_t<LhsMdspec>;
  using rhs_matrix = contraction_matrix_mdspec_t<RhsMdspec>;
  return requires(GemmSelector const& selector, output_matrix& output, Scalar alpha, lhs_matrix& lhs, rhs_matrix& rhs) {
    try_dispatch_kernel(selector, gemm_op{}, output, alpha, lhs, rhs, alpha);
  };
}

} // namespace detail::direct_gemm_contraction_backend

/// \brief Report direct-GEMM contraction eligibility for the retained GEMM selector.
template <
    class GemmSelector, std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
    MutableRankedContractionMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
    uni20::Scalar Scalar, RankedContractionMdspecLike<LhsRank> LhsMdspec,
    RankedContractionMdspecLike<RhsRank> RhsMdspec>
consteval auto kernel_accepts_types(DirectGemmContractionBackend<GemmSelector> const&,
                                    contract_op<LhsRank, RhsRank, ContractedRank> const&, OutputMdspec&, Scalar const&,
                                    LhsMdspec&, RhsMdspec&, Scalar const&)
{
  if constexpr (detail::direct_gemm_contraction_backend::gemm_dispatch_types_compatible<GemmSelector, OutputMdspec,
                                                                                        Scalar, LhsMdspec, RhsMdspec>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Project a contraction and execute it through one nested GEMM dispatch.
template <
    class GemmSelector, std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
    MutableRankedContractionMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
    uni20::Scalar Scalar, RankedContractionMdspecLike<LhsRank> LhsMdspec,
    RankedContractionMdspecLike<RhsRank> RhsMdspec>
KernelAttempt try_kernel(DirectGemmContractionBackend<GemmSelector> const& backend,
                         contract_op<LhsRank, RhsRank, ContractedRank> const& operation, OutputMdspec& output,
                         Scalar alpha, LhsMdspec& lhs, RhsMdspec& rhs, Scalar beta)
{
  auto plan = try_make_direct_contraction_gemm_plan(output, lhs, rhs, operation.axes);
  if (!plan) return KernelAttempt::unsupported_layout;

  auto output_matrix = make_contraction_matrix_mdspec(output, plan->output);
  auto lhs_matrix = make_contraction_matrix_mdspec(lhs, plan->lhs);
  auto rhs_matrix = make_contraction_matrix_mdspec(rhs, plan->rhs);
  if (!try_dispatch_kernel(backend.gemm_selector, gemm_op{}, output_matrix, alpha, lhs_matrix, rhs_matrix, beta))
    return KernelAttempt::unsupported_instance;
  return KernelAttempt::success;
}

} // namespace uni20::linalg
