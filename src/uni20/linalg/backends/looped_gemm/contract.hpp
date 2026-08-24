#pragma once

/**
 * \file contract.hpp
 * \ingroup linalg
 * \brief Residual-axis looped GEMM backend for pairwise tensor contraction.
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

/// \brief Contract through a loop of GEMMs selected from one retained execution selector.
/// \details This backend implements only `contract_op`. It accepts one
///          residual M or N stride group, offsets normalized mdspec operands
///          for each disjoint output slice, and delegates every rank-two slice
///          to `gemm_op`. Contracted K dimensions are not looped.
template <KernelBackendSelector GemmSelector> struct LoopedGemmContractionBackend
{
    static constexpr std::string_view name = "looped_gemm_contraction";

    GemmSelector gemm_selector;
};

template <class GemmSelector>
LoopedGemmContractionBackend(GemmSelector) -> LoopedGemmContractionBackend<std::remove_cvref_t<GemmSelector>>;

namespace detail::looped_gemm_contraction_backend
{

template <class GemmSelector, class OutputMdspec, class Scalar, class LhsMdspec, class RhsMdspec>
consteval bool gemm_dispatch_types_compatible()
{
  using output_matrix = offset_contraction_matrix_mdspec_t<OutputMdspec>;
  using lhs_matrix = offset_contraction_matrix_mdspec_t<LhsMdspec>;
  using rhs_matrix = offset_contraction_matrix_mdspec_t<RhsMdspec>;
  return requires(GemmSelector const& selector, OutputMdspec& output_source, LhsMdspec& lhs_source,
                  RhsMdspec& rhs_source, ContractionMatrixProjection const& projection, uni20::index_type offset,
                  output_matrix& output, Scalar alpha, lhs_matrix& lhs, rhs_matrix& rhs) {
    { make_offset_contraction_matrix_mdspec(output_source, projection, offset) } -> std::same_as<output_matrix>;
    { make_offset_contraction_matrix_mdspec(lhs_source, projection, offset) } -> std::same_as<lhs_matrix>;
    { make_offset_contraction_matrix_mdspec(rhs_source, projection, offset) } -> std::same_as<rhs_matrix>;
    try_dispatch_kernel(selector, gemm_op{}, output, alpha, lhs, rhs, alpha);
  };
}

} // namespace detail::looped_gemm_contraction_backend

/// \brief Report looped-GEMM contraction eligibility for the retained GEMM selector.
template <
    class GemmSelector, std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
    MutableRankedContractionMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
    uni20::Scalar Scalar, RankedContractionMdspecLike<LhsRank> LhsMdspec,
    RankedContractionMdspecLike<RhsRank> RhsMdspec>
consteval auto kernel_accepts_types(LoopedGemmContractionBackend<GemmSelector> const&,
                                    contract_op<LhsRank, RhsRank, ContractedRank> const&, OutputMdspec&, Scalar const&,
                                    LhsMdspec&, RhsMdspec&, Scalar const&)
{
  if constexpr (detail::looped_gemm_contraction_backend::gemm_dispatch_types_compatible<GemmSelector, OutputMdspec,
                                                                                        Scalar, LhsMdspec, RhsMdspec>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Execute one residual M or N loop through nested GEMM dispatch.
/// \details The first nested GEMM may cleanly decline before any output is
///          modified. Later slices differ only in valid descriptor offsets;
///          after the first succeeds, a later decline violates the backend's
///          offset-invariant GEMM acceptance requirement and is terminal.
template <
    class GemmSelector, std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
    MutableRankedContractionMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
    uni20::Scalar Scalar, RankedContractionMdspecLike<LhsRank> LhsMdspec,
    RankedContractionMdspecLike<RhsRank> RhsMdspec>
  requires(detail::looped_gemm_contraction_backend::gemm_dispatch_types_compatible<GemmSelector, OutputMdspec, Scalar,
                                                                                   LhsMdspec, RhsMdspec>())
KernelAttempt try_kernel(LoopedGemmContractionBackend<GemmSelector> const& backend,
                         contract_op<LhsRank, RhsRank, ContractedRank> const& operation, OutputMdspec& output,
                         Scalar alpha, LhsMdspec& lhs, RhsMdspec& rhs, Scalar beta)
{
  auto plan = try_make_looped_contraction_gemm_plan(output, lhs, rhs, operation.axes);
  if (!plan) return KernelAttempt::unsupported_layout;
  if (plan->loop_extent == 0) return KernelAttempt::success;

  bool output_modified = false;
  for (uni20::index_type index = 0; index < plan->loop_extent; ++index)
  {
    auto output_matrix =
        make_offset_contraction_matrix_mdspec(output, plan->gemm.output, index * plan->output_offset_stride);
    auto lhs_matrix = make_offset_contraction_matrix_mdspec(lhs, plan->gemm.lhs, index * plan->lhs_offset_stride);
    auto rhs_matrix = make_offset_contraction_matrix_mdspec(rhs, plan->gemm.rhs, index * plan->rhs_offset_stride);
    bool const succeeded =
        try_dispatch_kernel(backend.gemm_selector, gemm_op{}, output_matrix, alpha, lhs_matrix, rhs_matrix, beta);
    if (!succeeded)
    {
      if (!output_modified) return KernelAttempt::unsupported_instance;
      CHECK(false, "GEMM backend acceptance changed across offset-only contraction slices", index);
    }
    output_modified = true;
  }
  return KernelAttempt::success;
}

} // namespace uni20::linalg
