#pragma once

/**
 * \file contract.hpp
 * \ingroup linalg
 * \brief Direct no-copy BLAS lowering for pairwise tensor contraction.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/gemm.hpp>
#include <uni20/linalg/contraction_strides.hpp>
#include <uni20/linalg/kernel_attempt.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <concepts>

namespace uni20::linalg::blas
{

/// \brief Execute a preplanned contraction as one BLAS GEMM call.
/// \details The caller must complete all contraction grouping before this
///          function. The three resolved operands are projected to rank two
///          without changing their handles or accessors.
template <uni20::BlasScalar Scalar, uni20::MdspanLike OutputMdspan, uni20::MdspanLike LhsMdspan,
          uni20::MdspanLike RhsMdspan>
  requires requires(contraction_matrix_mdspan_t<OutputMdspan>& output, Scalar alpha,
                    contraction_matrix_mdspan_t<LhsMdspan>& lhs, contraction_matrix_mdspan_t<RhsMdspan>& rhs) {
    { uni20::linalg::blas::try_gemm(output, alpha, lhs, rhs, alpha) } -> std::same_as<KernelAttempt>;
  }
KernelAttempt try_contract(DirectContractionGemmPlan const& plan, OutputMdspan& output, Scalar alpha, LhsMdspan& lhs,
                           RhsMdspan& rhs, Scalar beta)
{
  auto output_matrix = make_contraction_matrix_mdspan(output, plan.output);
  auto lhs_matrix = make_contraction_matrix_mdspan(lhs, plan.lhs);
  auto rhs_matrix = make_contraction_matrix_mdspan(rhs, plan.rhs);
  return uni20::linalg::blas::try_gemm(output_matrix, alpha, lhs_matrix, rhs_matrix, beta);
}

} // namespace uni20::linalg::blas
