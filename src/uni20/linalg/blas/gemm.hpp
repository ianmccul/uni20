#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Direct mdspan GEMM wrappers over the configured BLAS provider.
 */

#include <uni20/backend/blas/backend_blas.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/mdspan_matrix_operand.hpp>

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20::linalg::blas
{

namespace detail
{
template <class Mdspan, class Scalar>
concept readable_blas_mdspan_for =
    uni20::StridedMdspan<Mdspan> && std::same_as<std::remove_cv_t<typename Mdspan::element_type>, Scalar> &&
    std::convertible_to<typename Mdspan::data_handle_type, Scalar const*>;

template <class Mdspan, class Scalar>
concept writable_blas_mdspan_for =
    uni20::MutableStridedMdspan<Mdspan> && std::same_as<std::remove_cv_t<typename Mdspan::element_type>, Scalar> &&
    std::convertible_to<typename Mdspan::data_handle_type, Scalar*>;

template <typename Scalar> struct GemmPlan
{
    char transa = 'N';
    char transb = 'N';
    blas_int m = 0;
    blas_int n = 0;
    blas_int k = 0;
    Scalar const* a = nullptr;
    blas_int lda = 0;
    Scalar const* b = nullptr;
    blas_int ldb = 0;
    Scalar* c = nullptr;
    blas_int ldc = 0;
};

constexpr blas_int logical_rows(blas_int rows, blas_int cols, MatrixTransform transform)
{
  return transformed_rows(rows, cols, transform);
}

constexpr blas_int logical_cols(blas_int rows, blas_int cols, MatrixTransform transform)
{
  return transformed_cols(rows, cols, transform);
}

template <typename Scalar>
std::optional<GemmPlan<Scalar>> make_column_major_output_plan(BlasWritableMatrix<Scalar, Scalar*> output,
                                                              BlasReadableMatrix<Scalar, Scalar const*> lhs,
                                                              BlasReadableMatrix<Scalar, Scalar const*> rhs)
{
  auto transa = standard_blas_trans_char<Scalar>(lhs.transform);
  auto transb = standard_blas_trans_char<Scalar>(rhs.transform);
  if (!transa || !transb)
  {
    return std::nullopt;
  }

  blas_int const lhs_rows = logical_rows(lhs.rows, lhs.cols, lhs.transform);
  blas_int const lhs_cols = logical_cols(lhs.rows, lhs.cols, lhs.transform);
  blas_int const rhs_rows = logical_rows(rhs.rows, rhs.cols, rhs.transform);
  blas_int const rhs_cols = logical_cols(rhs.rows, rhs.cols, rhs.transform);
  if (lhs_cols != rhs_rows || output.rows != lhs_rows || output.cols != rhs_cols)
  {
    return std::nullopt;
  }

  return GemmPlan<Scalar>{.transa = *transa,
                          .transb = *transb,
                          .m = output.rows,
                          .n = output.cols,
                          .k = lhs_cols,
                          .a = lhs.data,
                          .lda = lhs.leading_dimension,
                          .b = rhs.data,
                          .ldb = rhs.leading_dimension,
                          .c = output.data,
                          .ldc = output.leading_dimension};
}

template <typename Scalar>
std::optional<GemmPlan<Scalar>> make_gemm_plan(MdspanMatrixStage<Scalar, Scalar*> const& output_stage,
                                               MdspanMatrixStage<Scalar, Scalar const*> const& lhs_stage,
                                               MdspanMatrixStage<Scalar, Scalar const*> const& rhs_stage,
                                               MatrixTransform lhs_transform, MatrixTransform rhs_transform)
{
  if (output_stage.needs_conjugation)
  {
    return std::nullopt;
  }

  auto const output = blas_writable_matrix(output_stage);
  if (output_stage.unit_stride_axis == 0)
  {
    auto const lhs = blas_readable_matrix(lhs_stage, lhs_transform);
    auto const rhs = blas_readable_matrix(rhs_stage, rhs_transform);
    return make_column_major_output_plan(output, lhs, rhs);
  }

  auto const lhs = blas_readable_matrix(rhs_stage, transpose_result_transform(rhs_transform));
  auto const rhs = blas_readable_matrix(lhs_stage, transpose_result_transform(lhs_transform));
  return make_column_major_output_plan(output, lhs, rhs);
}

template <typename Scalar> void call_gemm(GemmPlan<Scalar> const& plan, Scalar alpha, Scalar beta)
{
  if (plan.m == 0 || plan.n == 0)
  {
    return;
  }
  ::uni20::blas::gemm(plan.transa, plan.transb, plan.m, plan.n, plan.k, alpha, plan.a, plan.lda, plan.b, plan.ldb, beta,
                      plan.c, plan.ldc);
}
} // namespace detail

/// \brief Try a direct no-copy BLAS GEMM from mdspan-like operands.
template <uni20::BlasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires detail::writable_blas_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
bool try_gemm(OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs, Scalar beta,
              MatrixTransform lhs_transform = MatrixTransform::normal,
              MatrixTransform rhs_transform = MatrixTransform::normal)
{
  auto output_stage = try_mdspan_matrix_stage(output);
  auto lhs_stage = try_mdspan_matrix_stage(lhs);
  auto rhs_stage = try_mdspan_matrix_stage(rhs);
  if (!output_stage || !lhs_stage || !rhs_stage)
  {
    return false;
  }

  using output_stage_type = MdspanMatrixStage<Scalar, Scalar*>;
  using input_stage_type = MdspanMatrixStage<Scalar, Scalar const*>;

  auto const normalized_output = output_stage_type{.data = output_stage->data,
                                                   .extent0 = output_stage->extent0,
                                                   .extent1 = output_stage->extent1,
                                                   .nonunit_stride = output_stage->nonunit_stride,
                                                   .unit_stride_axis = output_stage->unit_stride_axis,
                                                   .needs_conjugation = output_stage->needs_conjugation};
  auto const normalized_lhs = input_stage_type{.data = lhs_stage->data,
                                               .extent0 = lhs_stage->extent0,
                                               .extent1 = lhs_stage->extent1,
                                               .nonunit_stride = lhs_stage->nonunit_stride,
                                               .unit_stride_axis = lhs_stage->unit_stride_axis,
                                               .needs_conjugation = lhs_stage->needs_conjugation};
  auto const normalized_rhs = input_stage_type{.data = rhs_stage->data,
                                               .extent0 = rhs_stage->extent0,
                                               .extent1 = rhs_stage->extent1,
                                               .nonunit_stride = rhs_stage->nonunit_stride,
                                               .unit_stride_axis = rhs_stage->unit_stride_axis,
                                               .needs_conjugation = rhs_stage->needs_conjugation};

  auto plan = detail::make_gemm_plan(normalized_output, normalized_lhs, normalized_rhs, lhs_transform, rhs_transform);
  if (!plan)
  {
    return false;
  }

  detail::call_gemm(*plan, alpha, beta);
  return true;
}

/// \brief Direct no-copy BLAS GEMM from mdspan-like operands, throwing if unsupported.
template <uni20::BlasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires detail::writable_blas_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
void gemm_or_throw(OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs, Scalar beta,
                   MatrixTransform lhs_transform = MatrixTransform::normal,
                   MatrixTransform rhs_transform = MatrixTransform::normal)
{
  if (!try_gemm(std::forward<OutputMdspan>(output), alpha, std::forward<LhsMdspan>(lhs), std::forward<RhsMdspan>(rhs),
                beta, lhs_transform, rhs_transform))
  {
    throw std::invalid_argument("BLAS GEMM operands cannot be represented without copies");
  }
}

} // namespace uni20::linalg::blas
