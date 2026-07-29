#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Direct mdspan GEMM wrappers over the configured BLAS provider.
 */

#include <uni20/backend/blas/backend_blas.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/kernel_attempt.hpp>

#include <concepts>
#include <type_traits>

namespace uni20::linalg::blas
{

namespace detail
{
template <class Mdspan, class Scalar>
concept readable_blas_mdspan_for =
    blas_readable_mdspan<Mdspan, 2> && std::same_as<std::remove_cv_t<typename Mdspan::element_type>, Scalar> &&
    std::convertible_to<typename Mdspan::data_handle_type, Scalar const*>;

template <class Mdspan, class Scalar>
concept writable_blas_mdspan_for =
    blas_writable_mdspan<Mdspan, 2> && std::same_as<std::remove_cv_t<typename Mdspan::element_type>, Scalar> &&
    std::convertible_to<typename Mdspan::data_handle_type, Scalar*>;

constexpr blas_int logical_rows(blas_int rows, blas_int cols, MatrixTransform transform)
{
  return transformed_rows(rows, cols, transform);
}

constexpr blas_int logical_cols(blas_int rows, blas_int cols, MatrixTransform transform)
{
  return transformed_cols(rows, cols, transform);
}

template <class Scalar, class Handle>
constexpr auto with_transform(BlasReadableMatrix<Scalar, Handle> matrix,
                              MatrixTransform transform) -> BlasReadableMatrix<Scalar, Handle>
{
  matrix.transform = compose(transform, matrix.transform);
  return matrix;
}

template <uni20::BlasScalar Scalar, class LhsHandle, class RhsHandle>
constexpr bool provider_transforms_are_supported(BlasReadableMatrix<Scalar, LhsHandle> lhs,
                                                 BlasReadableMatrix<Scalar, RhsHandle> rhs)
{
  return blas_trans_char_is_supported<Scalar>(lhs.transform) && blas_trans_char_is_supported<Scalar>(rhs.transform);
}

template <uni20::BlasScalar Scalar, class OutputHandle, class LhsHandle, class RhsHandle>
  requires std::convertible_to<OutputHandle, Scalar*> && std::convertible_to<LhsHandle, Scalar const*> &&
           std::convertible_to<RhsHandle, Scalar const*>
void gemm(BlasWritableMatrix<Scalar, OutputHandle> output, BlasReadableMatrix<Scalar, LhsHandle> lhs,
          BlasReadableMatrix<Scalar, RhsHandle> rhs, Scalar alpha, Scalar beta)
{
  auto const transa = blas_trans_char<Scalar>(lhs.transform);
  auto const transb = blas_trans_char<Scalar>(rhs.transform);

  blas_int const lhs_rows = logical_rows(lhs.rows, lhs.cols, lhs.transform);
  blas_int const lhs_cols = logical_cols(lhs.rows, lhs.cols, lhs.transform);
  blas_int const rhs_rows = logical_rows(rhs.rows, rhs.cols, rhs.transform);
  blas_int const rhs_cols = logical_cols(rhs.rows, rhs.cols, rhs.transform);
  CHECK_EQUAL(lhs_cols, rhs_rows);
  CHECK_EQUAL(output.rows, lhs_rows);
  CHECK_EQUAL(output.cols, rhs_cols);

  if (output.rows == 0 || output.cols == 0)
  {
    return;
  }

  Scalar const* const lhs_data = lhs.data;
  Scalar const* const rhs_data = rhs.data;
  Scalar* const output_data = output.data;
  ::uni20::blas::gemm(transa, transb, output.rows, output.cols, lhs_cols, alpha, lhs_data, lhs.leading_dimension,
                      rhs_data, rhs.leading_dimension, beta, output_data, output.leading_dimension);
}

template <uni20::BlasScalar Scalar> struct GemmPlan
{
    BlasWritableMatrix<Scalar, Scalar*> output{};
    BlasReadableMatrix<Scalar, Scalar const*> lhs{};
    BlasReadableMatrix<Scalar, Scalar const*> rhs{};
    bool has_work = false;
};

template <uni20::BlasScalar Scalar> struct GemmPreparation
{
    KernelAttempt attempt = KernelAttempt::unsupported_instance;
    GemmPlan<Scalar> plan{};
};

template <uni20::BlasScalar Scalar, class Handle>
auto provider_writable_matrix(BlasWritableMatrix<Scalar, Handle> matrix) -> BlasWritableMatrix<Scalar, Scalar*>
{
  return {.data = matrix.data, .rows = matrix.rows, .cols = matrix.cols, .leading_dimension = matrix.leading_dimension};
}

template <uni20::BlasScalar Scalar, class Handle>
auto provider_readable_matrix(BlasReadableMatrix<Scalar, Handle> matrix) -> BlasReadableMatrix<Scalar, Scalar const*>
{
  return {.data = matrix.data,
          .rows = matrix.rows,
          .cols = matrix.cols,
          .leading_dimension = matrix.leading_dimension,
          .transform = matrix.transform};
}

template <uni20::BlasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires writable_blas_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           readable_blas_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           readable_blas_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
auto prepare_gemm(OutputMdspan&& output, LhsMdspan&& lhs, RhsMdspan&& rhs)
{
  using preparation_type = GemmPreparation<Scalar>;

  CHECK_EQUAL(lhs.extent(1), rhs.extent(0));
  CHECK_EQUAL(output.extent(0), lhs.extent(0));
  CHECK_EQUAL(output.extent(1), rhs.extent(1));

  if (output.extent(0) == 0 || output.extent(1) == 0)
  {
    return preparation_type{.attempt = KernelAttempt::success};
  }
  if (lhs.extent(1) == 0)
  {
    return preparation_type{.attempt = KernelAttempt::unsupported_instance};
  }

  auto output_stage = try_mdspan_matrix_stage(output);
  auto lhs_stage = try_mdspan_matrix_stage(lhs);
  auto rhs_stage = try_mdspan_matrix_stage(rhs);
  if (!output_stage || !lhs_stage || !rhs_stage)
  {
    return preparation_type{.attempt = KernelAttempt::unsupported_layout};
  }

  // The user-provided direct GEMM output must be ordinary writable storage.
  // Prepared fallbacks may still conjugate that storage before/after a provider
  // call as an internal workaround for unsupported readable transforms.
  if (output_stage->needs_conjugation)
  {
    return preparation_type{.attempt = KernelAttempt::unsupported_transform};
  }

  auto const output_matrix = provider_writable_matrix(blas_writable_matrix(*output_stage));
  if (output_stage->unit_stride_axis == 0)
  {
    auto const lhs_matrix = provider_readable_matrix(blas_readable_matrix(*lhs_stage));
    auto const rhs_matrix = provider_readable_matrix(blas_readable_matrix(*rhs_stage));
    if (!provider_transforms_are_supported(lhs_matrix, rhs_matrix))
    {
      return preparation_type{.attempt = KernelAttempt::unsupported_transform};
    }

    return preparation_type{.attempt = KernelAttempt::success,
                            .plan = {.output = output_matrix, .lhs = lhs_matrix, .rhs = rhs_matrix, .has_work = true}};
  }

  auto const lhs_matrix =
      provider_readable_matrix(with_transform(blas_readable_matrix(*rhs_stage), MatrixTransform::transpose));
  auto const rhs_matrix =
      provider_readable_matrix(with_transform(blas_readable_matrix(*lhs_stage), MatrixTransform::transpose));
  if (!provider_transforms_are_supported(lhs_matrix, rhs_matrix))
  {
    return preparation_type{.attempt = KernelAttempt::unsupported_transform};
  }

  return preparation_type{.attempt = KernelAttempt::success,
                          .plan = {.output = output_matrix, .lhs = lhs_matrix, .rhs = rhs_matrix, .has_work = true}};
}

template <uni20::BlasScalar Scalar> void execute_gemm(GemmPlan<Scalar> const& plan, Scalar alpha, Scalar beta)
{
  if (plan.has_work) gemm(plan.output, plan.lhs, plan.rhs, alpha, beta);
}

} // namespace detail

/// \brief Probe direct no-copy BLAS GEMM without invoking the provider.
/// \details This performs the same instance, layout, and transform checks as
///          `try_gemm`, but does not read or write matrix elements.
/// \return `success`, `unsupported_instance`, `unsupported_layout`, or
///         `unsupported_transform`.
template <uni20::BlasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires detail::writable_blas_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
KernelAttempt probe_gemm(OutputMdspan&& output, LhsMdspan&& lhs, RhsMdspan&& rhs)
{
  return detail::prepare_gemm<Scalar>(std::forward<OutputMdspan>(output), std::forward<LhsMdspan>(lhs),
                                      std::forward<RhsMdspan>(rhs))
      .attempt;
}

/// \brief Try a direct no-copy BLAS GEMM from mdspan-like operands.
/// \return `success`, `unsupported_instance`, `unsupported_layout`, or
///         `unsupported_transform`.
template <uni20::BlasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires detail::writable_blas_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
KernelAttempt try_gemm(OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs, Scalar beta)
{
  auto preparation = detail::prepare_gemm<Scalar>(std::forward<OutputMdspan>(output), std::forward<LhsMdspan>(lhs),
                                                  std::forward<RhsMdspan>(rhs));
  if (!kernel_attempt_succeeded(preparation.attempt)) return preparation.attempt;

  detail::execute_gemm(preparation.plan, alpha, beta);
  return KernelAttempt::success;
}

/// \brief Direct no-copy BLAS GEMM from mdspan-like operands.
template <uni20::BlasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires detail::writable_blas_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
void gemm(OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs, Scalar beta)
{
  CHECK(kernel_attempt_succeeded(try_gemm(output, alpha, lhs, rhs, beta)));
}

} // namespace uni20::linalg::blas
