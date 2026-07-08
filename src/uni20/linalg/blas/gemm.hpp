#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Direct mdspan GEMM wrappers over the configured BLAS provider.
 */

#include <uni20/backend/blas/backend_blas.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/mdspan_matrix_operand.hpp>

#include <concepts>
#include <type_traits>

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
  return blas_trans_char<Scalar>(lhs.transform).has_value() && blas_trans_char<Scalar>(rhs.transform).has_value();
}

template <uni20::BlasScalar Scalar, class OutputHandle, class LhsHandle, class RhsHandle>
  requires std::convertible_to<OutputHandle, Scalar*> && std::convertible_to<LhsHandle, Scalar const*> &&
           std::convertible_to<RhsHandle, Scalar const*>
void gemm(BlasWritableMatrix<Scalar, OutputHandle> output, BlasReadableMatrix<Scalar, LhsHandle> lhs,
          BlasReadableMatrix<Scalar, RhsHandle> rhs, Scalar alpha, Scalar beta)
{
  auto const transa = blas_trans_char<Scalar>(lhs.transform);
  auto const transb = blas_trans_char<Scalar>(rhs.transform);
  CHECK(transa.has_value());
  CHECK(transb.has_value());

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
  ::uni20::blas::gemm(*transa, *transb, output.rows, output.cols, lhs_cols, alpha, lhs_data, lhs.leading_dimension,
                      rhs_data, rhs.leading_dimension, beta, output_data, output.leading_dimension);
}

} // namespace detail

/// \brief Try a direct no-copy BLAS GEMM from mdspan-like operands.
template <uni20::BlasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires detail::writable_blas_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
bool try_gemm(OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs, Scalar beta)
{
  auto output_stage = try_mdspan_matrix_stage(output);
  auto lhs_stage = try_mdspan_matrix_stage(lhs);
  auto rhs_stage = try_mdspan_matrix_stage(rhs);
  if (!output_stage || !lhs_stage || !rhs_stage)
  {
    return false;
  }

  // The user-provided direct GEMM output must be ordinary writable storage.
  // Prepared fallbacks may still conjugate that storage before/after a provider
  // call as an internal workaround for unsupported readable transforms.
  if (output_stage->needs_conjugation)
  {
    return false;
  }

  auto const output_matrix = blas_writable_matrix(*output_stage);
  if (output_stage->unit_stride_axis == 0)
  {
    auto const lhs_matrix = blas_readable_matrix(*lhs_stage);
    auto const rhs_matrix = blas_readable_matrix(*rhs_stage);
    if (!detail::provider_transforms_are_supported(lhs_matrix, rhs_matrix))
    {
      return false;
    }

    detail::gemm(output_matrix, lhs_matrix, rhs_matrix, alpha, beta);
    return true;
  }

  auto const lhs_matrix = detail::with_transform(blas_readable_matrix(*rhs_stage), MatrixTransform::transpose);
  auto const rhs_matrix = detail::with_transform(blas_readable_matrix(*lhs_stage), MatrixTransform::transpose);
  if (!detail::provider_transforms_are_supported(lhs_matrix, rhs_matrix))
  {
    return false;
  }

  detail::gemm(output_matrix, lhs_matrix, rhs_matrix, alpha, beta);
  return true;
}

/// \brief Direct no-copy BLAS GEMM from mdspan-like operands.
template <uni20::BlasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires detail::writable_blas_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           detail::readable_blas_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
void gemm(OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs, Scalar beta)
{
  CHECK(try_gemm(output, alpha, lhs, rhs, beta));
}

} // namespace uni20::linalg::blas
