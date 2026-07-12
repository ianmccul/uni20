#pragma once

/**
 * \file gemv.hpp
 * \ingroup linalg
 * \brief Direct mdspan GEMV wrappers over the configured BLAS provider.
 */

#include <uni20/backend/blas/backend_blas.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/blas/mdspan_vector.hpp>
#include <uni20/linalg/kernel_attempt.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace uni20::linalg::blas
{

namespace detail
{
template <class Mdspan, class Scalar, std::size_t Rank>
concept readable_gemv_mdspan_for =
    blas_readable_mdspan<Mdspan, Rank> && std::same_as<std::remove_cv_t<typename Mdspan::element_type>, Scalar> &&
    std::convertible_to<typename Mdspan::data_handle_type, Scalar const*>;

template <class Mdspan, class Scalar>
concept writable_gemv_vector_for =
    blas_writable_mdspan<Mdspan, 1> && std::same_as<std::remove_cv_t<typename Mdspan::element_type>, Scalar> &&
    std::convertible_to<typename Mdspan::data_handle_type, Scalar*>;

template <uni20::BlasScalar Scalar, class Handle>
  requires std::convertible_to<Handle, Scalar*>
void scale_vector(BlasWritableVector<Scalar, Handle> output, Scalar beta)
{
  if (beta == Scalar{1})
  {
    return;
  }

  Scalar* value = output.data;
  for (blas_int index = 0; index < output.size; ++index, value += output.increment)
  {
    if (beta == Scalar{})
    {
      *value = Scalar{};
    }
    else
    {
      *value *= beta;
    }
  }
}

template <uni20::BlasScalar Scalar, class OutputHandle, class MatrixHandle, class InputHandle>
  requires std::convertible_to<OutputHandle, Scalar*> && std::convertible_to<MatrixHandle, Scalar const*> &&
           std::convertible_to<InputHandle, Scalar const*>
void gemv(BlasWritableVector<Scalar, OutputHandle> output, BlasReadableMatrix<Scalar, MatrixHandle> matrix,
          BlasReadableVector<Scalar, InputHandle> input, Scalar alpha, Scalar beta)
{
  blas_int const matrix_rows = transformed_rows(matrix.rows, matrix.cols, matrix.transform);
  blas_int const matrix_cols = transformed_cols(matrix.rows, matrix.cols, matrix.transform);
  CHECK_EQUAL(output.size, matrix_rows);
  CHECK_EQUAL(input.size, matrix_cols);

  if (output.size == 0)
  {
    return;
  }

  // BLAS GEMV permits a quick return for an empty provider dimension. Preserve
  // the linalg operation's y = beta*y semantics explicitly when no product is read.
  if (alpha == Scalar{} || input.size == 0)
  {
    scale_vector(output, beta);
    return;
  }

  CHECK(blas_trans_char_is_supported<Scalar>(matrix.transform));
  char const trans = blas_trans_char<Scalar>(matrix.transform);
  ::uni20::blas::gemv(trans, matrix.rows, matrix.cols, alpha, matrix.data, matrix.leading_dimension, input.data,
                      input.increment, beta, output.data, output.increment);
}

} // namespace detail

/// \brief Try a direct no-copy BLAS GEMV from mdspan-like operands.
/// \return `success`, `unsupported_layout`, or `unsupported_transform`.
template <uni20::BlasScalar Scalar, class OutputMdspan, class MatrixMdspan, class InputMdspan>
  requires detail::writable_gemv_vector_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_gemv_mdspan_for<std::remove_cvref_t<MatrixMdspan>, Scalar, 2> &&
           detail::readable_gemv_mdspan_for<std::remove_cvref_t<InputMdspan>, Scalar, 1>
KernelAttempt try_gemv(OutputMdspan&& output, Scalar alpha, MatrixMdspan&& matrix, InputMdspan&& input, Scalar beta)
{
  auto output_stage = try_mdspan_vector_stage(output);
  auto matrix_stage = try_mdspan_matrix_stage(matrix);
  auto input_stage = try_mdspan_vector_stage(input);
  if (!output_stage || !matrix_stage || !input_stage)
  {
    return KernelAttempt::unsupported_layout;
  }

  if (output_stage->needs_conjugation || input_stage->needs_conjugation)
  {
    return KernelAttempt::unsupported_transform;
  }

  auto const matrix_operand = blas_readable_matrix(*matrix_stage);
  if (!blas_trans_char_is_supported<Scalar>(matrix_operand.transform))
  {
    return KernelAttempt::unsupported_transform;
  }

  detail::gemv(blas_writable_vector(*output_stage), matrix_operand, blas_readable_vector(*input_stage), alpha, beta);
  return KernelAttempt::success;
}

/// \brief Direct no-copy BLAS GEMV from mdspan-like operands.
template <uni20::BlasScalar Scalar, class OutputMdspan, class MatrixMdspan, class InputMdspan>
  requires detail::writable_gemv_vector_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_gemv_mdspan_for<std::remove_cvref_t<MatrixMdspan>, Scalar, 2> &&
           detail::readable_gemv_mdspan_for<std::remove_cvref_t<InputMdspan>, Scalar, 1>
void gemv(OutputMdspan&& output, Scalar alpha, MatrixMdspan&& matrix, InputMdspan&& input, Scalar beta)
{
  CHECK(kernel_attempt_succeeded(try_gemv(output, alpha, matrix, input, beta)));
}

} // namespace uni20::linalg::blas
