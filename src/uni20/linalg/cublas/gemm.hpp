#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Checked provider-ready matrix wrappers over cuBLAS GEMM.
 */

#include <uni20/backend/cublas/gemm.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/blas/blas_matrix.hpp>
#include <uni20/linalg/kernel_attempt.hpp>

#include <concepts>
#include <limits>

namespace uni20::linalg::cublas
{
namespace detail
{

inline int cublas_int(blas_int value)
{
  CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), value);
  return static_cast<int>(value);
}

template <uni20::cublas::CublasScalar Scalar, class LhsHandle, class RhsHandle>
constexpr bool provider_transforms_are_supported(blas::BlasReadableMatrix<Scalar, LhsHandle> lhs,
                                                 blas::BlasReadableMatrix<Scalar, RhsHandle> rhs)
{
  return blas::blas_trans_char_is_supported<Scalar>(lhs.transform) &&
         blas::blas_trans_char_is_supported<Scalar>(rhs.transform);
}

template <uni20::cublas::CublasScalar Scalar, class OutputHandle, class LhsHandle, class RhsHandle>
blas_int require_gemm_shape(blas::BlasWritableMatrix<Scalar, OutputHandle> output,
                            blas::BlasReadableMatrix<Scalar, LhsHandle> lhs,
                            blas::BlasReadableMatrix<Scalar, RhsHandle> rhs)
{
  blas_int const lhs_rows = blas::transformed_rows(lhs.rows, lhs.cols, lhs.transform);
  blas_int const lhs_cols = blas::transformed_cols(lhs.rows, lhs.cols, lhs.transform);
  blas_int const rhs_rows = blas::transformed_rows(rhs.rows, rhs.cols, rhs.transform);
  blas_int const rhs_cols = blas::transformed_cols(rhs.rows, rhs.cols, rhs.transform);
  CHECK_EQUAL(lhs_cols, rhs_rows);
  CHECK_EQUAL(output.rows, lhs_rows);
  CHECK_EQUAL(output.cols, rhs_cols);
  return lhs_cols;
}

} // namespace detail

/// \brief Try provider-ready column-major GEMM through cuBLAS.
/// \return `success` or `unsupported_transform`; invalid dimensions are logic errors.
template <uni20::cublas::CublasScalar Scalar, class OutputHandle, class LhsHandle, class RhsHandle>
  requires std::convertible_to<OutputHandle, Scalar*> && std::convertible_to<LhsHandle, Scalar const*> &&
           std::convertible_to<RhsHandle, Scalar const*>
KernelAttempt try_gemm(uni20::cublas::ExecutionLease& execution, blas::BlasWritableMatrix<Scalar, OutputHandle> output,
                       blas::BlasReadableMatrix<Scalar, LhsHandle> lhs, blas::BlasReadableMatrix<Scalar, RhsHandle> rhs,
                       Scalar alpha, Scalar beta)
{
  if (!detail::provider_transforms_are_supported(lhs, rhs)) return KernelAttempt::unsupported_transform;

  blas_int const lhs_cols = detail::require_gemm_shape(output, lhs, rhs);
  if (output.rows == 0 || output.cols == 0) return KernelAttempt::success;

  uni20::cublas::gemm(execution, blas::blas_trans_char<Scalar>(lhs.transform),
                      blas::blas_trans_char<Scalar>(rhs.transform), detail::cublas_int(output.rows),
                      detail::cublas_int(output.cols), detail::cublas_int(lhs_cols), alpha,
                      static_cast<Scalar const*>(lhs.data), detail::cublas_int(lhs.leading_dimension),
                      static_cast<Scalar const*>(rhs.data), detail::cublas_int(rhs.leading_dimension), beta,
                      static_cast<Scalar*>(output.data), detail::cublas_int(output.leading_dimension));
  return KernelAttempt::success;
}

/// \brief Run provider-ready column-major GEMM through cuBLAS.
template <uni20::cublas::CublasScalar Scalar, class OutputHandle, class LhsHandle, class RhsHandle>
  requires std::convertible_to<OutputHandle, Scalar*> && std::convertible_to<LhsHandle, Scalar const*> &&
           std::convertible_to<RhsHandle, Scalar const*>
void gemm(uni20::cublas::ExecutionLease& execution, blas::BlasWritableMatrix<Scalar, OutputHandle> output,
          blas::BlasReadableMatrix<Scalar, LhsHandle> lhs, blas::BlasReadableMatrix<Scalar, RhsHandle> rhs,
          Scalar alpha, Scalar beta)
{
  CHECK(kernel_attempt_succeeded(try_gemm(execution, output, lhs, rhs, alpha, beta)));
}

} // namespace uni20::linalg::cublas
