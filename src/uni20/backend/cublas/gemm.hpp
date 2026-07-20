#pragma once

/**
 * \file gemm.hpp
 * \ingroup backend_cublas
 * \brief Checked column-major GEMM wrappers over a leased cuBLAS execution context.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/backend/cublas/cublas_error.hpp>
#include <uni20/backend/cublas/execution.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/types.hpp>

#include <cuComplex.h>
#include <cublas_v2.h>

#include <algorithm>
#include <concepts>
#include <type_traits>

namespace uni20::cublas
{

/// \brief Scalar types supported by the cuBLAS GEMM wrappers.
template <class Scalar>
concept CublasScalar = std::same_as<Scalar, float> || std::same_as<Scalar, double> ||
                       std::same_as<Scalar, uni20::cfloat> || std::same_as<Scalar, uni20::cdouble>;

namespace detail
{

inline cublasOperation_t operation(char transform)
{
  switch (transform)
  {
    case 'N':
    case 'n':
      return CUBLAS_OP_N;
    case 'T':
    case 't':
      return CUBLAS_OP_T;
    case 'C':
    case 'c':
      return CUBLAS_OP_C;
  }
  PANIC("invalid cuBLAS matrix transform", transform);
}

template <CublasScalar Scalar>
void gemm_call(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int rows, int cols, int inner,
               Scalar const& alpha, Scalar const* lhs, int lhs_leading_dimension, Scalar const* rhs,
               int rhs_leading_dimension, Scalar const& beta, Scalar* output, int output_leading_dimension, int device)
{
  cublasStatus_t status = CUBLAS_STATUS_INTERNAL_ERROR;
  if constexpr (std::same_as<Scalar, float>)
  {
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasSgemm, rows, cols, inner);
    status = cublasSgemm(handle, transa, transb, rows, cols, inner, &alpha, lhs, lhs_leading_dimension, rhs,
                         rhs_leading_dimension, &beta, output, output_leading_dimension);
  }
  else if constexpr (std::same_as<Scalar, double>)
  {
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasDgemm, rows, cols, inner);
    status = cublasDgemm(handle, transa, transb, rows, cols, inner, &alpha, lhs, lhs_leading_dimension, rhs,
                         rhs_leading_dimension, &beta, output, output_leading_dimension);
  }
  else if constexpr (std::same_as<Scalar, uni20::cfloat>)
  {
    static_assert(sizeof(Scalar) == sizeof(cuComplex));
    cuComplex const provider_alpha = make_cuComplex(alpha.real(), alpha.imag());
    cuComplex const provider_beta = make_cuComplex(beta.real(), beta.imag());
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasCgemm, rows, cols, inner);
    status =
        cublasCgemm(handle, transa, transb, rows, cols, inner, &provider_alpha, reinterpret_cast<cuComplex const*>(lhs),
                    lhs_leading_dimension, reinterpret_cast<cuComplex const*>(rhs), rhs_leading_dimension,
                    &provider_beta, reinterpret_cast<cuComplex*>(output), output_leading_dimension);
  }
  else
  {
    static_assert(std::same_as<Scalar, uni20::cdouble>);
    static_assert(sizeof(Scalar) == sizeof(cuDoubleComplex));
    cuDoubleComplex const provider_alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
    cuDoubleComplex const provider_beta = make_cuDoubleComplex(beta.real(), beta.imag());
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasZgemm, rows, cols, inner);
    status = cublasZgemm(handle, transa, transb, rows, cols, inner, &provider_alpha,
                         reinterpret_cast<cuDoubleComplex const*>(lhs), lhs_leading_dimension,
                         reinterpret_cast<cuDoubleComplex const*>(rhs), rhs_leading_dimension, &provider_beta,
                         reinterpret_cast<cuDoubleComplex*>(output), output_leading_dimension);
  }
  check(status, "cublasGemm", device);
}

} // namespace detail

/// \brief Enqueue column-major GEMM through an exclusive handle/stream lease.
template <CublasScalar Scalar>
void gemm(ExecutionLease& execution, char lhs_transform, char rhs_transform, int rows, int cols, int inner,
          Scalar alpha, Scalar const* lhs, int lhs_leading_dimension, Scalar const* rhs, int rhs_leading_dimension,
          Scalar beta, Scalar* output, int output_leading_dimension)
{
  CHECK(execution);
  CHECK(rows >= 0 && cols >= 0 && inner >= 0, rows, cols, inner);
  auto const transa = detail::operation(lhs_transform);
  auto const transb = detail::operation(rhs_transform);
  if (rows == 0 || cols == 0) return;

  int const minimum_lhs_leading_dimension = std::max(1, transa == CUBLAS_OP_N ? rows : inner);
  int const minimum_rhs_leading_dimension = std::max(1, transb == CUBLAS_OP_N ? inner : cols);
  int const minimum_output_leading_dimension = std::max(1, rows);
  CHECK(lhs_leading_dimension >= minimum_lhs_leading_dimension, lhs_leading_dimension, minimum_lhs_leading_dimension);
  CHECK(rhs_leading_dimension >= minimum_rhs_leading_dimension, rhs_leading_dimension, minimum_rhs_leading_dimension);
  CHECK(output_leading_dimension >= minimum_output_leading_dimension, output_leading_dimension,
        minimum_output_leading_dimension);
  cuda::ScopedDevice guard(execution.stream().device());
  detail::gemm_call(execution.handle().native_handle(), transa, transb, rows, cols, inner, alpha, lhs,
                    lhs_leading_dimension, rhs, rhs_leading_dimension, beta, output, output_leading_dimension,
                    execution.stream().device());
}

} // namespace uni20::cublas
