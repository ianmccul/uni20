// tests/backend/test_blas_reference_gem.cpp

#include "backend/blas/backend_blas.hpp"
#include "common/types.hpp"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

using namespace uni20;
using namespace std;

//----------------------------------------------------------------------
// Reference implementations
//----------------------------------------------------------------------

/// Simple reference GEMM: C = alpha * op(A) * op(B) + beta * C
template <typename T>
void gemm_ref(char transA, char transB, int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta,
              T* C, int ldc)
{
  for (int i = 0; i < m; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      T sum = T{};
      for (int p = 0; p < k; ++p)
      {
        // column-major indexing:
        T a = (transA == 'T' || transA == 't') ? A[p + i * lda] : A[i + p * lda];
        T b = (transB == 'T' || transB == 't') ? B[j + p * ldb] : B[p + j * ldb];
        sum += a * b;
      }
      C[i + j * ldc] = alpha * sum + beta * C[i + j * ldc];
    }
  }
}

/// Simple reference GEMV: y = alpha * op(A) * x + beta * y
template <typename T>
void gemv_ref(char trans, int m, int n, T alpha, const T* A, int lda, const T* x, int incx, T beta, T* y, int incy)
{
  if (trans == 'N' || trans == 'n')
  {
    // y_i = α ∑_j A(i,j)*x_j + β*y_i    for i=0..m-1
    for (int i = 0; i < m; ++i)
    {
      T sum = T{};
      for (int j = 0; j < n; ++j)
      {
        sum += A[i + j * lda] * x[j * incx]; // col-major: A(i,j)=A[i+j*lda]
      }
      y[i * incy] = alpha * sum + beta * y[i * incy];
    }
  }
  else
  {
    // y_i = α ∑_j A(j,i)*x_j + β*y_i    for i=0..n-1
    for (int i = 0; i < n; ++i)
    {
      T sum = T{};
      for (int j = 0; j < m; ++j)
      {
        sum += A[j + i * lda] * x[j * incx]; // A^T entry
      }
      y[i * incy] = alpha * sum + beta * y[i * incy];
    }
    // leave y[n..m-1] untouched, as BLAS does
  }
}

//----------------------------------------------------------------------
// Tests: real gemm/gemv
//----------------------------------------------------------------------

TEST(BLAS_Wrapper, GemmFloat32)
{
  using T = float32;
  const int m = 2, n = 2, k = 2;
  T alpha = 2, beta = 3;

  std::vector<T> A = {1, 2, 3, 4}; // 2×2 row-major
  std::vector<T> B = {5, 6, 7, 8};
  std::vector<T> C = {1, 1, 1, 1}; // initial C

  // make a copy for reference
  std::vector<T> Creference = C;

  // compute
  uni20::blas::gemm('N', 'N', m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);

  gemm_ref('N', 'N', m, n, k, alpha, A.data(), m, B.data(), k, beta, Creference.data(), m);

  for (int i = 0; i < m * n; ++i)
    EXPECT_FLOAT_EQ(C[i], Creference[i]);
}

TEST(BLAS_Wrapper, GemmFloat64)
{
  using T = float64;
  const int m = 2, n = 2, k = 2;
  T alpha = 1.5, beta = -0.5;

  std::vector<T> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<T> B = {5.0, 6.0, 7.0, 8.0};
  std::vector<T> C = {2.0, 2.0, 2.0, 2.0};
  auto Creference = C;

  uni20::blas::gemm('N', 'T', m, n, k, alpha, A.data(), m, B.data(), n, // note transpose B
                    beta, C.data(), m);

  gemm_ref('N', 'T', m, n, k, alpha, A.data(), m, B.data(), n, beta, Creference.data(), m);

  for (int i = 0; i < m * n; ++i)
    EXPECT_DOUBLE_EQ(C[i], Creference[i]);
}

TEST(BLAS_Wrapper, GemvFloat32)
{
  using T = float32;
  const int m = 2, n = 2;
  T alpha = 2, beta = 1;

  // A is 2×2:
  std::vector<T> A = {1, 2, 3, 4};
  std::vector<T> x = {1, -1};
  std::vector<T> y = {0, 5};
  auto yref = y;

  uni20::blas::gemv('N', m, n, alpha, A.data(), m, x.data(), 1, beta, y.data(), 1);

  gemv_ref('N', m, n, alpha, A.data(), m, x.data(), 1, beta, yref.data(), 1);

  for (int i = 0; i < m; ++i)
    EXPECT_FLOAT_EQ(y[i], yref[i]);
}

TEST(BLAS_Wrapper, GemvFloat64)
{
  using T = float64;
  const int m = 3, n = 2;
  T alpha = 0.5, beta = 2;

  std::vector<T> A = {1, 2, 3, 4, 5, 6}; // 3×2
  std::vector<T> x = {1, 2};
  std::vector<T> y = {0, 1, 2};
  auto yref = y;

  uni20::blas::gemv('T', m, n, alpha, A.data(), m, x.data(), 1, beta, y.data(), 1);

  gemv_ref('T', m, n, alpha, A.data(), m, x.data(), 1, beta, yref.data(), 1);

  for (int i = 0; i < m; ++i)
    EXPECT_DOUBLE_EQ(y[i], yref[i]);
}

//----------------------------------------------------------------------
// Tests: complex gemm/gemv
//----------------------------------------------------------------------

TEST(BLAS_Wrapper, GemmComplex64)
{
  using T = complex64;
  const int m = 2, n = 2, k = 2;
  T alpha{1, 1}, beta{0, 1};

  std::vector<T> A = {{1, 1}, {2, 0}, {0, 3}, {-1, 2}};
  std::vector<T> B = {{1, 0}, {0, 1}, {2, 2}, {-2, 0}};
  std::vector<T> C = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
  auto Cref = C;

  uni20::blas::gemm('N', 'N', m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);

  gemm_ref('N', 'N', m, n, k, alpha, A.data(), m, B.data(), k, beta, Cref.data(), m);

  for (int i = 0; i < m * n; ++i)
  {
    EXPECT_NEAR(real(C[i]), real(Cref[i]), 1e-6);
    EXPECT_NEAR(imag(C[i]), imag(Cref[i]), 1e-6);
  }
}

TEST(BLAS_Wrapper, GemvComplex128)
{
  using T = complex128;
  const int m = 2, n = 2;
  T alpha{2, -1}, beta{-1, 2};

  std::vector<T> A = {{1, 0}, {0, 1}, {1, -1}, {2, 2}};
  std::vector<T> x = {{1, 1}, {-1, 0}};
  std::vector<T> y = {{0, 0}, {1, 1}};
  auto yref = y;

  uni20::blas::gemv('T', m, n, alpha, A.data(), m, x.data(), 1, beta, y.data(), 1);

  gemv_ref('T', m, n, alpha, A.data(), m, x.data(), 1, beta, yref.data(), 1);

  for (int i = 0; i < m; ++i)
  {
    EXPECT_NEAR(real(y[i]), real(yref[i]), 1e-12);
    EXPECT_NEAR(imag(y[i]), imag(yref[i]), 1e-12);
  }
}
