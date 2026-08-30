#include <uni20/linalg/ops/matrix_norm.hpp>
#include <uni20/mdspan/transform_view.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include "deferred_host_tensor.hpp"

#include <cmath>
#include <concepts>
#include <type_traits>
#include <utility>

namespace
{

struct DoubleValue
{
    double operator()(double value) const { return 2.0 * value; }
};

template <class Matrix> void initialize_reference_matrix(Matrix& matrix)
{
  matrix[0, 0] = 1.0;
  matrix[0, 1] = -2.0;
  matrix[0, 2] = 3.0;
  matrix[1, 0] = -4.0;
  matrix[1, 1] = 5.0;
  matrix[1, 2] = -6.0;
}

template <class Matrix> void check_reference_norms(Matrix const& matrix)
{
  using uni20::linalg::MatrixNorm;
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::MaxAbs), 6.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::One), 9.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::Infinity), 15.0);
  EXPECT_NEAR(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::Frobenius), std::sqrt(91.0), 1.0e-14);
}

} // namespace

TEST(MatrixNormTest, ComputesAllNormsForColumnMajorMatrix)
{
  uni20::DenseMatrix<double> matrix(2, 3);
  initialize_reference_matrix(matrix);

  check_reference_norms(matrix);

  auto result = uni20::linalg::matrix_norm(matrix, uni20::linalg::MatrixNorm::One);
  static_assert(std::same_as<decltype(result), uni20::ScalarTensor<double>>);
  EXPECT_DOUBLE_EQ(result[], 9.0);
}

TEST(MatrixNormTest, LapackAccountsForRowMajorTranspose)
{
  uni20::DenseMatrix<double, uni20::RowMajor> matrix(2, 3);
  initialize_reference_matrix(matrix);

  using uni20::linalg::LapackBackend;
  using uni20::linalg::MatrixNorm;
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(LapackBackend{}, matrix, MatrixNorm::One), 9.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(LapackBackend{}, matrix, MatrixNorm::Infinity), 15.0);
}

TEST(MatrixNormTest, ComplexNormsUseMathematicalMagnitude)
{
  using complex_type = uni20::complex<double>;
  uni20::DenseMatrix<complex_type> matrix(1, 1);
  matrix[0, 0] = complex_type{3.0, 4.0};

  using uni20::linalg::MatrixNorm;
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::MaxAbs), 5.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::One), 5.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::Infinity), 5.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::Frobenius), 5.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(uni20::linalg::LapackBackend{}, matrix, MatrixNorm::Frobenius), 5.0);
}

TEST(MatrixNormTest, ComplexLapackDeclinePreservesOutputForComponentSumNorms)
{
  using complex_type = uni20::complex<double>;
  uni20::DenseMatrix<complex_type> matrix(1, 1);
  matrix[0, 0] = complex_type{3.0, 4.0};
  double output = 17.0;
  auto descriptor = uni20::mdspec_of(std::as_const(matrix));

  bool const accepted = uni20::linalg::try_dispatch_kernel(
      uni20::linalg::LapackBackend{}, uni20::linalg::matrix_norm_op{.kind = uni20::linalg::MatrixNorm::MaxAbs}, output,
      descriptor);

  EXPECT_FALSE(accepted);
  EXPECT_DOUBLE_EQ(output, 17.0);
}

TEST(MatrixNormTest, AcquiresDeferredHostInput)
{
  uni20::test::DeferredHostTensor<double, 2> matrix(2, 3);
  auto access = uni20::test::acquire_host_write_access_sync(matrix);
  auto span = access.mdspan();
  initialize_reference_matrix(span);
  access.release();

  check_reference_norms(matrix);
}

TEST(MatrixNormTest, CpuReferenceObservesTransformAccessorSemantics)
{
  uni20::DenseMatrix<double> matrix(2, 3);
  initialize_reference_matrix(matrix);
  auto transformed = uni20::transform_view(DoubleValue{}, std::as_const(matrix).mdspan());

  double const result = uni20::linalg::matrix_norm_host(uni20::linalg::CpuReferenceBackend{}, transformed,
                                                        uni20::linalg::MatrixNorm::One);

  EXPECT_DOUBLE_EQ(result, 18.0);
}

TEST(MatrixNormTest, EmptyMatricesHaveZeroNorm)
{
  uni20::DenseMatrix<double> matrix(0, 3);
  using uni20::linalg::MatrixNorm;
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::MaxAbs), 0.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::One), 0.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::Infinity), 0.0);
  EXPECT_DOUBLE_EQ(uni20::linalg::matrix_norm_host(matrix, MatrixNorm::Frobenius), 0.0);
}
