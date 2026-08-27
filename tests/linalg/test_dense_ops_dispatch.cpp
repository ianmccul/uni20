#include <uni20/linalg/linalg.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include "deferred_host_tensor.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace
{

TEST(LinalgDenseOpsDispatchTest, OperationTagsProvideCentralDiagnosticNames)
{
  EXPECT_EQ(uni20::linalg::conjugate_inplace_op::name, "conjugate_inplace");
  EXPECT_EQ(uni20::linalg::transform_op<std::plus<>>::name, "transform");
  EXPECT_EQ(uni20::linalg::transform_inplace_op<std::plus<>>::name, "transform_inplace");
  EXPECT_EQ(uni20::linalg::assign_product_op::name, "assign_product");
  EXPECT_EQ(uni20::linalg::gemm_op::name, "gemm");
  EXPECT_EQ(uni20::linalg::gemv_op::name, "gemv");
  EXPECT_EQ(uni20::linalg::inner_product_op::name, "inner_product");
  EXPECT_EQ(uni20::linalg::singular_values_op::name, "singular_values");
  EXPECT_EQ(uni20::linalg::svd_left_op::name, "svd_left");
  EXPECT_EQ(uni20::linalg::svd_right_op::name, "svd_right");
  EXPECT_EQ(uni20::linalg::svd_op::name, "svd");
  EXPECT_EQ(uni20::linalg::norm_op::name, "norm");
  EXPECT_EQ((uni20::linalg::sum_reduction_op<3, 1>::name), "sum");
  EXPECT_EQ(uni20::linalg::matrix_set_op::name, "matrix_set");
  EXPECT_EQ(uni20::linalg::matrix_exponential_op::name, "matrix_exponential");
  EXPECT_EQ(uni20::linalg::symmetric_tridiagonal_eigen_op::name, "symmetric_tridiagonal_eigen");
  EXPECT_EQ(uni20::linalg::nonsymmetric_eigen_op::name, "nonsymmetric_eigen");
  EXPECT_EQ(uni20::linalg::schur_op::name, "schur");
  EXPECT_EQ(uni20::linalg::hessenberg_schur_op::name, "hessenberg_schur");
  EXPECT_EQ(uni20::linalg::schur_reorder_op::name, "reorder_schur");
}

TEST(LinalgDenseOpsDispatchTest, LapackTemporarySizeProductRejectsOverflow)
{
  auto const ordinary = uni20::linalg::lapack_detail::try_size_product(3, 4);
  ASSERT_TRUE(ordinary.has_value());
  EXPECT_EQ(*ordinary, 12);

  auto const overflow = uni20::linalg::lapack_detail::try_size_product(std::numeric_limits<std::size_t>::max(), 2);
  EXPECT_FALSE(overflow.has_value());
}

TEST(LinalgDenseOpsDispatchTest, DenseMatrixCallsGemmWithoutExposingMdspan)
{
  uni20::DenseMatrix<double> lhs(2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);
  uni20::DenseMatrix<double> output(2, 2);
  lhs[0, 0] = 1.0;
  lhs[0, 1] = 2.0;
  lhs[1, 0] = 3.0;
  lhs[1, 1] = 4.0;
  rhs[0, 0] = 5.0;
  rhs[0, 1] = 6.0;
  rhs[1, 0] = 7.0;
  rhs[1, 1] = 8.0;

  uni20::linalg::gemm(output, 1.0, lhs, rhs, 0.0);

  EXPECT_DOUBLE_EQ((output[0, 0]), 19.0);
  EXPECT_DOUBLE_EQ((output[0, 1]), 22.0);
  EXPECT_DOUBLE_EQ((output[1, 0]), 43.0);
  EXPECT_DOUBLE_EQ((output[1, 1]), 50.0);
}

TEST(LinalgDenseOpsDispatchTest, MatrixSetUsesCpuReferenceForEitherOwningLayout)
{
  uni20::DenseMatrix<double> column_major(2, 3);
  uni20::DenseMatrix<double, uni20::RowMajor> row_major(2, 3);

  uni20::linalg::set_matrix(column_major, 4.0, -1.0);
  uni20::linalg::set_matrix(row_major, 4.0, -1.0);

  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type col = 0; col < 3; ++col)
    {
      double const expected = row == col ? 4.0 : -1.0;
      EXPECT_DOUBLE_EQ((column_major[row, col]), expected);
      EXPECT_DOUBLE_EQ((row_major[row, col]), expected);
    }
  }
}

TEST(LinalgDenseOpsDispatchTest, MatrixSetAcquiresDeferredWritableStorage)
{
  uni20::test::DeferredHostTensor<double, 2> matrix(2, 3);

  uni20::linalg::set_matrix(matrix, 4.0, -1.0);

  auto access = uni20::test::acquire_host_read_access_sync(matrix);
  auto span = access.mdspan();
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type col = 0; col < 3; ++col)
      EXPECT_DOUBLE_EQ((span[row, col]), row == col ? 4.0 : -1.0);
}

TEST(LinalgDenseOpsDispatchTest, MatrixExponentialUsesFixedOutputMatrixFrontEnd)
{
  uni20::DenseMatrix<double> input(1, 1);
  uni20::DenseMatrix<double> output(1, 1);
  input[0, 0] = 2.0;

  uni20::linalg::matrix_exponential(output, input, 0.5);

  EXPECT_NEAR((output[0, 0]), std::exp(1.0), 1.0e-14);
}

TEST(LinalgDenseOpsDispatchTest, MatrixExponentialAcquiresDeferredOperands)
{
  uni20::test::DeferredHostTensor<double, 2> input(1, 1);
  uni20::test::DeferredHostTensor<double, 2> output(1, 1);
  input.storage()[0] = 2.0;

  uni20::linalg::matrix_exponential(output, input, 0.5);

  EXPECT_NEAR(output.storage()[0], std::exp(1.0), 1.0e-14);
}

TEST(LinalgDenseOpsDispatchTest, TridiagonalEigenUsesLapackBackend)
{
  std::vector<double> diagonal{2.0, 2.0};
  std::vector<double> subdiagonal{1.0};
  uni20::DenseMatrix<double> eigenvectors(2, 2);

  uni20::linalg::symmetric_tridiagonal_eigen(std::span<double>(diagonal), std::span<double>(subdiagonal), eigenvectors,
                                             true);

  EXPECT_NEAR(diagonal[0], 1.0, 1.0e-14);
  EXPECT_NEAR(diagonal[1], 3.0, 1.0e-14);
  for (uni20::index_type col = 0; col < 2; ++col)
  {
    double const norm = eigenvectors[0, col] * eigenvectors[0, col] + eigenvectors[1, col] * eigenvectors[1, col];
    EXPECT_NEAR(norm, 1.0, 1.0e-14);
  }
}

TEST(LinalgDenseOpsDispatchTest, LapackDeclinePreservesRowMajorTridiagonalOperands)
{
  std::vector<double> diagonal{2.0, 2.0};
  std::vector<double> subdiagonal{1.0};
  uni20::DenseMatrix<double, uni20::RowMajor> eigenvectors(2, 2);
  eigenvectors[0, 0] = 7.0;
  eigenvectors[1, 1] = 9.0;

  std::span<double> diagonal_span(diagonal);
  std::span<double> subdiagonal_span(subdiagonal);
  auto eigenvector_span = eigenvectors.mdspan();
  bool const accepted = uni20::linalg::try_dispatch_kernel(
      uni20::linalg::LapackBackend{}, uni20::linalg::symmetric_tridiagonal_eigen_op{.compute_vectors = true},
      diagonal_span, subdiagonal_span, eigenvector_span);

  EXPECT_FALSE(accepted);
  EXPECT_EQ(diagonal, (std::vector<double>{2.0, 2.0}));
  EXPECT_EQ(subdiagonal, (std::vector<double>{1.0}));
  EXPECT_DOUBLE_EQ((eigenvectors[0, 0]), 7.0);
  EXPECT_DOUBLE_EQ((eigenvectors[1, 1]), 9.0);
}

TEST(LinalgDenseOpsDispatchTest, NonsymmetricEigenUnpacksRealConjugatePairs)
{
  using Complex = uni20::complex<double>;
  uni20::DenseMatrix<double> matrix(2, 2);
  uni20::fill(matrix, 0.0);
  matrix[0, 0] = 1.0;
  matrix[0, 1] = -2.0;
  matrix[1, 0] = 2.0;
  matrix[1, 1] = 1.0;
  std::vector<Complex> eigenvalues(2);
  uni20::DenseMatrix<Complex> right_eigenvectors(2, 2);

  uni20::linalg::nonsymmetric_eigen(matrix, std::span<Complex>(eigenvalues), right_eigenvectors, true);

  EXPECT_NEAR(eigenvalues[0].real(), 1.0, 1.0e-14);
  EXPECT_NEAR(eigenvalues[1].real(), 1.0, 1.0e-14);
  EXPECT_NEAR(std::abs(eigenvalues[0].imag()), 2.0, 1.0e-14);
  EXPECT_EQ(eigenvalues[1], uni20::conj(eigenvalues[0]));
}

TEST(LinalgDenseOpsDispatchTest, SchurAndReorderOperateOnDenseMatrices)
{
  using Complex = uni20::complex<double>;
  uni20::DenseMatrix<double> matrix(2, 2);
  uni20::fill(matrix, 0.0);
  matrix[0, 0] = 1.0;
  matrix[1, 1] = 3.0;
  std::vector<Complex> eigenvalues(2);
  uni20::DenseMatrix<double> schur_vectors(2, 2);

  uni20::linalg::schur(matrix, std::span<Complex>(eigenvalues), schur_vectors, true);
  uni20::linalg::reorder_schur(matrix, schur_vectors, 1, 0, true);

  EXPECT_NEAR((matrix[0, 0]), 3.0, 1.0e-14);
  EXPECT_NEAR((matrix[1, 1]), 1.0, 1.0e-14);
}

} // namespace
