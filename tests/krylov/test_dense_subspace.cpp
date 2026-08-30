#include <uni20/krylov/dense_subspace.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

namespace
{

TEST(KrylovDenseSubspace, SolvesSymmetricTridiagonalProjection)
{
  auto result =
      uni20::krylov::symmetric_tridiagonal_eigensystem(std::vector<double>{2.0, 2.0}, std::vector<double>{1.0}, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1.0e-14);
  EXPECT_NEAR(result.eigenvalues[1], 3.0, 1.0e-14);
}

TEST(KrylovDenseSubspace, ReordersRealSchurBlocks)
{
  uni20::DenseMatrix<double> matrix(2, 2);
  uni20::krylov::laset(matrix, 0.0, 0.0, uni20::krylov::MatrixFill::All);
  matrix[0, 0] = 1.0;
  matrix[1, 1] = 3.0;

  auto schur = uni20::krylov::real_schur(std::move(matrix), true);
  auto reordered = uni20::krylov::reorder_real_schur(std::move(schur), std::vector<std::size_t>{1});

  ASSERT_EQ(reordered.eigenvalues.size(), 2);
  EXPECT_NEAR(reordered.eigenvalues[0].real(), 3.0, 1.0e-14);
  EXPECT_NEAR(reordered.eigenvalues[1].real(), 1.0, 1.0e-14);
}

TEST(KrylovDenseSubspace, SolvesRealNonsymmetricProjection)
{
  uni20::DenseMatrix<double> matrix(2, 2);
  matrix[0, 0] = 1.0;
  matrix[0, 1] = -2.0;
  matrix[1, 0] = 2.0;
  matrix[1, 1] = 1.0;

  auto result = uni20::krylov::real_nonsymmetric_eigensystem(std::move(matrix), true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(result.eigenvalues[0].real(), 1.0, 1.0e-14);
  EXPECT_NEAR(std::abs(result.eigenvalues[0].imag()), 2.0, 1.0e-14);
  EXPECT_NEAR(result.eigenvalues[1].real(), 1.0, 1.0e-14);
  EXPECT_NEAR(std::abs(result.eigenvalues[1].imag()), 2.0, 1.0e-14);
  EXPECT_EQ(result.right_eigenvectors.rows(), 2);
  EXPECT_EQ(result.right_eigenvectors.cols(), 2);
}

TEST(KrylovDenseSubspace, SolvesComplexNonsymmetricProjection)
{
  using Complex = uni20::complex<double>;

  uni20::DenseMatrix<Complex> matrix(2, 2);
  uni20::krylov::laset(matrix, Complex{}, Complex{}, uni20::krylov::MatrixFill::All);
  matrix[0, 0] = Complex{1.0, 0.0};
  matrix[1, 1] = Complex{3.0, 0.0};

  auto result = uni20::krylov::complex_nonsymmetric_eigensystem<double>(std::move(matrix), true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(result.eigenvalues[0].real(), 1.0, 1.0e-14);
  EXPECT_NEAR(result.eigenvalues[1].real(), 3.0, 1.0e-14);
  EXPECT_EQ(result.right_eigenvectors.rows(), 2);
  EXPECT_EQ(result.right_eigenvectors.cols(), 2);
}

} // namespace
