#include <uni20/core/math.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace
{
uni20::DenseMatrix<double> make_diagonal_matrix()
{
  uni20::DenseMatrix<double> matrix(4, 4);
  std::ranges::fill(matrix.storage(), 0.0);
  matrix[0, 0] = 4.0;
  matrix[1, 1] = 3.0;
  matrix[2, 2] = 1.0;
  matrix[3, 3] = 0.5;
  return matrix;
}

template <class Matrix, class Result> double relative_reconstruction_error(Matrix const& matrix, Result const& result)
{
  double error_squared = 0.0;
  double norm_squared = 0.0;
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
  {
    for (uni20::index_type column = 0; column < matrix.cols(); ++column)
    {
      typename Matrix::value_type reconstructed{};
      for (uni20::index_type inner = 0; inner < result.singular_values.extent(0); ++inner)
      {
        reconstructed += result.left_singular_vectors[row, inner] * result.singular_values[inner] *
                         result.right_singular_vectors_adjoint[inner, column];
      }
      error_squared += std::norm(reconstructed - matrix[row, column]);
      norm_squared += std::norm(matrix[row, column]);
    }
  }
  return norm_squared == 0.0 ? 0.0 : error_squared / norm_squared;
}
} // namespace

TEST(TruncatedSvdTest, DefaultPolicyReturnsExactReducedDecomposition)
{
  auto const matrix = make_diagonal_matrix();
  auto result = uni20::linalg::truncated_svd(matrix);

  ASSERT_EQ(result.singular_values.extent(0), 4);
  EXPECT_EQ(result.left_singular_vectors.rows(), 4);
  EXPECT_EQ(result.left_singular_vectors.cols(), 4);
  EXPECT_EQ(result.right_singular_vectors_adjoint.rows(), 4);
  EXPECT_EQ(result.right_singular_vectors_adjoint.cols(), 4);
  EXPECT_EQ(result.truncation.available_rank, 4);
  EXPECT_EQ(result.truncation.retained_rank, 4);
  EXPECT_DOUBLE_EQ(result.truncation.original_squared_norm, 26.25);
  EXPECT_DOUBLE_EQ(result.truncation.discarded_weight, 0.0);
  ASSERT_TRUE(result.truncation.smallest_retained_singular_value);
  EXPECT_DOUBLE_EQ(*result.truncation.smallest_retained_singular_value, 0.5);
  EXPECT_FALSE(result.truncation.largest_discarded_singular_value);
  EXPECT_LE(relative_reconstruction_error(matrix, result), 1.0e-28);
}

TEST(TruncatedSvdTest, CutoffsAndDiscardedWeightSelectExpectedRank)
{
  auto const matrix = make_diagonal_matrix();

  auto absolute =
      uni20::linalg::truncated_svd(matrix, uni20::linalg::SvdTruncationPolicy<double>{.singular_value_cutoff = 3.0});
  EXPECT_EQ(absolute.truncation.retained_rank, 2);

  auto normalized = uni20::linalg::truncated_svd(
      matrix, uni20::linalg::SvdTruncationPolicy<double>{.normalized_squared_singular_value_cutoff = 0.1});
  EXPECT_EQ(normalized.truncation.retained_rank, 2);

  auto discarded = uni20::linalg::truncated_svd(
      matrix, uni20::linalg::SvdTruncationPolicy<double>{.maximum_discarded_weight = 0.05});
  EXPECT_EQ(discarded.truncation.retained_rank, 2);
  EXPECT_NEAR(discarded.truncation.discarded_weight, 1.25 / 26.25, 1.0e-15);
  EXPECT_NEAR(relative_reconstruction_error(matrix, discarded), discarded.truncation.discarded_weight, 1.0e-15);
  ASSERT_TRUE(discarded.truncation.smallest_retained_singular_value);
  ASSERT_TRUE(discarded.truncation.largest_discarded_singular_value);
  EXPECT_DOUBLE_EQ(*discarded.truncation.smallest_retained_singular_value, 3.0);
  EXPECT_DOUBLE_EQ(*discarded.truncation.largest_discarded_singular_value, 1.0);
}

TEST(TruncatedSvdTest, ExtentBoundsApplyAfterAccuracyCriteria)
{
  auto const matrix = make_diagonal_matrix();

  auto minimum = uni20::linalg::truncated_svd(
      matrix, uni20::linalg::SvdTruncationPolicy<double>{.minimum_retained_extent = 2, .singular_value_cutoff = 3.5});
  EXPECT_EQ(minimum.truncation.retained_rank, 2);

  auto maximum = uni20::linalg::truncated_svd(
      matrix,
      uni20::linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = 1, .maximum_discarded_weight = 0.01});
  EXPECT_EQ(maximum.truncation.retained_rank, 1);
  EXPECT_GT(maximum.truncation.discarded_weight, 0.01);
}

TEST(TruncatedSvdTest, PositiveCutoffCanReturnZeroRank)
{
  uni20::DenseMatrix<double> matrix(3, 2);
  std::ranges::fill(matrix.storage(), 0.0);
  auto result =
      uni20::linalg::truncated_svd(matrix, uni20::linalg::SvdTruncationPolicy<double>{.singular_value_cutoff = 1.0});

  EXPECT_EQ(result.left_singular_vectors.rows(), 3);
  EXPECT_EQ(result.left_singular_vectors.cols(), 0);
  EXPECT_EQ(result.singular_values.extent(0), 0);
  EXPECT_EQ(result.right_singular_vectors_adjoint.rows(), 0);
  EXPECT_EQ(result.right_singular_vectors_adjoint.cols(), 2);
  EXPECT_EQ(result.truncation.available_rank, 2);
  EXPECT_EQ(result.truncation.retained_rank, 0);
  EXPECT_DOUBLE_EQ(result.truncation.original_squared_norm, 0.0);
  EXPECT_DOUBLE_EQ(result.truncation.discarded_weight, 0.0);
  EXPECT_FALSE(result.truncation.smallest_retained_singular_value);
  ASSERT_TRUE(result.truncation.largest_discarded_singular_value);
  EXPECT_DOUBLE_EQ(*result.truncation.largest_discarded_singular_value, 0.0);
}

TEST(TruncatedSvdTest, HandlesComplexInputAndConsumingOwner)
{
  using scalar_type = uni20::complex<double>;
  uni20::DenseMatrix<scalar_type> matrix(2, 3);
  matrix[0, 0] = scalar_type{1.0, 2.0};
  matrix[0, 1] = scalar_type{-3.0, 0.5};
  matrix[0, 2] = scalar_type{0.0, -1.0};
  matrix[1, 0] = scalar_type{2.0, -1.0};
  matrix[1, 1] = scalar_type{0.5, 4.0};
  matrix[1, 2] = scalar_type{-2.0, 3.0};

  auto result = uni20::linalg::truncated_svd(std::move(matrix),
                                             uni20::linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = 1});

  static_assert(std::same_as<typename decltype(result.singular_values)::value_type, double>);
  EXPECT_EQ(result.left_singular_vectors.rows(), 2);
  EXPECT_EQ(result.left_singular_vectors.cols(), 1);
  EXPECT_EQ(result.singular_values.extent(0), 1);
  EXPECT_EQ(result.right_singular_vectors_adjoint.rows(), 1);
  EXPECT_EQ(result.right_singular_vectors_adjoint.cols(), 3);
  EXPECT_EQ(result.truncation.available_rank, 2);
  EXPECT_EQ(result.truncation.retained_rank, 1);
  EXPECT_GT(result.truncation.discarded_weight, 0.0);
  EXPECT_LT(result.truncation.discarded_weight, 1.0);
}

TEST(TruncatedSvdTest, ScaledStatisticsAvoidIntermediateOverflow)
{
  uni20::DenseMatrix<double> matrix(2, 2);
  std::ranges::fill(matrix.storage(), 0.0);
  matrix[0, 0] = 1.0e200;
  matrix[1, 1] = 1.0e199;

  auto result =
      uni20::linalg::truncated_svd(matrix, uni20::linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = 1});

  EXPECT_FALSE(uni20::isfinite(result.truncation.original_squared_norm));
  EXPECT_GT(result.truncation.original_squared_norm, 0.0);
  EXPECT_NEAR(result.truncation.discarded_weight, 1.0 / 101.0, 1.0e-15);
}

TEST(TruncatedSvdTest, ZeroAvailableRankReturnsEmptyFactors)
{
  uni20::DenseMatrix<double> matrix(0, 3);
  auto result = uni20::linalg::truncated_svd(matrix);

  EXPECT_EQ(result.left_singular_vectors.rows(), 0);
  EXPECT_EQ(result.left_singular_vectors.cols(), 0);
  EXPECT_EQ(result.singular_values.extent(0), 0);
  EXPECT_EQ(result.right_singular_vectors_adjoint.rows(), 0);
  EXPECT_EQ(result.right_singular_vectors_adjoint.cols(), 3);
  EXPECT_EQ(result.truncation.available_rank, 0);
  EXPECT_EQ(result.truncation.retained_rank, 0);
}
