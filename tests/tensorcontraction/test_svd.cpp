#include <uni20/tensorcontraction/svd.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace utc = uni20::tensorcontraction;

namespace
{

utc::MatrixFamily make_matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> values)
{
  std::array blocks{utc::MatrixFamily::Block{rows, cols}};
  utc::MatrixFamily matrix(blocks);
  matrix.assign(0, std::span{values.begin(), values.size()});
  return matrix;
}

std::vector<double> reconstruct(utc::SingleBlockSvd const& svd)
{
  auto const u_block = svd.u.block(0);
  auto const vt_block = svd.vt.block(0);
  auto const rows = u_block.rows;
  auto const rank = u_block.cols;
  auto const cols = vt_block.cols;
  auto const u = svd.u.values(0);
  auto const vt = svd.vt.values(0);
  std::vector<double> result(rows * cols, 0.0);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      for (std::size_t k = 0; k < rank; ++k)
      {
        result[row * cols + col] += u[row * rank + k] * svd.singular_values[k] * vt[k * cols + col];
      }
    }
  }
  return result;
}

double dot_columns(utc::MatrixFamily const& matrix, std::size_t lhs, std::size_t rhs)
{
  auto const block = matrix.block(0);
  auto const values = matrix.values(0);
  double result = 0.0;
  for (std::size_t row = 0; row < block.rows; ++row)
  {
    result += values[row * block.cols + lhs] * values[row * block.cols + rhs];
  }
  return result;
}

} // namespace

TEST(TensorContractionSvdTest, ReconstructsFullRankSingleBlock)
{
  auto matrix = make_matrix(3, 2, {3.0, 1.0, 0.0, 2.0, 0.0, 0.0});

  auto svd = utc::single_block_svd(matrix);
  auto reconstructed = reconstruct(svd);

  EXPECT_EQ(svd.u.block(0), (utc::MatrixFamily::Block{3, 2}));
  EXPECT_EQ(svd.vt.block(0), (utc::MatrixFamily::Block{2, 2}));
  EXPECT_EQ(svd.singular_values.size(), 2);
  EXPECT_GE(svd.singular_values[0], svd.singular_values[1]);
  EXPECT_NEAR(svd.discarded_weight, 0.0, 1.0e-12);
  EXPECT_EQ(svd.full_rank, 2);
  for (std::size_t i = 0; i < reconstructed.size(); ++i)
  {
    EXPECT_NEAR(reconstructed[i], matrix.values(0)[i], 1.0e-12);
  }
}

TEST(TensorContractionSvdTest, ProducesOrthonormalKeptLeftVectors)
{
  auto matrix = make_matrix(3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 7.0});

  auto svd = utc::single_block_svd(matrix);

  EXPECT_NEAR(dot_columns(svd.u, 0, 0), 1.0, 1.0e-12);
  EXPECT_NEAR(dot_columns(svd.u, 1, 1), 1.0, 1.0e-12);
  EXPECT_NEAR(dot_columns(svd.u, 0, 1), 0.0, 1.0e-12);
}

TEST(TensorContractionSvdTest, TruncatesByMaximumRankAndReportsDiscardedWeight)
{
  auto matrix = make_matrix(3, 3, {4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0});

  auto svd = utc::single_block_svd(matrix, utc::SvdOptions{.max_rank = 2});
  auto reconstructed = reconstruct(svd);

  EXPECT_EQ(svd.singular_values.size(), 2);
  EXPECT_NEAR(svd.singular_values[0], 4.0, 1.0e-12);
  EXPECT_NEAR(svd.singular_values[1], 2.0, 1.0e-12);
  EXPECT_NEAR(svd.discarded_weight, 1.0, 1.0e-12);
  EXPECT_NEAR(reconstructed[0], 4.0, 1.0e-12);
  EXPECT_NEAR(reconstructed[4], 2.0, 1.0e-12);
  EXPECT_NEAR(reconstructed[8], 0.0, 1.0e-12);
}

TEST(TensorContractionSvdTest, TruncatesByCutoff)
{
  auto matrix = make_matrix(3, 3, {5.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.125});

  auto svd = utc::single_block_svd(matrix, utc::SvdOptions{.cutoff = 0.2});

  EXPECT_EQ(svd.singular_values.size(), 2);
  EXPECT_NEAR(svd.singular_values[0], 5.0, 1.0e-12);
  EXPECT_NEAR(svd.singular_values[1], 0.25, 1.0e-12);
  EXPECT_NEAR(svd.discarded_weight, 0.125 * 0.125, 1.0e-12);
}

TEST(TensorContractionSvdTest, RejectsUnsupportedShapesAndOptions)
{
  std::array blocks{utc::MatrixFamily::Block{1, 1}, utc::MatrixFamily::Block{1, 1}};
  utc::MatrixFamily two_blocks(blocks);
  EXPECT_THROW(utc::single_block_svd(two_blocks), std::invalid_argument);

  auto matrix = make_matrix(1, 1, {1.0});
  EXPECT_THROW(utc::single_block_svd(matrix, utc::SvdOptions{.max_rank = 0}), std::invalid_argument);
  EXPECT_THROW(utc::single_block_svd(matrix, utc::SvdOptions{.cutoff = -1.0}), std::invalid_argument);
}
