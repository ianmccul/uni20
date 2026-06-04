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

std::vector<double> reconstruct(utc::SingleBlockSvdSplit const& split, utc::SingleBlockSvdSplitLayout layout)
{
  auto const rank = split.spectrum.singular_values.size();
  auto const rows = layout.left_bond_dim * layout.left_physical_dim;
  auto const cols = layout.right_physical_dim * layout.right_bond_dim;
  std::vector<double> result(rows * cols, 0.0);
  for (std::size_t left_bond = 0; left_bond < layout.left_bond_dim; ++left_bond)
  {
    for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
    {
      auto const left_values = split.left.values(left_phys);
      auto const row = left_bond * layout.left_physical_dim + left_phys;
      for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
      {
        auto const right_values = split.right.values(right_phys);
        for (std::size_t right_bond = 0; right_bond < layout.right_bond_dim; ++right_bond)
        {
          auto const col = right_phys * layout.right_bond_dim + right_bond;
          for (std::size_t bond = 0; bond < rank; ++bond)
          {
            result[row * cols + col] +=
                left_values[left_bond * rank + bond] * right_values[bond * layout.right_bond_dim + right_bond];
          }
        }
      }
    }
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

TEST(TensorContractionSvdTest, ReconstructsWideSingleBlock)
{
  auto matrix = make_matrix(2, 4, {1.0, 2.0, 0.0, 1.0, 0.5, -1.0, 3.0, 0.0});

  auto svd = utc::single_block_svd(matrix);
  auto reconstructed = reconstruct(svd);

  EXPECT_EQ(svd.u.block(0), (utc::MatrixFamily::Block{2, 2}));
  EXPECT_EQ(svd.vt.block(0), (utc::MatrixFamily::Block{2, 4}));
  EXPECT_EQ(svd.singular_values.size(), 2);
  EXPECT_GE(svd.singular_values[0], svd.singular_values[1]);
  EXPECT_NEAR(svd.discarded_weight, 0.0, 1.0e-12);
  EXPECT_EQ(svd.full_rank, 2);
  for (std::size_t i = 0; i < reconstructed.size(); ++i)
  {
    EXPECT_NEAR(reconstructed[i], matrix.values(0)[i], 1.0e-12);
  }
}

TEST(TensorContractionSvdTest, ReferenceFallbackReconstructsFullRankSingleBlock)
{
  auto matrix = make_matrix(3, 2, {3.0, 1.0, 0.0, 2.0, 0.0, 0.0});

  auto svd = utc::single_block_svd_reference(matrix);
  auto reconstructed = reconstruct(svd);

  EXPECT_EQ(svd.singular_values.size(), 2);
  EXPECT_NEAR(svd.discarded_weight, 0.0, 1.0e-12);
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

TEST(TensorContractionSvdTest, SplitsPhysicalBlocksAndAbsorbsSingularValues)
{
  utc::SingleBlockSvdSplitLayout layout{
      .left_bond_dim = 2, .left_physical_dim = 3, .right_physical_dim = 2, .right_bond_dim = 2};
  auto matrix = make_matrix(6, 4, {1.0, 0.2,  -0.4, 0.7, 0.3, 1.1, 0.5,  -0.2, -0.6, 0.8,  1.7, 0.4,
                                   0.9, -0.1, 0.2,  1.3, 0.4, 0.6, -1.2, 0.5,  1.5,  -0.7, 0.3, 0.9});

  for (auto absorb : {utc::SvdAbsorbSingularValues::Left, utc::SvdAbsorbSingularValues::Right})
  {
    auto split = utc::single_block_svd_split(matrix, layout, absorb);
    auto reconstructed = reconstruct(split, layout);

    ASSERT_EQ(split.left.size(), layout.left_physical_dim);
    ASSERT_EQ(split.right.size(), layout.right_physical_dim);
    ASSERT_EQ(split.spectrum.singular_values.size(), 4);
    EXPECT_EQ(split.left.block(0), (utc::MatrixFamily::Block{layout.left_bond_dim, 4}));
    EXPECT_EQ(split.right.block(0), (utc::MatrixFamily::Block{4, layout.right_bond_dim}));
    EXPECT_NEAR(split.spectrum.discarded_weight, 0.0, 1.0e-12);
    for (std::size_t i = 0; i < reconstructed.size(); ++i)
    {
      EXPECT_NEAR(reconstructed[i], matrix.values(0)[i], 1.0e-10);
    }
  }
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
