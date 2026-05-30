#include <uni20/tensorcontraction/matrix_family.hpp>

#include "Matrix.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

namespace utc = uni20::tensorcontraction;

TEST(TensorContractionMatrixFamilyTest, OwnsHostStorageAndDescriptors)
{
  std::array blocks{utc::MatrixFamily::Block{2, 3}, utc::MatrixFamily::Block{1, 2}};

  utc::MatrixFamily family(blocks);
  ASSERT_EQ(family.size(), 2);
  EXPECT_FALSE(family.empty());
  EXPECT_EQ(family.block(0).rows, 2);
  EXPECT_EQ(family.block(0).cols, 3);

  family.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  family.fill(4.0);

  auto values = family.values(0);
  ASSERT_EQ(values.size(), 6);
  EXPECT_EQ(values[0], 4.0);
  EXPECT_EQ(values[5], 4.0);

  auto const& matrices = utc::raw_matrices(family);
  ASSERT_EQ(matrices.size(), 2);
  EXPECT_EQ(matrices[0].getFirstDim(), 2);
  EXPECT_EQ(matrices[0].getSecondDim(), 3);
  EXPECT_EQ(matrices[0].getPtr(), values.data());
  EXPECT_EQ(matrices[1].getFirstDim(), 1);
  EXPECT_EQ(matrices[1].getSecondDim(), 2);
}

TEST(TensorContractionMatrixFamilyTest, RejectsWrongSizedAssignment)
{
  std::array blocks{utc::MatrixFamily::Block{2, 2}};
  utc::MatrixFamily family(blocks);
  std::vector<double> values{1.0, 2.0, 3.0};

  EXPECT_THROW(family.assign(0, values), std::invalid_argument);
}
