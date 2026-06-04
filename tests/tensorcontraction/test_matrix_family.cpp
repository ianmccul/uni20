#include <uni20/tensorcontraction/matrix_family.hpp>

#include "Matrix.hpp"
#include "MatrixAllocator.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
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

TEST(TensorContractionMatrixFamilyTest, ExposesLogicalHandleAndHostView)
{
  std::array blocks{utc::MatrixFamily::Block{2, 3}};
  utc::MatrixFamily family(blocks);
  auto const& matrix = utc::raw_matrices(family).front();

  auto handle = matrix.handle();
  EXPECT_EQ(handle.id(), matrix.getId());
  EXPECT_EQ(handle.rows(), 2);
  EXPECT_EQ(handle.cols(), 3);
  EXPECT_EQ(handle.nodeId(), matrix.getNodeId());
  EXPECT_EQ(handle.size(), 6);
  EXPECT_EQ(handle.sizeInByte(), 6 * sizeof(double));

  auto host = matrix.hostView();
  EXPECT_TRUE(host.valid());
  EXPECT_EQ(host.handle().id(), handle.id());
  EXPECT_EQ(host.data(), matrix.getPtr());
  EXPECT_EQ(host.memoryKind(), tensor::HostMemoryKind::Pageable);
}

TEST(TensorContractionMatrixFamilyTest, MatrixAllocatorMarksPinnedHostStorage)
{
  tensor::MatrixAllocator allocator;
  auto matrices = allocator.allocateMatrices(std::vector<std::pair<int, int>>{{2, 2}}, false, false);
  ASSERT_EQ(matrices.size(), 1);

  auto host = matrices.front().hostView();
  EXPECT_TRUE(host.valid());
  EXPECT_EQ(host.memoryKind(), tensor::HostMemoryKind::Pinned);
  EXPECT_TRUE(host.pinned());
  allocator.freeAll({}, {}, {});
}

TEST(TensorContractionMatrixFamilyTest, RejectsWrongSizedAssignment)
{
  std::array blocks{utc::MatrixFamily::Block{2, 2}};
  utc::MatrixFamily family(blocks);
  std::vector<double> values{1.0, 2.0, 3.0};

  EXPECT_THROW(family.assign(0, values), std::invalid_argument);
}

TEST(TensorContractionMatrixFamilyTest, CopiesCompatibleFamilyValues)
{
  std::array blocks{utc::MatrixFamily::Block{2, 2}, utc::MatrixFamily::Block{1, 3}};
  utc::MatrixFamily source(blocks);
  utc::MatrixFamily target(blocks);

  source.assign(0, std::array{1.0, 2.0, 3.0, 4.0});
  source.assign(1, std::array{5.0, 6.0, 7.0});
  target.fill(-1.0);
  target.assign(source);

  EXPECT_EQ(target.blocks().size(), blocks.size());
  EXPECT_EQ(target.block(0), blocks[0]);
  EXPECT_EQ(target.block(1), blocks[1]);
  EXPECT_EQ(target.values(0)[0], 1.0);
  EXPECT_EQ(target.values(0)[3], 4.0);
  EXPECT_EQ(target.values(1)[0], 5.0);
  EXPECT_EQ(target.values(1)[2], 7.0);
}

TEST(TensorContractionMatrixFamilyTest, RejectsIncompatibleFamilyCopy)
{
  std::array source_blocks{utc::MatrixFamily::Block{2, 2}};
  std::array target_blocks{utc::MatrixFamily::Block{4, 1}};
  utc::MatrixFamily source(source_blocks);
  utc::MatrixFamily target(target_blocks);

  EXPECT_THROW(target.assign(source), std::invalid_argument);
}

TEST(TensorContractionMatrixFamilyTest, MatrixIdsDoNotHitLegacyTenThousandLimit)
{
  std::vector<utc::MatrixFamily::Block> blocks(10001, utc::MatrixFamily::Block{1, 1});

  utc::MatrixFamily family(blocks);

  ASSERT_EQ(family.size(), blocks.size());
  auto const& matrices = utc::raw_matrices(family);
  ASSERT_EQ(matrices.size(), blocks.size());
  for (std::size_t i = 1; i < matrices.size(); ++i)
  {
    EXPECT_NE(matrices[i - 1].getId(), matrices[i].getId());
  }
}
