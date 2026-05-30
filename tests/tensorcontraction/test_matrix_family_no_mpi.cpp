#include <uni20/tensorcontraction/matrix_family.hpp>

#include "Matrix.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <array>

namespace utc = uni20::tensorcontraction;

TEST(TensorContractionMatrixFamilyNoMpiTest, ConstructsDescriptorsBeforeMpiInit)
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  ASSERT_EQ(initialized, 0);

  std::array blocks{utc::MatrixFamily::Block{2, 3}};
  utc::MatrixFamily family(blocks);

  auto const& matrices = utc::raw_matrices(family);
  ASSERT_EQ(matrices.size(), 1);
  EXPECT_EQ(matrices[0].getFirstDim(), 2);
  EXPECT_EQ(matrices[0].getSecondDim(), 3);
  EXPECT_GE(matrices[0].getId(), 0);
}
