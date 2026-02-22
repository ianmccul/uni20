#include "linalg/linalg.hpp"
#include "tensor/basic_tensor.hpp"

#include <gtest/gtest.h>

namespace
{
using index_t = uni20::index_type;
using extents_2d = stdex::dextents<index_t, 2>;
using tensor_type = uni20::BasicTensor<double, extents_2d, uni20::VectorStorage>;
} // namespace

TEST(CpuOpsTest, FillIdentity)
{
  extents_2d exts{3, 3};
  tensor_type tensor(exts);

  uni20::linalg::fill_identity(tensor.view());

  for (index_t i = 0; i < exts.extent(0); ++i)
  {
    for (index_t j = 0; j < exts.extent(1); ++j)
    {
      double const expected = (i == j) ? 1.0 : 0.0;
      EXPECT_DOUBLE_EQ((tensor[i, j]), expected);
    }
  }
}

TEST(CpuOpsTest, Multiply)
{
  extents_2d lhs_exts{2, 3};
  extents_2d rhs_exts{3, 2};

  tensor_type lhs(lhs_exts);
  tensor_type rhs(rhs_exts);

  lhs[0, 0] = 1.0;
  lhs[0, 1] = 2.0;
  lhs[0, 2] = 3.0;
  lhs[1, 0] = 4.0;
  lhs[1, 1] = 5.0;
  lhs[1, 2] = 6.0;

  rhs[0, 0] = 7.0;
  rhs[0, 1] = 8.0;
  rhs[1, 0] = 9.0;
  rhs[1, 1] = 10.0;
  rhs[2, 0] = 11.0;
  rhs[2, 1] = 12.0;

  auto result = uni20::linalg::multiply(lhs.view(), rhs.view());
  ASSERT_EQ(result.extents().extent(0), 2);
  ASSERT_EQ(result.extents().extent(1), 2);

  EXPECT_DOUBLE_EQ((result[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 154.0);
}

TEST(CpuOpsTest, SolveLinearSystem)
{
  extents_2d a_exts{2, 2};
  extents_2d b_exts{2, 1};

  tensor_type A(a_exts);
  tensor_type B(b_exts);

  A[0, 0] = 3.0;
  A[0, 1] = 1.0;
  A[1, 0] = 1.0;
  A[1, 1] = 2.0;

  B[0, 0] = 9.0;
  B[1, 0] = 8.0;

  auto solution = uni20::linalg::solve_linear_system(A.view(), B.view());
  ASSERT_EQ(solution.extents().extent(0), 2);
  ASSERT_EQ(solution.extents().extent(1), 1);

  EXPECT_NEAR((solution[0, 0]), 2.0, 1e-12);
  EXPECT_NEAR((solution[1, 0]), 3.0, 1e-12);
}

TEST(CpuOpsTest, MatrixPowerMatchesRepeatedMultiplication)
{
  extents_2d exts{2, 2};
  tensor_type base(exts);

  base[0, 0] = 1.0;
  base[0, 1] = 1.0;
  base[1, 0] = 1.0;
  base[1, 1] = 0.0;

  auto squared = uni20::linalg::matrix_power(base.view(), 2);
  auto cubed = uni20::linalg::matrix_power(base.view(), 3);

  auto manual_squared = uni20::linalg::multiply(base.view(), base.view());
  auto manual_cubed = uni20::linalg::multiply(manual_squared.view(), base.view());

  for (index_t i = 0; i < exts.extent(0); ++i)
  {
    for (index_t j = 0; j < exts.extent(1); ++j)
    {
      EXPECT_DOUBLE_EQ((squared[i, j]), (manual_squared[i, j])) << "Mismatch at squared(" << i << ", " << j << ")";
      EXPECT_DOUBLE_EQ((cubed[i, j]), (manual_cubed[i, j])) << "Mismatch at cubed(" << i << ", " << j << ")";
    }
  }
}
