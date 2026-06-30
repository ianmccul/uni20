#include <mplapack_config.h>
#include <uni20/linalg/linalg.hpp>
#include <uni20/tensor/basic_tensor.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>

#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_LDBL)

TEST(MplapackBinary128CpuOpsTest, SkipsLongDoubleAliasMode)
{
  GTEST_SKIP() << "configured MPLAPACK binary128 mode aliases long double";
}

#else

namespace
{

using Binary128 = mplapack_binary128_t;
using index_t = uni20::index_type;
using extents_2d = stdex::dextents<index_t, 2>;
using tensor_type = uni20::BasicTensor<Binary128, extents_2d, uni20::VectorStorage>;

Binary128 abs_error(Binary128 actual, Binary128 expected) { return std::abs(actual - expected); }

Binary128 tolerance() { return static_cast<Binary128>(1.0e-25L); }

Binary128 binary_power_of_two(int exponent)
{
  Binary128 result{1};
  if (exponent >= 0)
  {
    for (int i = 0; i < exponent; ++i)
    {
      result *= Binary128{2};
    }
  }
  else
  {
    for (int i = 0; i < -exponent; ++i)
    {
      result /= Binary128{2};
    }
  }
  return result;
}

Binary128 below_double_resolution_gap() { return binary_power_of_two(-80); }

Binary128 below_double_minimum_value() { return binary_power_of_two(-1200); }

void expect_gap_is_binary128_only(Binary128 gap)
{
  EXPECT_TRUE(Binary128{1} + gap > Binary128{1});
  EXPECT_EQ(static_cast<double>(Binary128{1} + gap), 1.0);
}

void expect_value_underflows_to_double_zero(Binary128 value)
{
  EXPECT_GT(value, Binary128{});
  EXPECT_EQ(static_cast<double>(value), 0.0);
}

} // namespace

TEST(MplapackBinary128CpuOpsTest, TensorMatrixOneNormPreservesBinary128Accumulation)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const one_plus_delta = Binary128{1} + delta;
  expect_gap_is_binary128_only(delta);

  tensor_type matrix(extents_2d{2, 2});
  matrix[0, 0] = one_plus_delta;
  matrix[1, 0] = Binary128{1};
  matrix[0, 1] = Binary128{1};
  matrix[1, 1] = Binary128{};

  auto const norm = uni20::linalg::matrix_one_norm(matrix.const_view());
  static_assert(std::same_as<decltype(norm), Binary128 const>);

  EXPECT_EQ(static_cast<double>(norm), 2.0);
  EXPECT_TRUE(norm > Binary128{2});
  EXPECT_TRUE(abs_error(norm, Binary128{2} + delta) <= tolerance());
}

TEST(MplapackBinary128CpuOpsTest, TensorSolveAcceptsPivotsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  tensor_type matrix(extents_2d{2, 2});
  matrix[0, 0] = tiny;
  matrix[0, 1] = Binary128{};
  matrix[1, 0] = Binary128{};
  matrix[1, 1] = tiny;

  tensor_type rhs(extents_2d{2, 1});
  rhs[0, 0] = tiny;
  rhs[1, 0] = Binary128{2} * tiny;

  auto solution = uni20::linalg::solve_linear_system(matrix.const_view(), rhs.const_view());

  ASSERT_EQ(solution.extents().extent(0), 2);
  ASSERT_EQ(solution.extents().extent(1), 1);
  EXPECT_TRUE(abs_error(solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 0], Binary128{2}) <= tolerance());
}

#endif
