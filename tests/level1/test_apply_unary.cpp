#include "helpers.hpp"
#include "level1/apply_unary.hpp"
#include "gtest/gtest.h"
#include <numeric>

using namespace uni20;

TEST(MakeIterationPlanTest, SimpleContiguousPlan)
{
  auto mapping = make_mapping(std::array<std::size_t, 1>{10}, std::array<index_t, 1>{1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(plan[0].extent, 10);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, CoalescedContiguousPlan)
{
  auto mapping = make_mapping(std::array<std::size_t, 3>{10, 20, 30}, std::array<index_t, 3>{1, 10, 200});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 10 * 20 * 30);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, OutOfOrderStrides)
{
  auto mapping = make_mapping(std::array<std::size_t, 3>{30, 20, 10}, std::array<index_t, 3>{200, 10, 1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 10 * 20 * 30);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, InnerNegativeStride)
{
  auto mapping = make_mapping(std::array<std::size_t, 1>{10}, std::array<index_t, 1>{-1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(plan[0].extent, 10);
  EXPECT_EQ(offset, -9);
}

TEST(MakeIterationPlanTest, OuterNegativeStride)
{
  auto mapping = make_mapping(std::array<std::size_t, 2>{4, 2}, std::array<index_t, 2>{-8, 1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 2);
  EXPECT_EQ(plan[0].extent, 4);
  EXPECT_EQ(plan[0].stride, 8);
  EXPECT_EQ(offset, -24);
}

TEST(MakeIterationPlanTest, NegativeStrideMiddleDimension)
{
  auto mapping = make_mapping(std::array<std::size_t, 3>{4, 3, 2}, std::array<index_t, 3>{1, -4, 20});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_EQ(plan.size(), 2);
  EXPECT_EQ(offset, -8);
}

TEST(MakeIterationPlanTest, MixedSignsNoMerge)
{
  auto mapping = make_mapping(std::array<std::size_t, 2>{4, 3}, std::array<index_t, 2>{-7, 1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_EQ(plan.size(), 2);    // Can't merge
  EXPECT_EQ(plan[0].extent, 4); // Outer
  EXPECT_EQ(plan[0].stride, 7);
  EXPECT_EQ(plan[1].extent, 3); // Inner
  EXPECT_EQ(plan[1].stride, 1);
  EXPECT_EQ(offset, -21); // offset = (extent-1)*stride = (4-1)*-7 = -21
}

TEST(MakeIterationPlanTest, CoalescableNegativeStrides)
{
  auto mapping = make_mapping(std::array<std::size_t, 2>{4, 5}, std::array<index_t, 2>{-1, -4});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 20);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(offset, -19);
}

TEST(ApplyUnaryInplace, MultiplyBy2_1DContiguous)
{
  std::vector<double> v(10);
  std::iota(v.begin(), v.end(), 0.0);
  auto m = make_mdspan_1d(v);

  apply_unary_inplace(m, [](double x) { return x * 2; });

  for (size_t i = 0; i < 10; ++i)
  {
    EXPECT_DOUBLE_EQ(v[i], static_cast<double>(i * 2));
  }
}

TEST(ApplyUnaryInplace, Add5_2DRowMajor)
{
  std::size_t R = 3, C = 4;
  std::vector<double> v(R * C);
  for (size_t i = 0; i < R; i++)
    for (size_t j = 0; j < C; j++)
      v[i * C + j] = static_cast<double>(i * 10 + j);

  auto m = make_mdspan_2d(v, R, C);
  apply_unary_inplace(m, [](double x) { return x + 5; });

  for (size_t i = 0; i < R; i++)
    for (size_t j = 0; j < C; j++)
      EXPECT_DOUBLE_EQ(v[i * C + j], static_cast<double>(i * 10 + j + 5));
}

TEST(ApplyUnaryInplace, Square_Reversed1D)
{
  std::vector<double> v(8);
  std::iota(v.begin(), v.end(), 1.0); // 1,2,3,...,8
  auto m = make_reversed_1d(v);

  apply_unary_inplace(m, [](double x) { return x * x; });

  // reversed view means m[i] = original v[7-i]
  // after squaring, the underlying container should be:
  // v[7] = 1^2, v[6] = 2^2, ..., v[0] = 8^2
  for (size_t i = 0; i < 8; i++)
  {
    EXPECT_DOUBLE_EQ(v[i], std::pow(static_cast<double>(i + 1), 2));
  }
}

TEST(ApplyUnaryInplace, ScaleAndShiftMixedStrides)
{
  // Create a 3×3 buffer, but view it with non‐unit stride in one dim:
  std::vector<double> buf(3 * 5, 0.0);
  // Fill buffer row‑major in the first 3 columns of each row
  for (index_t r = 0; r < 3; ++r)
    for (index_t c = 0; c < 3; ++c)
      buf[r * 5 + c * 2] = static_cast<double>(r * 3 + c);

  using extents_t = stdex::dextents<index_t, 2>;
  // extents = (3,3), strides = (5,2)  —  row step = 5, col step = 2
  std::array<std::ptrdiff_t, 2> strides{5, 2};
  auto mapping = stdex::layout_stride::mapping<extents_t>(extents_t{3, 3}, strides);
  stdex::mdspan<double, extents_t, stdex::layout_stride> m(buf.data(), mapping);

  apply_unary_inplace(m, [](double x) { return x * 10 - 1; });

  for (index_t r = 0; r < 3; ++r)
  {
    for (index_t c = 0; c < 3; ++c)
    {
      // the element at (r,c) lives in buf[r*5 + c*2]
      auto v = buf[r * 5 + c * 2];
      EXPECT_DOUBLE_EQ(v, static_cast<double>((r * 3 + c) * 10 - 1));
    }
  }
}
