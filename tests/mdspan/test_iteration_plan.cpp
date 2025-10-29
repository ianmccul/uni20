#include "../helpers.hpp"
#include "level1/apply_unary.hpp"
#include "mdspan/iteration_plan.hpp"
#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <vector>

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

TEST(MakeIterationPlanTest, MergedContiguousPlan)
{
  auto mapping = make_mapping(std::array<std::size_t, 3>{10, 20, 30}, std::array<index_t, 3>{1, 10, 200});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 10 * 20 * 30);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, ZeroExtentProducesEmptyPlan)
{
  auto mapping = make_mapping(std::array<std::size_t, 1>{0}, std::array<index_t, 1>{1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_TRUE(plan.empty());
  EXPECT_EQ(offset, 0);

  std::array<double, 3> buffer{1.0, 2.0, 3.0};
  using extents_t = stdex::dextents<index_t, 1>;
  std::array<std::ptrdiff_t, 1> strides{1};
  auto zero_map = stdex::layout_stride::mapping<extents_t>(extents_t{0}, strides);
  stdex::mdspan<double, extents_t, stdex::layout_stride> span(buffer.data(), zero_map);

  apply_unary_inplace(span, [](double x) { return x + 10.0; });

  EXPECT_DOUBLE_EQ(buffer[0], 1.0);
  EXPECT_DOUBLE_EQ(buffer[1], 2.0);
  EXPECT_DOUBLE_EQ(buffer[2], 3.0);
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

TEST(MakeIterationPlanTest, MergeableNegativeStrides)
{
  auto mapping = make_mapping(std::array<std::size_t, 2>{4, 5}, std::array<index_t, 2>{-1, -4});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 20);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(offset, -19);
}
