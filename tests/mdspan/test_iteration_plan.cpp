#include "../helpers.hpp"
#include "gtest/gtest.h"
#include <uni20/mdspan/iteration_plan.hpp>

#include <array>
#include <cstddef>
#include <tuple>

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

TEST(MakeIterationPlanTest, ZeroExtentProducesRetainedZeroDim)
{
  // A zero-extent dim means 0 elements. It is carried as a single retained
  // extent-0 dim -- NOT an empty plan, which denotes a rank-0 scalar (1
  // element). The driver then loops that dim zero times.
  auto mapping = make_mapping(std::array<std::size_t, 1>{0}, std::array<index_t, 1>{1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 0);
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

TEST(MakeIterationPlanTest, MergeableNegativeStrides)
{
  auto mapping = make_mapping(std::array<std::size_t, 2>{4, 5}, std::array<index_t, 2>{-1, -4});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 20);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(offset, -19);
}

TEST(MakeIterationPlanTest, SizeOneDimsAreDropped)
{
  // Size-1 surviving dims contribute index 0 only, so they are dropped from the
  // plan; only the extent-10 dim survives.
  auto mapping = make_mapping(std::array<std::size_t, 3>{1, 10, 1}, std::array<index_t, 3>{999, 1, 7});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 10);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, AllSizeOneIsScalarEmptyPlan)
{
  // Every dim size-1 => rank-0 scalar => empty plan (which the driver reads as
  // exactly one element, never zero).
  auto mapping = make_mapping(std::array<std::size_t, 3>{1, 1, 1}, std::array<index_t, 3>{5, 9, 2});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_TRUE(plan.empty());
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, RankZeroMdspanIsScalar)
{
  using extents_t = stdex::dextents<index_t, 0>;
  using mapping_t = stdex::layout_stride::mapping<extents_t>;

  std::array<std::ptrdiff_t, 0> strides{};
  mapping_t mapping{extents_t{}, strides};

  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_TRUE(plan.empty());
  EXPECT_EQ(offset, 0);
}

TEST(MakeMultiIterationPlanTest, SimpleMatchingLayouts)
{
  auto output = make_mapping(std::array<std::size_t, 2>{10, 2}, std::array<index_t, 2>{2, 1});
  auto input = make_mapping(std::array<std::size_t, 2>{10, 2}, std::array<index_t, 2>{20, 10});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::tuple{output, input});

  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 20);
  EXPECT_EQ(plan[0].strides[0], 1);
  EXPECT_EQ(plan[0].strides[1], 10);
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 0);
}

TEST(MakeMultiIterationPlanTest, MismatchedButMergeable)
{
  auto output = make_mapping(std::array<std::size_t, 2>{3, 4}, std::array<index_t, 2>{4, 1});
  auto input = make_mapping(std::array<std::size_t, 2>{3, 4}, std::array<index_t, 2>{40, 10});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::tuple{output, input});

  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 12);
  EXPECT_EQ(plan[0].strides[0], 1);
  EXPECT_EQ(plan[0].strides[1], 10);
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 0);
}

TEST(MakeMultiIterationPlanTest, OutputNegativeStrideFlipsEveryOperand)
{
  auto output = make_mapping(std::array<std::size_t, 1>{5}, std::array<index_t, 1>{-2});
  auto input = make_mapping(std::array<std::size_t, 1>{5}, std::array<index_t, 1>{3});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::tuple{output, input});

  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 5);
  EXPECT_EQ(plan[0].strides[0], 2);
  EXPECT_EQ(plan[0].strides[1], -3);
  EXPECT_EQ(offsets[0], -8);
  EXPECT_EQ(offsets[1], 12);
}

TEST(MakeMultiIterationPlanTest, MixedSignsPreventMerge)
{
  auto output = make_mapping(std::array<std::size_t, 2>{4, 2}, std::array<index_t, 2>{1, -4});
  auto input = make_mapping(std::array<std::size_t, 2>{4, 2}, std::array<index_t, 2>{10, 40});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::tuple{output, input});

  ASSERT_EQ(plan.size(), 2);
  EXPECT_EQ(plan[0].extent, 2);
  EXPECT_EQ(plan[0].strides[0], 4);
  EXPECT_EQ(plan[0].strides[1], -40);
  EXPECT_EQ(plan[1].extent, 4);
  EXPECT_EQ(plan[1].strides[0], 1);
  EXPECT_EQ(plan[1].strides[1], 10);
  EXPECT_EQ(offsets[0], -4);
  EXPECT_EQ(offsets[1], 40);
}

TEST(MakeMultiIterationPlanTest, ZeroExtentProducesRetainedZeroDim)
{
  auto output = make_mapping(std::array<std::size_t, 1>{0}, std::array<index_t, 1>{1});
  auto input = make_mapping(std::array<std::size_t, 1>{0}, std::array<index_t, 1>{3});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::tuple{output, input});

  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 0);
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 0);
}

TEST(MakeMultiIterationPlanTest, AllSizeOneIsScalarEmptyPlan)
{
  auto output = make_mapping(std::array<std::size_t, 2>{1, 1}, std::array<index_t, 2>{7, 3});
  auto input = make_mapping(std::array<std::size_t, 2>{1, 1}, std::array<index_t, 2>{11, 5});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::tuple{output, input});

  EXPECT_TRUE(plan.empty());
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 0);
}
