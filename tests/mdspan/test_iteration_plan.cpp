#include "../helpers.hpp"
#include "gtest/gtest.h"
#include <uni20/level1/assign.hpp>
#include <uni20/level1/transform.hpp>
#include <uni20/mdspan/iteration_plan.hpp>

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

  std::array<double, 3> buffer{1.0, 2.0, 3.0};
  using extents_t = stdex::dextents<index_t, 1>;
  std::array<std::ptrdiff_t, 1> strides{1};
  auto zero_map = stdex::layout_stride::mapping<extents_t>(extents_t{0}, strides);
  stdex::mdspan<double, extents_t, stdex::layout_stride> span(buffer.data(), zero_map);

  transform_inplace(span, [](double x) { return x + 10.0; });

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

  double dst_value = 2.0;
  stdex::mdspan<double, extents_t, stdex::layout_stride> dst{&dst_value, mapping};
  transform_inplace(dst, [](double x) { return x + 3.0; });
  EXPECT_DOUBLE_EQ(dst_value, 5.0);

  double src_value = 11.0;
  stdex::mdspan<double, extents_t, stdex::layout_stride> src{&src_value, mapping};
  assign(dst, src);
  EXPECT_DOUBLE_EQ(dst_value, 11.0);
}

TEST(TransformInplaceTest, ScalarEmptyPlanAppliesExactlyOnce)
{
  // An in-place transform on an all-size-1 span must touch the single base element once.
  // (empty plan -> 0-dim scalar terminal), not skip it.
  std::vector<double> buffer{10.0, 20.0, 30.0};
  std::array<std::size_t, 2> extents{1, 1};
  std::array<index_t, 2> strides{7, 3}; // only index 0 exists; strides irrelevant
  auto span = make_mdspan_strided(buffer, extents, strides);
  transform_inplace(span, [](double x) { return x + 100.0; });
  EXPECT_DOUBLE_EQ(buffer[0], 110.0); // applied exactly once
  EXPECT_DOUBLE_EQ(buffer[1], 20.0);  // untouched
  EXPECT_DOUBLE_EQ(buffer[2], 30.0);
}

TEST(AssignTest, ScalarEmptyPlanCopiesSingleElement)
{
  // assign between all-size-1 spans => empty multi-plan => copies the one element.
  std::vector<double> src{7.0, 0.0};
  std::vector<double> dst{0.0, 99.0};
  std::array<std::size_t, 2> extents{1, 1};
  std::array<index_t, 2> strides{3, 5};
  auto s = make_mdspan_strided(src, extents, strides);
  auto d = make_mdspan_strided(dst, extents, strides);
  assign(d, s);
  EXPECT_DOUBLE_EQ(dst[0], 7.0);  // copied
  EXPECT_DOUBLE_EQ(dst[1], 99.0); // untouched
}

TEST(TransformInplaceTest, NonMergeableFourDimVisitsEachOnce)
{
  // Gapped strides prevent every merge => a genuine 4-dim plan, exercising the
  // depth-4 run_dynamic handoff. Each addressed element is visited exactly once.
  std::array<std::size_t, 4> extents{3, 3, 3, 3};
  std::array<index_t, 4> strides{100, 20, 4, 1}; // no adjacent pair coalesces
  std::vector<double> buffer(span_size_for(extents, strides), 0.0);
  auto span = make_mdspan_strided(buffer, extents, strides);

  auto [plan, offset] = make_iteration_plan_with_offset(span.mapping());
  EXPECT_EQ(plan.size(), 4); // nothing merged
  EXPECT_EQ(offset, 0);

  transform_inplace(span, [](double x) { return x + 1.0; });

  double sum = 0.0;
  for (double v : buffer)
  {
    EXPECT_LE(v, 1.0); // none visited twice
    sum += v;
  }
  EXPECT_DOUBLE_EQ(sum, 81.0); // 3^4 elements, each +1 once
}

TEST(TransformInplaceTest, NonMergeableFiveDimVisitsEachOnce)
{
  // Five non-mergeable dims: run_dynamic peels twice before the static 3-dim
  // unroll.
  std::array<std::size_t, 5> extents{2, 2, 2, 2, 2};
  std::array<index_t, 5> strides{100, 30, 9, 4, 1}; // no adjacent pair coalesces
  std::vector<double> buffer(span_size_for(extents, strides), 0.0);
  auto span = make_mdspan_strided(buffer, extents, strides);

  auto [plan, offset] = make_iteration_plan_with_offset(span.mapping());
  EXPECT_EQ(plan.size(), 5);
  EXPECT_EQ(offset, 0);

  transform_inplace(span, [](double x) { return x + 1.0; });

  double sum = 0.0;
  for (double v : buffer)
  {
    EXPECT_LE(v, 1.0);
    sum += v;
  }
  EXPECT_DOUBLE_EQ(sum, 32.0); // 2^5
}
