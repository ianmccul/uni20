#include "../helpers.hpp"
#include "level1/assign.hpp"
#include "level1/zip_transform.hpp"
#include "gtest/gtest.h"
#include <numeric>

using namespace uni20;

TEST(MultiIterationPlanTest, SimpleMatchingLayouts)
{
  auto a = make_mapping(std::array<std::size_t, 2>{10, 2}, std::array<index_t, 2>{2, 1});
  auto b = make_mapping(std::array<std::size_t, 2>{10, 2}, std::array<index_t, 2>{20, 10});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::array{a, b});

  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 20);     // 10×2
  EXPECT_EQ(plan[0].strides[0], 1);  // from tensor A (innermost stride)
  EXPECT_EQ(plan[0].strides[1], 10); // from tensor B
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 0);
}

TEST(MultiIterationPlanTest, MismatchedButCoalescable)
{
  auto a = make_mapping(std::array<std::size_t, 2>{3, 4}, std::array<index_t, 2>{4, 1});
  auto b = make_mapping(std::array<std::size_t, 2>{3, 4}, std::array<index_t, 2>{40, 10});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::array{a, b});

  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 12); // 3 × 4
  EXPECT_EQ(plan[0].strides[0], 1);
  EXPECT_EQ(plan[0].strides[1], 10);
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 0);
}

TEST(MultiIterationPlanTest, WithNegativeStride)
{
  auto a = make_mapping(std::array<std::size_t, 1>{4}, std::array<index_t, 1>{-1});
  auto b = make_mapping(std::array<std::size_t, 1>{4}, std::array<index_t, 1>{-10});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::array{a, b});

  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 4);
  EXPECT_EQ(plan[0].strides[0], 1);  // flipped from -1
  EXPECT_EQ(plan[0].strides[1], 10); // flipped from -10
  EXPECT_EQ(offsets[0], -3);         // (1-4) * 1
  EXPECT_EQ(offsets[1], -30);        // (1-4) * 10
}

TEST(MultiIterationPlanTest, MixedSignsPreventMerge)
{
  auto a = make_mapping(std::array<std::size_t, 2>{4, 2}, std::array<index_t, 2>{1, -4});
  auto b = make_mapping(std::array<std::size_t, 2>{4, 2}, std::array<index_t, 2>{10, 40});
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::array{a, b});

  EXPECT_EQ(plan.size(), 2);
  // Check the flipped‐stride dimension; also swapped order to make largest stride of a the outer dimension
  EXPECT_EQ(plan[0].extent, 2); // outer dim extent
  EXPECT_EQ(plan[0].strides[0], 4);
  EXPECT_EQ(plan[0].strides[1], -40);
  EXPECT_EQ(plan[1].extent, 4); // inner dim extent
  EXPECT_EQ(plan[1].strides[0], 1);
  EXPECT_EQ(plan[1].strides[1], 10);
  EXPECT_EQ(offsets[0], -4);
  EXPECT_EQ(offsets[1], 40);
}

TEST(MultiIterationPlanTest, AllStridesFlippedWhenOutputIsNegative)
{
  auto a = make_mapping(std::array<std::size_t, 1>{5}, std::array<index_t, 1>{-2}); // output
  auto b = make_mapping(std::array<std::size_t, 1>{5}, std::array<index_t, 1>{3});  // input

  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::array{a, b});

  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 5);
  EXPECT_EQ(plan[0].strides[0], 2);  // flipped from -2
  EXPECT_EQ(plan[0].strides[1], -3); // flipped from 3
  EXPECT_EQ(offsets[0], -8);         // (5-1) * 2
  EXPECT_EQ(offsets[1], 12);         // (5-1) * -3
}

TEST(Assign, Simple1D)
{
  std::vector<double> src_data = {1, 2, 3, 4};
  std::vector<double> dst_data(4, 0);

  auto src = make_mdspan_1d(src_data);
  auto dst = make_mdspan_1d(dst_data);

  uni20::assign(src, dst);

  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(dst_data[i], src_data[i]);
}

TEST(Assign, Strided2D)
{
  std::vector<double> buf1(25, 0.0), buf2(25, 0.0);

  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      buf1[r * 5 + c * 2] = static_cast<double>(r * 3 + c + 1);

  auto m1 = make_mdspan_2d(buf1, 3, 3, {5, 2});
  auto m2 = make_mdspan_2d(buf2, 3, 3, {5, 2});

  uni20::assign(m1, m2);

  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      EXPECT_EQ(buf2[r * 5 + c * 2], buf1[r * 5 + c * 2]);
}

TEST(Assign, Reversed1D)
{
  std::vector<double> v = {1, 2, 3, 4};
  std::vector<double> result(4, 0);

  auto src = make_reversed_1d(v);
  auto dst = make_mdspan_1d(result);

  uni20::assign(src, dst);

  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(result[i], v[3 - i]);
}

TEST(Assign, TransformNegate)
{
  std::vector<double> v = {1, 2, 3, 4};
  std::vector<double> out(4, 0);

  auto src = make_mdspan_1d(v);
  auto dst = make_mdspan_1d(out);

  auto neg = zip_transform([](double x) { return -x; }, src);
  uni20::assign(neg, dst);

  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(out[i], -v[i]);
}

TEST(Assign, TransformScaleShift)
{
  std::vector<double> v = {1, 2, 3, 4};
  std::vector<double> out(4, 0);

  auto src = make_mdspan_1d(v);
  auto dst = make_mdspan_1d(out);

  auto chain = zip_transform([](double x) { return x + 1; }, zip_transform([](double x) { return 2 * x; }, src));

  uni20::assign(chain, dst);

  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(out[i], 2 * v[i] + 1);
}
