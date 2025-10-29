#include "../helpers.hpp"
#include "level1/apply_unary.hpp"
#include "gtest/gtest.h"
#include <numeric>

using namespace uni20;

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

TEST(ApplyUnaryInplace, NonMergeable4DDispatchesDynamically)
{
  using extents_t = stdex::dextents<index_t, 4>;
  extents_t extents{2, 3, 4, 5};
  std::array<std::ptrdiff_t, 4> strides{500, 60, 7, 1};
  auto mapping = stdex::layout_stride::mapping<extents_t>(extents, strides);
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_GE(plan.size(), 4u);
  EXPECT_EQ(offset, 0);

  std::vector<double> storage(mapping.required_span_size(), -1.0);
  stdex::mdspan<double, extents_t, stdex::layout_stride> tensor(storage.data(), mapping);

  for (index_t i0 = 0; i0 < extents.extent(0); ++i0)
    for (index_t i1 = 0; i1 < extents.extent(1); ++i1)
      for (index_t i2 = 0; i2 < extents.extent(2); ++i2)
        for (index_t i3 = 0; i3 < extents.extent(3); ++i3)
        {
          auto idx = mapping(i0, i1, i2, i3);
          storage[idx] = static_cast<double>(i0 * 1000 + i1 * 100 + i2 * 10 + i3);
        }

  apply_unary_inplace(tensor, [](double x) { return x - 2.5; });

  for (index_t i0 = 0; i0 < extents.extent(0); ++i0)
    for (index_t i1 = 0; i1 < extents.extent(1); ++i1)
      for (index_t i2 = 0; i2 < extents.extent(2); ++i2)
        for (index_t i3 = 0; i3 < extents.extent(3); ++i3)
        {
          auto idx = mapping(i0, i1, i2, i3);
          double expected = static_cast<double>(i0 * 1000 + i1 * 100 + i2 * 10 + i3) - 2.5;
          EXPECT_DOUBLE_EQ(storage[idx], expected);
        }
}
