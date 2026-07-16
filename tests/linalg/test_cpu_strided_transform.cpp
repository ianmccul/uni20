#include "../helpers.hpp"
#include "gtest/gtest.h"
#include <numeric>
#include <stdexcept>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/mdspan/iteration_plan.hpp>
#include <uni20/tensor/transform.hpp>

using namespace uni20;

TEST(TransformInplace, MultiplyBy2_1DContiguous)
{
  std::vector<double> v(10);
  std::iota(v.begin(), v.end(), 0.0);
  auto m = make_mdspan_1d(v);

  transform_inplace(linalg::CpuReferenceBackend{}, m, [](double x) { return x * 2; });

  for (size_t i = 0; i < 10; ++i)
  {
    EXPECT_DOUBLE_EQ(v[i], static_cast<double>(i * 2));
  }
}

TEST(TransformInplace, Add5_2DRowMajor)
{
  std::size_t R = 3, C = 4;
  std::vector<double> v(R * C);
  for (size_t i = 0; i < R; i++)
    for (size_t j = 0; j < C; j++)
      v[i * C + j] = static_cast<double>(i * 10 + j);

  auto m = make_mdspan_2d(v, R, C);
  transform_inplace(linalg::CpuReferenceBackend{}, m, [](double x) { return x + 5; });

  for (size_t i = 0; i < R; i++)
    for (size_t j = 0; j < C; j++)
      EXPECT_DOUBLE_EQ(v[i * C + j], static_cast<double>(i * 10 + j + 5));
}

TEST(TransformInplace, Square_Reversed1D)
{
  std::vector<double> v(8);
  std::iota(v.begin(), v.end(), 1.0); // 1,2,3,...,8
  auto m = make_reversed_1d(v);

  transform_inplace(linalg::CpuReferenceBackend{}, m, [](double x) { return x * x; });

  // reversed view means m[i] = original v[7-i]
  // after squaring, the underlying container should be:
  // v[7] = 1^2, v[6] = 2^2, ..., v[0] = 8^2
  for (size_t i = 0; i < 8; i++)
  {
    EXPECT_DOUBLE_EQ(v[i], std::pow(static_cast<double>(i + 1), 2));
  }
}

TEST(TransformInplace, ScaleAndShiftMixedStrides)
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

  transform_inplace(linalg::CpuReferenceBackend{}, m, [](double x) { return x * 10 - 1; });

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

TEST(TransformInplace, NonMergeable4DDispatchesDynamically)
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

  transform_inplace(linalg::CpuReferenceBackend{}, tensor, [](double x) { return x - 2.5; });

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

TEST(Transform, UnaryMixedCanonicalLayouts)
{
  using extents_type = stdex::dextents<index_t, 2>;

  std::vector<double> input_storage(12);
  std::vector<double> output_storage(12, 0.0);
  stdex::mdspan<double, extents_type, stdex::layout_left> input(input_storage.data(), 3, 4);
  stdex::mdspan<double, extents_type, stdex::layout_right> output(output_storage.data(), 3, 4);

  for (index_t row = 0; row < 3; ++row)
    for (index_t column = 0; column < 4; ++column)
      input[row, column] = 10.0 * row + column;

  assign_transform(linalg::CpuReferenceBackend{}, output, [](double value) { return 2.0 * value + 1.0; }, input);

  for (index_t row = 0; row < 3; ++row)
    for (index_t column = 0; column < 4; ++column)
    {
      double const expected = 2.0 * input[row, column] + 1.0;
      EXPECT_DOUBLE_EQ((output[row, column]), expected);
    }
}

TEST(Transform, BinaryMixedLayouts)
{
  using extents_type = stdex::dextents<index_t, 2>;

  std::vector<double> lhs_storage(12);
  std::vector<double> rhs_storage(24, 0.0);
  std::vector<double> output_storage(12, 0.0);
  stdex::mdspan<double, extents_type, stdex::layout_left> lhs(lhs_storage.data(), 3, 4);
  auto rhs = make_mdspan_2d(rhs_storage, 3, 4, {8, 2});
  stdex::mdspan<double, extents_type, stdex::layout_right> output(output_storage.data(), 3, 4);

  for (index_t row = 0; row < 3; ++row)
    for (index_t column = 0; column < 4; ++column)
    {
      lhs[row, column] = 10.0 * row + column;
      rhs[row, column] = 100.0 + row - 2.0 * column;
    }

  assign_transform(
      linalg::CpuReferenceBackend{}, output, [](double left, double right) { return left - right; }, lhs, rhs);

  for (index_t row = 0; row < 3; ++row)
    for (index_t column = 0; column < 4; ++column)
    {
      double const expected = lhs[row, column] - rhs[row, column];
      EXPECT_DOUBLE_EQ((output[row, column]), expected);
    }
}

TEST(TransformInplace, BinaryTracksOutputOnce)
{
  using extents_type = stdex::dextents<index_t, 2>;

  std::vector<double> output_storage(12);
  std::vector<double> rhs_storage(12);
  stdex::mdspan<double, extents_type, stdex::layout_right> output(output_storage.data(), 3, 4);
  stdex::mdspan<double, extents_type, stdex::layout_left> rhs(rhs_storage.data(), 3, 4);

  for (index_t row = 0; row < 3; ++row)
    for (index_t column = 0; column < 4; ++column)
    {
      output[row, column] = 10.0 * row + column;
      rhs[row, column] = 100.0 + row + column;
    }

  transform_inplace(
      linalg::CpuReferenceBackend{}, output, [](double current, double value) { return current + value; }, rhs);

  for (index_t row = 0; row < 3; ++row)
    for (index_t column = 0; column < 4; ++column)
      EXPECT_DOUBLE_EQ((output[row, column]), 100.0 + 11.0 * row + 2.0 * column);
}

TEST(Transform, CallableExceptionsPropagate)
{
  std::vector<double> storage{1.0, 2.0, 3.0};
  auto span = make_mdspan_1d(storage);

  EXPECT_THROW(transform_inplace(linalg::CpuReferenceBackend{}, span,
                                 [](double value) {
                                   if (value == 2.0) throw std::runtime_error("transform failed");
                                   return value + 1.0;
                                 }),
               std::runtime_error);
}

TEST(CpuStridedTransformTest, ZeroExtentDoesNotVisitStorage)
{
  std::array<std::size_t, 1> extents{0};
  std::array<index_t, 1> strides{1};
  std::vector<double> storage{1.0, 2.0, 3.0};
  auto span = make_mdspan_strided(storage, extents, strides);

  transform_inplace(linalg::CpuReferenceBackend{}, span, [](double value) { return value + 10.0; });

  EXPECT_EQ(storage, (std::vector<double>{1.0, 2.0, 3.0}));
}

TEST(CpuStridedTransformTest, RankZeroAppliesExactlyOnce)
{
  using extents_type = stdex::dextents<index_t, 0>;
  using mapping_type = stdex::layout_stride::mapping<extents_type>;

  std::array<std::ptrdiff_t, 0> strides{};
  mapping_type mapping{extents_type{}, strides};
  double input_value = 11.0;
  double output_value = 2.0;
  stdex::mdspan<double, extents_type, stdex::layout_stride> input{&input_value, mapping};
  stdex::mdspan<double, extents_type, stdex::layout_stride> output{&output_value, mapping};

  assign_transform(linalg::CpuReferenceBackend{}, output, [](double value) { return value; }, input);
  EXPECT_DOUBLE_EQ(output_value, 11.0);

  transform_inplace(linalg::CpuReferenceBackend{}, output, [](double value) { return value + 3.0; });
  EXPECT_DOUBLE_EQ(output_value, 14.0);
}

TEST(CpuStridedTransformTest, AllSizeOneAppliesExactlyOnce)
{
  std::vector<double> storage{10.0, 20.0, 30.0};
  std::array<std::size_t, 2> extents{1, 1};
  std::array<index_t, 2> strides{7, 3};
  auto span = make_mdspan_strided(storage, extents, strides);

  transform_inplace(linalg::CpuReferenceBackend{}, span, [](double value) { return value + 100.0; });

  EXPECT_DOUBLE_EQ(storage[0], 110.0);
  EXPECT_DOUBLE_EQ(storage[1], 20.0);
  EXPECT_DOUBLE_EQ(storage[2], 30.0);
}

TEST(CpuStridedTransformTest, NonMergeableFiveDimVisitsEachElementOnce)
{
  std::array<std::size_t, 5> extents{2, 2, 2, 2, 2};
  std::array<index_t, 5> strides{100, 30, 9, 4, 1};
  std::vector<double> storage(span_size_for(extents, strides), 0.0);
  auto span = make_mdspan_strided(storage, extents, strides);

  auto [plan, offset] = make_iteration_plan_with_offset(span.mapping());
  ASSERT_EQ(plan.size(), 5);
  EXPECT_EQ(offset, 0);

  transform_inplace(linalg::CpuReferenceBackend{}, span, [](double value) { return value + 1.0; });

  double sum = 0.0;
  for (double value : storage)
  {
    EXPECT_LE(value, 1.0);
    sum += value;
  }
  EXPECT_DOUBLE_EQ(sum, 32.0);
}
