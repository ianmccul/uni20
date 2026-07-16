#include "../helpers.hpp"
#include "gtest/gtest.h"
#include <numeric>
#include <stdexcept>
#include <uni20/mdspan/transform_view.hpp>
#include <uni20/mdspan/zip_layout.hpp>

using namespace uni20;

/// \brief Zero-state functor that returns the sum of any number of arguments.
/// \tparam Ts  Types of each summand (all must support operator+).
struct plus_n
{
    /// \brief Return x₀ + x₁ + … + xₙ.
    template <typename... Ts> constexpr auto operator()(Ts const&... xs) const
    {
      return (xs + ...); // fold-expression
    }
};

struct final_scale final
{
    double factor;

    constexpr double operator()(double value) const { return factor * value; }
};

//----------------------------------------------------------------------
// 1D: simple plus_n over two contiguous spans
//----------------------------------------------------------------------

TEST(TransformView1D, SimplePlusN)
{
  std::vector<double> a(5), b(5);
  std::iota(a.begin(), a.end(), 0.0);  // {0,1,2,3,4}
  std::iota(b.begin(), b.end(), 10.0); // {10,11,12,13,14}

  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);

  // plus_n{} sums any number of args
  auto Z = transform_view(plus_n{}, A, B);

  static_assert(std::is_const_v<typename decltype(Z)::element_type>);
  static_assert(!MutableSpanLike<decltype(Z)>);
  ASSERT_EQ(Z.rank(), 1);
  EXPECT_EQ(Z.extent(0), 5);

  for (index_t i = 0; i < 5; ++i)
  {
    EXPECT_DOUBLE_EQ(Z[i], a[i] + b[i]);
  }

  // mapping should be strided with stride==1
  auto m = Z.mapping();
  EXPECT_TRUE(m.is_strided());
  EXPECT_EQ(m.stride(0), 1);
}

TEST(TransformView1D, NestedUnaryViewsOwnTheirAccessorState)
{
  std::vector<double> values{1.0, 2.0, 3.0};
  auto input = make_mdspan_1d(values);

  auto view = transform_view([](double value) { return value + 1.0; }, transform_view(final_scale{2.0}, input));

  EXPECT_DOUBLE_EQ(view[0], 3.0);
  EXPECT_DOUBLE_EQ(view[1], 5.0);
  EXPECT_DOUBLE_EQ(view[2], 7.0);
}

TEST(TransformView1D, CallableExceptionsPropagateOnAccess)
{
  std::vector<double> values{1.0};
  auto input = make_mdspan_1d(values);
  auto view = transform_view([](double) -> double { throw std::runtime_error("transform view failed"); }, input);

  EXPECT_THROW(static_cast<void>(view[0]), std::runtime_error);
}

//----------------------------------------------------------------------
// 1D: three-span weighted sum
//----------------------------------------------------------------------

TEST(TransformView1D, ThreeSpanWeighted)
{
  std::vector<double> a{1, 2, 3, 4}, b{2, 4, 6, 8}, c{3, 6, 9, 12};
  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);
  auto C = make_mdspan_1d(c);

  auto Z = transform_view(plus_n{}, A, B, C);

  // plus_n{} folds x+y+z
  for (index_t i = 0; i < 4; ++i)
  {
    EXPECT_DOUBLE_EQ(Z[i], a[i] + b[i] + c[i]);
  }

  // still strided
  EXPECT_TRUE(Z.mapping().is_strided());
}

//----------------------------------------------------------------------
// 1D reversed + normal → not strided mapping
//----------------------------------------------------------------------

TEST(TransformView1D, MixedStrideNotStrided)
{
  std::vector<double> v(6);
  std::iota(v.begin(), v.end(), 1.0);
  auto A = make_mdspan_1d(v);
  auto R = make_reversed_1d(v);

  auto Z = transform_view(plus_n{}, A, R);

  // Z[i] = v[i] + v[5-i]
  for (index_t i = 0; i < 6; ++i)
  {
    EXPECT_DOUBLE_EQ(Z[i], v[i] + v[5 - i]);
  }

  EXPECT_FALSE(Z.mapping().is_strided());
}

//----------------------------------------------------------------------
// A unary transform_view should preserve the existsing layout and just transform the accessor
//----------------------------------------------------------------------

TEST(TransformView1D, UnaryPreservesLayoutAndValues)
{
  std::vector<double> v{5, 6, 7, 8};
  auto M = make_mdspan_1d(v);

  // a simple unary op: multiply by 10
  auto U = transform_view([](double x) { return x * 10.0; }, M);

  // shape must be unchanged
  ASSERT_EQ(U.rank(), 1);
  EXPECT_EQ(U.extent(0), M.extent(0));

  // values should be exactly 10× the input
  for (index_t i = 0; i < (index_t)M.extent(0); ++i)
    EXPECT_DOUBLE_EQ(U[i], 10.0 * M[i]);

  // mapping must be the same type and have the same behavior
  using OrigMap = decltype(M.mapping());
  using NewMap = decltype(U.mapping());
  static_assert(std::is_same_v<OrigMap, NewMap>, "Unary transform_view must preserve layout_type");

  // exercise the mapping offsets too
  auto m0 = M.mapping();
  auto m1 = U.mapping();
  for (index_t i = 0; i < (index_t)M.extent(0); ++i)
  {
    auto o0 = m0(i);
    auto o1 = m1(i);
    EXPECT_EQ(o0, o1);
  }
}

//----------------------------------------------------------------------
// Data_handle tuple is passed through accessor
//----------------------------------------------------------------------

TEST(TransformView1D, DataHandleTuple)
{
  std::vector<double> a{0, 1, 2}, b{10, 11, 12};
  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);

  auto Z = transform_view(plus_n{}, A, B);

  // mdspan..data_handle() should be tuple of A.data_handle(), B.data_handle()
  auto dh = Z.data_handle();
  static_assert(std::is_same_v<decltype(dh), std::tuple<std::remove_cvref_t<decltype(A.data_handle())>,
                                                        std::remove_cvref_t<decltype(B.data_handle())>>>);
  // And its values should compare equal
  EXPECT_EQ(std::get<0>(dh), A.data_handle());
  EXPECT_EQ(std::get<1>(dh), B.data_handle());
}

//----------------------------------------------------------------------
// 2D: transform_view on 2D row-major spans
//----------------------------------------------------------------------

TEST(TransformView2D, RowMajorSum)
{
  std::size_t R = 3, C = 4;
  std::vector<double> a(R * C), b(R * C);
  for (std::size_t i = 0; i < R * C; ++i)
  {
    a[i] = double(i);
    b[i] = 100 + double(i);
  }
  auto A = make_mdspan_2d(a, R, C);
  auto B = make_mdspan_2d(b, R, C);

  auto Z = transform_view(plus_n{}, A, B);

  ASSERT_EQ(Z.rank(), 2);
  EXPECT_EQ(Z.extent(0), R);
  EXPECT_EQ(Z.extent(1), C);

  for (index_t i = 0; i < (index_t)R; ++i)
    for (index_t j = 0; j < (index_t)C; ++j)
      EXPECT_DOUBLE_EQ((Z[i, j]), a[i * C + j] + b[i * C + j]);

  // mapping strided, row-major stride = C for dim0, 1 for dim1
  auto m = Z.mapping();
  EXPECT_TRUE(m.is_strided());
  EXPECT_EQ(m.stride(0), C);
  EXPECT_EQ(m.stride(1), 1);
}
