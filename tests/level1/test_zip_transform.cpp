#include "helpers.hpp"
#include "level1/zip_layout.hpp"
#include "level1/zip_transform.hpp"
#include "gtest/gtest.h"
#include <numeric>

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

//----------------------------------------------------------------------
// 1D: simple plus_n over two contiguous spans
//----------------------------------------------------------------------

TEST(ZipTransform1D, SimplePlusN)
{
  std::vector<double> a(5), b(5);
  std::iota(a.begin(), a.end(), 0.0);  // {0,1,2,3,4}
  std::iota(b.begin(), b.end(), 10.0); // {10,11,12,13,14}

  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);

  // plus_n{} sums any number of args
  auto Z = zip_transform(plus_n{}, A, B);

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

//----------------------------------------------------------------------
// 1D: three-span weighted sum
//----------------------------------------------------------------------

TEST(ZipTransform1D, ThreeSpanWeighted)
{
  std::vector<double> a{1, 2, 3, 4}, b{2, 4, 6, 8}, c{3, 6, 9, 12};
  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);
  auto C = make_mdspan_1d(c);

  auto Z = zip_transform(plus_n{}, A, B, C);

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

TEST(ZipTransform1D, MixedStrideNotStrided)
{
  std::vector<double> v(6);
  std::iota(v.begin(), v.end(), 1.0);
  auto A = make_mdspan_1d(v);
  auto R = make_reversed_1d(v);

  auto Z = zip_transform(plus_n{}, A, R);

  // Z[i] = v[i] + v[5-i]
  for (index_t i = 0; i < 6; ++i)
  {
    EXPECT_DOUBLE_EQ(Z[i], v[i] + v[5 - i]);
  }

  EXPECT_FALSE(Z.mapping().is_strided());
}

//----------------------------------------------------------------------
// Data_handle tuple is passed through accessor
//----------------------------------------------------------------------

TEST(ZipTransform1D, DataHandleTuple)
{
  std::vector<double> a{0, 1, 2}, b{10, 11, 12};
  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);

  auto Z = zip_transform(plus_n{}, A, B);

  // mdspan..data_handle() should be tuple of A.data_handle(), B.data_handle()
  auto dh = Z.data_handle();
  static_assert(std::is_same_v<decltype(dh), std::tuple<std::remove_cvref_t<decltype(A.data_handle())>,
                                                        std::remove_cvref_t<decltype(B.data_handle())>>>);
  // And its values should compare equal
  EXPECT_EQ(std::get<0>(dh), A.data_handle());
  EXPECT_EQ(std::get<1>(dh), B.data_handle());
}

//----------------------------------------------------------------------
// 2D: zip_transform on 2D row-major spans
//----------------------------------------------------------------------

TEST(ZipTransform2D, RowMajorSum)
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

  auto Z = zip_transform(plus_n{}, A, B);

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
