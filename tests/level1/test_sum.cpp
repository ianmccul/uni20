#include "../helpers.hpp"
#include "level1/sum.hpp"
#include "gtest/gtest.h"
#include <numeric>

using namespace uni20;

TEST(SumView1D, SimpleContiguous)
{
  std::vector<double> a(5), b(5);
  std::iota(a.begin(), a.end(), 0.0);  // 0,1,2,3,4
  std::iota(b.begin(), b.end(), 10.0); // 10,11,12,13,14

  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);
  auto S = sum_view(A, B);

  ASSERT_EQ(S.rank(), 1);
  EXPECT_EQ(S.extent(0), 5);

  for (index_t i = 0; i < 5; ++i)
  {
    EXPECT_DOUBLE_EQ(S[i], a[i] + b[i]);
  }
}

TEST(SumView2D, RowMajor)
{
  std::size_t R = 3, C = 4;
  std::vector<double> a(R * C), b(R * C);
  for (std::size_t i = 0; i < R * C; ++i)
  {
    a[i] = i;
    b[i] = 100 + i;
  }

  auto A = make_mdspan_2d(a, R, C);
  auto B = make_mdspan_2d(b, R, C);
  auto S = sum_view(A, B);

  ASSERT_EQ(S.rank(), 2);
  EXPECT_EQ(S.extent(0), R);
  EXPECT_EQ(S.extent(1), C);

  for (index_t i = 0; i < (index_t)R; ++i)
    for (index_t j = 0; j < (index_t)C; ++j)
      EXPECT_DOUBLE_EQ((S[i, j]), a[i * C + j] + b[i * C + j]);
}

TEST(SumViewReversed, MixedStrides)
{
  // original data
  std::vector<double> a{1, 2, 3, 4}, b{10, 20, 30, 40};
  // reversed view for a, contiguous for b
  auto A = make_reversed_1d(a); // A[0]=4, A[1]=3, …
  auto B = make_mdspan_1d(b);

  auto S = sum_view(A, B);
  ASSERT_EQ(S.extent(0), 4);

  // expected S[i] = A[i] + B[i]
  for (index_t i = 0; i < 4; ++i)
  {
    double exp = (a[3 - i] + b[i]);
    EXPECT_DOUBLE_EQ(S[i], exp);
  }
}

TEST(SumViewVariadic, ThreeInputs)
{
  std::vector<double> a(3), b(3), c(3);
  std::iota(a.begin(), a.end(), 1.0);   // 1,2,3
  std::iota(b.begin(), b.end(), 10.0);  // 10,11,12
  std::iota(c.begin(), c.end(), 100.0); // 100,101,102

  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);
  auto C = make_mdspan_1d(c);

  auto S = sum_view(A, B, C);
  ASSERT_EQ(S.extent(0), 3);

  for (index_t i = 0; i < 3; ++i)
  {
    EXPECT_DOUBLE_EQ(S[i], a[i] + b[i] + c[i]);
  }
}
TEST(SumViewNested, NestedRight)
{
  std::vector<double> a{1, 2}, b{10, 20}, c{100, 200}, d{1000, 2000};
  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);
  auto C = make_mdspan_1d(c);

  // nested sum: A + (B+C)
  auto S1 = sum_view(B, C);
  auto S = sum_view(A, S1);

  ASSERT_EQ(S.extent(0), 2);
  for (index_t i = 0; i < 2; ++i)
  {
    double exp = a[i] + b[i] + c[i];
    EXPECT_DOUBLE_EQ(S[i], exp);
  }
}

TEST(SumViewNested, NestedLeft)
{
  std::vector<double> a{1, 2}, b{10, 20}, c{100, 200}, d{1000, 2000};
  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);
  auto C = make_mdspan_1d(c);

  // nested sum: (A + B) + C
  auto S1 = sum_view(A, B);
  auto S = sum_view(S1, C);

  ASSERT_EQ(S.extent(0), 2);
  for (index_t i = 0; i < 2; ++i)
  {
    double exp = a[i] + b[i] + c[i];
    EXPECT_DOUBLE_EQ(S[i], exp);
  }
}

TEST(SumViewNested, CombinedAndNested)
{
  std::vector<double> a{1, 2}, b{10, 20}, c{100, 200}, d{1000, 2000};
  auto A = make_mdspan_1d(a);
  auto B = make_mdspan_1d(b);
  auto C = make_mdspan_1d(c);
  auto D = make_mdspan_1d(d);

  // nested sum: (A+B) + (C+D)
  auto S1 = sum_view(A, B);
  auto S2 = sum_view(C, D);
  auto S = sum_view(S1, S2);

  ASSERT_EQ(S.extent(0), 2);
  for (index_t i = 0; i < 2; ++i)
  {
    double exp = a[i] + b[i] + c[i] + d[i];
    EXPECT_DOUBLE_EQ(S[i], exp);
  }
}
