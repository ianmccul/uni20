#include "helpers.hpp"
#include "level1/zip_layout.hpp"
#include "gtest/gtest.h"

using namespace uni20;

//----------------------------------------------------------------------
// Test StridedZipLayout<2>::mapping<...> in 1D
//----------------------------------------------------------------------

TEST(StridedZipLayoutMapping1D, OffsetsAndIsStrided)
{
  using Layout = StridedZipLayout<2>;
  using extents_t = stdex::dextents<index_t, 1>;
  extents_t exts{4};

  // pick two different strides
  std::array<std::array<index_t, 1>, 2> strides_pack{{{2}, {3}}};
  Layout::mapping<extents_t> m(exts, strides_pack);

  // operator()(i) → { i*2, i*3 }
  auto o0 = m(0);
  EXPECT_EQ(std::get<0>(o0), 0);
  EXPECT_EQ(std::get<1>(o0), 0);

  auto o2 = m(2);
  EXPECT_EQ(std::get<0>(o2), 2 * 2);
  EXPECT_EQ(std::get<1>(o2), 2 * 3);

  // static queries
  EXPECT_TRUE(Layout::mapping<extents_t>::is_always_unique());
  EXPECT_FALSE(Layout::mapping<extents_t>::is_always_exhaustive());
  EXPECT_FALSE(Layout::mapping<extents_t>::is_always_strided());

  // runtime is_strided() is false when strides differ
  EXPECT_FALSE(m.is_strided());

  // now test the strided case
  std::array<std::array<index_t, 1>, 2> eq_pack{{{5}, {5}}};
  Layout::mapping<extents_t> m2(exts, eq_pack);
  EXPECT_TRUE(m2.is_strided());
  EXPECT_EQ(m2.stride(0), 5);
}

//----------------------------------------------------------------------
// Test StridedZipLayout<3>::mapping<...> merging constructors
//----------------------------------------------------------------------

TEST(StridedZipLayoutMapping1D, MergePrependAppend)
{
  using Layout = StridedZipLayout<3>;
  using extents_t = stdex::dextents<index_t, 1>;
  extents_t exts{3};

  // start with a 2-span mapping:
  StridedZipLayout<2>::mapping<extents_t> m2(exts, std::array<std::array<index_t, 1>, 2>{{{1}, {10}}});
  // prepend stride=100
  Layout::mapping<extents_t> mp(std::array<index_t, 1>{100}, m2);
  EXPECT_EQ(std::get<0>(mp(2)), 2 * 100);
  EXPECT_EQ(std::get<1>(mp(2)), 2 * 1);
  EXPECT_EQ(std::get<2>(mp(2)), 2 * 10);

  // append stride=1000
  Layout::mapping<extents_t> ma(m2, std::array<index_t, 1>{1000});
  EXPECT_EQ(std::get<0>(ma(2)), 2 * 1);
  EXPECT_EQ(std::get<1>(ma(2)), 2 * 10);
  EXPECT_EQ(std::get<2>(ma(2)), 2 * 1000);
}

//----------------------------------------------------------------------
// Test GeneralZipLayout<layout_stride,layout_stride>::mapping<...> in 1D
//----------------------------------------------------------------------

TEST(GeneralZipLayoutMapping1D, DefaultStrides)
{
  using L1 = stdex::layout_stride;
  using L2 = stdex::layout_stride;
  using Layout = GeneralZipLayout<L1, L2>;
  using extents_t = stdex::dextents<index_t, 1>;

  extents_t exts{5};
  // child mappings default to stride=1
  Layout::mapping<extents_t> m(exts, L1::mapping<extents_t>(exts, std::array{1}),
                               L2::mapping<extents_t>(exts, std::array{1}));

  // offset(i) == (i,i)
  auto o3 = m(3);
  EXPECT_EQ(std::get<0>(o3), 3);
  EXPECT_EQ(std::get<1>(o3), 3);

  // static queries
  EXPECT_TRUE(Layout::mapping<extents_t>::is_always_unique());
  EXPECT_FALSE(Layout::mapping<extents_t>::is_always_exhaustive());
  EXPECT_FALSE(Layout::mapping<extents_t>::is_always_strided());
}

//----------------------------------------------------------------------
// Test zip_layout_t selects the right layout
//----------------------------------------------------------------------

TEST(ZipLayoutSelector, StridedMdspanPacksToStridedZipLayout)
{
  std::vector<double> v(4);
  auto A = make_mdspan_1d(v);
  auto B = make_mdspan_1d(v);

  // both A,B are StridedMdspan → zip_layout_t should be StridedZipLayout<2>
  using Z = zip_layout_t<decltype(A), decltype(B)>;
  static_assert(std::is_same_v<Z, StridedZipLayout<2>>);
}

TEST(ZipLayoutSelector, MixedMdspanAlsoChoosesStridedZipLayout)
{
  std::vector<double> v(4);
  auto A = make_mdspan_1d(v);
  auto R = make_reversed_1d(v);

  // reversed_1d is still layout_stride → still StridedZipLayout<2>
  using Z = zip_layout_t<decltype(A), decltype(R)>;
  static_assert(std::is_same_v<Z, StridedZipLayout<2>>);
}
