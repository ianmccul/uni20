#include "mdspan/concepts.hpp"
#include "mdspan/strides.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>
#include <tuple>

using namespace uni20;

TEST(ExtentStrides, MergeWithInnerSuccessAndFailure)
{
  extent_strides<2> outer{2, std::array<std::ptrdiff_t, 2>{6, 12}};
  extent_strides<2> inner{3, std::array<std::ptrdiff_t, 2>{2, 4}};

  EXPECT_TRUE(outer.can_merge_with_inner(inner));
  outer.merge_with_inner(inner);
  EXPECT_EQ(outer.extent, 6);
  EXPECT_EQ(outer.strides[0], 2);
  EXPECT_EQ(outer.strides[1], 4);

  extent_strides<2> incompatible{2, std::array<std::ptrdiff_t, 2>{8, 12}};
  EXPECT_FALSE(incompatible.can_merge_with_inner(inner));
}

TEST(MergeStrides, LeftOrdersAscendingAndPreservesPairs)
{
  static_vector<extent_strides<2>, 3> dims;
  dims.emplace_back(2, std::array<std::ptrdiff_t, 2>{100, 1000});
  dims.emplace_back(3, std::array<std::ptrdiff_t, 2>{10, 100});
  dims.emplace_back(5, std::array<std::ptrdiff_t, 2>{1, 10});

  merge_strides_left(dims);

  ASSERT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0].strides[0], 1);
  EXPECT_EQ(dims[1].strides[0], 10);
  EXPECT_EQ(dims[2].strides[0], 100);

  EXPECT_EQ(dims[0].strides[1], 10);
  EXPECT_EQ(dims[1].strides[1], 100);
  EXPECT_EQ(dims[2].strides[1], 1000);
}

TEST(MergeStrides, RightOrdersDescendingAndPreservesPairs)
{
  static_vector<extent_strides<2>, 3> dims;
  dims.emplace_back(2, std::array<std::ptrdiff_t, 2>{1, 10});
  dims.emplace_back(3, std::array<std::ptrdiff_t, 2>{10, 100});
  dims.emplace_back(5, std::array<std::ptrdiff_t, 2>{100, 1000});

  merge_strides_right(dims);

  ASSERT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0].strides[0], 100);
  EXPECT_EQ(dims[1].strides[0], 10);
  EXPECT_EQ(dims[2].strides[0], 1);

  EXPECT_EQ(dims[0].strides[1], 1000);
  EXPECT_EQ(dims[1].strides[1], 100);
  EXPECT_EQ(dims[2].strides[1], 10);
}

namespace
{
template <class Mdspan> struct StaticRankMdspan : Mdspan
{
    using Mdspan::Mdspan;

    static constexpr std::size_t rank = Mdspan::rank();
};
} // namespace

TEST(MergeStridesLeft, MergesAllBroadcastDimensions)
{
  static_vector<extent_strides<2>, 3> dims;
  dims.emplace_back(2, std::array<std::ptrdiff_t, 2>{0, 0});
  dims.emplace_back(3, std::array<std::ptrdiff_t, 2>{0, 0});
  dims.emplace_back(4, std::array<std::ptrdiff_t, 2>{0, 0});

  merge_strides_left(dims);

  ASSERT_EQ(dims.size(), 1);
  EXPECT_EQ(dims[0].extent, 24);
  EXPECT_EQ(dims[0].strides[0], 0);
  EXPECT_EQ(dims[0].strides[1], 0);
}

TEST(MergeStridesLeft, PartiallyMergesBroadcastDimensions)
{
  static_vector<extent_strides<2>, 3> dims;
  dims.emplace_back(7, std::array<std::ptrdiff_t, 2>{14, 21});
  dims.emplace_back(2, std::array<std::ptrdiff_t, 2>{0, 0});
  dims.emplace_back(5, std::array<std::ptrdiff_t, 2>{0, 0});

  merge_strides_left(dims);

  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].extent, 10);
  EXPECT_EQ(dims[0].strides[0], 0);
  EXPECT_EQ(dims[0].strides[1], 0);
  EXPECT_EQ(dims[1].extent, 7);
  EXPECT_EQ(dims[1].strides[0], 14);
  EXPECT_EQ(dims[1].strides[1], 21);
}

TEST(MergeStridesLeft, MergesContiguousLayoutLeftDimensions)
{
  static_vector<extent_strides<2>, 3> dims;
  dims.emplace_back(3, std::array<std::ptrdiff_t, 2>{1, 5});
  dims.emplace_back(4, std::array<std::ptrdiff_t, 2>{3, 15});
  dims.emplace_back(5, std::array<std::ptrdiff_t, 2>{12, 60});

  merge_strides_left(dims);

  ASSERT_EQ(dims.size(), 1);
  EXPECT_EQ(dims[0].extent, 60);
  EXPECT_EQ(dims[0].strides[0], 1);
  EXPECT_EQ(dims[0].strides[1], 5);
}

TEST(MergeStridesLeft, PartiallyMergesContiguousLayoutLeftDimensions)
{
  static_vector<extent_strides<2>, 3> dims;
  dims.emplace_back(3, std::array<std::ptrdiff_t, 2>{1, 5});
  dims.emplace_back(4, std::array<std::ptrdiff_t, 2>{3, 15});
  dims.emplace_back(2, std::array<std::ptrdiff_t, 2>{100, 500});

  merge_strides_left(dims);

  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].extent, 12);
  EXPECT_EQ(dims[0].strides[0], 1);
  EXPECT_EQ(dims[0].strides[1], 5);
  EXPECT_EQ(dims[1].extent, 2);
  EXPECT_EQ(dims[1].strides[0], 100);
  EXPECT_EQ(dims[1].strides[1], 500);
}

TEST(MergeStridesRight, MergesAllContiguousDimensions)
{
  static_vector<extent_strides<2>, 3> dims;
  dims.emplace_back(2, std::array<std::ptrdiff_t, 2>{12, 60});
  dims.emplace_back(3, std::array<std::ptrdiff_t, 2>{4, 20});
  dims.emplace_back(4, std::array<std::ptrdiff_t, 2>{1, 5});

  merge_strides_right(dims);

  ASSERT_EQ(dims.size(), 1);
  EXPECT_EQ(dims[0].extent, 24);
  EXPECT_EQ(dims[0].strides[0], 1);
  EXPECT_EQ(dims[0].strides[1], 5);
}

TEST(MergeStridesRight, PartiallyMergesContiguousDimensions)
{
  static_vector<extent_strides<2>, 3> dims;
  dims.emplace_back(2, std::array<std::ptrdiff_t, 2>{15, 45});
  dims.emplace_back(3, std::array<std::ptrdiff_t, 2>{5, 15});
  dims.emplace_back(5, std::array<std::ptrdiff_t, 2>{2, 7});

  merge_strides_right(dims);

  ASSERT_EQ(dims.size(), 2);
  EXPECT_EQ(dims[0].extent, 6);
  EXPECT_EQ(dims[0].strides[0], 5);
  EXPECT_EQ(dims[0].strides[1], 15);
  EXPECT_EQ(dims[1].extent, 5);
  EXPECT_EQ(dims[1].strides[0], 2);
  EXPECT_EQ(dims[1].strides[1], 7);
}

TEST(StridesHelpers, StridesOverloadsReturnExpectedArrays)
{
  using right_layout_extents = stdex::extents<std::size_t, 2, 3>;
  using right_layout_base = stdex::mdspan<int, right_layout_extents, stdex::layout_right>;
  std::array<int, 6> contiguous{};
  StaticRankMdspan<right_layout_base> right_layout(contiguous.data());

  auto right_strides = strides(right_layout);
  EXPECT_EQ(right_strides[0], 3);
  EXPECT_EQ(right_strides[1], 1);

  using dyn_extents = stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>;
  using strided_base = stdex::mdspan<int, dyn_extents, stdex::layout_stride>;
  dyn_extents dynamic_shape{2, 3};
  std::array<std::ptrdiff_t, 2> custom{5, 1};
  auto mapping = stdex::layout_stride::mapping<dyn_extents>(dynamic_shape, custom);

  std::array<int, 16> storage{};
  StaticRankMdspan<strided_base> strided(storage.data(), mapping);

  auto stride_array = strides(strided);
  EXPECT_EQ(stride_array[0], custom[0]);
  EXPECT_EQ(stride_array[1], custom[1]);

  auto formatted = fmt::format("{}", dynamic_shape);
  EXPECT_EQ(formatted, "[2,3]");
}

TEST(ExtractStrides, ContractsAndMergesGroups)
{
  using AExtents = stdex::extents<std::size_t, 2, 3, 4>;
  using BExtents = stdex::extents<std::size_t, 4, 5, 6>;
  using CExtents = stdex::extents<std::size_t, 2, 3, 5, 6>;

  std::array<double, 2 * 3 * 4> Adata{};
  std::array<double, 4 * 5 * 6> Bdata{};
  std::array<double, 2 * 3 * 5 * 6> Cdata{};

  stdex::mdspan<double, AExtents> A(Adata.data());
  stdex::mdspan<double, BExtents> B(Bdata.data());
  stdex::mdspan<double, CExtents> C(Cdata.data());

  std::array<std::pair<std::size_t, std::size_t>, 1> contract{{{2, 0}}};

  auto [Mgroup, Ngroup, Kgroup] = extract_strides(A, B, contract, C);

  ASSERT_EQ(Mgroup.size(), 1);
  EXPECT_EQ(Mgroup[0].extent, 6);
  EXPECT_EQ(Mgroup[0].strides[0], 4);
  EXPECT_EQ(Mgroup[0].strides[1], 30);

  ASSERT_EQ(Ngroup.size(), 1);
  EXPECT_EQ(Ngroup[0].extent, 30);
  EXPECT_EQ(Ngroup[0].strides[0], 1);
  EXPECT_EQ(Ngroup[0].strides[1], 1);

  ASSERT_EQ(Kgroup.size(), 1);
  EXPECT_EQ(Kgroup[0].extent, 4);
  EXPECT_EQ(Kgroup[0].strides[0], 1);
  EXPECT_EQ(Kgroup[0].strides[1], 30);
}
