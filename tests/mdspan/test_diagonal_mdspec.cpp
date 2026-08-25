#include <uni20/mdspan/diagonal_accessor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <type_traits>

namespace
{

using namespace uni20;

TEST(DiagonalMdspecTest, PresentsStridedComponentsAsAFullLogicalTensor)
{
  using component_extents_type = stdex::dextents<index_type, 1>;
  using component_mapping_type = stdex::layout_stride::mapping<component_extents_type>;
  using component_mdspan_type = stdex::mdspan<double, component_extents_type, stdex::layout_stride>;
  using logical_extents_type = stdex::dextents<index_type, 2>;

  std::array<double, 6> storage{10.0, 0.0, 20.0, 0.0, 30.0, 0.0};
  auto const component_mapping = component_mapping_type{component_extents_type{3}, std::array<index_type, 1>{-2}};
  component_mdspan_type components{storage.data() + 4, component_mapping};
  auto diagonal = make_diagonal_mdspan(logical_extents_type{3, 4}, components);

  static_assert(DiagonalMdspecLike<decltype(diagonal)>);
  static_assert(MutableDiagonalMdspecLike<decltype(diagonal)>);
  static_assert(!StridedMdspecLike<decltype(diagonal)>);
  EXPECT_EQ(diagonal.extent(0), 3);
  EXPECT_EQ(diagonal.extent(1), 4);
  EXPECT_DOUBLE_EQ((diagonal[0, 0]), 30.0);
  EXPECT_DOUBLE_EQ((diagonal[1, 1]), 20.0);
  EXPECT_DOUBLE_EQ((diagonal[2, 2]), 10.0);
  EXPECT_DOUBLE_EQ((diagonal[0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((diagonal[2, 3]), 0.0);

  diagonal[1, 1] = 25.0;
  EXPECT_DOUBLE_EQ(storage[2], 25.0);
  diagonal[0, 3] = 0.0;

  auto observed_components = diagonal_components(diagonal);
  static_assert(MutableRankedStridedMdspanLike<decltype(observed_components), 1>);
  EXPECT_EQ(observed_components.extent(0), 3);
  EXPECT_EQ(observed_components.stride(0), -2);
  EXPECT_EQ(observed_components.data_handle(), storage.data() + 4);
  EXPECT_DOUBLE_EQ(observed_components[0], 30.0);
  EXPECT_DOUBLE_EQ(observed_components[1], 25.0);
  EXPECT_DOUBLE_EQ(observed_components[2], 10.0);

  auto const_diagonal = make_const_mdspan(diagonal);
  static_assert(DiagonalMdspecLike<decltype(const_diagonal)>);
  static_assert(!MutableDiagonalMdspecLike<decltype(const_diagonal)>);
  auto const_components = diagonal_components(const_diagonal);
  static_assert(!MutableMdspecLike<decltype(const_components)>);
  EXPECT_DOUBLE_EQ(const_components[1], 25.0);
}

TEST(DiagonalMdspecTest, RejectsTheWrongNumberOfComponents)
{
  using component_extents_type = stdex::dextents<index_type, 1>;
  using logical_extents_type = stdex::dextents<index_type, 3>;
  std::array<double, 1> storage{};
  stdex::mdspan<double, component_extents_type> components{storage.data(), component_extents_type{1}};

  EXPECT_THROW(static_cast<void>(make_diagonal_mdspan(logical_extents_type{2, 3, 4}, components)),
               std::invalid_argument);
}

} // namespace
