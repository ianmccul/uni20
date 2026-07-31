#include <uni20/mdspan/mdspec.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace
{

struct RegionDescriptor
{
    int storage_id = 0;
    std::size_t element_offset = 0;
};

struct MoveOnlyDescriptor
{
    MoveOnlyDescriptor() = default;
    MoveOnlyDescriptor(MoveOnlyDescriptor&&) = default;
    MoveOnlyDescriptor& operator=(MoveOnlyDescriptor&&) = default;
    MoveOnlyDescriptor(MoveOnlyDescriptor const&) = delete;
    MoveOnlyDescriptor& operator=(MoveOnlyDescriptor const&) = delete;
};

struct StatefulAccessor
{
    using element_type = int;
    using data_handle_type = int*;
    using reference = int&;
    using offset_policy = StatefulAccessor;
    using offset_type = std::ptrdiff_t;

    std::ptrdiff_t bias = 0;

    constexpr data_handle_type offset(data_handle_type pointer, offset_type delta) const noexcept
    {
      return pointer + delta;
    }

    constexpr reference access(data_handle_type pointer, offset_type delta) const noexcept
    {
      return *(pointer + delta + bias);
    }
};

using extents_type = stdex::dextents<std::size_t, 2>;
using mdspec_type = uni20::mdspec<int, extents_type, stdex::layout_stride, StatefulAccessor, RegionDescriptor>;
using move_only_mdspec_type =
    uni20::mdspec<int, extents_type, stdex::layout_stride, StatefulAccessor, MoveOnlyDescriptor>;
using immediate_span_type = stdex::mdspan<int, extents_type, stdex::layout_stride>;

template <class Span>
concept HasDataHandle = requires(Span const& span) { span.data_handle(); };

template <class Span>
concept HasMatrixSubscript =
    requires(Span const& span, typename Span::index_type index) { span.operator[](index, index); };

struct IndependentMdspecFacade
{
    using element_type = typename mdspec_type::element_type;
    using value_type = typename mdspec_type::value_type;
    using index_type = typename mdspec_type::index_type;
    using extents_type = typename mdspec_type::extents_type;
    using layout_type = typename mdspec_type::layout_type;
    using mapping_type = typename mdspec_type::mapping_type;
    using accessor_type = typename mdspec_type::accessor_type;
    using data_handle_type = typename mdspec_type::data_handle_type;
    using data_descriptor_type = typename mdspec_type::data_descriptor_type;
    using reference = typename mdspec_type::reference;

    [[nodiscard]] static constexpr std::size_t rank() noexcept { return extents_type::rank(); }
    [[nodiscard]] static constexpr bool is_always_strided() noexcept { return true; }

    [[nodiscard]] extents_type const& extents() const;
    [[nodiscard]] index_type extent(std::size_t axis) const;
    [[nodiscard]] mapping_type const& mapping() const;
    [[nodiscard]] accessor_type const& accessor() const;
    [[nodiscard]] data_descriptor_type const& data_descriptor() const;
    [[nodiscard]] index_type stride(std::size_t axis) const;
};

struct MissingDataDescriptorObserver : IndependentMdspecFacade
{
    data_descriptor_type const& data_descriptor() const = delete;
};

static_assert(uni20::AccessorPolicy<StatefulAccessor>);
static_assert(uni20::MdspecLike<mdspec_type>);
static_assert(uni20::MdspecLike<mdspec_type const&>);
static_assert(uni20::MutableMdspecLike<mdspec_type>);
static_assert(uni20::MutableStridedMdspecLike<mdspec_type>);
static_assert(uni20::RankedMdspecLike<mdspec_type, 2>);
static_assert(uni20::MutableRankedMdspecLike<mdspec_type, 2>);
static_assert(!uni20::RankedMdspecLike<mdspec_type, 1>);
static_assert(uni20::StridedMdspecLike<mdspec_type>);
static_assert(uni20::RankedStridedMdspecLike<mdspec_type, 2>);
static_assert(uni20::MutableRankedStridedMdspecLike<mdspec_type, 2>);
static_assert(uni20::MdspecLike<move_only_mdspec_type>);
static_assert(std::movable<move_only_mdspec_type>);
static_assert(!std::copy_constructible<move_only_mdspec_type>);

static_assert(uni20::MdspecLike<immediate_span_type>);
static_assert(uni20::RankedMdspecLike<immediate_span_type, 2>);
static_assert(uni20::StridedMdspecLike<immediate_span_type>);
static_assert(uni20::MutableRankedStridedMdspecLike<immediate_span_type, 2>);

static_assert(uni20::MdspecLike<IndependentMdspecFacade>);
static_assert(uni20::RankedStridedMdspecLike<IndependentMdspecFacade, 2>);
static_assert(!uni20::MdspecLike<MissingDataDescriptorObserver>);

static_assert(!uni20::MdspanLike<mdspec_type>);
static_assert(!HasDataHandle<mdspec_type>);
static_assert(!HasMatrixSubscript<mdspec_type>);

static_assert(std::same_as<typename mdspec_type::element_type, int>);
static_assert(std::same_as<typename mdspec_type::value_type, int>);
static_assert(std::same_as<typename mdspec_type::data_handle_type, int*>);
static_assert(std::same_as<typename mdspec_type::reference, int&>);

TEST(MdspecTest, PreservesDescriptorMappingAndAccessor)
{
  extents_type const extents{2, 3};
  std::array<std::size_t, 2> const strides{1, 5};
  mdspec_type::mapping_type const mapping{extents, strides};
  RegionDescriptor const descriptor{.storage_id = 17, .element_offset = 4};
  StatefulAccessor const accessor{.bias = 2};

  mdspec_type const span{descriptor, mapping, accessor};

  EXPECT_EQ(span.data_descriptor().storage_id, 17);
  EXPECT_EQ(span.data_descriptor().element_offset, 4U);
  EXPECT_EQ(span.extent(0), 2U);
  EXPECT_EQ(span.extent(1), 3U);
  EXPECT_EQ(span.mapping().stride(0), 1U);
  EXPECT_EQ(span.mapping().stride(1), 5U);
  EXPECT_EQ(span.stride(0), 1U);
  EXPECT_EQ(span.stride(1), 5U);
  EXPECT_EQ(span.accessor().bias, 2);
  EXPECT_TRUE(span.is_unique());
  EXPECT_TRUE(span.is_strided());
}

TEST(MdspecTest, MutableDescriptorObserverPreservesDescriptorIdentity)
{
  extents_type const extents{2, 3};
  std::array<std::size_t, 2> const strides{1, 2};
  mdspec_type span{RegionDescriptor{.storage_id = 17, .element_offset = 4}, mdspec_type::mapping_type{extents, strides},
                   StatefulAccessor{}};

  span.data_descriptor().element_offset = 9;

  EXPECT_EQ(std::as_const(span).data_descriptor().element_offset, 9U);
}

TEST(MdspecTest, ExposesMdspanCompatibleStaticMetadata)
{
  static_assert(mdspec_type::rank() == 2);
  static_assert(mdspec_type::rank_dynamic() == 2);
  static_assert(mdspec_type::static_extent(0) == stdex::dynamic_extent);
  static_assert(mdspec_type::static_extent(1) == stdex::dynamic_extent);
  static_assert(mdspec_type::is_always_unique());
  static_assert(!mdspec_type::is_always_exhaustive());
  static_assert(mdspec_type::is_always_strided());
}

TEST(MdspecTest, MaterializesStridesWithoutADataHandle)
{
  extents_type const extents{2, 3};
  std::array<std::size_t, 2> const expected{1, 5};
  mdspec_type const span{RegionDescriptor{}, mdspec_type::mapping_type{extents, expected}, StatefulAccessor{}};

  EXPECT_EQ(uni20::strides(span), expected);
}

} // namespace
