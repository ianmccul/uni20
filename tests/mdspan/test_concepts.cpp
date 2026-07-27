#include <gtest/gtest.h>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/mdspan.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

using namespace uni20;

namespace
{

struct AccessorWithOffset
{
    using element_type = int;
    using data_handle_type = int*;
    using reference = int&;
    using offset_policy = AccessorWithOffset;
    using offset_type = std::ptrdiff_t;

    constexpr data_handle_type offset(data_handle_type ptr, offset_type delta) const noexcept { return ptr + delta; }

    constexpr reference access(data_handle_type ptr, offset_type delta) const noexcept { return *(ptr + delta); }
};

struct AccessorWithoutOffset
{
    using element_type = int;
    using data_handle_type = int*;
    using reference = int&;
    using offset_policy = AccessorWithoutOffset;

    constexpr data_handle_type offset(data_handle_type ptr, std::size_t delta) const noexcept
    {
      return ptr + static_cast<std::ptrdiff_t>(delta);
    }

    constexpr reference access(data_handle_type ptr, std::size_t delta) const noexcept
    {
      return *(ptr + static_cast<std::ptrdiff_t>(delta));
    }
};

struct StatefulAccessor
{
    using element_type = int;
    using data_handle_type = int*;
    using reference = int&;
    using offset_policy = StatefulAccessor;
    using offset_type = std::ptrdiff_t;

    std::ptrdiff_t bias = 0;

    constexpr data_handle_type offset(data_handle_type ptr, offset_type delta) const noexcept { return ptr + delta; }

    constexpr reference access(data_handle_type ptr, offset_type delta) const noexcept { return *(ptr + delta + bias); }
};

struct ThrowingIndex
{
    operator std::size_t() const noexcept(false);
};

template <class Span, class... Index>
concept HasConstAccess =
    requires(Span const& span, Index... indices) { uni20::const_access(span, std::move(indices)...); };

} // namespace

static_assert(std::is_same_v<span_offset_t<AccessorWithOffset>, AccessorWithOffset::offset_type>);
static_assert(std::is_same_v<span_offset_t<AccessorWithoutOffset>, std::size_t>);

using MutableAccessor = AccessorWithoutOffset;
using ExpectedConstAccessor = const_accessor_adaptor<MutableAccessor>;
static_assert(std::is_same_v<const_accessor_t<MutableAccessor>, ExpectedConstAccessor>);
static_assert(std::is_same_v<const_accessor_t<ExpectedConstAccessor>, ExpectedConstAccessor>);
static_assert(is_default_accessor_v<stdex::default_accessor<int>>);
static_assert(is_default_accessor_v<stdex::default_accessor<int const>>);
static_assert(!is_default_accessor_v<MutableAccessor>);

TEST(MdspanConcepts, ConstAccessorAdaptorYieldsConstReference)
{
  MutableAccessor accessor{};
  auto const_accessor_policy = const_accessor(accessor);

  int values[] = {1, 2, 3, 4};
  auto* const handle = values;

  auto* const advanced_handle = const_accessor_policy.offset(handle, 2);
  EXPECT_EQ(values + 2, advanced_handle);

  auto&& ref = const_accessor_policy.access(handle, 1);
  static_assert(std::is_same_v<decltype(ref), int const&>);
  EXPECT_EQ(2, ref);
}

using DynamicExtent = stdex::extents<std::size_t, stdex::dynamic_extent>;
using DynamicMatrixExtents = stdex::dextents<std::size_t, 2>;
using StaticSpan = stdex::mdspan<int, stdex::extents<std::size_t, 2, 3>>;
using ConstStaticSpan = stdex::mdspan<int const, stdex::extents<std::size_t, 2, 3>>;
using StridedSpan = stdex::mdspan<int, DynamicExtent, stdex::layout_stride>;
using ConstStridedSpan = stdex::mdspan<int const, DynamicExtent, stdex::layout_stride>;
using StridedMatrixSpan = stdex::mdspan<int, DynamicMatrixExtents, stdex::layout_stride>;
using ConstStridedMatrixSpan = stdex::mdspan<int const, DynamicMatrixExtents, stdex::layout_stride>;
using CustomAccessorSpan = stdex::mdspan<int, DynamicMatrixExtents, stdex::layout_stride, MutableAccessor>;
using StatefulSpan = stdex::mdspan<int, DynamicExtent, stdex::layout_right, StatefulAccessor>;

static_assert(HasConstAccess<StaticSpan, int, int>);
static_assert(!HasConstAccess<StaticSpan, int>);
static_assert(!HasConstAccess<StaticSpan, int, int, int>);
static_assert(!HasConstAccess<StaticSpan, ThrowingIndex, int>);

TEST(MdspanConcepts, ConstAccessPreservesStatefulAccessor)
{
  int values[] = {10, 20, 30, 40};
  StatefulSpan::mapping_type mapping{DynamicExtent{3}};
  StatefulSpan span{values, mapping, StatefulAccessor{.bias = 1}};

  auto&& ref = const_access(span, 1);
  static_assert(std::is_same_v<decltype(ref), int const&>);
  EXPECT_EQ(30, ref);
}

struct SpanDescriptorWithoutSubscript
{
    using storage_span = StaticSpan;
    using element_type = typename storage_span::element_type;
    using value_type = typename storage_span::value_type;
    using index_type = typename storage_span::index_type;
    using extents_type = typename storage_span::extents_type;
    using layout_type = typename storage_span::layout_type;
    using mapping_type = typename storage_span::mapping_type;
    using accessor_type = typename storage_span::accessor_type;
    using data_handle_type = typename storage_span::data_handle_type;
    using reference = typename storage_span::reference;

    static constexpr std::size_t rank() noexcept { return extents_type::rank(); }

    extents_type const& extents() const;
    index_type extent(std::size_t axis) const;
    mapping_type const& mapping() const;
    data_handle_type data_handle() const;
    accessor_type const& accessor() const;
};

struct CompleteSpanFacade : SpanDescriptorWithoutSubscript
{
    reference operator[](index_type row, index_type column) const;
};

struct MissingExtentSpan : CompleteSpanFacade
{
    index_type extent(std::size_t axis) const = delete;
};

struct ClaimedStridedSpanWithoutStride : CompleteSpanFacade
{
    static constexpr bool is_always_strided() noexcept { return true; }
};

static_assert(MdspanLike<StaticSpan>);
static_assert(MutableMdspanLike<StaticSpan>);
static_assert(MdspanLike<ConstStaticSpan>);
static_assert(!MutableMdspanLike<ConstStaticSpan>);

static_assert(StridedMdspanLike<StridedSpan>);
static_assert(MutableStridedMdspanLike<StridedSpan>);
static_assert(StridedMdspanLike<ConstStridedSpan>);
static_assert(!MutableStridedMdspanLike<ConstStridedSpan>);
static_assert(DefaultAccessorMdspanLike<StridedSpan>);
static_assert(DefaultAccessorMdspanLike<ConstStridedSpan>);
static_assert(!DefaultAccessorMdspanLike<CustomAccessorSpan>);

static_assert(RankedMdspanLike<StridedMatrixSpan, 2>);
static_assert(RankedMdspanLike<StridedMatrixSpan&, 2>);
static_assert(!RankedMdspanLike<StridedMatrixSpan, 1>);
static_assert(MutableRankedMdspanLike<StridedMatrixSpan, 2>);
static_assert(RankedMdspanLike<ConstStridedMatrixSpan, 2>);
static_assert(!MutableRankedMdspanLike<ConstStridedMatrixSpan, 2>);
static_assert(RankedStridedMdspanLike<StridedMatrixSpan, 2>);
static_assert(RankedStridedMdspanLike<StridedMatrixSpan&, 2>);
static_assert(!RankedStridedMdspanLike<StridedMatrixSpan, 1>);
static_assert(MutableRankedStridedMdspanLike<StridedMatrixSpan, 2>);
static_assert(MutableRankedStridedMdspanLike<StridedMatrixSpan&, 2>);
static_assert(RankedStridedMdspanLike<ConstStridedMatrixSpan, 2>);
static_assert(!MutableRankedStridedMdspanLike<ConstStridedMatrixSpan, 2>);

static_assert(!MdspanLike<SpanDescriptorWithoutSubscript>);
static_assert(MdspanLike<CompleteSpanFacade>);
static_assert(MutableMdspanLike<CompleteSpanFacade>);
static_assert(!MdspanLike<MissingExtentSpan>);
static_assert(!StridedMdspanLike<ClaimedStridedSpanWithoutStride>);

struct NotSpanLike
{};

static_assert(!MdspanLike<NotSpanLike>);
static_assert(!MutableMdspanLike<NotSpanLike>);
static_assert(!StridedMdspanLike<NotSpanLike>);
static_assert(!MutableStridedMdspanLike<NotSpanLike>);
static_assert(!RankedMdspanLike<NotSpanLike, 2>);
static_assert(!MutableRankedMdspanLike<NotSpanLike, 2>);
static_assert(!RankedStridedMdspanLike<NotSpanLike, 2>);
static_assert(!MutableRankedStridedMdspanLike<NotSpanLike, 2>);
