#include "common/mdspan.hpp"
#include "mdspan/concepts.hpp"
#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

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

} // namespace

static_assert(std::is_same_v<span_offset_t<AccessorWithOffset>, AccessorWithOffset::offset_type>);
static_assert(std::is_same_v<span_offset_t<AccessorWithoutOffset>, std::size_t>);

using MutableAccessor = AccessorWithoutOffset;
using ExpectedConstAccessor = const_accessor_adaptor<MutableAccessor, MutableAccessor::element_type const&>;
static_assert(std::is_same_v<const_accessor_t<MutableAccessor>, ExpectedConstAccessor>);

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
using StaticSpan = stdex::mdspan<int, stdex::extents<std::size_t, 2, 3>>;
using ConstStaticSpan = stdex::mdspan<int const, stdex::extents<std::size_t, 2, 3>>;
using StridedSpan = stdex::mdspan<int, DynamicExtent, stdex::layout_stride>;
using ConstStridedSpan = stdex::mdspan<int const, DynamicExtent, stdex::layout_stride>;

static_assert(SpanLike<StaticSpan>);
static_assert(MutableSpanLike<StaticSpan>);
static_assert(SpanLike<ConstStaticSpan>);
static_assert(!MutableSpanLike<ConstStaticSpan>);

static_assert(StridedMdspan<StridedSpan>);
static_assert(MutableStridedMdspan<StridedSpan>);
static_assert(StridedMdspan<ConstStridedSpan>);
static_assert(!MutableStridedMdspan<ConstStridedSpan>);

struct NotSpanLike
{};

static_assert(!SpanLike<NotSpanLike>);
static_assert(!MutableSpanLike<NotSpanLike>);
static_assert(!StridedMdspan<NotSpanLike>);
static_assert(!MutableStridedMdspan<NotSpanLike>);
