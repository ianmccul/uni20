#include <uni20/mdspan/device_mdspan.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

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
using device_span_type =
    uni20::device_mdspan<int, extents_type, stdex::layout_stride, StatefulAccessor, RegionDescriptor>;
using move_only_device_span_type =
    uni20::device_mdspan<int, extents_type, stdex::layout_stride, StatefulAccessor, MoveOnlyDescriptor>;
using immediate_span_type = stdex::mdspan<int, extents_type, stdex::layout_stride>;

template <class Span>
concept HasDataHandle = requires(Span const& span) { span.data_handle(); };

template <class Span>
concept HasMatrixSubscript =
    requires(Span const& span, typename Span::index_type index) { span.operator[](index, index); };

struct IndependentDeviceSpanFacade
{
    using element_type = typename device_span_type::element_type;
    using value_type = typename device_span_type::value_type;
    using index_type = typename device_span_type::index_type;
    using extents_type = typename device_span_type::extents_type;
    using layout_type = typename device_span_type::layout_type;
    using mapping_type = typename device_span_type::mapping_type;
    using accessor_type = typename device_span_type::accessor_type;
    using data_handle_type = typename device_span_type::data_handle_type;
    using data_descriptor_type = typename device_span_type::data_descriptor_type;
    using reference = typename device_span_type::reference;

    [[nodiscard]] static constexpr std::size_t rank() noexcept { return extents_type::rank(); }
    [[nodiscard]] static constexpr bool is_always_strided() noexcept { return true; }

    [[nodiscard]] extents_type const& extents() const;
    [[nodiscard]] index_type extent(std::size_t axis) const;
    [[nodiscard]] mapping_type const& mapping() const;
    [[nodiscard]] accessor_type const& accessor() const;
    [[nodiscard]] data_descriptor_type const& data_descriptor() const;
    [[nodiscard]] index_type stride(std::size_t axis) const;
};

struct MissingDataDescriptorObserver : IndependentDeviceSpanFacade
{
    data_descriptor_type const& data_descriptor() const = delete;
};

static_assert(uni20::AccessorPolicy<StatefulAccessor>);
static_assert(uni20::DeviceSpanLike<device_span_type>);
static_assert(uni20::DeviceSpanLike<device_span_type const&>);
static_assert(uni20::RankedDeviceSpanLike<device_span_type, 2>);
static_assert(!uni20::RankedDeviceSpanLike<device_span_type, 1>);
static_assert(uni20::StridedDeviceSpanLike<device_span_type>);
static_assert(uni20::RankedStridedDeviceSpanLike<device_span_type, 2>);
static_assert(uni20::DeviceSpanLike<move_only_device_span_type>);
static_assert(std::movable<move_only_device_span_type>);
static_assert(!std::copy_constructible<move_only_device_span_type>);

static_assert(uni20::DeviceSpanLike<immediate_span_type>);
static_assert(uni20::RankedDeviceSpanLike<immediate_span_type, 2>);
static_assert(uni20::StridedDeviceSpanLike<immediate_span_type>);

static_assert(uni20::DeviceSpanLike<IndependentDeviceSpanFacade>);
static_assert(uni20::RankedStridedDeviceSpanLike<IndependentDeviceSpanFacade, 2>);
static_assert(!uni20::DeviceSpanLike<MissingDataDescriptorObserver>);

static_assert(!uni20::SpanLike<device_span_type>);
static_assert(!HasDataHandle<device_span_type>);
static_assert(!HasMatrixSubscript<device_span_type>);

static_assert(std::same_as<typename device_span_type::element_type, int>);
static_assert(std::same_as<typename device_span_type::value_type, int>);
static_assert(std::same_as<typename device_span_type::data_handle_type, int*>);
static_assert(std::same_as<typename device_span_type::reference, int&>);

TEST(DeviceMdspanTest, PreservesDescriptorMappingAndAccessor)
{
  extents_type const extents{2, 3};
  std::array<std::size_t, 2> const strides{1, 5};
  device_span_type::mapping_type const mapping{extents, strides};
  RegionDescriptor const descriptor{.storage_id = 17, .element_offset = 4};
  StatefulAccessor const accessor{.bias = 2};

  device_span_type const span{descriptor, mapping, accessor};

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

TEST(DeviceMdspanTest, ExposesMdspanCompatibleStaticMetadata)
{
  static_assert(device_span_type::rank() == 2);
  static_assert(device_span_type::rank_dynamic() == 2);
  static_assert(device_span_type::static_extent(0) == stdex::dynamic_extent);
  static_assert(device_span_type::static_extent(1) == stdex::dynamic_extent);
  static_assert(device_span_type::is_always_unique());
  static_assert(!device_span_type::is_always_exhaustive());
  static_assert(device_span_type::is_always_strided());
}

} // namespace
