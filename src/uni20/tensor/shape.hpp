#pragma once

/**
 * \file shape.hpp
 * \ingroup tensor
 * \brief Checked helpers for constructing runtime tensor extents.
 */

#include <uni20/common/mdspan.hpp>
#include <uni20/common/trace.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace uni20::detail
{

template <std::integral Extent> [[nodiscard]] constexpr bool tensor_extent_is_valid(Extent extent) noexcept
{
  if constexpr (std::signed_integral<Extent>)
    if (extent < 0) return false;
  return std::in_range<index_type>(extent);
}

template <std::integral... Extents> [[nodiscard]] auto make_tensor_extents(Extents... extents)
{
  ERROR_IF(!(tensor_extent_is_valid(extents) && ...),
           "tensor extents must be nonnegative and representable by uni20::index_type");
  using extents_type = stdex::dextents<index_type, sizeof...(Extents)>;
  return extents_type{static_cast<index_type>(extents)...};
}

template <class Extents> [[nodiscard]] std::size_t checked_element_count(Extents const& extents)
{
  bool has_zero_extent = false;
  for (std::size_t axis = 0; axis < Extents::rank(); ++axis)
  {
    auto const extent = extents.extent(axis);
    ERROR_IF(!tensor_extent_is_valid(extent), "tensor extents must be nonnegative and representable");
    has_zero_extent = has_zero_extent || extent == 0;
  }
  if (has_zero_extent) return 0;

  std::size_t result = 1;
  for (std::size_t axis = 0; axis < Extents::rank(); ++axis)
  {
    auto const value = static_cast<std::size_t>(extents.extent(axis));
    ERROR_IF(result > std::numeric_limits<std::size_t>::max() / value, "tensor element count overflows size_t");
    result *= value;
  }
  return result;
}

template <std::integral Extent> [[nodiscard]] constexpr bool is_inferred_extent(Extent extent) noexcept
{
  if constexpr (std::signed_integral<Extent>) return extent == static_cast<Extent>(-1);
  return false;
}

template <std::integral... Extents>
[[nodiscard]] auto make_reshape_extents(std::size_t source_size, Extents... requested_extents)
{
  constexpr std::size_t rank = sizeof...(Extents);
  std::array<index_type, rank> values{};
  std::array<bool, rank> inferred{};
  std::size_t axis = 0;
  auto record_extent = [&](auto requested) {
    bool const infer = is_inferred_extent(requested);
    ERROR_IF(!infer && !tensor_extent_is_valid(requested),
             "reshape extents must be nonnegative, except for one inferred -1 extent");
    inferred[axis] = infer;
    values[axis] = infer ? index_type{0} : static_cast<index_type>(requested);
    ++axis;
  };
  (record_extent(requested_extents), ...);

  std::size_t inferred_axis = rank;
  std::size_t known_size = 1;
  for (axis = 0; axis < rank; ++axis)
  {
    if (inferred[axis])
    {
      ERROR_IF(inferred_axis != rank, "reshape accepts at most one inferred extent");
      inferred_axis = axis;
      continue;
    }
    auto const extent = static_cast<std::size_t>(values[axis]);
    if (extent == 0)
    {
      known_size = 0;
      continue;
    }
    ERROR_IF(known_size > std::numeric_limits<std::size_t>::max() / extent, "reshape element count overflows size_t");
    known_size *= extent;
  }

  if (inferred_axis != rank)
  {
    ERROR_IF(known_size == 0, "cannot infer a reshape extent when the specified extents have zero product");
    ERROR_IF(source_size % known_size != 0, "inferred reshape extent does not divide the source element count");
    auto const inferred_value = source_size / known_size;
    ERROR_IF(!std::in_range<index_type>(inferred_value), "inferred reshape extent is not representable");
    values[inferred_axis] = static_cast<index_type>(inferred_value);
  }
  else
  {
    ERROR_IF(known_size != source_size, "reshape source and destination element counts differ");
  }

  using extents_type = stdex::dextents<index_type, rank>;
  return [&]<std::size_t... Axis>(std::index_sequence<Axis...>) {
    return extents_type{values[Axis]...};
  }(std::make_index_sequence<rank>{});
}

template <class TargetExtents, class SourceExtents>
[[nodiscard]] auto convert_reshape_extents(SourceExtents const& source) -> TargetExtents
{
  static_assert(TargetExtents::rank() == SourceExtents::rank());
  for (std::size_t axis = 0; axis < TargetExtents::rank(); ++axis)
  {
    auto const fixed_extent = TargetExtents::static_extent(axis);
    ERROR_IF(fixed_extent != stdex::dynamic_extent && static_cast<std::size_t>(source.extent(axis)) != fixed_extent,
             "reshape extent does not match the tensor's static extent");
  }

  using target_index = typename TargetExtents::index_type;
  return [&]<std::size_t... Axis>(std::index_sequence<Axis...>) {
    return TargetExtents{static_cast<target_index>(source.extent(Axis))...};
  }(std::make_index_sequence<TargetExtents::rank()>{});
}

} // namespace uni20::detail
