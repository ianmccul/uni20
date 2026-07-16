#pragma once

/**
 * \file reduction_axes.hpp
 * \ingroup linalg
 * \brief Normalized input and surviving axes for fixed-rank tensor reductions.
 */

#include <uni20/common/trace.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Integral runtime axis accepted by reduction front ends.
template <class T>
concept ReductionAxis = std::integral<std::remove_cvref_t<T>> && (!std::same_as<std::remove_cvref_t<T>, bool>);

/// \brief Normalized axes for reducing a fixed-rank input.
/// \details Reduced and surviving axes are both sorted in ascending input-axis
///          order. Negative user axes are normalized before this descriptor is
///          constructed.
template <std::size_t InputRank, std::size_t ReducedRank> struct ReductionAxes
{
    static_assert(ReducedRank <= InputRank);

    static constexpr std::size_t input_rank = InputRank;
    static constexpr std::size_t reduced_rank = ReducedRank;
    static constexpr std::size_t output_rank = InputRank - ReducedRank;

    std::array<std::size_t, reduced_rank> reduced{};
    std::array<std::size_t, output_rank> surviving{};
};

namespace detail
{

template <std::size_t InputRank, ReductionAxis Axis>
[[nodiscard]] auto normalize_reduction_axis(Axis axis) -> std::size_t
{
  using axis_type = std::remove_cvref_t<Axis>;
  if constexpr (std::signed_integral<axis_type>)
  {
    if (axis < 0)
    {
      using unsigned_axis = std::make_unsigned_t<axis_type>;
      unsigned_axis magnitude = static_cast<unsigned_axis>(-(axis + 1));
      ++magnitude;
      ERROR_IF(std::cmp_greater(magnitude, InputRank), "reduction axis is out of range");
      return InputRank - static_cast<std::size_t>(magnitude);
    }
  }

  ERROR_IF(std::cmp_greater_equal(axis, InputRank), "reduction axis is out of range");
  return static_cast<std::size_t>(axis);
}

} // namespace detail

/// \brief Return whether a reduction-axis descriptor partitions every input axis.
template <std::size_t InputRank, std::size_t ReducedRank>
[[nodiscard]] constexpr bool reduction_axes_are_valid(ReductionAxes<InputRank, ReducedRank> const& axes) noexcept
{
  std::array<bool, InputRank> present{};
  std::size_t previous = 0;
  bool have_previous = false;

  for (std::size_t axis : axes.reduced)
  {
    if (axis >= InputRank || present[axis] || (have_previous && axis <= previous)) return false;
    present[axis] = true;
    previous = axis;
    have_previous = true;
  }

  previous = 0;
  have_previous = false;
  for (std::size_t axis : axes.surviving)
  {
    if (axis >= InputRank || present[axis] || (have_previous && axis <= previous)) return false;
    present[axis] = true;
    previous = axis;
    have_previous = true;
  }

  return std::ranges::all_of(present, [](bool value) { return value; });
}

/// \brief Normalize and validate a runtime reduction-axis pack.
/// \details Negative axes count backward from `InputRank`. Duplicate and
///          out-of-range axes are rejected before backend dispatch.
template <std::size_t InputRank, ReductionAxis... Axes>
[[nodiscard]] auto make_reduction_axes(Axes... requested_axes) -> ReductionAxes<InputRank, sizeof...(Axes)>
{
  static_assert(sizeof...(Axes) <= InputRank);
  ReductionAxes<InputRank, sizeof...(Axes)> result;
  result.reduced = {detail::normalize_reduction_axis<InputRank>(requested_axes)...};
  std::ranges::sort(result.reduced);

  for (std::size_t index = 1; index < result.reduced.size(); ++index)
    ERROR_IF(result.reduced[index - 1] == result.reduced[index], "reduction axes must be unique");

  std::size_t reduced_index = 0;
  std::size_t surviving_index = 0;
  for (std::size_t axis = 0; axis < InputRank; ++axis)
  {
    if (reduced_index < result.reduced.size() && result.reduced[reduced_index] == axis)
      ++reduced_index;
    else
      result.surviving[surviving_index++] = axis;
  }

  CHECK(reduction_axes_are_valid(result));
  return result;
}

/// \brief Construct the full-reduction descriptor for a fixed input rank.
template <std::size_t InputRank>
[[nodiscard]] constexpr auto all_reduction_axes() -> ReductionAxes<InputRank, InputRank>
{
  ReductionAxes<InputRank, InputRank> result;
  for (std::size_t axis = 0; axis < InputRank; ++axis)
    result.reduced[axis] = axis;
  return result;
}

} // namespace uni20::linalg
