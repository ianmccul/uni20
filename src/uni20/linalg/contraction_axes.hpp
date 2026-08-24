#pragma once

/**
 * \file contraction_axes.hpp
 * \ingroup linalg
 * \brief Normalized axis metadata for fixed-rank pairwise tensor contractions.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/types.hpp>
#include <uni20/mdspan/mdspan.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace uni20::linalg
{

/// \brief Canonical contracted and surviving axes for two fixed-rank operands.
/// \details Contracted pairs are ordered by the left operand axis. Surviving
///          axes retain their original order. The corresponding output order
///          is all surviving left axes followed by all surviving right axes.
template <std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank> struct ContractionAxes
{
    static_assert(ContractedRank <= LhsRank);
    static_assert(ContractedRank <= RhsRank);

    static constexpr std::size_t lhs_rank = LhsRank;
    static constexpr std::size_t rhs_rank = RhsRank;
    static constexpr std::size_t contracted_rank = ContractedRank;
    static constexpr std::size_t lhs_surviving_rank = lhs_rank - contracted_rank;
    static constexpr std::size_t rhs_surviving_rank = rhs_rank - contracted_rank;
    static constexpr std::size_t output_rank = lhs_surviving_rank + rhs_surviving_rank;

    std::array<std::size_t, contracted_rank> lhs_contracted{};
    std::array<std::size_t, contracted_rank> rhs_contracted{};
    std::array<std::size_t, lhs_surviving_rank> lhs_surviving{};
    std::array<std::size_t, rhs_surviving_rank> rhs_surviving{};
};

/// \brief Return whether contraction axes form canonical partitions of both inputs.
template <std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank>
[[nodiscard]] constexpr bool
contraction_axes_are_valid(ContractionAxes<LhsRank, RhsRank, ContractedRank> const& axes) noexcept
{
  std::array<bool, LhsRank> lhs_present{};
  std::array<bool, RhsRank> rhs_present{};

  for (std::size_t index = 0; index < ContractedRank; ++index)
  {
    auto const lhs_axis = axes.lhs_contracted[index];
    auto const rhs_axis = axes.rhs_contracted[index];
    if (lhs_axis >= LhsRank || rhs_axis >= RhsRank || lhs_present[lhs_axis] || rhs_present[rhs_axis]) return false;
    if (index > 0 && axes.lhs_contracted[index - 1] >= lhs_axis) return false;
    lhs_present[lhs_axis] = true;
    rhs_present[rhs_axis] = true;
  }

  for (std::size_t index = 0; index < axes.lhs_surviving.size(); ++index)
  {
    auto const axis = axes.lhs_surviving[index];
    if (axis >= LhsRank || lhs_present[axis]) return false;
    if (index > 0 && axes.lhs_surviving[index - 1] >= axis) return false;
    lhs_present[axis] = true;
  }

  for (std::size_t index = 0; index < axes.rhs_surviving.size(); ++index)
  {
    auto const axis = axes.rhs_surviving[index];
    if (axis >= RhsRank || rhs_present[axis]) return false;
    if (index > 0 && axes.rhs_surviving[index - 1] >= axis) return false;
    rhs_present[axis] = true;
  }

  return std::ranges::all_of(lhs_present, [](bool present) { return present; }) &&
         std::ranges::all_of(rhs_present, [](bool present) { return present; });
}

/// \brief Normalize and validate a fixed-size array of contraction-axis pairs.
/// \details Each pair contains a left axis followed by its matching right
///          axis. Repeated and out-of-range axes are rejected. Pair order does
///          not affect the normalized result.
template <std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank>
[[nodiscard]] auto make_contraction_axes(std::array<std::pair<std::size_t, std::size_t>, ContractedRank> requested_axes)
    -> ContractionAxes<LhsRank, RhsRank, ContractedRank>
{
  for (auto const [lhs_axis, rhs_axis] : requested_axes)
  {
    ERROR_IF(lhs_axis >= LhsRank, "left contraction axis is out of range", lhs_axis, LhsRank);
    ERROR_IF(rhs_axis >= RhsRank, "right contraction axis is out of range", rhs_axis, RhsRank);
  }

  std::ranges::sort(requested_axes);
  ContractionAxes<LhsRank, RhsRank, ContractedRank> result;
  std::array<bool, RhsRank> rhs_contracted{};

  for (std::size_t index = 0; index < ContractedRank; ++index)
  {
    auto const [lhs_axis, rhs_axis] = requested_axes[index];
    if (index > 0) ERROR_IF(result.lhs_contracted[index - 1] == lhs_axis, "left contraction axes must be unique");
    ERROR_IF(rhs_contracted[rhs_axis], "right contraction axes must be unique");
    result.lhs_contracted[index] = lhs_axis;
    result.rhs_contracted[index] = rhs_axis;
    rhs_contracted[rhs_axis] = true;
  }

  std::size_t lhs_contracted_index = 0;
  std::size_t lhs_surviving_index = 0;
  for (std::size_t axis = 0; axis < LhsRank; ++axis)
  {
    if (lhs_contracted_index < ContractedRank && result.lhs_contracted[lhs_contracted_index] == axis)
      ++lhs_contracted_index;
    else
      result.lhs_surviving[lhs_surviving_index++] = axis;
  }

  std::size_t rhs_surviving_index = 0;
  for (std::size_t axis = 0; axis < RhsRank; ++axis)
  {
    if (!rhs_contracted[axis]) result.rhs_surviving[rhs_surviving_index++] = axis;
  }

  CHECK(contraction_axes_are_valid(result));
  return result;
}

namespace detail
{
template <std::size_t Rank, std::size_t... Axis>
[[nodiscard]] auto make_contraction_extents(std::array<uni20::index_type, Rank> const& values,
                                            std::index_sequence<Axis...>) -> stdex::dextents<uni20::index_type, Rank>
{
  return stdex::dextents<uni20::index_type, Rank>{values[Axis]...};
}
} // namespace detail

/// \brief Validate paired extents and return the canonical contraction output shape.
template <class Lhs, class Rhs, std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank>
[[nodiscard]] auto contraction_output_extents(Lhs const& lhs, Rhs const& rhs,
                                              ContractionAxes<LhsRank, RhsRank, ContractedRank> const& axes)
    -> stdex::dextents<uni20::index_type, ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank>
{
  CHECK(contraction_axes_are_valid(axes));
  for (std::size_t index = 0; index < ContractedRank; ++index)
  {
    auto const lhs_axis = axes.lhs_contracted[index];
    auto const rhs_axis = axes.rhs_contracted[index];
    ERROR_IF(lhs.extent(lhs_axis) != rhs.extent(rhs_axis), "paired contraction extents do not agree", lhs_axis,
             rhs_axis, lhs.extent(lhs_axis), rhs.extent(rhs_axis));
  }

  constexpr std::size_t output_rank = ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank;
  std::array<uni20::index_type, output_rank> values{};
  std::size_t output_axis = 0;
  for (auto const axis : axes.lhs_surviving)
    values[output_axis++] = static_cast<uni20::index_type>(lhs.extent(axis));
  for (auto const axis : axes.rhs_surviving)
    values[output_axis++] = static_cast<uni20::index_type>(rhs.extent(axis));

  return detail::make_contraction_extents(values, std::make_index_sequence<output_rank>{});
}

} // namespace uni20::linalg
