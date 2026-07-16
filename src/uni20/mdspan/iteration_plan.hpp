#pragma once

/**
 * \file iteration_plan.hpp
 * \ingroup mdspan_ext
 * \brief Helpers for constructing merged iteration plans over strided tensors.
 */

#include <uni20/common/static_vector.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/strides.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <tuple>
#include <utility>

namespace uni20
{

/// \brief Represents a single dimension's extent and stride for iteration planning.
/// \tparam ExtentT Integral type used to store the extent.
/// \tparam StrideT Integral type used to store the stride.
/// \ingroup internal
template <typename ExtentT = std::size_t, typename StrideT = std::ptrdiff_t> struct extent_stride
{
    ExtentT extent{};
    StrideT stride{};

    constexpr extent_stride() = default;
    constexpr extent_stride(ExtentT extent_, StrideT stride_) : extent(extent_), stride(stride_) {}

    /// \brief Returns true when two adjacent dimensions can be merged.
    /// \param inner Metadata describing the inner dimension.
    /// \ingroup internal
    [[nodiscard]] constexpr bool can_merge_with_inner(extent_stride inner) const noexcept
    {
      return stride == inner.stride * static_cast<StrideT>(inner.extent);
    }

    /// \brief Merge an inner dimension into this one.
    /// \param inner The dimension that satisfies can_merge_with_inner.
    /// \ingroup internal
    constexpr void merge_with_inner(extent_stride inner) noexcept
    {
      extent *= inner.extent;
      stride = inner.stride;
    }
};

/// \brief Construct a merged iteration plan and offset for a single mapping.
/// \tparam Mapping Layout mapping type modelling the mdspan mapping interface.
/// \param mapping Mapping used to compute strides and extents.
/// \return Pair consisting of the merged plan and the starting offset adjustment.
/// \ingroup internal
template <typename Mapping> auto make_iteration_plan_with_offset(Mapping const& mapping)
{
  using size_type = typename Mapping::size_type;
  using index_type = typename Mapping::index_type;
  static constexpr size_type Rank = Mapping::extents_type::rank();

  static_vector<extent_strides<1>, Rank> raw_plan;
  static_vector<extent_stride<size_type, index_type>, Rank> plan;
  index_type offset = 0;

  for (size_type i = 0; i < Rank; ++i)
  {
    size_type const extent = mapping.extents().extent(i);

    // A zero-extent dimension makes the whole iteration empty (0 elements).
    // Encode that as a single retained extent-0 dim so the loop nest runs it
    // zero times; no other dimension's stride matters. This is deliberately
    // distinct from an *empty* plan, which denotes a single element at the base
    // offset (a rank-0 scalar) -- see the empty-plan return below.
    if (extent == 0)
    {
      plan.emplace_back(size_type{0}, index_type{0});
      return std::pair{plan, index_type{0}};
    }

    // A size-1 dimension spans index 0 only: it adds 0*stride to every offset
    // and a factor of 1 to the element count, so it coalesces with anything by
    // simply vanishing. Drop it here rather than leaving it to the arithmetic
    // merge, whose merge_with_inner would otherwise adopt this dim's
    // meaningless stride.
    if (extent == 1) continue;

    index_type stride = mapping.stride(i);
    if (stride < 0)
    {
      offset += stride * static_cast<index_type>(extent - 1);
      stride = -stride;
    }

    raw_plan.emplace_back(extent, std::array<index_type, 1>{stride});
  }

  // An empty plan denotes a single element at the base offset (rank-0 scalar):
  // every dimension was size-1, or the mapping is rank-0, so there is exactly
  // one element to visit.
  if (raw_plan.empty()) return std::pair{plan, offset};

  std::sort(raw_plan.begin(), raw_plan.end(),
            [](auto const& lhs, auto const& rhs) { return std::abs(lhs.strides[0]) > std::abs(rhs.strides[0]); });

  merge_strides_right(raw_plan);

  for (auto const& dim : raw_plan)
  {
    plan.emplace_back(static_cast<size_type>(dim.extent), static_cast<index_type>(dim.strides[0]));
  }

  return std::pair{plan, offset};
}

/// \brief Build a merged iteration plan for multiple tensors sharing the same extents.
/// \details The first mapping determines iteration order and negative-stride
///          normalization. Remaining mappings may have different layout and
///          extents types, but must have the same rank and runtime extents.
/// \tparam Mappings Layout mapping types modelling the mdspan mapping interface.
/// \param mappings Tuple of mappings, with the writable output mapping first.
/// \return Pair of the merged plan and the per-tensor offset corrections.
/// \ingroup internal
template <typename... Mappings> auto make_multi_iteration_plan_with_offset(std::tuple<Mappings...> const& mappings)
{
  static_assert(sizeof...(Mappings) >= 1, "At least one mapping is required");

  using first_mapping = std::tuple_element_t<0, std::tuple<Mappings...>>;
  using size_type = typename first_mapping::size_type;
  using index_type = std::ptrdiff_t;
  static constexpr std::size_t N = sizeof...(Mappings);
  static constexpr size_type Rank = first_mapping::extents_type::rank();

  static_assert(((Mappings::extents_type::rank() == Rank) && ...),
                "All iteration-plan mappings must have the same rank");

  auto const& base_mapping = std::get<0>(mappings);
  auto const& base_extents = base_mapping.extents();

  auto check_extents = [&](auto const& mapping) {
    for (size_type axis = 0; axis < Rank; ++axis)
      PRECONDITION_EQUAL(mapping.extents().extent(axis), base_extents.extent(axis),
                         "elementwise iteration requires equal extents");
  };
  std::apply([&](auto const&... mapping) { (check_extents(mapping), ...); }, mappings);

  static_vector<extent_strides<N>, Rank> raw_plan;
  std::array<index_type, N> offsets{};

  for (size_type i = 0; i < Rank; ++i)
  {
    size_type const extent = base_extents.extent(i);

    // A zero-extent dimension makes the whole iteration empty (0 elements):
    // retain a single extent-0 dim so the loop nest runs zero times, and reset
    // the offset corrections (unused once nothing is visited). Distinct from an
    // empty plan, which denotes a single element (rank-0 scalar).
    if (extent == 0)
    {
      raw_plan.clear();
      raw_plan.emplace_back(size_type{0}, std::array<index_type, N>{});
      offsets = std::array<index_type, N>{};
      return std::pair{raw_plan, offsets};
    }

    // Size-1 dimensions span index 0 only (0*stride, count factor 1), so drop
    // them; they coalesce with anything by vanishing.
    if (extent == 1) continue;

    auto strides = std::apply(
        [i](auto const&... mapping) {
          return std::array<index_type, N>{static_cast<index_type>(mapping.stride(i))...};
        },
        mappings);

    if (strides[0] < 0)
    {
      for (std::size_t k = 0; k < N; ++k)
      {
        offsets[k] += strides[k] * static_cast<index_type>(extent - 1);
        strides[k] = -strides[k];
      }
    }

    raw_plan.emplace_back(extent, strides);
  }

  // Empty plan => single element at the base offset (rank-0 scalar).
  if (raw_plan.empty()) return std::pair{raw_plan, offsets};

  std::sort(raw_plan.begin(), raw_plan.end(),
            [](auto const& lhs, auto const& rhs) { return std::abs(lhs.strides[0]) > std::abs(rhs.strides[0]); });

  merge_strides_right(raw_plan);

  return std::pair{raw_plan, offsets};
}

} // namespace uni20
