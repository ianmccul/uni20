#pragma once

#include "common/mdspan.hpp"
#include "common/static_vector.hpp"
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{

/// @brief Represents a single dimension's extent and its corresponding stride.
///
/// Used for creating iteration plans for multidimensional memory traversal.
/// The extent is the size along that dimension, and the stride is the number
/// of memory units to jump to reach the next element in that dimension.
///
/// @tparam Index Integer type for memory strides (usually signed).
/// @tparam Size  Integer type for extents (usually unsigned).
template <typename ExtentT = std::size_t, typename StrideT = std::ptrdiff_t> struct extent_stride
{
    ExtentT extent;
    StrideT stride;

    constexpr extent_stride() = default;
    constexpr extent_stride(ExtentT extent, StrideT stride) : extent(extent), stride(stride) {}

    /// @brief Returns true if the current (outer) dimension and the given inner dimension can be merged.
    ///
    /// Coalescing is allowed if the inner dimension's stride equals the outer stride multiplied by the outer extent.
    /// This ensures a contiguous memory layout between the two.
    constexpr bool can_merge_with_inner(extent_stride inner) const noexcept
    {
      return stride == inner.stride * static_cast<StrideT>(inner.extent);
    }

    /// @brief Merge an inner dimension into this one (assuming can_merge_with_inner returns true).
    ///
    /// After merging, the extent becomes the product, and the stride is updated to the inner stride.
    constexpr void merge_with_inner(extent_stride inner) noexcept
    {
      extent *= inner.extent;
      stride = inner.stride;
    }
};

/// @brief Represents a multi-tensor stride plan for one dimension.
///        All tensors share the same extent, with individual strides.
template <typename ExtentT = std::size_t, typename StrideT = std::ptrdiff_t, std::size_t N = 2>
struct multi_extent_stride
{
    ExtentT extent;
    std::array<StrideT, N> strides;

    constexpr multi_extent_stride() = default;

    constexpr multi_extent_stride(ExtentT e, std::array<StrideT, N> const& s) : extent(e), strides(s) {}

    /// Check if this (outer) dimension can be merged with the inner one
    constexpr bool can_merge_with_inner(const multi_extent_stride& inner) const noexcept
    {
      for (std::size_t i = 0; i < N; ++i)
        if (strides[i] != inner.strides[i] * static_cast<StrideT>(inner.extent)) return false;
      return true;
    }

    /// Merge an inner dimension into this one (assuming `can_merge_with_inner` is true)
    constexpr void merge_with_inner(const multi_extent_stride& inner) noexcept
    {
      extent *= inner.extent;
      strides = inner.strides;
    }
};

/// @brief Create a coalesced iteration plan for looping over a strided layout.
///
/// This function analyzes the layout of a tensor (represented by an mdspan mapping)
/// and produces a compact, optimized loop plan:
///   - Negative strides are flipped and an offset is computed
///   - Dimensions with zero extent are dropped
///   - Dimensions are sorted by stride (largest first = outermost loop)
///   - Contiguous dimensions are coalesced into a single loop
///
/// @tparam Mapping A layout mapping type (e.g. layout_stride::mapping<Extents>)
/// @param mapping The layout mapping to analyze
/// @return A pair of:
///   - `static_vector<extent_stride<size_type, index_type>, Rank>`: the compact loop plan
///   - `index_type offset`: the base offset from the data pointer
///
template <typename Mapping> auto make_iteration_plan_with_offset(Mapping const& mapping)
{
  using size_type = typename Mapping::size_type;
  using index_type = typename Mapping::index_type;
  static constexpr size_type Rank = Mapping::extents_type::rank();

  static_vector<extent_stride<size_type, index_type>, Rank> plan;
  index_type offset = 0;

  // Build plan with flipped negative strides
  for (size_type i = 0; i < Rank; ++i)
  {
    size_type extent = mapping.extents().extent(i);
    index_type stride = mapping.stride(i);

    if (stride < 0)
    {
      offset += stride * static_cast<index_type>(extent - 1);
      stride = -stride;
    }

    if (extent != 0)
    {
      plan.push_back({extent, stride});
    }
  }

  // Sort by stride descending (largest stride = outermost loop)
  std::sort(plan.begin(), plan.end(), [](auto const& a, auto const& b) { return a.stride > b.stride; });

  // Coalesce adjacent dimensions
  static_vector<extent_stride<size_type, index_type>, Rank> merged;
  for (auto const& dim : plan)
  {
    if (!merged.empty() && merged.back().can_merge_with_inner(dim))
    {
      merged.back().merge_with_inner(dim);
    }
    else
    {
      merged.push_back(dim);
    }
  }

  return std::pair{merged, offset};
}

// @brief Represents a multi-tensor strided iteration plan for a fixed-rank layout.
///
/// This variant handles multiple tensors with the same extents but independent strides.
/// It coalesces contiguous dimensions across all tensors and flips stride signs so
/// that the first tensor (typically the output) has positive strides.
///
/// @tparam Mapping A layout mapping type (e.g. layout_stride::mapping<Extents>)
/// @tparam N       Number of tensors (e.g., 2 for A = B + C)
/// @param mappings Array of mappings, one per tensor. All must have the same extents.
/// @return A pair of:
///   - `static_vector<multi_extent_stride<size, index, N>, Rank>`: coalesced loop plan
///   - `std::array<index_type, N>`: base offset for each tensor (based on stride sign flips)
template <typename Mapping, std::size_t N>
auto make_multi_iteration_plan_with_offset(std::array<Mapping, N> const& mappings)
{
  using size_type = typename Mapping::size_type;
  using index_type = typename Mapping::index_type;
  static constexpr size_type Rank = Mapping::extents_type::rank();

  static_assert(N >= 1, "At least one mapping is required");

  auto const& base_extents = mappings[0].extents();

  // Validate all mappings have the same extents
  for (std::size_t k = 1; k < N; ++k)
  {
    for (size_type i = 0; i < Rank; ++i)
    {
      assert(mappings[k].extents().extent(i) == base_extents.extent(i));
    }
  }

  static_vector<multi_extent_stride<size_type, index_type, N>, Rank> plan;
  std::array<index_type, N> offsets = {};

  for (size_type i = 0; i < Rank; ++i)
  {
    size_type extent = base_extents.extent(i);
    std::array<index_type, N> strides;

    // Get original strides
    for (std::size_t k = 0; k < N; ++k)
      strides[k] = mappings[k].stride(i);

    // Determine sign of reference (first) tensor. We could use a different strategy for handling
    // negative strides, but the first tensor will typically be the output (for out-of-place operations),
    // so that is a good choice to make all +ve strides.
    if (strides[0] < 0)
    {
      for (std::size_t k = 0; k < N; ++k)
      {
        offsets[k] += strides[k] * static_cast<index_type>(extent - 1);
        strides[k] = -strides[k];
      }
    }

    if (extent != 0)
    {
      plan.push_back({extent, strides});
    }
  }

  // Sort by outermost loop (based on tensor 0's stride)
  std::sort(plan.begin(), plan.end(), [](auto const& a, auto const& b) { return a.strides[0] > b.strides[0]; });

  // Coalesce adjacent mergeable dimensions
  static_vector<multi_extent_stride<size_type, index_type, N>, Rank> merged;

  for (auto const& dim : plan)
  {
    if (!merged.empty() && merged.back().can_merge_with_inner(dim))
    {
      merged.back().merge_with_inner(dim);
    }
    else
    {
      merged.push_back(dim);
    }
  }

  return std::pair{merged, offsets};
}

// This is probably not needed - we can put the offset into the mdspan data handle itself
// // Function to return the offset of a mdspan-like object.
// template <typename MDS> constexpr auto get_offset(MDS const& mds)
// {
//   if constexpr (requires { mds.offset(); })
//   {
//     return mds.offset();
//   }
//   else
//   {
//     return typename MDS::index_type(0);
//   }
// }

template <typename T> constexpr bool HasOffset = requires(T const& t) { t.offset(); };

namespace detail
{

template <typename DataHandle, typename Accessor, typename Op> struct UnrollHelper
{
    DataHandle data_;
    Accessor acc_;
    Op& op_;

    using idx_t = std::ptrdiff_t;

    // Depth 0: innermost loop
    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::integral_constant<std::size_t, 0>)
    {
      auto this_extent = plan->extent;
      auto this_stride = plan->stride;
      if (this_stride == 1)
      {
        for (idx_t i = 0; i < this_extent; ++i)
        {
          acc_.access(data_, Offset + i) = op_(acc_.access(data_, Offset + i));
        }
      }
      else
      {
        for (idx_t i = 0; i < this_extent; ++i)
        {
          acc_.access(data_, Offset + i * this_stride) = op_(acc_.access(data_, Offset + i * this_stride));
        }
      }
    }

    template <std::size_t N>
    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::integral_constant<std::size_t, N>)
    {
      auto this_extent = plan->extent;
      auto this_stride = plan->stride;
      ++plan;
      for (idx_t i = 0; i < this_extent; ++i)
      {
        this->run(Offset + i * this_stride, plan, std::integral_constant<std::size_t, N - 1>{});
      }
    }

    void run_dynamic(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::size_t depth)
    {
      auto this_extent = plan->extent;
      auto this_stride = plan->stride;
      ++plan;
      if (depth == 3) // if we're on the last dynamic index, then dispatch to the unrolled version
      {
        for (idx_t i = 0; i < this_extent; ++i)
        {
          this->run(Offset + i * this_stride, plan, std::integral_constant<std::size_t, 2>{});
        }
      }
      else
      {
        for (idx_t i = 0; i < this_extent; ++i) // otherwise depth must be > 4 here, recurse into run_dynamic again
        {
          this->run_dynamic(Offset + i * this_stride, plan, depth - 1);
        }
      }
    }

    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::size_t depth)
    {
      switch (depth)
      {
        case 0:
          this->run(Offset, plan, std::integral_constant<std::size_t, 0>{});
          return;
        case 1:
          this->run(Offset, plan, std::integral_constant<std::size_t, 1>{});
          return;
        case 2:
          this->run(Offset, plan, std::integral_constant<std::size_t, 2>{});
          return;
        default:
          this->run_dynamic(Offset, plan, depth);
          return;
      }
    }
};

} // namespace detail

template <typename MDS, typename Op> void apply_unary_inplace(MDS a, Op&& op)
{
  using T = typename MDS::value_type;
  using idx_t = typename MDS::index_type;

  auto const& map = a.mapping();
  auto const& acc = a.accessor();
  auto data = a.data_handle();
  auto [plan, offset] = make_iteration_plan_with_offset(map);

  if (plan.empty()) return;

  detail::UnrollHelper helper{data, acc, op};
  helper.run(offset, plan.data(), plan.size() - 1);
}

namespace detail
{

template <typename Op, typename OutAcc, typename InAcc, typename OutHandle, typename InHandle, typename idx_t>
struct BinaryUnrollHelper
{
    OutHandle dst_;
    OutAcc acc_out_;
    InHandle src_;
    InAcc acc_in_;
    Op& op_;

    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::integral_constant<std::size_t, 0>)
    {
      auto Extent = plan->extent;
      auto Stride = plan->stride;
      for (idx_t i = 0; i < Extent; ++i)
      {
        idx_t index = Offset + i * Stride;
        acc_out_.reference(dst_, index) = op_(acc_in_.reference(src_, index));
      }
    }

    template <std::size_t N>
    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::integral_constant<std::size_t, N>)
    {
      auto Extent = plan->extent;
      auto Stride = plan->stride;
      ++plan;
      for (idx_t i = 0; i < Extent; ++i)
      {
        this->run(Offset + i * Stride, plan, std::integral_constant<std::size_t, N - 1>{});
      }
    }

    void run_dynamic(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::size_t depth)
    {
      auto Extent = plan->extent;
      auto Stride = plan->stride;
      ++plan;
      if (depth == 3)
      {
        for (idx_t i = 0; i < Extent; ++i)
          this->run(Offset + i * Stride, plan, std::integral_constant<std::size_t, 2>{});
      }
      else
      {
        for (idx_t i = 0; i < Extent; ++i)
          this->run_dynamic(Offset + i * Stride, plan, depth - 1);
      }
    }

    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::size_t depth)
    {
      switch (depth)
      {
        case 0:
          this->run(Offset, plan, std::integral_constant<std::size_t, 0>{});
          return;
        case 1:
          this->run(Offset, plan, std::integral_constant<std::size_t, 1>{});
          return;
        case 2:
          this->run(Offset, plan, std::integral_constant<std::size_t, 2>{});
          return;
        default:
          this->run_dynamic(Offset, plan, depth);
          return;
      }
    }
};

} // namespace detail

template <typename MDSin, typename MDSout, typename Op> void apply_unary(MDSin src, Op&& op, MDSout dst)
{
  static_assert(std::is_same_v<typename MDSin::extents_type, typename MDSout::extents_type>);
  using idx_t = typename MDSin::index_type;

  auto [plan, offset] = make_multi_iteration_plan_with_offset(std::array{src.mapping(), dst.mapping()});

  if (plan.empty()) return;

  detail::BinaryUnrollHelper helper{dst.data_handle(), dst.accessor(), src.data_handle(), src.accessor(), op};

  helper.run(offset[1], plan.data(), plan.size() - 1);
}

} // namespace uni20
