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

/// \brief Represents a single dimension's extent and its corresponding stride.
///
/// Used for creating iteration plans for multidimensional memory traversal.
/// The extent is the size along that dimension, and the stride is the number
/// of memory units to jump to reach the next element in that dimension.
///
/// \param Index Integer type for memory strides (usually signed).
/// \param Size  Integer type for extents (usually unsigned).
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

/// \brief Create a coalesced iteration plan for looping over a strided layout.
///
/// This function analyzes the layout of a tensor (represented by an mdspan mapping)
/// and produces a compact, optimized loop plan:
///   - Negative strides are flipped and an offset is computed
///   - Dimensions with zero extent are dropped
///   - Dimensions are sorted by stride (largest first = outermost loop)
///   - Contiguous dimensions are coalesced into a single loop
///
/// \tparam Mapping A layout mapping type (e.g. layout_stride::mapping<Extents>)
/// \param mapping The layout mapping to analyze
/// \return A pair of:
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
        for (idx_t i = 0; i < idx_t(this_extent); ++i)
        {
          acc_.access(data_, Offset + i) = op_(acc_.access(data_, Offset + i));
        }
      }
      else
      {
        for (idx_t i = 0; i < idx_t(this_extent); ++i)
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
      for (idx_t i = 0; i < idx_t(this_extent); ++i)
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
        for (idx_t i = 0; i < idx_t(this_extent); ++i)
        {
          this->run(Offset + i * this_stride, plan, std::integral_constant<std::size_t, 2>{});
        }
      }
      else
      {
        for (idx_t i = 0; i < idx_t(this_extent); ++i)
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
  auto const& map = a.mapping();
  auto const& acc = a.accessor();
  auto data = a.data_handle();
  auto [plan, offset] = make_iteration_plan_with_offset(map);

  if (plan.empty()) return;

  detail::UnrollHelper helper{data, acc, op};
  helper.run(offset, plan.data(), plan.size() - 1);
}

} // namespace uni20
