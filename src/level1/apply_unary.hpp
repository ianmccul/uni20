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
/// \details Used for creating iteration plans for multidimensional memory traversal.
/// The extent is the size along that dimension, and the stride is the number of
/// memory units to jump to reach the next element in that dimension.
/// \tparam ExtentT Integer type for extents (usually unsigned).
/// \tparam StrideT Integer type for memory strides (usually signed).
/// \ingroup internal
template <typename ExtentT = std::size_t, typename StrideT = std::ptrdiff_t> struct extent_stride
{
    ExtentT extent;
    StrideT stride;

    constexpr extent_stride() = default;
    constexpr extent_stride(ExtentT extent, StrideT stride) : extent(extent), stride(stride) {}

    /// \brief Check if the current outer dimension and the inner dimension can be merged.
    /// \details Coalescing is allowed when the inner stride equals the outer stride multiplied by the outer extent.
    /// This ensures a contiguous memory layout between the two.
    /// \param inner Candidate inner dimension.
    /// \return True when the contiguity requirement holds.
    /// \ingroup internal
    constexpr bool can_merge_with_inner(extent_stride inner) const noexcept
    {
      return stride == inner.stride * static_cast<StrideT>(inner.extent);
    }

    /// \brief Merge an inner dimension into this one.
    /// \details After merging, the extent becomes the product and the stride is updated to the inner stride.
    /// \param inner Dimension that satisfies can_merge_with_inner.
    /// \ingroup internal
    constexpr void merge_with_inner(extent_stride inner) noexcept
    {
      extent *= inner.extent;
      stride = inner.stride;
    }
};

/// \brief Create a coalesced iteration plan for looping over a strided layout.
/// \details Negative strides are flipped to positive by adjusting the offset, zero-extent
/// dimensions are dropped, and contiguous dimensions are coalesced.
/// \tparam Mapping Layout mapping type (for example, layout_stride::mapping<Extents>).
/// \param mapping Layout mapping to analyze.
/// \return Pair of the compact loop plan and the base offset from the data pointer.
/// \ingroup internal
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

/// \brief Trait detecting whether an mdspan-like accessor exposes offset().
/// \tparam T Accessor policy type to probe.
/// \ingroup internal
template <typename T> constexpr bool HasOffset = requires(T const& t) { t.offset(); };

namespace detail
{

/// \brief Helper that executes nested loops according to an iteration plan.
/// \tparam DataHandle Data handle type from the mdspan.
/// \tparam Accessor   Accessor policy associated with the mdspan.
/// \tparam Op         Unary callable applied to each element.
/// \ingroup internal
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

    /// \brief Dispatch to the unrolled or dynamic traversal depending on depth.
    /// \param Offset Current offset in the data handle.
    /// \param plan   Pointer to the first dimension in the iteration plan.
    /// \param depth  Remaining depth to traverse.
    /// \ingroup internal
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

/// \brief Apply a unary operation to every element of a strided mdspan in-place.
/// \tparam MDS Mdspan-like type that models StridedMdspan.
/// \tparam Op  Unary callable applied to each element.
/// \param a    Target span whose elements are mutated.
/// \param op   Unary functor invoked for each element.
/// \ingroup level1_ops
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
