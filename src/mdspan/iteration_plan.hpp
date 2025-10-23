#pragma once

/**
 * \file iteration_plan.hpp
 * \ingroup mdspan_ext
 * \brief Helpers for constructing merged iteration plans over strided tensors.
 */

#include "common/static_vector.hpp"
#include "mdspan/concepts.hpp"
#include "mdspan/strides.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <tuple>
#include <type_traits>
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
    constexpr bool can_merge_with_inner(extent_stride inner) const noexcept
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

/// \brief Alias that mirrors the legacy multi-extent stride descriptor.
/// \tparam ExtentT Unused extent parameter retained for backwards compatibility.
/// \tparam StrideT Unused stride parameter retained for backwards compatibility.
/// \tparam N       Number of tensors tracked by the descriptor.
/// \ingroup internal
template <typename ExtentT = std::size_t, typename StrideT = std::ptrdiff_t, std::size_t N = 2>
using multi_extent_stride = extent_strides<N>;

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
    index_type stride = mapping.stride(i);

    if (stride < 0)
    {
      offset += stride * static_cast<index_type>(extent - 1);
      stride = -stride;
    }

    if (extent == 0) continue;

    raw_plan.emplace_back(extent, std::array<index_type, 1>{stride});
  }

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
/// \tparam Mapping Layout mapping type modelling the mdspan mapping interface.
/// \tparam N       Number of tensors participating in the iteration.
/// \param mappings Array of mappings, one per tensor.
/// \return Pair of the merged plan and the per-tensor offset corrections.
/// \ingroup internal
template <typename Mapping, std::size_t N>
auto make_multi_iteration_plan_with_offset(std::array<Mapping, N> const& mappings)
{
  using size_type = typename Mapping::size_type;
  using index_type = typename Mapping::index_type;
  static constexpr size_type Rank = Mapping::extents_type::rank();

  static_assert(N >= 1, "At least one mapping is required");

  auto const& base_extents = mappings[0].extents();

  for (std::size_t k = 1; k < N; ++k)
  {
    for (size_type i = 0; i < Rank; ++i)
    {
      assert(mappings[k].extents().extent(i) == base_extents.extent(i));
    }
  }

  static_vector<extent_strides<N>, Rank> raw_plan;
  std::array<index_type, N> offsets{};

  for (size_type i = 0; i < Rank; ++i)
  {
    size_type const extent = base_extents.extent(i);
    if (extent == 0) continue;

    std::array<index_type, N> strides{};
    for (std::size_t k = 0; k < N; ++k)
    {
      strides[k] = mappings[k].stride(i);
    }

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

  if (raw_plan.empty()) return std::pair{raw_plan, offsets};

  std::sort(raw_plan.begin(), raw_plan.end(),
            [](auto const& lhs, auto const& rhs) { return std::abs(lhs.strides[0]) > std::abs(rhs.strides[0]); });

  merge_strides_right(raw_plan);

  return std::pair{raw_plan, offsets};
}

namespace detail
{

/// \brief Helper that executes nested loops according to a single-span iteration plan.
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

    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::integral_constant<std::size_t, 0>)
    {
      auto const this_extent = plan->extent;
      auto const this_stride = plan->stride;
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
      auto const this_extent = plan->extent;
      auto const this_stride = plan->stride;
      ++plan;
      for (idx_t i = 0; i < idx_t(this_extent); ++i)
      {
        this->run(Offset + i * this_stride, plan, std::integral_constant<std::size_t, N - 1>{});
      }
    }

    void run_dynamic(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::size_t depth)
    {
      auto const this_extent = plan->extent;
      auto const this_stride = plan->stride;
      ++plan;
      if (depth == 3)
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

/// \brief Helper to unroll nested loops for multiple tensors of identical extent.
/// \tparam Op        Callable taking N element values and returning the result.
/// \tparam Spans...  StridedMdspan types that share identical extents and rank.
/// \ingroup internal
template <typename Op, StridedMdspan... Spans> struct MultiUnrollHelper
{
    std::tuple<typename Spans::data_handle_type...> dh_;
    std::tuple<typename Spans::accessor_type...> accs_;

    using op_type = std::decay_t<Op>;
    op_type op_;

    static constexpr std::size_t num_spans = sizeof...(Spans);
    using index_type = std::ptrdiff_t;

    using offset_type = std::array<index_type, num_spans>;

    template <typename Span>
    using raw_arg_t = decltype(std::declval<typename Span::accessor_type>().access(
        std::declval<typename Span::data_handle_type>(), index_type()));

    template <typename FwdOp>
    MultiUnrollHelper(FwdOp&& op, Spans const&... spans)
        : dh_{spans.data_handle()...}, accs_{spans.accessor()...}, op_(std::forward<FwdOp>(op))
    {}

    template <typename Plan> void run(Plan const& plan, offset_type offsets) noexcept
    {
      std::size_t depth = plan.size() - 1;
      switch (depth)
      {
        case 0:
          run_unrolled(offsets, plan.data(), std::integral_constant<std::size_t, 0>{});
          return;
        case 1:
          run_unrolled(offsets, plan.data(), std::integral_constant<std::size_t, 1>{});
          return;
        case 2:
          run_unrolled(offsets, plan.data(), std::integral_constant<std::size_t, 2>{});
          return;
        default:
          this->run_dynamic(offsets, plan.data(), depth);
          return;
      }
    }

  private:
    static constexpr std::size_t MaxUnrollDepth = 3;

    void run_dynamic(offset_type offsets, const extent_strides<num_spans>* plan, std::size_t depth) noexcept
    {
      index_type const N = plan->extent;
      for (index_type i = 0; i < N; ++i)
      {
        if (depth == MaxUnrollDepth)
        {
          this->run_unrolled(offsets, plan + 1, std::integral_constant<std::size_t, MaxUnrollDepth - 1>{});
        }
        else
        {
          this->run_dynamic(offsets, plan + 1, depth - 1);
        }

        for (std::size_t s = 0; s < num_spans; ++s)
        {
          offsets[s] += plan->strides[s];
        }
      }
    }

    void run_unrolled(offset_type offsets, const extent_strides<num_spans>* plan,
                      std::integral_constant<std::size_t, 0>) noexcept
    {
      index_type const N = plan->extent;
      for (index_type i = 0; i < N; ++i)
      {
        [&]<std::size_t... I>(std::index_sequence<I...>)
        {
          std::get<0>(accs_).access(std::get<0>(dh_), offsets[0]) =
              op_(std::get<I>(accs_).access(std::get<I>(dh_), offsets[I])...);
        }
        (std::make_index_sequence<num_spans>{});

        for (std::size_t s = 0; s < num_spans; ++s)
        {
          offsets[s] += plan->strides[s];
        }
      }
    }

    template <std::size_t D>
    void run_unrolled(offset_type offsets, const extent_strides<num_spans>* plan,
                      std::integral_constant<std::size_t, D>) noexcept
    {
      index_type const N = plan->extent;
      for (index_type i = 0; i < N; ++i)
      {
        run_unrolled(offsets, plan + 1, std::integral_constant<std::size_t, D - 1>{});

        for (std::size_t s = 0; s < num_spans; ++s)
        {
          offsets[s] += plan->strides[s];
        }
      }
    }

    template <std::size_t I> auto& dh_get() noexcept { return std::get<I>(dh_); }
    template <std::size_t I> auto& accs_get() noexcept { return std::get<I>(accs_); }
};

template <typename Op, typename... Spans> MultiUnrollHelper(Op&&, Spans...) -> MultiUnrollHelper<Op, Spans...>;

} // namespace detail

} // namespace uni20
