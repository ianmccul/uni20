#pragma once

#include "apply_unary.hpp"
#include "common/mdspan.hpp"
#include "common/trace.hpp"
#include "core/types.hpp"
#include "mdspan/concepts.hpp"

namespace uni20
{

/// \brief Represents a multi-tensor stride plan for one dimension.
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

/// \brief Represents a multi-tensor strided iteration plan for a fixed-rank layout.
///
/// This variant handles multiple tensors with the same extents but independent strides.
/// It coalesces contiguous dimensions across all tensors and flips stride signs so
/// that the first tensor (typically the output) has positive strides.
///
/// \tparam Mapping A layout mapping type (e.g. layout_stride::mapping<Extents>)
/// \tparam N       Number of tensors (e.g., 2 for A = B + C)
/// \param mappings Array of mappings, one per tensor. All must have the same extents.
/// \return A pair of:
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

/// \brief Helper to unroll nested loops over a fixed‐rank N‐span view.
/// \tparam Op        Callable taking N element‐values and returning the result.
/// \tparam Spans...  StridedMdspan types (all share the same extents/rank).
template <typename Op, StridedMdspan... Spans> struct MultiUnrollHelper
{
    std::tuple<typename Spans::data_handle_type...> dh_;
    std::tuple<typename Spans::accessor_type...> accs_;

    // — store a decayed copy of the functor
    using op_type = std::decay_t<Op>;
    op_type op_;

    static constexpr std::size_t num_spans = sizeof...(Spans);
    using index_type = std::ptrdiff_t;

    using offset_type = std::array<index_type, num_spans>;

    // For each Span, what type do we get back from accessor.access(handle, offset)?
    template <typename Span>
    using raw_arg_t = decltype(std::declval<typename Span::accessor_type>().access(
        std::declval<typename Span::data_handle_type>(), index_type()));

    using op_result_type = std::invoke_result_t<Op, raw_arg_t<Spans>...>;

    /// \brief Build from each span’s data_handle and accessor.
    template <typename FwdOp>
    MultiUnrollHelper(FwdOp&& op, Spans const&... spans)
        : dh_{spans.data_handle()...}, accs_{spans.accessor()...}, op_(std::forward<FwdOp>(op))
    {}

    //======================================================================
    //  Compile‐time unrolled depths

    //======================================================================
    //  Dynamic‐depth dispatcher (for ranks > MaxUnrollDepth)

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

    void run_dynamic(offset_type offsets, const multi_extent_stride<std::size_t, index_type>* plan,
                     std::size_t depth) noexcept
    {
      // fully dynamic: peel one dimension and recurse
      index_type N = plan->extent;
      for (index_type i = 0; i < N; ++i)
      {
        if (depth == MaxUnrollDepth)
          this->run_unrolled(offsets, plan + 1, std::integral_constant<std::size_t, MaxUnrollDepth - 1>{});
        else
          this->run_dynamic(offsets, plan + 1, depth - 1);

        for (std::size_t s = 0; s < num_spans; ++s)
          offsets[s] += plan->strides[s];
      }
    }

    // Depth == 0 (innermost loop), handle single‐dim unroll
    void run_unrolled(offset_type offsets, const multi_extent_stride<std::size_t, index_type>* plan,
                      std::integral_constant<std::size_t, 0>) noexcept
    {
      index_type N = plan->extent;
      for (index_type i = 0; i < N; ++i)
      {
        [&]<std::size_t... I>(std::index_sequence<I...>)
        {
          std::get<0>(accs_).access(std::get<0>(dh_), offsets[0]) =
              op_(std::get<I>(accs_).access(std::get<I>(dh_), offsets[I])...);
        }
        (std::make_index_sequence<num_spans>{});

        for (std::size_t s = 0; s < num_spans; ++s)
          offsets[s] += plan->strides[s];
      }
    }

    // Depth == D>0: peel off one loop and recurse
    template <std::size_t D>
    void run_unrolled(offset_type offsets, const multi_extent_stride<std::size_t, index_type>* plan,
                      std::integral_constant<std::size_t, D>) noexcept
    {
      index_type N = plan->extent;
      for (index_type i = 0; i < N; ++i)
      {
        run_unrolled(offsets, plan + 1, std::integral_constant<std::size_t, D - 1>{});

        for (std::size_t s = 0; s < num_spans; ++s)
          offsets[s] += plan->strides[s];
      }
    }
};

template <typename Op, typename... Spans> MultiUnrollHelper(Op&&, Spans...) -> MultiUnrollHelper<Op, Spans...>;

template <StridedMdspan MDS1, StridedMdspan MDS2> void assign(MDS1 const& src, MDS2 dst)
{
  static_assert(MDS1::rank() == MDS2::rank(), "assign: rank mismatch");
  PRECONDITION_EQUAL(src.extents(), dst.extents(), "assign: shape mismatch");

  auto [plan, offset] = make_multi_iteration_plan_with_offset(std::array{dst.mapping(), src.mapping()});

  if (plan.empty()) return;

  MultiUnrollHelper helper{[](auto&& dst, auto&& src) { return src; }, dst, src};
  helper.run(plan, offset);
}

} // namespace uni20
