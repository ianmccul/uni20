#pragma once

/**
 * \file contraction_strides.hpp
 * \ingroup linalg
 * \brief Stride grouping and direct-GEMM projections for tensor contraction.
 */

#include <uni20/common/static_vector.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/contraction_axes.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/mdspan/mdspec.hpp>
#include <uni20/mdspan/strides.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Ranked mdspec usable by contraction stride grouping.
/// \details Positive-rank operands must expose strides. Rank-zero operands
///          are accepted without a stride observer because they contribute no
///          physical axes to any M, N, or K group.
template <class Mdspec, std::size_t Rank>
concept RankedContractionMdspecLike =
    uni20::RankedMdspecLike<Mdspec, Rank> && (Rank == 0 || uni20::StridedMdspecLike<Mdspec>);

/// \brief Mutable ranked mdspec usable as a contraction output.
template <class Mdspec, std::size_t Rank>
concept MutableRankedContractionMdspecLike =
    uni20::MutableRankedMdspecLike<Mdspec, Rank> && (Rank == 0 || uni20::MutableStridedMdspecLike<Mdspec>);

/// \brief M, N, and K stride groups for a normalized pairwise contraction.
/// \details M dimensions pair left-input and output strides, N dimensions
///          pair right-input and output strides, and K dimensions pair the
///          two input strides. Dimensions merge only when both participating
///          mappings admit the same linearization.
template <std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank> struct ContractionStrideGroups
{
    using axes_type = ContractionAxes<LhsRank, RhsRank, ContractedRank>;

    static_vector<extent_strides<2>, axes_type::lhs_surviving_rank> m;
    static_vector<extent_strides<2>, axes_type::rhs_surviving_rank> n;
    static_vector<extent_strides<2>, axes_type::contracted_rank> k;
};

/// \brief Build jointly merged M, N, and K stride groups without acquiring data handles.
template <
    std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
    MutableRankedContractionMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
    RankedContractionMdspecLike<LhsRank> LhsMdspec, RankedContractionMdspecLike<RhsRank> RhsMdspec>
[[nodiscard]] auto make_contraction_stride_groups(OutputMdspec const& output, LhsMdspec const& lhs,
                                                  RhsMdspec const& rhs,
                                                  ContractionAxes<LhsRank, RhsRank, ContractedRank> const& axes)
    -> ContractionStrideGroups<LhsRank, RhsRank, ContractedRank>
{
  CHECK(contraction_axes_are_valid(axes));
  auto const required_extents = contraction_output_extents(lhs, rhs, axes);
  CHECK(uni20::tensor_extents_equal(output.extents(), required_extents),
        "contraction output extents do not match the surviving input axes");

  using axes_type = ContractionAxes<LhsRank, RhsRank, ContractedRank>;
  ContractionStrideGroups<LhsRank, RhsRank, ContractedRank> result;
  if constexpr (axes_type::lhs_surviving_rank > 0)
  {
    for (std::size_t index = 0; index < axes.lhs_surviving.size(); ++index)
    {
      auto const lhs_axis = axes.lhs_surviving[index];
      result.m.emplace_back(lhs.extent(lhs_axis), lhs.stride(lhs_axis), output.stride(index));
    }
  }
  if constexpr (axes_type::rhs_surviving_rank > 0)
  {
    for (std::size_t index = 0; index < axes.rhs_surviving.size(); ++index)
    {
      auto const rhs_axis = axes.rhs_surviving[index];
      auto const output_axis = axes.lhs_surviving.size() + index;
      result.n.emplace_back(rhs.extent(rhs_axis), rhs.stride(rhs_axis), output.stride(output_axis));
    }
  }
  if constexpr (ContractedRank > 0)
  {
    for (std::size_t index = 0; index < ContractedRank; ++index)
    {
      auto const lhs_axis = axes.lhs_contracted[index];
      auto const rhs_axis = axes.rhs_contracted[index];
      result.k.emplace_back(lhs.extent(lhs_axis), lhs.stride(lhs_axis), rhs.stride(rhs_axis));
    }
  }

  merge_strides_right(result.m);
  merge_strides_right(result.n);
  merge_strides_right(result.k);
  return result;
}

/// \brief Extents and strides for one rank-two projection of a contraction operand.
struct ContractionMatrixProjection
{
    std::array<uni20::index_type, 2> extents{};
    std::array<uni20::index_type, 2> strides{};
};

/// \brief Handle-independent projection of a contraction to one GEMM.
struct DirectContractionGemmPlan
{
    ContractionMatrixProjection output;
    ContractionMatrixProjection lhs;
    ContractionMatrixProjection rhs;
};

/// \brief One residual M or N loop around a rank-two GEMM projection.
/// \details Each loop iteration addresses one disjoint output slice. The
///          offset strides are measured in elements from the original mdspec
///          descriptors or immediate handles.
struct LoopedContractionGemmPlan
{
    uni20::index_type loop_extent = 0;
    uni20::index_type output_offset_stride = 0;
    uni20::index_type lhs_offset_stride = 0;
    uni20::index_type rhs_offset_stride = 0;
    DirectContractionGemmPlan gemm;
};

namespace detail::contraction_strides
{

struct GroupedDimension
{
    uni20::index_type extent = 1;
    std::array<uni20::index_type, 2> strides{1, 1};
};

template <std::size_t Capacity>
[[nodiscard]] auto grouped_dimension(static_vector<extent_strides<2>, Capacity> const& group) -> GroupedDimension
{
  if constexpr (Capacity == 0)
  {
    return {};
  }
  else
  {
    if (group.empty()) return {};
    auto result = GroupedDimension{.extent = group[0].extent, .strides = group[0].strides};
    if (result.extent <= 1) result.strides = {1, 1};
    return result;
  }
}

[[nodiscard]] inline auto grouped_dimension(extent_strides<2> const& dimension) -> GroupedDimension
{
  auto result = GroupedDimension{.extent = dimension.extent, .strides = dimension.strides};
  if (result.extent <= 1) result.strides = {1, 1};
  return result;
}

[[nodiscard]] inline bool projection_is_directly_strided(ContractionMatrixProjection const& projection) noexcept
{
  for (std::size_t axis = 0; axis < 2; ++axis)
  {
    if (projection.extents[axis] < 0 || projection.strides[axis] <= 0) return false;
  }
  return projection.strides[0] == 1 || projection.strides[1] == 1;
}

[[nodiscard]] inline auto try_make_direct_gemm_plan(GroupedDimension const& m, GroupedDimension const& n,
                                                    GroupedDimension const& k)
    -> std::optional<DirectContractionGemmPlan>
{
  DirectContractionGemmPlan plan{.output = {.extents = {m.extent, n.extent}, .strides = {m.strides[1], n.strides[1]}},
                                 .lhs = {.extents = {m.extent, k.extent}, .strides = {m.strides[0], k.strides[0]}},
                                 .rhs = {.extents = {k.extent, n.extent}, .strides = {k.strides[1], n.strides[0]}}};
  if (!projection_is_directly_strided(plan.output) || !projection_is_directly_strided(plan.lhs) ||
      !projection_is_directly_strided(plan.rhs))
    return std::nullopt;
  return plan;
}

template <class Mdspec> consteval bool can_offset_contraction_mdspec()
{
  using span_type = std::remove_cvref_t<Mdspec>;
  if constexpr (uni20::MdspanLike<span_type>)
  {
    using accessor_type = typename span_type::accessor_type;
    using offset_accessor_type = typename accessor_type::offset_policy;
    return std::same_as<typename offset_accessor_type::element_type, typename span_type::element_type> &&
           std::constructible_from<offset_accessor_type, accessor_type const&>&&
             requires(span_type & span, span_offset_t<accessor_type> offset)
    {
      {span.accessor().offset(span.data_handle(), offset)}
          ->std::convertible_to<typename offset_accessor_type::data_handle_type>;
    };
  }
  else
  {
    using descriptor_type = typename span_type::data_descriptor_type;
    return requires(descriptor_type const& descriptor, std::size_t offset) {
      { descriptor.offset_by(offset) } -> std::same_as<descriptor_type>;
    };
  }
}

} // namespace detail::contraction_strides

/// \brief Try to collapse merged contraction groups to three rank-two GEMM operands.
/// \return A direct plan when every group has at most one dimension and each
///         projected matrix has a unit-stride axis; otherwise `std::nullopt`.
template <
    std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
    MutableRankedContractionMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
    RankedContractionMdspecLike<LhsRank> LhsMdspec, RankedContractionMdspecLike<RhsRank> RhsMdspec>
[[nodiscard]] auto try_make_direct_contraction_gemm_plan(OutputMdspec const& output, LhsMdspec const& lhs,
                                                         RhsMdspec const& rhs,
                                                         ContractionAxes<LhsRank, RhsRank, ContractedRank> const& axes)
    -> std::optional<DirectContractionGemmPlan>
{
  if (!output.mapping().is_unique() || !lhs.mapping().is_unique() || !rhs.mapping().is_unique()) return std::nullopt;

  auto groups = make_contraction_stride_groups(output, lhs, rhs, axes);
  if (groups.m.size() > 1 || groups.n.size() > 1 || groups.k.size() > 1) return std::nullopt;

  auto const m = detail::contraction_strides::grouped_dimension(groups.m);
  auto const n = detail::contraction_strides::grouped_dimension(groups.n);
  auto const k = detail::contraction_strides::grouped_dimension(groups.k);

  return detail::contraction_strides::try_make_direct_gemm_plan(m, n, k);
}

/// \brief Try to leave one residual M or N dimension around a GEMM projection.
/// \details The first implementation deliberately leaves K unlooped so every
///          GEMM writes a disjoint output slice and uses the original `beta`.
///          Exactly one of M or N must contain two merged stride descriptors;
///          the other groups must each contain at most one. When either
///          descriptor can serve as the loop, the plan with fewer GEMM calls
///          is selected.
/// \return A looped plan for one residual M or N dimension, or `std::nullopt`.
template <
    std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank,
    MutableRankedContractionMdspecLike<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> OutputMdspec,
    RankedContractionMdspecLike<LhsRank> LhsMdspec, RankedContractionMdspecLike<RhsRank> RhsMdspec>
[[nodiscard]] auto try_make_looped_contraction_gemm_plan(OutputMdspec const& output, LhsMdspec const& lhs,
                                                         RhsMdspec const& rhs,
                                                         ContractionAxes<LhsRank, RhsRank, ContractedRank> const& axes)
    -> std::optional<LoopedContractionGemmPlan>
{
  if (!output.mapping().is_unique() || !lhs.mapping().is_unique() || !rhs.mapping().is_unique()) return std::nullopt;

  auto groups = make_contraction_stride_groups(output, lhs, rhs, axes);
  if (groups.k.size() > 1) return std::nullopt;

  bool const loop_m = groups.m.size() == 2 && groups.n.size() <= 1;
  bool const loop_n = groups.n.size() == 2 && groups.m.size() <= 1;
  if (loop_m == loop_n) return std::nullopt;

  auto const k = detail::contraction_strides::grouped_dimension(groups.k);
  std::optional<LoopedContractionGemmPlan> result;
  for (std::size_t loop_index = 0; loop_index < 2; ++loop_index)
  {
    std::size_t const matrix_index = 1 - loop_index;
    auto const& loop = loop_m ? groups.m[loop_index] : groups.n[loop_index];
    if (loop.extent < 0 || loop.strides[0] < 0 || loop.strides[1] < 0) continue;

    auto const m = loop_m ? detail::contraction_strides::grouped_dimension(groups.m[matrix_index])
                          : detail::contraction_strides::grouped_dimension(groups.m);
    auto const n = loop_n ? detail::contraction_strides::grouped_dimension(groups.n[matrix_index])
                          : detail::contraction_strides::grouped_dimension(groups.n);
    auto gemm = detail::contraction_strides::try_make_direct_gemm_plan(m, n, k);
    if (!gemm) continue;

    LoopedContractionGemmPlan candidate{
        .loop_extent = loop.extent,
        .output_offset_stride = loop.strides[1],
        .lhs_offset_stride = loop_m ? loop.strides[0] : 0,
        .rhs_offset_stride = loop_n ? loop.strides[0] : 0,
        .gemm = *gemm,
    };
    if (!result || candidate.loop_extent < result->loop_extent) result = std::move(candidate);
  }
  return result;
}

/// \brief Rank-two `layout_stride` mdspan type retaining a resolved operand accessor.
template <uni20::MdspanLike Mdspan>
using contraction_matrix_mdspan_t =
    stdex::mdspan<typename std::remove_cvref_t<Mdspan>::element_type,
                  stdex::dextents<typename std::remove_cvref_t<Mdspan>::index_type, 2>, stdex::layout_stride,
                  typename std::remove_cvref_t<Mdspan>::accessor_type>;

namespace detail::contraction_strides
{

template <class Mdspec, bool = uni20::MdspanLike<Mdspec>> struct ContractionMatrixMdspec;

template <class Mdspec> struct ContractionMatrixMdspec<Mdspec, true>
{
    using type = contraction_matrix_mdspan_t<Mdspec>;
};

template <class Mdspec> struct ContractionMatrixMdspec<Mdspec, false>
{
    using span_type = std::remove_cvref_t<Mdspec>;
    using type = uni20::mdspec<typename span_type::element_type, stdex::dextents<typename span_type::index_type, 2>,
                               stdex::layout_stride, typename span_type::accessor_type,
                               typename span_type::data_descriptor_type>;
};

template <class Mdspec, bool = uni20::MdspanLike<Mdspec>> struct OffsetContractionMatrixMdspec;

template <class Mdspec> struct OffsetContractionMatrixMdspec<Mdspec, true>
{
    using span_type = std::remove_cvref_t<Mdspec>;
    using accessor_type = typename span_type::accessor_type::offset_policy;
    using type = stdex::mdspan<typename accessor_type::element_type, stdex::dextents<typename span_type::index_type, 2>,
                               stdex::layout_stride, accessor_type>;
};

template <class Mdspec> struct OffsetContractionMatrixMdspec<Mdspec, false>
{
    using type = typename ContractionMatrixMdspec<Mdspec, false>::type;
};

} // namespace detail::contraction_strides

/// \brief Rank-two strided metadata type retaining an operand descriptor or immediate handle.
template <uni20::MdspecLike Mdspec>
using contraction_matrix_mdspec_t =
    typename detail::contraction_strides::ContractionMatrixMdspec<std::remove_cvref_t<Mdspec>>::type;

/// \brief Rank-two strided metadata type whose storage origin may be advanced.
template <uni20::MdspecLike Mdspec>
using offset_contraction_matrix_mdspec_t =
    typename detail::contraction_strides::OffsetContractionMatrixMdspec<std::remove_cvref_t<Mdspec>>::type;

/// \brief Project a resolved mdspan into one rank-two GEMM operand without changing storage.
template <uni20::MdspanLike Mdspan>
[[nodiscard]] auto make_contraction_matrix_mdspan(Mdspan& span, ContractionMatrixProjection const& projection)
    -> contraction_matrix_mdspan_t<Mdspan>
{
  using result_type = contraction_matrix_mdspan_t<Mdspan>;
  using extents_type = typename result_type::extents_type;
  using mapping_type = typename result_type::mapping_type;
  auto const extents = extents_type{projection.extents[0], projection.extents[1]};
  auto const mapping = mapping_type{extents, projection.strides};
  return result_type{span.data_handle(), mapping, span.accessor()};
}

/// \brief Project an mdspec into one rank-two GEMM operand without acquiring its handle.
/// \details Immediate mdspans remain immediate. Descriptor-backed inputs retain
///          a copied data descriptor, mapping, and accessor for acquisition by
///          the selected GEMM backend.
template <uni20::MdspecLike Mdspec>
[[nodiscard]] auto make_contraction_matrix_mdspec(Mdspec& span, ContractionMatrixProjection const& projection)
    -> contraction_matrix_mdspec_t<Mdspec>
{
  using result_type = contraction_matrix_mdspec_t<Mdspec>;
  using extents_type = typename result_type::extents_type;
  using mapping_type = typename result_type::mapping_type;
  auto const extents = extents_type{projection.extents[0], projection.extents[1]};
  auto const mapping = mapping_type{extents, projection.strides};
  if constexpr (uni20::MdspanLike<Mdspec>)
    return result_type{span.data_handle(), mapping, span.accessor()};
  else
    return result_type{span.data_descriptor(), mapping, span.accessor()};
}

/// \brief Project a rank-two mdspec slice at one nonnegative element offset.
/// \details Immediate storage advances its handle through the accessor and
///          uses the accessor's offset policy. Deferred storage advances the
///          copied data descriptor through `offset_by`, preserving acquisition
///          and resource identity for the selected GEMM backend.
/// \pre `element_offset` is nonnegative and identifies a valid element origin
///      within the source descriptor.
template <uni20::MdspecLike Mdspec>
  requires(detail::contraction_strides::can_offset_contraction_mdspec<Mdspec>())
[[nodiscard]] auto make_offset_contraction_matrix_mdspec(Mdspec& span, ContractionMatrixProjection const& projection,
                                                         uni20::index_type element_offset)
    -> offset_contraction_matrix_mdspec_t<Mdspec>
{
  CHECK(element_offset >= 0);
  using result_type = offset_contraction_matrix_mdspec_t<Mdspec>;
  using extents_type = typename result_type::extents_type;
  using mapping_type = typename result_type::mapping_type;
  auto const extents = extents_type{projection.extents[0], projection.extents[1]};
  auto const mapping = mapping_type{extents, projection.strides};
  if constexpr (uni20::MdspanLike<Mdspec>)
  {
    using source_accessor_type = typename std::remove_cvref_t<Mdspec>::accessor_type;
    using offset_accessor_type = typename source_accessor_type::offset_policy;
    auto const offset = static_cast<span_offset_t<source_accessor_type>>(element_offset);
    return result_type{span.accessor().offset(span.data_handle(), offset), mapping,
                       offset_accessor_type{span.accessor()}};
  }
  else
  {
    auto descriptor = span.data_descriptor().offset_by(static_cast<std::size_t>(element_offset));
    return result_type{std::move(descriptor), mapping, span.accessor()};
  }
}

} // namespace uni20::linalg
