#pragma once

#include <uni20/common/mdspan.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/iteration_plan.hpp>

/**
 * \defgroup level1_ops Level-1 tensor algorithms
 * \brief Element-wise tensor kernels that operate on strided mdspan views.
 */

namespace uni20
{

/// \brief Copy elements from a source mdspan into a destination mdspan.
/// \tparam MDSDst Destination mdspan type that models StridedMdspan.
/// \tparam MDSSrc Source mdspan type that models StridedMdspan.
/// \param dst Destination view receiving the copied elements.
/// \param src Source view providing the element values.
/// \ingroup level1_ops
template <StridedMdspan MDSDst, StridedMdspan MDSSrc> void assign(MDSDst dst, MDSSrc const& src)
{
  static_assert(MDSDst::rank() == MDSSrc::rank(), "assign: rank mismatch");
  PRECONDITION_EQUAL(src.extents(), dst.extents(), "assign: shape mismatch");

  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::array{dst.mapping(), src.mapping()});

  // No empty-plan short-circuit: an empty plan denotes a rank-0 scalar (assign
  // the single element), which MultiUnrollHelper::run handles via its 0-dim
  // terminal. A zero-size iteration is carried as a retained extent-0 dim.
  detail::MultiUnrollHelper helper{[](auto&& dst_v, auto&& src_v) { return src_v; }, dst, src};
  helper.run(plan, offsets);
}

} // namespace uni20
