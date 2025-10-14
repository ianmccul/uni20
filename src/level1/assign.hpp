#pragma once

#include "common/mdspan.hpp"
#include "common/trace.hpp"
#include "mdspan/concepts.hpp"
#include "mdspan/iteration_plan.hpp"

/**
 * \defgroup level1_ops Level-1 tensor algorithms
 * \brief Element-wise tensor kernels that operate on strided mdspan views.
 */

namespace uni20
{

/// \brief Copy elements from a source mdspan into a destination mdspan.
/// \tparam MDS1 Source mdspan type that models StridedMdspan.
/// \tparam MDS2 Destination mdspan type that models StridedMdspan.
/// \param src Source view providing the element values.
/// \param dst Destination view receiving the copied elements.
/// \ingroup level1_ops
template <StridedMdspan MDS1, StridedMdspan MDS2> void assign(MDS1 const& src, MDS2 dst)
{
  static_assert(MDS1::rank() == MDS2::rank(), "assign: rank mismatch");
  PRECONDITION_EQUAL(src.extents(), dst.extents(), "assign: shape mismatch");

  auto [plan, offsets] = make_multi_iteration_plan_with_offset(std::array{dst.mapping(), src.mapping()});

  if (plan.empty()) return;

  detail::MultiUnrollHelper helper{[](auto&& dst_v, auto&& src_v) { return src_v; }, dst, src};
  helper.run(plan, offsets);
}

} // namespace uni20

