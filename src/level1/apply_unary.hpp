#pragma once

#include "common/mdspan.hpp"
#include "mdspan/iteration_plan.hpp"

namespace uni20
{

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

