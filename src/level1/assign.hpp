#pragma once

#include "apply_unary.hpp" // your loop unrolling helper, like apply_unary_inplace
#include "common/mdspan.hpp"
#include "level1/concepts.hpp"
#include <cassert>

namespace uni20
{

namespace detail
{

using idx_t = uni20::index_type;

template <typename MDS1, typename MDS2> struct MultiUnrollHelper
{
    using src_data_type = typename MDS1::data_type;
    using src_acc_type = typename MDS1::accessor_type;
    using dst_data_type = typename MDS2::data_type;
    using dst_acc_type = typename MDS2::accessor_type;

    src_data_type src_data_;
    src_acc_type src_acc_;
    dst_data_type dst_data_;
    dst_acc_type dst_acc_;

    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::integral_constant<std::size_t, 0>)
    {
      auto Extent = plan->extent;
      auto StrideSrc = plan->strides[0];
      auto StrideDst = plan->strides[1];
      for (idx_t i = 0; i < Extent; ++i)
      {
        acc_dst_.access(dst_, Offset + i * StrideDst) = acc_src_.access(src_, Offset + i * StrideSrc);
      }
    }

    template <std::size_t N>
    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::integral_constant<std::size_t, N>)
    {
      auto Extent = plan->extent;
      auto StrideSrc = plan->strides[0];
      auto StrideDst = plan->strides[1];
      ++plan;
      for (idx_t i = 0; i < Extent; ++i)
      {
        run(Offset + i * StrideDst, plan, std::integral_constant<std::size_t{N - 1}>());
      }
    }

    void run(idx_t Offset, extent_stride<std::size_t, idx_t> const* plan, std::size_t depth)
    {
      switch (depth)
      {
        case 0:
          run(Offset, plan, std::integral_constant<std::size_t, 0>{});
          break;
        case 1:
          run(Offset, plan, std::integral_constant<std::size_t, 1>{});
          break;
        case 2:
          run(Offset, plan, std::integral_constant<std::size_t, 2>{});
          break;
        default:
          // generic fallback or dynamic loop
          assert(false && "assign: unhandled depth > 2");
          break;
      }
    }
};

template <StridedMdspan MDS1, StridedMdspan MDS2> void assign(MDS1 const& src, MDS2 dst)
{
  static_assert(MDS1::rank() == MDS2::rank(), "assign: rank mismatch");
  assert(src.extents() == dst.extents() && "assign: shape mismatch");

  using idx_t = typename MDS1::index_type;
  auto [plan, offset] = make_multi_iteration_plan_with_offset(src.mapping(), dst.mapping());

  auto src_data = src.data_handle();
  auto dst_data = dst.data_handle();

  auto const& acc_src = src.accessor();
  auto const& acc_dst = dst.accessor();

  if (plan.empty()) return;

  MultiUnrollHelper helper{src_data, dst_data, acc_src, acc_dst};
  helper.run(offset, plan.data(), plan.size() - 1);
}

} // namespace detail

} // namespace uni20
