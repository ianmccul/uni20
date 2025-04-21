#pragma once

#include <mdspan>
#include <utility>

namespace uni20
{

template <class Op, class Accessor> struct UnaryTransformAccessor
{
    using data_handle_type = typename Accessor::data_handle_type;
    using element_type = std::invoke_result_t<Op, typename Accessor::element_type>;
    using reference = element_type;
    using offset_policy = UnaryTransformAccessor<Op, typename Accessor::offset_policy>;

    UnaryTransformAccessor() = default;
    UnaryTransformAccessor(Op op, Accessor acc) : op_(std::move(op)), acc_(std::move(acc)) {}

    constexpr reference access(data_handle_type p, std::ptrdiff_t i) const { return op_(acc_.access(p, i)); }

    constexpr auto offset(data_handle_type p, std::ptrdiff_t i) const
    {
      return UnaryTransformAccessor(op_, acc_.offset(p, i));
    }

  private:
    Op op_;
    Accessor acc_;
};

template <class MDS, class Op> auto transform_view(MDS const& mds, Op&& op)
{
  auto mapping = mds.mapping();
  auto acc = UnaryTransformAccessor<Op, typename MDS::accessor_type>(std::forward<Op>(op), mds.accessor());
  return std::mdspan<typename UnaryTransformAccessor<Op, typename MDS::accessor_type>::element_type,
                     typename MDS::extents_type, typename MDS::layout_type,
                     UnaryTransformAccessor<Op, typename MDS::accessor_type>>(mds.data_handle(), mapping, acc);
}

} // namespace uni20
