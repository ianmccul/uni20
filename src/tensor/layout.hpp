#pragma once

#include "common/mdspan.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20::layout
{

template <typename LayoutPolicy, typename Extents>
using mapping_for_t = typename LayoutPolicy::template mapping<Extents>;

template <typename Builder, typename LayoutPolicy, typename Extents>
concept mapping_builder_for = requires(Builder&& builder, Extents const& exts) {
                                {
                                  std::forward<Builder>(builder)(exts)
                                  } -> std::same_as<mapping_for_t<LayoutPolicy, Extents>>;
                              };
namespace detail
{

template <typename Extents>
constexpr auto layout_right_strides(Extents const& exts) -> std::array<typename Extents::index_type, Extents::rank()>
{
  using index_type = typename Extents::index_type;
  std::array<index_type, Extents::rank()> strides{};
  index_type run = 1;
  for (int d = static_cast<int>(Extents::rank()) - 1; d >= 0; --d)
  {
    strides[d] = run;
    run *= exts.extent(static_cast<std::size_t>(d));
  }
  return strides;
}

template <typename Extents>
constexpr auto layout_left_strides(Extents const& exts) -> std::array<typename Extents::index_type, Extents::rank()>
{
  using index_type = typename Extents::index_type;
  std::array<index_type, Extents::rank()> strides{};
  index_type run = 1;
  for (std::size_t d = 0; d < Extents::rank(); ++d)
  {
    strides[d] = run;
    run *= exts.extent(d);
  }
  return strides;
}

} // namespace detail

struct LayoutRight
{
    template <typename Extents>
    constexpr auto operator()(Extents const& exts) const -> mapping_for_t<stdex::layout_stride, Extents>
    {
      return {exts, detail::layout_right_strides(exts)};
    }
};

struct LayoutLeft
{
    template <typename Extents>
    constexpr auto operator()(Extents const& exts) const -> mapping_for_t<stdex::layout_stride, Extents>
    {
      return {exts, detail::layout_left_strides(exts)};
    }
};

} // namespace uni20::layout
