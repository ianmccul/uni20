#pragma once

#include "common/mdspan.hpp"
#include <cassert>
#include <concepts>
#include <fmt/format.h>

namespace uni20
{

template <class MDS>
concept StridedMdspan = std::same_as<typename MDS::layout_type, stdex::layout_stride>;

} // namespace uni20

namespace fmt
{

template <typename IndexType, std::size_t... StaticExts, typename CharT>
struct formatter<stdex::extents<IndexType, StaticExts...>, CharT>
{
    // No format‐specs, just use the default
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(stdex::extents<IndexType, StaticExts...> const& ex, FormatContext& ctx) const
    {
      // write: '[' n0 ',' n1 ',' … ']'
      auto out = ctx.out();
      *out++ = '[';
      constexpr std::size_t R = sizeof...(StaticExts);
      for (std::size_t d = 0; d < R; ++d)
      {
        if (d) *out++ = ',';
        out = fmt::format_to(out, "{}", ex.extent(d));
      }
      *out++ = ']';
      return out;
    }
};

} // namespace fmt
