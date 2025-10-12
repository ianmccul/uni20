#pragma once

// use reference implementation
#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
#error "cmake configuration error: mdspan namespace is not defined"
#endif

#include <fmt/format.h>
#include <format>
#include <mdspan/mdspan.hpp>

template <class IndexType, size_t... Extents> struct std::formatter<stdex::extents<IndexType, Extents...>>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const stdex::extents<IndexType, Extents...>& ex, FormatContext& ctx) const
    {
      fmt::format_to(ctx.out(), "extents(");
      for (size_t r = 0; r < ex.rank(); ++r)
      {
        if (r) fmt::format_to(ctx.out(), ",");
        fmt::format_to(ctx.out(), "{}", ex.extent(r));
      }
      return fmt::format_to(ctx.out(), ")");
    }
};
