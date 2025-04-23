#pragma once

#include "common/mdspan.hpp"
#include <cassert>
#include <concepts>
#include <fmt/format.h>

namespace uni20
{

template <class MDS>
concept StridedMdspan = std::same_as<typename MDS::layout_type, stdex::layout_stride>;

/// \brief Trait to pull an AccessorPolicy’s offset_type if present,
///        or fall back to std::size_t otherwise.
/// \tparam AP  The accessor policy to inspect.
/// Note: this is an extension to the standard mdspan AccessorPolicy
/// https://en.cppreference.com/w/cpp/named_req/AccessorPolicy
/// that doesn't name offset_type but simply uses std::size_t
template <typename AP, typename = void> struct span_offset_type
{
    using type = std::size_t;
};

template <typename AP> struct span_offset_type<AP, std::void_t<typename AP::offset_type>>
{
    using type = typename AP::offset_type;
};

/// \brief Convenience alias for accessor_offset_type<AP>::type.
/// \tparam AP  The accessor policy to inspect.
template <typename AP> using span_offset_t = typename span_offset_type<AP>::type;

} // namespace uni20

namespace fmt
{

template <typename IndexType, std::size_t... StaticExts, typename CharT>
struct formatter<stdex::extents<IndexType, StaticExts...>, CharT>
{
    // No format‐specs, just use the default
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    constexpr auto format(stdex::extents<IndexType, StaticExts...> const& ex, FormatContext& ctx) const
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
