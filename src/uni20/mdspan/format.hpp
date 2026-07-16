#pragma once

/**
 * \file format.hpp
 * \ingroup mdspan_ext
 * \brief Formatting support for mdspan extents.
 */

#include <uni20/mdspan/mdspan.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <format>

/// \brief Formatter specialization that prints extents in a human-readable form.
template <class IndexType, std::size_t... Extents> struct std::formatter<stdex::extents<IndexType, Extents...>>
{
    constexpr auto parse(format_parse_context& context) { return context.begin(); }

    template <class FormatContext>
    auto format(stdex::extents<IndexType, Extents...> const& extents, FormatContext& context) const
    {
      auto output = context.out();
      output = fmt::format_to(output, "extents(");
      for (std::size_t axis = 0; axis < extents.rank(); ++axis)
      {
        if (axis != 0) output = fmt::format_to(output, ",");
        output = fmt::format_to(output, "{}", extents.extent(axis));
      }
      return fmt::format_to(output, ")");
    }
};

namespace fmt
{

/// \brief Formatter specialization that prints mdspan extents as a comma-separated list.
template <class IndexType, std::size_t... StaticExtents> struct formatter<stdex::extents<IndexType, StaticExtents...>>
{
    constexpr auto parse(format_parse_context& context) { return context.begin(); }

    template <class FormatContext>
    constexpr auto format(stdex::extents<IndexType, StaticExtents...> const& extents, FormatContext& context) const
    {
      auto output = context.out();
      *output++ = '[';
      for (std::size_t axis = 0; axis < sizeof...(StaticExtents); ++axis)
      {
        if (axis != 0) *output++ = ',';
        output = fmt::format_to(output, "{}", extents.extent(axis));
      }
      *output++ = ']';
      return output;
    }
};

} // namespace fmt
