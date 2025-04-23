#pragma once

#include "common/mdspan.hpp"
#include <cassert>
#include <concepts>
#include <fmt/format.h>

namespace uni20
{

/// \concept SpanLike
/// \brief A “span-like” type usable by uni20’s zip_transform machinery.
///
/// A type \c S models SpanLike if it provides the minimal mdspan-like API:
///   - \c S::extents_type defines the shape type
///   - \c S::layout_type defines the layout policy
///   - \c S::accessor_type defines the accessor policy
///   - a const \c mapping() member returning something convertible to
///     \c layout_type::mapping<extents_type>
///   - a const \c data_handle() member returning something convertible to
///     \c accessor_type::data_handle_type
///   - an \c accessor() member returning something convertible to \c accessor_type
template <class S>
concept SpanLike = requires(S s) {
                     typename S::extents_type;
                     typename S::layout_type;
                     typename S::accessor_type;

                     // mapping() must return something convertible to mapping_t
                     {
                       s.mapping()
                       } -> std::convertible_to<typename S::layout_type::template mapping<typename S::extents_type>>;

                     // data_handle() must return something convertible to data_handle_type
                     {
                       s.data_handle()
                       } -> std::convertible_to<typename S::accessor_type::data_handle_type>;

                     // accessor() must return something convertible to accessor_type
                     {
                       s.accessor()
                       } -> std::convertible_to<typename S::accessor_type>;
                   };

/// \brief A “strided mdspan‐like” type:
///        – models SpanLike (has extents_type, layout_type, accessor_type, mapping(), data_handle())
///        – uses layout_stride as its layout_policy
template <class MDS>
concept StridedMdspan = SpanLike<MDS> && // must satisfy our mdspan‐like protocol
                        std::same_as<typename MDS::layout_type, stdex::layout_stride>; // must use layout_stride

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
