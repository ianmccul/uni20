#pragma once

/**
 * \file transform_view.hpp
 * \ingroup mdspan_ext
 * \brief Lazy read-only elementwise transform views over mdspan-like inputs.
 */

#include <uni20/core/types.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/mdspan/zip_layout.hpp>

#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class Function, SpanLike Span> class unary_transform_accessor {
  public:
    using function_type = Function;
    using span_type = Span;
    using wrapped_accessor_type = typename span_type::accessor_type;
    using data_handle_type = typename span_type::data_handle_type;
    using offset_type = span_offset_t<wrapped_accessor_type>;
    using reference = std::invoke_result_t<function_type const&, typename span_type::reference>;
    using element_type = remove_proxy_reference_t<reference> const;
    using offset_policy = unary_transform_accessor;

    template <class FwdFunction>
    constexpr unary_transform_accessor(FwdFunction&& function, span_type const& span)
        : function_(std::forward<FwdFunction>(function)), accessor_(span.accessor())
    {}

    [[nodiscard]] constexpr auto offset(data_handle_type const& handle,
                                        offset_type const& offset) const -> data_handle_type
    {
      return accessor_.offset(handle, offset);
    }

    [[nodiscard]] constexpr auto access(data_handle_type const& handle, offset_type const& offset) const -> reference
    {
      return std::invoke(function_, accessor_.access(handle, offset));
    }

  private:
    [[no_unique_address]] function_type function_;
    [[no_unique_address]] wrapped_accessor_type accessor_;
};

template <class Function, SpanLike... Spans> class transform_accessor {
  public:
    static_assert(sizeof...(Spans) >= 2);

    using function_type = Function;
    using accessor_tuple = std::tuple<typename Spans::accessor_type...>;
    using data_handle_type = std::tuple<typename Spans::data_handle_type...>;
    using offset_type = std::tuple<span_offset_t<typename Spans::accessor_type>...>;
    using reference = std::invoke_result_t<function_type const&, typename Spans::reference...>;
    using element_type = remove_proxy_reference_t<reference> const;
    using offset_policy = transform_accessor;

    template <class FwdFunction>
    constexpr transform_accessor(FwdFunction&& function, Spans const&... spans)
        : function_(std::forward<FwdFunction>(function)), accessors_(spans.accessor()...)
    {}

    [[nodiscard]] constexpr auto offset(data_handle_type const& handles,
                                        offset_type const& offsets) const -> data_handle_type
    {
      return offset_impl(handles, offsets, std::index_sequence_for<Spans...>{});
    }

    [[nodiscard]] constexpr auto access(data_handle_type const& handles, offset_type const& offsets) const -> reference
    {
      return access_impl(handles, offsets, std::index_sequence_for<Spans...>{});
    }

  private:
    template <std::size_t... Index>
    [[nodiscard]] constexpr auto offset_impl(data_handle_type const& handles, offset_type const& offsets,
                                             std::index_sequence<Index...>) const -> data_handle_type
    {
      return {std::get<Index>(accessors_).offset(std::get<Index>(handles), std::get<Index>(offsets))...};
    }

    template <std::size_t... Index>
    [[nodiscard]] constexpr auto access_impl(data_handle_type const& handles, offset_type const& offsets,
                                             std::index_sequence<Index...>) const -> reference
    {
      return std::invoke(function_,
                         std::get<Index>(accessors_).access(std::get<Index>(handles), std::get<Index>(offsets))...);
    }

    [[no_unique_address]] function_type function_;
    [[no_unique_address]] accessor_tuple accessors_;
};

} // namespace detail

/// \brief Create a lazy read-only unary transform view while preserving the input mapping.
template <class Function, SpanLike Span>
[[nodiscard]] constexpr auto transform_view(Function&& function, Span const& span)
{
  using function_type = std::decay_t<Function>;
  using accessor_type = detail::unary_transform_accessor<function_type, Span>;
  using element_type = typename accessor_type::element_type;
  using extents_type = typename Span::extents_type;
  using layout_type = typename Span::layout_type;

  return stdex::mdspan<element_type, extents_type, layout_type, accessor_type>{
      span.data_handle(), span.mapping(), accessor_type(std::forward<Function>(function), span)};
}

/// \brief Create a lazy read-only elementwise transform view over two or more inputs.
/// \pre Every input has the same runtime extents.
template <class Function, SpanLike First, SpanLike Second, SpanLike... Rest>
[[nodiscard]] constexpr auto transform_view(Function&& function, First const& first, Second const& second,
                                            Rest const&... rest)
{
  static_assert(First::rank() == Second::rank() && ((First::rank() == Rest::rank()) && ...));

  using function_type = std::decay_t<Function>;
  using extents_type = common_extents_t<First, Second, Rest...>;
  using layout_type = zip_layout_t<First, Second, Rest...>;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using accessor_type = detail::transform_accessor<function_type, First, Second, Rest...>;
  using element_type = typename accessor_type::element_type;

  auto extents = make_common_extents(first, second, rest...);
  return stdex::mdspan<element_type, extents_type, layout_type, accessor_type>{
      std::tuple{first.data_handle(), second.data_handle(), rest.data_handle()...},
      mapping_type{extents, first.mapping(), second.mapping(), rest.mapping()...},
      accessor_type(std::forward<Function>(function), first, second, rest...)};
}

} // namespace uni20
