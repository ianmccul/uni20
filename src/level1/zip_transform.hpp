#pragma once

#include "common/mdspan.hpp"
#include "core/types.hpp"
#include "mdspan/concepts.hpp"
#include "mdspan/zip_layout.hpp"
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief An mdspan accessor that applies an N‑ary functor to N child spans,
///        with maximal empty‑base optimizations.
/// \tparam Func   A callable taking Spans::reference... and returning R.
/// \tparam Spans  Zero or more mdspan types (must share extents & rank).
/// \ingroup internal
template <class Func, class... Spans>
struct TransformAccessor : private Func // EBO if Func is empty
    ,
                           private std::tuple<typename Spans::accessor_type...> // EBO for accessors
{
    static_assert(sizeof...(Spans) >= 1, "Need at least one span");

    /// \brief The tuple of child-accessor types we inherit.
    using accessor_tuple = std::tuple<typename Spans::accessor_type...>;

    /// \brief Bundle of child data handles.
    using data_handle_type = std::tuple<typename Spans::data_handle_type...>;

    /// \brief Per‑child offset_type (falls back to std::size_t).
    using offset_type = std::tuple<span_offset_t<typename Spans::accessor_type>...>;

    /// \brief Return type when calling func_ on each child::reference.
    using reference = std::invoke_result_t<Func, typename Spans::reference...>;

    /// \brief “True” element type, stripping any proxy/reference wrappers.
    using element_type = uni20::remove_proxy_reference_t<reference> const;

    /// \brief Alias so mdspan sees this as its offset_policy.
    using offset_policy = TransformAccessor;

    /// \brief Perfect‑forwarding CTAD constructor.
    /// \tparam FwdFunc  Deduced type of the functor (can bind to temporaries).
    template <class FwdFunc>
    /// \brief Perfect-forwarding constructor used by class template argument deduction.
    /// \param f      Callable object to store within the accessor.
    /// \param spans  Child spans whose accessors are captured.
    /// \ingroup internal
    TransformAccessor(FwdFunc&& f, Spans const&... spans) noexcept(std::is_nothrow_constructible_v<Func, FwdFunc>)
        : Func(std::forward<FwdFunc>(f)), accessor_tuple(spans.accessor()...)
    {}

    /// \brief Advance each child handle by its per‑span offset.
    /// \param handles  Tuple of current child data handles.
    /// \param rel      Tuple of per‑child offsets.
    /// \return         New tuple of advanced handles.
    /// \ingroup internal
    constexpr data_handle_type offset(data_handle_type const& handles, offset_type const& rel) const noexcept
    {
      return offset_impl(handles, rel, std::make_index_sequence<sizeof...(Spans)>{});
    }

    /// \brief Fetch each child element then invoke the stored functor.
    /// \param handles  Tuple of current child data handles.
    /// \param rel      Tuple of per‑child offsets.
    /// \return         Result of calling func_(child0, child1, ...).
    /// \ingroup internal
    constexpr reference access(data_handle_type const& handles, offset_type const& rel) const noexcept
    {
      return access_impl(handles, rel, std::make_index_sequence<sizeof...(Spans)>{});
    }

  private:
    /// \brief Helper: call each child.offset() and bundle new handles.
    /// \ingroup internal
    template <std::size_t... I>
    constexpr data_handle_type offset_impl(data_handle_type const& handles, offset_type const& rel,
                                           std::index_sequence<I...>) const noexcept
    {
      return {std::get<I>(static_cast<accessor_tuple const&>(*this)).offset(std::get<I>(handles), std::get<I>(rel))...};
    }

    /// \brief Helper: call each child.access() then Func::operator()(...).
    /// \ingroup internal
    template <std::size_t... I>
    constexpr reference access_impl(data_handle_type const& handles, offset_type const& rel,
                                    std::index_sequence<I...>) const noexcept
    {
      return Func::operator()(
          std::get<I>(static_cast<accessor_tuple const&>(*this)).access(std::get<I>(handles), std::get<I>(rel))...);
    }
};

/// \brief CTAD guide: deduce TransformAccessor<Func,Spans...>
/// \tparam Func   Functor type.
/// \tparam Spans  Span types.
/// \ingroup internal
template <typename Func, typename... Spans>
TransformAccessor(Func&&, Spans const&...) -> TransformAccessor<std::decay_t<Func>, Spans...>;

/// \brief Create an element-wise “zip + transform” view over N spans.
/// \tparam Func   A callable taking N arguments (one per span) and returning R.
/// \tparam Spans  One or more mdspan types (all must share the same extents).
/// \param  f      The N-ary functor to apply (e.g. plus_n{}, a lambda, etc.).
/// \param  spans  The input spans to zip (all must have identical extents).
/// \return        An mdspan whose element at multi-index I is
///                f(spans0(I), spans1(I), …).
/// \ingroup level1_ops
template <typename Func, typename... Spans> auto zip_transform(Func&& f, Spans const&... spans)
{
  static_assert(sizeof...(Spans) >= 1, "zip_transform needs at least one span");

  // all spans must share extents type and rank
  using first_span = std::tuple_element_t<0, std::tuple<Spans...>>;
  static_assert(((Spans::rank() == first_span::rank()) && ...), "All spans must have same rank");

  // merge the extents objects into a common extent, collapsing static ranks where possible
  using extents_t = common_extents_t<Spans...>;

  // pick the layout
  using layout_t = zip_layout_t<Spans...>;

  // build the concrete mapping for these extents
  using mapping_t = typename layout_t::template mapping<extents_t>;

  // build the accessor
  using accessor_t = TransformAccessor<std::decay_t<Func>, Spans...>;

  // construct the mdspan
  return stdex::mdspan<typename accessor_t::element_type, extents_t, layout_t, accessor_t>{
      std::tuple{spans.data_handle()...}, mapping_t{make_common_extents(spans...), spans.mapping()...},
      accessor_t(std::forward<Func>(f), spans...)};
}

/// \brief Unary “zip‐transform”: apply \p f element‐wise to one mdspan, preserving its layout.
/// \tparam Func  Callable taking one argument of Span’s element_type.
/// \tparam Span  A mdspan‐like type
/// \param  f     The unary operation.
/// \param  span  The input mdspan.
/// \return       An mdspan view whose element(i…) == f(span(i…)), with the same layout_type and extents_type as \p
/// span.
/// \ingroup level1_ops
template <typename Func, typename Span> auto zip_transform(Func&& f, Span const& span)
{
  // build the transform‐accessor
  TransformAccessor<std::decay_t<Func>, Span> acc{std::forward<Func>(f), span};

  using accessor_t = decltype(acc);
  using element_t = typename accessor_t::element_type;
  using extents_t = typename Span::extents_type;
  using layout_t = typename Span::layout_type;

  // construct the mdspan view with the original data_handle(), mapping, and our accessor
  return stdex::mdspan<element_t, extents_t, layout_t, accessor_t>{span.data_handle(), span.mapping(), std::move(acc)};
}

} // namespace uni20
