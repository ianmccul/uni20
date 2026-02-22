#pragma once

/**
 * \file concepts.hpp
 * \ingroup core
 * \brief Mdspan concept and accessor extensions for Uni20.
 */

/**
 * \defgroup mdspan_ext Mdspan extensions
 * \ingroup core
 * \brief Additional concepts, adaptors, and helpers that extend the reference mdspan implementation.
 */

#include "common/mdspan.hpp"
#include <cassert>
#include <concepts>
#include <fmt/format.h>

namespace uni20
{

/// \brief Trait to pull an AccessorPolicy’s offset_type if present, or fall back to std::size_t otherwise.
/// \note This is an extension to the standard mdspan AccessorPolicy.
/// \ingroup mdspan_ext
template <typename AP, typename = void> struct span_offset_type
{
    /// \brief The resulting offset type (std::size_t by default).
    /// \ingroup mdspan_ext
    using type = std::size_t;
};

/// \brief Partial specialization that uses an accessor policy’s declared offset_type.
/// \tparam AP The accessor policy that provides offset_type.
/// \ingroup mdspan_ext
template <typename AP> struct span_offset_type<AP, std::void_t<typename AP::offset_type>>
{
    /// \brief The resulting offset type defined by the accessor policy.
    /// \ingroup mdspan_ext
    using type = typename AP::offset_type;
};

/// \brief Convenience alias for accessor_offset_type<AP>::type.
/// \tparam AP The accessor policy to inspect.
/// \ingroup mdspan_ext
template <typename AP> using span_offset_t = typename span_offset_type<AP>::type;

/// \concept AccessorPolicy
/// \brief A model of the C++ mdspan AccessorPolicy named requirement.
/// \details Requirements for \c AP:
///          - nested types: element_type, data_handle_type, offset_policy, reference.
///          - \c offset(dh, off) must be constexpr noexcept and return a type convertible to
///            \c offset_policy::data_handle_type.
///          - \c access(dh, off) must be constexpr noexcept and return exactly \c AP::reference.
/// \tparam AP The accessor policy to test.
/// \ingroup mdspan_ext
template <class AP>
concept AccessorPolicy = requires
{
  typename AP::element_type;
  typename AP::data_handle_type;
  typename AP::offset_policy;
  typename AP::reference;
}
&&requires(AP a, typename AP::data_handle_type dh, span_offset_t<AP> off)
{
  {
    a.offset(dh, off)
    } -> std::convertible_to<typename AP::offset_policy::data_handle_type>;
  {
    a.access(dh, off)
    } -> std::same_as<typename AP::reference>;
};

/// \brief Generic adaptor that converts an AccessorPolicy’s reference type to a compatible const-qualified reference.
/// \details Example: to turn a mutable accessor (returning \c T&) into a read-only one returning \c T const&, use
///          \code
///          conversion_accessor_adaptor<MyAccessor, MyAccessor::element_type const&> rd_access{my_access};
///          \endcode
/// \tparam Accessor An AccessorPolicy whose \c reference type will be adapted.
/// \tparam NewReference The new reference type returned by \c access(); must be convertible from \c
/// Accessor::reference. \ingroup mdspan_ext
template <AccessorPolicy Accessor, typename NewReference>
requires std::is_same_v<typename Accessor::reference, typename Accessor::element_type&>
class const_accessor_adaptor {
  public:
    using element_type = typename Accessor::element_type const;
    using reference = element_type&;
    using data_handle_type = typename Accessor::data_handle_type;
    using offset_policy = const_accessor_adaptor;
    using offset_type = span_offset_t<Accessor>;

    const_accessor_adaptor(Accessor const& to_be_wrapped) : wrapped_{to_be_wrapped} {}

    constexpr reference access(data_handle_type p, offset_type i) const { return wrapped_.access(p, i); }

    constexpr data_handle_type offset(data_handle_type p, offset_type i) const { return wrapped_.offset(p, i); }

  private:
    Accessor wrapped_;
};

//
// const_accessor overloads: build a read-only accessor from a mutable one.
//

/// \brief Wrap a default_accessor<T> into a const_default_accessor<T>.
/// \tparam T Element type accessed by the policy.
/// \param accessor_policy The accessor policy to upgrade.
/// \return A default accessor for \c T const.
/// \ingroup mdspan_ext
template <typename T>
constexpr stdex::default_accessor<T const> const_accessor(stdex::default_accessor<T> const& accessor_policy)
{
  (void)accessor_policy;
  return stdex::default_accessor<T const>();
}

// /// \brief const_accessor on const_default_accessor yields itself.
// template <typename T> constexpr default_accessor<T const> const_accessor(default_accessor<T const> const&)
// {
//   return default_accessor<T const>();
// }

/// \brief Wrap any accessor whose reference is T& into a const adaptor.
/// \tparam Acc A mutable accessor policy with reference equal to \c element_type&.
/// \param acc The accessor to wrap.
/// \return A const-qualified accessor adaptor.
/// \ingroup mdspan_ext
template <AccessorPolicy Acc>
requires std::is_same_v<typename Acc::reference, typename Acc::element_type&>
constexpr auto const_accessor(Acc const& acc)
{
  return const_accessor_adaptor<Acc, typename Acc::element_type const&>{acc};
}

/// \brief If an accessor returns element_type const&, no change is required.
/// \tparam Acc A read-only accessor policy with reference equal to \c element_type const&.
/// \param acc The accessor (returned by value).
/// \return The original accessor policy.
/// \ingroup mdspan_ext
template <AccessorPolicy Acc>
requires std::is_same_v<typename Acc::reference, typename Acc::element_type const&>
constexpr Acc const_accessor(Acc const& acc) { return acc; }

/// \brief If an accessor returns element_type by value, no change is required.
/// \tparam Acc A read-only accessor policy with reference equal to \c element_type.
/// \param acc The accessor (returned by value).
/// \return The original accessor policy.
/// \ingroup mdspan_ext
template <AccessorPolicy Acc>
requires std::is_same_v<typename Acc::reference, typename Acc::element_type>
constexpr Acc const_accessor(Acc const& acc) { return acc; }

/// \brief Type alias that produces the const-qualified version of an accessor policy.
/// \tparam Acc An accessor policy.
/// \ingroup mdspan_ext
template <AccessorPolicy Acc> using const_accessor_t = decltype(const_accessor(std::declval<Acc>()));

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
/// \tparam S The type being tested for SpanLike requirements.
/// \ingroup mdspan_ext
template <class S>
concept SpanLike = requires(S s)
{
  typename S::element_type;
  typename S::extents_type;
  typename S::layout_type;
  typename S::accessor_type;
  typename S::reference;
  typename S::value_type;
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
}
&&AccessorPolicy<typename S::accessor_type>;

// ConstSpanLike
// Not sure that we need this, and its hard to be definitive that it is actually const
// template <class S>
// concept ConstSpanLike =
//     requires(S s) {
//       typename S::element_type;
//       typename S::extents_type;
//       typename S::layout_type;
//       typename S::accessor_type;
//       // mapping() must return something convertible to mapping_t
//       {
//         s.mapping()
//         } -> std::convertible_to<typename S::layout_type::template mapping<typename S::extents_type>>;
//
//       // data_handle() must return something convertible to data_handle_type
//       {
//         s.data_handle()
//         } -> std::convertible_to<typename S::accessor_type::data_handle_type>;
//
//       // accessor() must return something convertible to accessor_type
//       {
//         s.accessor()
//         } -> std::convertible_to<typename S::accessor_type>;
//     } && AccessorPolicy<typename S::accessor_type> && std::is_const_v<typename S::element_type>;

/// \concept MutableSpanLike
/// \brief SpanLike types whose reference type supports assignment.
/// \tparam S The type being evaluated for mutable access.
/// \ingroup mdspan_ext
template <class S>
concept MutableSpanLike = requires(S s)
{
  typename S::element_type;
  typename S::extents_type;
  typename S::layout_type;
  typename S::accessor_type;
  typename S::reference;
  typename S::value_type;
  typename S::index_type;
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
}
&&AccessorPolicy<typename S::accessor_type> &&
    (!std::is_const_v<typename S::element_type>)&&requires(typename S::reference ref, typename S::value_type val)
{
  ref = val;
}; // must be able to assign a value to a reference

/// \brief A “strided mdspan‐like” type that models SpanLike and reports layout_stride.
/// \tparam MDS The mdspan-like type under test.
/// \ingroup mdspan_ext
template <class MDS>
concept StridedMdspan = SpanLike<MDS> && // must satisfy our mdspan‐like protocol
    MDS::is_always_strided();

/// \concept MutableStridedMdspan
/// \brief Mutable span-like types whose layout reports they are always strided.
/// \tparam MDS The mdspan-like type under test.
/// \ingroup mdspan_ext
template <class MDS>
concept MutableStridedMdspan = MutableSpanLike<MDS> && // must satisfy our mdspan‐like protocol
    MDS::is_always_strided();

namespace detail
{

/// \brief Helper that materializes strides from the layout mapping.
/// \tparam S The strided mdspan-like type.
/// \tparam I Index sequence selecting the stride positions.
/// \param s The mdspan instance whose strides will be computed.
/// \return An array containing strides for each dimension in \c S.
/// \ingroup internal
template <StridedMdspan S, size_t... I> constexpr auto strides_impl(S const& s, std::index_sequence<I...>)
{
  using index_type = typename S::index_type;
  // fold the pack I... into an array by calling s.mapping().stride(I) for each I
  return std::array<index_type, sizeof...(I)>{s.mapping().stride(I)...};
}

} // namespace detail

/// \brief Retrieve the strides associated with a strided mdspan-like type.
/// \tparam S The strided mdspan-like type.
/// \param s The mdspan instance whose strides will be returned.
/// \return A std::array containing the strides for each rank.
/// \ingroup mdspan_ext
template <StridedMdspan S> auto strides(S const& s)
{
  return detail::strides_impl(s, std::make_index_sequence<S::rank>{});
}

/// \brief Retrieve the strides from a reference layout_stride mdspan.
/// \tparam T Element type stored by the mdspan.
/// \tparam Extents Extents type of the mdspan.
/// \tparam AccessorPolicy Accessor policy used by the mdspan.
/// \param s The mdspan instance whose native strides will be returned.
/// \return The strides exposed by the mdspan.
/// \ingroup mdspan_ext
template <typename T, typename Extents, typename AccessorPolicy>
constexpr auto strides(stdex::mdspan<T, Extents, stdex::layout_stride, AccessorPolicy> const& s) noexcept
{
  return s.strides();
}

} // namespace uni20

namespace fmt
{

/// \brief Formatter specialization that prints mdspan extents as a comma-separated list.
/// \tparam IndexType Integral type used for extents.
/// \tparam StaticExts Static extent values embedded in the extents type.
/// \tparam CharT Character type for the target formatter.
/// \ingroup mdspan_ext
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
