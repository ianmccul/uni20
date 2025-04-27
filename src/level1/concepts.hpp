#pragma once

#include "common/mdspan.hpp"
#include <cassert>
#include <concepts>
#include <fmt/format.h>

namespace uni20
{

/// \brief Trait to pull an AccessorPolicy’s offset_type if present,
///        or fall back to std::size_t otherwise.
/// \tparam AP  The accessor policy to inspect.
/// \note this is an extension to the standard mdspan AccessorPolicy
///       https://en.cppreference.com/w/cpp/named_req/AccessorPolicy
///       that has no offset_type but simply uses std::size_t
template <typename AP, typename = void> struct span_offset_type
{
    /// \brief The resulting offset type (std::size_t by default).
    using type = std::size_t;
};

/// \brief Partial specialization: when AP declares offset_type, use it.
/// \tparam AP  The accessor policy that provides offset_type.
template <typename AP> struct span_offset_type<AP, std::void_t<typename AP::offset_type>>
{
    using type = typename AP::offset_type;
};

/// \brief Convenience alias for accessor_offset_type<AP>::type.
/// \tparam AP  The accessor policy to inspect.
template <typename AP> using span_offset_t = typename span_offset_type<AP>::type;

/// \concept AccessorPolicy
/// \brief A model of the C++ mdspan AccessorPolicy named requirement.
/// Requirements for AP:
///  - nested types: element_type, data_handle_type, offset_policy, reference
///  - offset(dh, off) must be constexpr noexcept and return
///    a type convertible to offset_policy::data_handle_type.
///  - access(dh, off) must be constexpr noexcept and return exactly AP::reference.
/// \tparam AP  The accessor policy to test.
template <class AP>
concept AccessorPolicy = requires {
                           typename AP::element_type;
                           typename AP::data_handle_type;
                           typename AP::offset_policy;
                           typename AP::reference;
                         } && requires(AP a, typename AP::data_handle_type dh, span_offset_t<AP> off) {
                                {
                                  a.offset(dh, off)
                                  } -> std::convertible_to<typename AP::offset_policy::data_handle_type>;
                                {
                                  a.access(dh, off)
                                  } -> std::same_as<typename AP::reference>;
                              };

/// \brief Generic adaptor that converts an AccessorPolicy’s reference type
///        to an arbitrary compatible reference type.
///
/// \tparam Accessor     An AccessorPolicy whose \c reference type will be adapted.
/// \tparam NewReference The new reference type returned by \c access(); must be
///                      convertible from \c Accessor::reference.
///
/// Example: to turn a mutable accessor (returning \c T&) into a read-only one
/// returning \c T const&, use
/// \code
/// conversion_accessor_adaptor<MyAccessor, MyAccessor::element_type const&> rd_access{my_access};
/// \endcode
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
template <typename T> constexpr stdex::default_accessor<T const> const_accessor(stdex::default_accessor<T> const&)
{
  return stdex::default_accessor<T const>();
}

// /// \brief const_accessor on const_default_accessor yields itself.
// template <typename T> constexpr default_accessor<T const> const_accessor(default_accessor<T const> const&)
// {
//   return default_accessor<T const>();
// }

/// \brief Wrap any accessor whose reference is T& into a const adaptor.
/// \tparam Acc  A mutable accessor policy with reference == element_type&.
/// \param acc   The accessor to wrap.
/// \return      A conversion_accessor_adaptor<Acc, Acc::element_type const&>.
template <AccessorPolicy Acc>
  requires std::is_same_v<typename Acc::reference, typename Acc::element_type&>
constexpr auto const_accessor(Acc const& acc)
{
  return const_accessor_adaptor<Acc, typename Acc::element_type const&>{acc};
}

/// \brief if accessor returns element_type const&, no change needed.
/// \tparam Acc  A read-only accessor policy with reference == element_type const&.
/// \param acc   The accessor (returned by-value).
template <AccessorPolicy Acc>
  requires std::is_same_v<typename Acc::reference, typename Acc::element_type const&>
constexpr Acc const_accessor(Acc const& acc)
{
  return acc;
}

/// \brief if accessor returns element_type by-value, no change needed.
/// \tparam Acc  A read-only accessor policy with reference == element_type.
/// \param acc   The accessor (returned by-value).
template <AccessorPolicy Acc>
  requires std::is_same_v<typename Acc::reference, typename Acc::element_type>
constexpr Acc const_accessor(Acc const& acc)
{
  return acc;
}

/// \brief Type alias for the const version of AccessorPolicy
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
template <class S>
concept SpanLike = requires(S s) {
                     typename S::element_type;
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
                   } && AccessorPolicy<typename S::accessor_type>;

template <class S>
concept ConstSpanLike =
    requires(S s) {
      typename S::element_type;
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
    } && AccessorPolicy<typename S::accessor_type> && std::is_const_v<typename S::element_type>;

template <class S>
concept MutableSpanLike =
    requires(S s) {
      typename S::element_type;
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
    } && AccessorPolicy<typename S::accessor_type> && (!std::is_const_v<typename S::element_type>);

/// \brief A “strided mdspan‐like” type:
///        – models SpanLike (has extents_type, layout_type, accessor_type, mapping(), data_handle())
///        – uses layout_stride as its layout_policy
template <class MDS>
concept StridedMdspan = SpanLike<MDS> && // must satisfy our mdspan‐like protocol
                        std::same_as<typename MDS::layout_type, stdex::layout_stride>; // must use layout_stride

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
