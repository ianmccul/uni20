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

#include <concepts>
#include <type_traits>
#include <uni20/mdspan/mdspan.hpp>
#include <utility>

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
///          - \c offset(dh, off) returns a type convertible to
///            \c offset_policy::data_handle_type.
///          - \c access(dh, off) returns exactly \c AP::reference.
///          Access may throw when the accessor deliberately applies a throwing
///          transformation.
/// \tparam AP The accessor policy to test.
/// \ingroup mdspan_ext
template <class AP>
concept AccessorPolicy = requires {
  typename AP::element_type;
  typename AP::data_handle_type;
  typename AP::offset_policy;
  typename AP::reference;
} && requires(AP a, typename AP::data_handle_type dh, span_offset_t<AP> off) {
  { a.offset(dh, off) } -> std::convertible_to<typename AP::offset_policy::data_handle_type>;
  { a.access(dh, off) } -> std::same_as<typename AP::reference>;
};

/// \brief Execution domain in which ordinary host code may evaluate an accessor.
struct host_access_domain
{};

/// \brief Execution domain in which CUDA device code may evaluate an accessor.
struct cuda_access_domain
{};

/// \brief Opt-in customization for the execution domains supported by an accessor.
/// \details Specializations must describe where `access(...)` may be evaluated,
///          independently of whether `data_handle_type` is pointer-shaped.
/// \tparam Accessor Accessor policy being classified.
/// \tparam Domain Execution-domain tag.
template <class Accessor, class Domain> inline constexpr bool enable_accessor_in_domain = false;

/// \brief Standard mdspan accessors are directly host-accessible.
template <class ElementType>
inline constexpr bool enable_accessor_in_domain<stdex::default_accessor<ElementType>, host_access_domain> = true;

/// \concept AccessorInDomain
/// \brief Accessor policy explicitly valid in one execution domain.
/// \tparam Accessor Accessor policy being tested.
/// \tparam Domain Execution-domain tag.
template <class Accessor, class Domain>
concept AccessorInDomain = AccessorPolicy<std::remove_cvref_t<Accessor>> &&
                           enable_accessor_in_domain<std::remove_cvref_t<Accessor>, std::remove_cvref_t<Domain>>;

/// \concept HostAccessibleAccessor
/// \brief Accessor policy that may be evaluated directly by host code.
/// \tparam Accessor Accessor policy being tested.
template <class Accessor>
concept HostAccessibleAccessor = AccessorInDomain<Accessor, host_access_domain>;

/// \concept CudaAccessibleAccessor
/// \brief Accessor policy that may be evaluated directly by CUDA device code.
/// \tparam Accessor Accessor policy being tested.
template <class Accessor>
concept CudaAccessibleAccessor = AccessorInDomain<Accessor, cuda_access_domain>;

/// \brief Adaptor that exposes a mutable lvalue accessor as read-only.
/// \details Example: to turn a mutable accessor returning \c T& into a read-only
///          accessor returning \c T const&, use
///          \code
///          const_accessor_adaptor<MyAccessor> read_access{access};
///          \endcode
/// \tparam Accessor An AccessorPolicy whose \c reference type will be adapted.
/// \ingroup mdspan_ext
template <AccessorPolicy Accessor>
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

/// \brief A const accessor adaptor preserves its wrapped accessor's execution domains.
template <class Accessor, class Domain>
inline constexpr bool enable_accessor_in_domain<const_accessor_adaptor<Accessor>, Domain> =
    enable_accessor_in_domain<Accessor, Domain>;

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

/// \brief Trait that recognizes the standard mdspan default accessor.
/// \tparam Accessor Candidate accessor policy.
/// \ingroup mdspan_ext
template <class Accessor> struct is_default_accessor : std::false_type
{};

/// \brief Specialization for `stdex::default_accessor<T>`.
/// \tparam ElementType Element type of the default accessor.
/// \ingroup mdspan_ext
template <class ElementType> struct is_default_accessor<stdex::default_accessor<ElementType>> : std::true_type
{};

/// \brief True when an accessor is `stdex::default_accessor<T>`.
/// \tparam Accessor Candidate accessor policy.
/// \ingroup mdspan_ext
template <class Accessor>
inline constexpr bool is_default_accessor_v = is_default_accessor<std::remove_cvref_t<Accessor>>::value;

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
  requires(!std::is_const_v<typename Acc::element_type> &&
           std::is_same_v<typename Acc::reference, typename Acc::element_type&>)
constexpr auto const_accessor(Acc const& acc)
{
  return const_accessor_adaptor<Acc>{acc};
}

/// \brief Return an accessor unchanged when its element type is already const.
/// \details Read-only calculated accessors commonly return an unqualified value
///          while declaring const `element_type`; no additional adaptor is needed.
/// \tparam Acc A read-only accessor policy with const \c element_type.
/// \param acc The accessor (returned by value).
/// \return The original accessor policy.
/// \ingroup mdspan_ext
template <AccessorPolicy Acc>
  requires(std::is_const_v<typename Acc::element_type> && !is_default_accessor_v<Acc>)
constexpr Acc const_accessor(Acc const& acc)
{
  return acc;
}

/// \brief Type alias that produces the const-qualified version of an accessor policy.
/// \tparam Acc An accessor policy.
/// \ingroup mdspan_ext
template <AccessorPolicy Acc> using const_accessor_t = decltype(const_accessor(std::declval<Acc>()));

namespace detail
{

template <class S> using span_type_t = std::remove_cvref_t<S>;

template <class S, std::size_t... Axis> consteval bool span_has_ranked_subscript(std::index_sequence<Axis...>)
{
  using span_type = span_type_t<S>;
  return requires(span_type const& span, typename span_type::index_type index) {
    { span.operator[](((void)Axis, index)...) } -> std::same_as<typename span_type::reference>;
  };
}

template <class S, std::size_t... Axis> consteval bool span_has_ranked_assignment(std::index_sequence<Axis...>)
{
  using span_type = span_type_t<S>;
  return requires(span_type& span, typename span_type::index_type index, typename span_type::value_type value) {
    span.operator[](((void)Axis, index)...) = value;
  };
}

template <class S>
concept SpanMetadata =
    requires(span_type_t<S> const& span, std::size_t axis) {
      typename span_type_t<S>::element_type;
      typename span_type_t<S>::value_type;
      typename span_type_t<S>::index_type;
      typename span_type_t<S>::extents_type;
      typename span_type_t<S>::layout_type;
      typename span_type_t<S>::mapping_type;
      typename span_type_t<S>::accessor_type;
      typename span_type_t<S>::data_handle_type;
      typename span_type_t<S>::reference;
      typename std::integral_constant<std::size_t, span_type_t<S>::rank()>;

      { span.extents() } -> std::convertible_to<typename span_type_t<S>::extents_type>;
      { span.extent(axis) } -> std::convertible_to<typename span_type_t<S>::index_type>;
      { span.mapping() } -> std::convertible_to<typename span_type_t<S>::mapping_type>;
      { span.accessor() } -> std::convertible_to<typename span_type_t<S>::accessor_type>;
    } && std::same_as<typename span_type_t<S>::value_type, std::remove_cv_t<typename span_type_t<S>::element_type>> &&
    std::same_as<typename span_type_t<S>::index_type, typename span_type_t<S>::extents_type::index_type> &&
    std::same_as<typename span_type_t<S>::mapping_type,
                 typename span_type_t<S>::layout_type::template mapping<typename span_type_t<S>::extents_type>> &&
    std::same_as<typename span_type_t<S>::data_handle_type, typename span_type_t<S>::accessor_type::data_handle_type> &&
    std::same_as<typename span_type_t<S>::reference, typename span_type_t<S>::accessor_type::reference> &&
    (span_type_t<S>::rank() == span_type_t<S>::extents_type::rank());

template <class S>
concept SpanDescriptor = SpanMetadata<S> && requires(span_type_t<S> const& span) {
  { span.data_handle() } -> std::convertible_to<typename span_type_t<S>::data_handle_type>;
};

template <class S>
concept DeviceSpanDescriptor = SpanMetadata<S> && requires(span_type_t<S> const& span) {
  typename span_type_t<S>::data_descriptor_type;
  { span.data_descriptor() } -> std::convertible_to<typename span_type_t<S>::data_descriptor_type const&>;
};

} // namespace detail

/// \concept MdspanLike
/// \brief A readable type implementing the mdspan observers and multidimensional subscript contract.
/// \details In addition to the standard mdspan descriptor aliases, a model
///          exposes its static rank, extents, mapping, accessor, data handle,
///          and an `operator[]` accepting one index per rank whose result is
///          exactly `reference`.
/// \tparam S The type being tested for MdspanLike requirements.
/// \ingroup mdspan_ext
template <class S>
concept MdspanLike = detail::SpanDescriptor<S> && AccessorPolicy<typename detail::span_type_t<S>::accessor_type> &&
                     detail::span_has_ranked_subscript<S>(std::make_index_sequence<detail::span_type_t<S>::rank()>{});

/// \concept DeviceMdspanLike
/// \brief Mdspan metadata whose data handle is immediate or available through a data descriptor.
/// \details Ordinary `MdspanLike` values model this concept as the immediate case.
///          A deferred model instead exposes `data_descriptor_type` and
///          `data_descriptor()` while retaining the actual mapping and accessor
///          that will be used after handle acquisition. The concept does not
///          require the concrete `device_mdspan` class.
/// \tparam S The type being tested for device-mdspan requirements.
/// \ingroup mdspan_ext
template <class S>
concept DeviceMdspanLike = MdspanLike<S> || (detail::DeviceSpanDescriptor<S> &&
                                             AccessorPolicy<typename detail::span_type_t<S>::accessor_type>);

/// \concept HostAccessibleMdspan
/// \brief Mdspan whose accessor may be evaluated directly by host code.
/// \tparam S Mdspan-like type being tested.
template <class S>
concept HostAccessibleMdspan = MdspanLike<S> && HostAccessibleAccessor<typename detail::span_type_t<S>::accessor_type>;

/// \concept CudaAccessibleMdspan
/// \brief Mdspan whose accessor may be evaluated directly by CUDA device code.
/// \tparam S Mdspan-like type being tested.
template <class S>
concept CudaAccessibleMdspan = MdspanLike<S> && CudaAccessibleAccessor<typename detail::span_type_t<S>::accessor_type>;

/// \concept HostAccessibleDeviceMdspan
/// \brief Immediate or descriptor-backed mdspan metadata targeting host access.
/// \tparam S Device-mdspan-like type being tested.
template <class S>
concept HostAccessibleDeviceMdspan =
    DeviceMdspanLike<S> && HostAccessibleAccessor<typename detail::span_type_t<S>::accessor_type>;

/// \concept CudaAccessibleDeviceMdspan
/// \brief Immediate or descriptor-backed mdspan metadata targeting CUDA access.
/// \tparam S Device-mdspan-like type being tested.
template <class S>
concept CudaAccessibleDeviceMdspan =
    DeviceMdspanLike<S> && CudaAccessibleAccessor<typename detail::span_type_t<S>::accessor_type>;

namespace detail
{

template <class Span, class... Index>
concept MdspanIndexPack = MdspanLike<Span> && (sizeof...(Index) == span_type_t<Span>::rank()) &&
                          (std::is_convertible_v<Index, typename span_type_t<Span>::index_type> && ...) &&
                          (std::is_nothrow_constructible_v<typename span_type_t<Span>::index_type, Index> && ...);

template <class Accessor>
concept AccessorReferenceAssignable =
    requires(Accessor const& accessor, typename Accessor::data_handle_type handle, span_offset_t<Accessor> offset,
             std::remove_const_t<typename Accessor::element_type> value) { accessor.access(handle, offset) = value; };

} // namespace detail

/// \brief Access one mdspan element through a read-only adaptation of its stored accessor.
/// \details This preserves the mapping and accessor state without constructing
///          a second mdspan descriptor. Index eligibility and conversion match
///          the multidimensional mdspan subscript operation.
/// \tparam Span The mdspan-like descriptor type.
/// \tparam Index One index type per mdspan rank.
/// \param span Descriptor whose mapping, handle, and accessor are used.
/// \param indices Coordinates for every mdspan axis.
/// \return The reference or calculated value produced by the read-only accessor.
/// \ingroup mdspan_ext
template <MdspanLike Span, class... Index>
  requires detail::MdspanIndexPack<Span, Index...>
constexpr decltype(auto) const_access(Span const& span, Index... indices)
{
  using index_type = typename detail::span_type_t<Span>::index_type;
  auto accessor = const_accessor(span.accessor());
  return accessor.access(span.data_handle(), span.mapping()(static_cast<index_type>(std::move(indices))...));
}

/// \concept MutableMdspanLike
/// \brief MdspanLike types whose accessor supports indexed assignment.
/// \tparam S The type being evaluated for mutable access.
/// \ingroup mdspan_ext
template <class S>
concept MutableMdspanLike =
    MdspanLike<S> && (!std::is_const_v<typename detail::span_type_t<S>::element_type>) &&
    detail::span_has_ranked_assignment<S>(std::make_index_sequence<detail::span_type_t<S>::rank()>{});

/// \concept MutableDeviceMdspanLike
/// \brief DeviceMdspanLike types whose eventual storage supports writes.
/// \details Immediate spans use `MutableMdspanLike`. Deferred spans use the same
///          accessor-reference assignment that will govern the resolved span.
/// \tparam S The device-mdspan-like type being evaluated.
/// \ingroup mdspan_ext
template <class S>
concept MutableDeviceMdspanLike =
    DeviceMdspanLike<S> && (!std::is_const_v<typename detail::span_type_t<S>::element_type>) &&
    (MutableMdspanLike<S> || detail::AccessorReferenceAssignable<typename detail::span_type_t<S>::accessor_type>);

/// \brief An mdspan-like type whose mapping is always strided and exposes stride observers.
/// \tparam MDS The mdspan-like type under test.
/// \ingroup mdspan_ext
template <class MDS>
concept StridedMdspanLike = MdspanLike<MDS> && detail::span_type_t<MDS>::is_always_strided() &&
                            requires(detail::span_type_t<MDS> const& span, std::size_t axis) {
                              {
                                span.stride(axis)
                              } -> std::convertible_to<typename detail::span_type_t<MDS>::index_type>;
                              {
                                span.mapping().stride(axis)
                              } -> std::convertible_to<typename detail::span_type_t<MDS>::index_type>;
                            };

/// \concept MutableStridedMdspanLike
/// \brief Mutable mdspan-like types whose layout reports they are always strided.
/// \tparam MDS The mdspan-like type under test.
/// \ingroup mdspan_ext
template <class MDS>
concept MutableStridedMdspanLike = MutableMdspanLike<MDS> && StridedMdspanLike<MDS>;

/// \brief A device-mdspan-like type whose mapping exposes multidimensional strides.
/// \tparam S The device-mdspan-like type under test.
/// \ingroup mdspan_ext
template <class S>
concept StridedDeviceMdspanLike = DeviceMdspanLike<S> && detail::span_type_t<S>::is_always_strided() &&
                                  requires(detail::span_type_t<S> const& span, std::size_t axis) {
                                    {
                                      span.stride(axis)
                                    } -> std::convertible_to<typename detail::span_type_t<S>::index_type>;
                                    {
                                      span.mapping().stride(axis)
                                    } -> std::convertible_to<typename detail::span_type_t<S>::index_type>;
                                  };

/// \concept MutableStridedDeviceMdspanLike
/// \brief Mutable device-mdspan-like type whose mapping is always strided.
/// \tparam S The device-mdspan-like type under test.
/// \ingroup mdspan_ext
template <class S>
concept MutableStridedDeviceMdspanLike = MutableDeviceMdspanLike<S> && StridedDeviceMdspanLike<S>;

/// \brief An mdspan-like type with a specified static rank.
/// \tparam MDS The mdspan-like type under test.
/// \tparam Rank Required rank.
/// \ingroup mdspan_ext
template <class MDS, std::size_t Rank>
concept RankedMdspanLike = MdspanLike<std::remove_cvref_t<MDS>> && (std::remove_cvref_t<MDS>::rank() == Rank);

/// \brief A device-mdspan-like type with a specified static rank.
/// \tparam S The device-mdspan-like type under test.
/// \tparam Rank Required rank.
/// \ingroup mdspan_ext
template <class S, std::size_t Rank>
concept RankedDeviceMdspanLike = DeviceMdspanLike<std::remove_cvref_t<S>> && (std::remove_cvref_t<S>::rank() == Rank);

/// \concept MutableRankedDeviceMdspanLike
/// \brief Mutable device-mdspan-like type with a specified static rank.
/// \tparam S The device-mdspan-like type under test.
/// \tparam Rank Required rank.
/// \ingroup mdspan_ext
template <class S, std::size_t Rank>
concept MutableRankedDeviceMdspanLike = MutableDeviceMdspanLike<S> && RankedDeviceMdspanLike<S, Rank>;

/// \brief A mutable mdspan-like type with a specified static rank.
/// \tparam MDS The mdspan-like type under test.
/// \tparam Rank Required rank.
/// \ingroup mdspan_ext
template <class MDS, std::size_t Rank>
concept MutableRankedMdspanLike =
    MutableMdspanLike<std::remove_cvref_t<MDS>> && (std::remove_cvref_t<MDS>::rank() == Rank);

/// \brief An mdspan-like type whose accessor is `stdex::default_accessor`.
/// \tparam MDS The mdspan-like type under test.
/// \ingroup mdspan_ext
template <class MDS>
concept DefaultAccessorMdspanLike =
    MdspanLike<std::remove_cvref_t<MDS>> && is_default_accessor_v<typename std::remove_cvref_t<MDS>::accessor_type>;

/// \brief A strided mdspan-like type with a specified static rank.
/// \tparam MDS The mdspan-like type under test.
/// \tparam Rank Required rank.
/// \ingroup mdspan_ext
template <class MDS, std::size_t Rank>
concept RankedStridedMdspanLike = RankedMdspanLike<MDS, Rank> && StridedMdspanLike<std::remove_cvref_t<MDS>>;

/// \brief A strided device-mdspan-like type with a specified static rank.
/// \tparam S The device-mdspan-like type under test.
/// \tparam Rank Required rank.
/// \ingroup mdspan_ext
template <class S, std::size_t Rank>
concept RankedStridedDeviceMdspanLike =
    RankedDeviceMdspanLike<S, Rank> && StridedDeviceMdspanLike<std::remove_cvref_t<S>>;

/// \concept MutableRankedStridedDeviceMdspanLike
/// \brief Mutable strided device-mdspan-like type with a specified static rank.
/// \tparam S The device-mdspan-like type under test.
/// \tparam Rank Required rank.
/// \ingroup mdspan_ext
template <class S, std::size_t Rank>
concept MutableRankedStridedDeviceMdspanLike =
    MutableRankedDeviceMdspanLike<S, Rank> && MutableStridedDeviceMdspanLike<std::remove_cvref_t<S>>;

/// \brief A mutable strided mdspan-like type with a specified static rank.
/// \tparam MDS The mdspan-like type under test.
/// \tparam Rank Required rank.
/// \ingroup mdspan_ext
template <class MDS, std::size_t Rank>
concept MutableRankedStridedMdspanLike =
    MutableRankedMdspanLike<MDS, Rank> && MutableStridedMdspanLike<std::remove_cvref_t<MDS>>;

namespace detail
{

/// \brief Helper that materializes strides from the layout mapping.
/// \tparam S The strided device-mdspan-like type.
/// \tparam I Index sequence selecting the stride positions.
/// \param s The multidimensional descriptor whose strides will be computed.
/// \return An array containing strides for each dimension in \c S.
/// \ingroup internal
template <StridedDeviceMdspanLike S, size_t... I> constexpr auto strides_impl(S const& s, std::index_sequence<I...>)
{
  using index_type = typename S::index_type;
  // fold the pack I... into an array by calling s.mapping().stride(I) for each I
  return std::array<index_type, sizeof...(I)>{s.mapping().stride(I)...};
}

} // namespace detail

/// \brief Retrieve the strides associated with a strided device-mdspan-like type.
/// \tparam S The strided device-mdspan-like type.
/// \param s The multidimensional descriptor whose strides will be returned.
/// \return A std::array containing the strides for each rank.
/// \ingroup mdspan_ext
template <StridedDeviceMdspanLike S> auto strides(S const& s)
{
  return detail::strides_impl(s, std::make_index_sequence<S::rank()>{});
}

} // namespace uni20
