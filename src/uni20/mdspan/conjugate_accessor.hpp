#pragma once

/**
 * \file conjugate_accessor.hpp
 * \ingroup mdspan_ext
 * \brief Mdspan accessor adaptor for lazy complex conjugation.
 */

#include <uni20/core/math.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/device_mdspan.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Trait used by mdspan accessors that present conjugated values on read.
template <class Accessor> struct accessor_applies_conjugation : std::false_type
{};

/// \brief True when an accessor presents conjugated values.
template <class Accessor>
inline constexpr bool accessor_applies_conjugation_v =
    accessor_applies_conjugation<std::remove_cvref_t<Accessor>>::value;

/// \brief True when an mdspan-like type's accessor presents conjugated values.
template <class Mdspan>
inline constexpr bool mdspan_needs_conjugation_v =
    accessor_applies_conjugation_v<typename std::remove_cvref_t<Mdspan>::accessor_type>;

/// \brief Read-only accessor that returns `uni20::conj` of the wrapped accessor value.
/// \details The adaptor preserves the wrapped accessor's data handle and offset policy, so
///          backend code can still recover the original storage pointer while seeing that the
///          accessor semantically conjugates values. The name follows the
///          C++26 `std::linalg::conjugated_accessor` terminology.
template <AccessorPolicy Accessor>
  requires uni20::Complex<std::remove_cv_t<typename Accessor::element_type>>
class conjugated_accessor {
  public:
    using wrapped_accessor_type = Accessor;
    using value_type = std::remove_cv_t<typename Accessor::element_type>;
    using element_type = value_type const;
    using reference = value_type;
    using data_handle_type = typename Accessor::data_handle_type;
    using offset_policy = conjugated_accessor;
    using offset_type = span_offset_t<Accessor>;

    constexpr conjugated_accessor()
      requires std::default_initializable<Accessor>
    = default;

    /// \brief Construct from the accessor whose values should be conjugated on read.
    constexpr explicit conjugated_accessor(Accessor accessor) : accessor_(std::move(accessor)) {}

    /// \brief Return the conjugated value at a handle-relative offset.
    [[nodiscard]] constexpr reference access(data_handle_type ptr, offset_type offset) const
    {
      return uni20::conj(accessor_.access(ptr, offset));
    }

    /// \brief Delegate handle offsetting to the wrapped accessor.
    [[nodiscard]] constexpr data_handle_type offset(data_handle_type ptr, offset_type offset) const
    {
      return accessor_.offset(ptr, offset);
    }

    /// \brief Return the wrapped accessor.
    [[nodiscard]] constexpr auto wrapped_accessor() const& -> Accessor const& { return accessor_; }

    /// \brief Move out the wrapped accessor.
    [[nodiscard]] constexpr auto wrapped_accessor() && -> Accessor { return std::move(accessor_); }

  private:
    [[no_unique_address]] Accessor accessor_{};
};

template <AccessorPolicy Accessor> struct accessor_applies_conjugation<conjugated_accessor<Accessor>> : std::true_type
{};

/// \brief Conjugation preserves the execution domains of the wrapped accessor.
template <AccessorPolicy Accessor, class Domain>
inline constexpr bool enable_accessor_in_domain<conjugated_accessor<Accessor>, Domain> =
    enable_accessor_in_domain<Accessor, Domain>;

/// \brief Return a read-only mdspan view that presents conjugated complex values.
template <MdspanLike Span>
  requires(uni20::Complex<std::remove_cv_t<typename Span::element_type>> &&
           !accessor_applies_conjugation_v<typename std::remove_cvref_t<Span>::accessor_type>)
[[nodiscard]] constexpr auto conj(Span const& span)
{
  auto accessor = conjugated_accessor{const_accessor(span.accessor())};
  using accessor_type = decltype(accessor);
  using extents_type = typename Span::extents_type;
  using layout_type = typename Span::layout_type;
  using element_type = typename accessor_type::element_type;

  return stdex::mdspan<element_type, extents_type, layout_type, accessor_type>{span.data_handle(), span.mapping(),
                                                                               std::move(accessor)};
}

/// \brief Return a read-only deferred view that presents conjugated complex values.
template <DeviceMdspanLike Span>
  requires(!MdspanLike<Span> && uni20::Complex<std::remove_cv_t<typename Span::element_type>> &&
           !accessor_applies_conjugation_v<typename std::remove_cvref_t<Span>::accessor_type>)
[[nodiscard]] constexpr auto conj(Span const& span)
{
  auto accessor = conjugated_accessor{const_accessor(span.accessor())};
  using accessor_type = decltype(accessor);
  using extents_type = typename Span::extents_type;
  using layout_type = typename Span::layout_type;
  using element_type = typename accessor_type::element_type;
  using descriptor_type = typename Span::data_descriptor_type;

  return device_mdspan<element_type, extents_type, layout_type, accessor_type, descriptor_type>{
      span.data_descriptor(), span.mapping(), std::move(accessor)};
}

/// \brief Cancel a conjugating mdspan accessor and return the underlying view.
template <MdspanLike Span>
  requires(uni20::Complex<std::remove_cv_t<typename Span::element_type>> &&
           accessor_applies_conjugation_v<typename std::remove_cvref_t<Span>::accessor_type>)
[[nodiscard]] constexpr auto conj(Span const& span)
{
  auto accessor = span.accessor().wrapped_accessor();
  using accessor_type = decltype(accessor);
  using extents_type = typename Span::extents_type;
  using layout_type = typename Span::layout_type;
  using element_type = typename accessor_type::element_type;

  return stdex::mdspan<element_type, extents_type, layout_type, accessor_type>{span.data_handle(), span.mapping(),
                                                                               std::move(accessor)};
}

/// \brief Cancel conjugation on a deferred multidimensional view.
template <DeviceMdspanLike Span>
  requires(!MdspanLike<Span> && uni20::Complex<std::remove_cv_t<typename Span::element_type>> &&
           accessor_applies_conjugation_v<typename std::remove_cvref_t<Span>::accessor_type>)
[[nodiscard]] constexpr auto conj(Span const& span)
{
  auto accessor = span.accessor().wrapped_accessor();
  using accessor_type = decltype(accessor);
  using extents_type = typename Span::extents_type;
  using layout_type = typename Span::layout_type;
  using element_type = typename accessor_type::element_type;
  using descriptor_type = typename Span::data_descriptor_type;

  return device_mdspan<element_type, extents_type, layout_type, accessor_type, descriptor_type>{
      span.data_descriptor(), span.mapping(), std::move(accessor)};
}

/// \brief Return a read-only identity view for non-complex mdspan-like views.
template <MdspanLike Span>
  requires(!uni20::Complex<std::remove_cv_t<typename Span::element_type>>)
[[nodiscard]] constexpr auto conj(Span const& span)
{
  auto accessor = const_accessor(span.accessor());
  using accessor_type = decltype(accessor);
  using extents_type = typename Span::extents_type;
  using layout_type = typename Span::layout_type;
  using element_type = typename accessor_type::element_type;

  return stdex::mdspan<element_type, extents_type, layout_type, accessor_type>{span.data_handle(), span.mapping(),
                                                                               std::move(accessor)};
}

/// \brief Return a read-only identity view for a non-complex deferred view.
template <DeviceMdspanLike Span>
  requires(!MdspanLike<Span> && !uni20::Complex<std::remove_cv_t<typename Span::element_type>>)
[[nodiscard]] constexpr auto conj(Span const& span)
{
  auto accessor = const_accessor(span.accessor());
  using accessor_type = decltype(accessor);
  using extents_type = typename Span::extents_type;
  using layout_type = typename Span::layout_type;
  using element_type = typename accessor_type::element_type;
  using descriptor_type = typename Span::data_descriptor_type;

  return device_mdspan<element_type, extents_type, layout_type, accessor_type, descriptor_type>{
      span.data_descriptor(), span.mapping(), std::move(accessor)};
}

} // namespace uni20
