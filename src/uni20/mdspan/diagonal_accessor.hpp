/**
 * \file diagonal_accessor.hpp
 * \ingroup mdspan_ext
 * \brief Accessor support for compressed generalized diagonal tensors.
 */

#pragma once

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/core/types.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/generated_layout.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Handle for one compressed generalized diagonal value sequence.
/// \tparam ElementType Stored element type, possibly const-qualified.
template <class ElementType> struct diagonal_data_handle
{
    ElementType* data = nullptr;
    std::size_t base_offset = 0;

    UNI20_HOST_DEVICE constexpr diagonal_data_handle() noexcept = default;

    UNI20_HOST_DEVICE constexpr diagonal_data_handle(ElementType* data, std::size_t base_offset) noexcept
        : data(data), base_offset(base_offset)
    {}

    template <class OtherElementType>
      requires std::convertible_to<OtherElementType*, ElementType*>
    UNI20_HOST_DEVICE constexpr diagonal_data_handle(diagonal_data_handle<OtherElementType> const& other) noexcept
        : data(other.data), base_offset(other.base_offset)
    {}

    friend UNI20_HOST_DEVICE constexpr bool operator==(diagonal_data_handle const&,
                                                       diagonal_data_handle const&) = default;
};

/// \brief Writable proxy for one structural diagonal or off-diagonal element.
/// \details Reading an off-diagonal proxy returns zero. Assigning an
///          off-diagonal proxy has the precondition that the assigned value is
///          zero, preserving the compressed diagonal representation.
/// \tparam ElementType Stored non-const element type.
template <class ElementType> class diagonal_reference {
  public:
    static_assert(!std::is_const_v<ElementType>);

    UNI20_HOST_DEVICE constexpr diagonal_reference() noexcept = default;
    UNI20_HOST_DEVICE constexpr explicit diagonal_reference(ElementType* element) noexcept : element_(element) {}

    [[nodiscard]] UNI20_HOST_DEVICE constexpr operator ElementType() const noexcept
    {
      return element_ == nullptr ? ElementType{} : *element_;
    }

    UNI20_HOST_DEVICE constexpr auto operator=(ElementType const& value) noexcept -> diagonal_reference&
    {
      if (element_ == nullptr)
      {
        assert(value == ElementType{});
        return *this;
      }
      *element_ = value;
      return *this;
    }

    UNI20_HOST_DEVICE constexpr auto operator=(diagonal_reference const& other) noexcept -> diagonal_reference&
    {
      return *this = static_cast<ElementType>(other);
    }

  private:
    ElementType* element_ = nullptr;
};

/// \brief Mdspan accessor over a compressed rank-N generalized diagonal.
/// \details The associated mapping encodes logical tensor indices as offsets.
///          This accessor decodes those offsets and addresses the compressed
///          value only when every logical index is equal. Rectangular tensors
///          store `min(extents...)` values.
/// \tparam ElementType Presented element type, possibly const-qualified.
/// \tparam Extents Logical tensor extents used to decode mapping offsets.
template <class ElementType, class Extents> class diagonal_accessor {
  public:
    using value_type = std::remove_cv_t<ElementType>;
    using element_type = ElementType;
    using reference = std::conditional_t<std::is_const_v<element_type>, value_type, diagonal_reference<value_type>>;
    using data_handle_type = diagonal_data_handle<element_type>;
    using offset_policy = diagonal_accessor;
    using offset_type = std::size_t;
    using extents_type = Extents;
    using index_type = typename extents_type::index_type;

    /// \brief Decoded component coordinate for one logical mapping offset.
    struct component_index_result
    {
        std::size_t index = 0;
        bool present = true;
    };

    UNI20_HOST_DEVICE constexpr diagonal_accessor()
      requires std::default_initializable<extents_type>
    = default;

    /// \brief Construct an accessor for one logical index space and component stride.
    /// \param extents Full logical tensor extents.
    /// \param component_stride Offset between consecutive stored diagonal components.
    UNI20_HOST_DEVICE explicit constexpr diagonal_accessor(extents_type extents, index_type component_stride = 1)
        : extents_(std::move(extents)), component_stride_(component_stride)
    {}

    /// \brief Read or write the logical element represented by an encoded offset.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference access(data_handle_type handle, offset_type offset) const
    {
      auto const diagonal_index = this->component_index(handle, offset);
      auto const component_offset = static_cast<index_type>(diagonal_index.index) * component_stride_;
      if constexpr (std::is_const_v<element_type>)
      {
        return diagonal_index.present ? handle.data[component_offset] : value_type{};
      }
      else
      {
        return reference{diagonal_index.present ? handle.data + component_offset : nullptr};
      }
    }

    /// \brief Advance a logical handle without changing its original index space.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr data_handle_type offset(data_handle_type handle,
                                                                      offset_type offset) const noexcept
    {
      handle.base_offset += offset;
      return handle;
    }

    /// \brief Return the extents used to decode logical offsets.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto extents() const noexcept -> extents_type const& { return extents_; }

    /// \brief Return the physical offset between consecutive diagonal components.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto component_stride() const noexcept -> index_type
    {
      return component_stride_;
    }

    /// \brief Decode whether a logical offset addresses a stored component and which one.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto
    component_index(data_handle_type handle, offset_type offset = 0) const noexcept -> component_index_result
    {
      return this->diagonal_index(handle.base_offset + offset);
    }

  private:
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto
    diagonal_index(std::size_t linear) const noexcept -> component_index_result
    {
      if constexpr (extents_type::rank() == 0)
      {
        return {};
      }
      else
      {
        std::array<index_type, extents_type::rank()> indices{};
        for (std::size_t axis = extents_type::rank(); axis > 0; --axis)
        {
          auto const extent = static_cast<std::size_t>(extents_.extent(axis - 1));
          indices[axis - 1] = static_cast<index_type>(linear % extent);
          linear /= extent;
        }
        auto const first = indices[0];
        for (std::size_t axis = 1; axis < extents_type::rank(); ++axis)
          if (indices[axis] != first) return {.present = false};
        return {.index = static_cast<std::size_t>(first), .present = true};
      }
    }

    [[no_unique_address]] extents_type extents_{};
    index_type component_stride_ = 1;
};

/// \brief Convert a writable diagonal accessor to its read-only form.
template <class ElementType, class Extents>
  requires(!std::is_const_v<ElementType>)
[[nodiscard]] UNI20_HOST_DEVICE constexpr auto const_accessor(diagonal_accessor<ElementType, Extents> const& accessor)
{
  return diagonal_accessor<ElementType const, Extents>{accessor.extents(), accessor.component_stride()};
}

/// \brief Return an already read-only diagonal accessor unchanged.
template <class ElementType, class Extents>
  requires std::is_const_v<ElementType>
[[nodiscard]] UNI20_HOST_DEVICE constexpr auto const_accessor(diagonal_accessor<ElementType, Extents> const& accessor)
{
  return accessor;
}

/// \brief Diagonal accessors may be evaluated by ordinary host code.
template <class ElementType, class Extents>
inline constexpr bool enable_accessor_in_domain<diagonal_accessor<ElementType, Extents>, host_access_domain> = true;

namespace detail
{

template <class Extents>
[[nodiscard]] UNI20_HOST_DEVICE constexpr auto diagonal_component_extent(Extents const& extents) noexcept ->
    typename Extents::index_type
{
  using index_type = typename Extents::index_type;
  if constexpr (Extents::rank() == 0)
  {
    return index_type{1};
  }
  else
  {
    index_type result = extents.extent(0);
    for (std::size_t axis = 1; axis < Extents::rank(); ++axis)
      if (extents.extent(axis) < result) result = extents.extent(axis);
    return result;
  }
}

template <class Mapping, std::size_t... Axis>
[[nodiscard]] constexpr auto diagonal_mapping_step(Mapping const& mapping, std::index_sequence<Axis...>)
{
  using index_type = typename Mapping::extents_type::index_type;
  return mapping(((void)Axis, index_type{1})...) - mapping(((void)Axis, index_type{0})...);
}

} // namespace detail

/// \brief Expose the physically stored values of a generalized-diagonal mdspan.
/// \details The returned rank-one mdspan preserves the component origin and
///          stride of aligned offset or stepped views. The source remains the
///          full logical rank-N mdspan whose off-diagonal accesses observe zero.
/// \tparam ElementType Presented component type, possibly const-qualified.
/// \tparam Extents Full logical tensor extents.
/// \tparam LayoutPolicy Logical mdspan layout policy.
/// \tparam AccessorExtents Original extents used by the diagonal accessor to decode offsets.
/// \param span Full generalized-diagonal mdspan.
/// \return Rank-one strided mdspan over the stored diagonal components.
/// \throws std::invalid_argument If the logical view does not begin on or
///         advance along the represented diagonal.
template <class ElementType, class Extents, class LayoutPolicy, class AccessorExtents>
  requires(Extents::rank() == AccessorExtents::rank())
[[nodiscard]] constexpr auto diagonal_components(
    stdex::mdspan<ElementType, Extents, LayoutPolicy, diagonal_accessor<ElementType, AccessorExtents>> const& span)
{
  using index_type = typename Extents::index_type;
  using component_extents_type = stdex::dextents<index_type, 1>;
  using component_mapping_type = stdex::layout_stride::mapping<component_extents_type>;
  using component_mdspan_type = stdex::mdspan<ElementType, component_extents_type, stdex::layout_stride>;
  auto const extents = component_extents_type{detail::diagonal_component_extent(span.extents())};
  index_type component_stride = span.accessor().component_stride();
  auto* data = span.data_handle().data;
  if (extents.extent(0) > 0)
  {
    auto const origin = span.accessor().component_index(span.data_handle());
    if (!origin.present)
      throw std::invalid_argument("diagonal component view does not begin on the represented diagonal");
    data += static_cast<index_type>(origin.index) * component_stride;
    if (extents.extent(0) > 1)
    {
      auto const step = detail::diagonal_mapping_step(span.mapping(), std::make_index_sequence<Extents::rank()>{});
      if constexpr (std::signed_integral<std::remove_cv_t<decltype(step)>>)
        if (step < 0) throw std::invalid_argument("diagonal component view has a negative logical step");
      auto const next = span.accessor().component_index(span.data_handle(), static_cast<std::size_t>(step));
      if (!next.present || next.index < origin.index)
        throw std::invalid_argument("diagonal component view does not preserve the represented diagonal");
      component_stride *= static_cast<index_type>(next.index - origin.index);
    }
  }
  auto const mapping = component_mapping_type{extents, std::array<index_type, 1>{component_stride}};
  return component_mdspan_type{data, mapping};
}

/// \brief Present rank-one strided components as a full generalized-diagonal mdspan.
/// \details The resulting mdspan has the supplied logical rank-N extents.
///          It reads as zero unless every logical index is equal and delegates
///          diagonal values to the supplied component view.
/// \tparam Extents Full logical extents type.
/// \tparam Components Default-accessor rank-one strided mdspan type.
/// \param logical_extents Full logical tensor extents.
/// \param components Stored component view with `min(logical_extents)` entries.
/// \return Full logical mdspan modelling `DiagonalMdspecLike`.
/// \throws std::invalid_argument If the component extent does not match the logical diagonal extent.
template <class Extents, class Components>
  requires RankedStridedMdspanLike<Components, 1> && DefaultAccessorMdspanLike<Components>
[[nodiscard]] constexpr auto make_diagonal_mdspan(Extents logical_extents, Components const& components)
{
  using component_type = std::remove_cvref_t<Components>;
  using element_type = typename component_type::element_type;
  using index_type = typename Extents::index_type;
  if (static_cast<index_type>(components.extent(0)) != detail::diagonal_component_extent(logical_extents))
  {
    throw std::invalid_argument("diagonal component extent does not match the logical tensor extents");
  }
  using accessor_type = diagonal_accessor<element_type, Extents>;
  using result_type = stdex::mdspan<element_type, Extents, GeneratedLayout, accessor_type>;
  return result_type{typename accessor_type::data_handle_type{components.data_handle(), 0},
                     typename result_type::mapping_type{logical_extents},
                     accessor_type{std::move(logical_extents), static_cast<index_type>(components.stride(0))}};
}

/// \brief Recover the stored value type from a writable diagonal proxy.
template <class ElementType> struct remove_proxy_reference<diagonal_reference<ElementType>>
{
    using type = ElementType;
};

} // namespace uni20
