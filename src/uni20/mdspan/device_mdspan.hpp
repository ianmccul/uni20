#pragma once

/**
 * \file device_mdspan.hpp
 * \ingroup mdspan_ext
 * \brief Mdspan-shaped metadata for storage whose data handle requires acquisition.
 */

#include <uni20/mdspan/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Store a data descriptor with the mapping and accessor for an eventual mdspan.
/// \details Unlike an mdspan, this type has no data-handle value and provides no
///          element indexing. Domain-specific access acquisition uses
///          `data_descriptor()`, then combines the acquired handle with
///          `mapping()` and `accessor()` to construct a genuine mdspan.
/// \tparam ElementType Element type presented by the eventual mdspan.
/// \tparam Extents Extents type of the eventual mdspan.
/// \tparam LayoutPolicy Layout policy whose mapping is stored.
/// \tparam Accessor Accessor policy stored for the eventual mdspan.
/// \tparam DataDescriptor Descriptor identifying the region from which a handle can be acquired.
template <class ElementType, class Extents, class LayoutPolicy, AccessorPolicy Accessor, class DataDescriptor>
  requires std::same_as<ElementType, typename Accessor::element_type>
class device_mdspan {
  public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using extents_type = Extents;
    using layout_type = LayoutPolicy;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using accessor_type = Accessor;
    using data_descriptor_type = DataDescriptor;
    using data_handle_type = typename accessor_type::data_handle_type;
    using reference = typename accessor_type::reference;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;

    /// \brief Construct unresolved mdspan metadata from its three stored components.
    /// \param descriptor Descriptor used by later access acquisition.
    /// \param mapping Mapping installed in the eventual mdspan.
    /// \param accessor Accessor installed in the eventual mdspan.
    constexpr explicit device_mdspan(data_descriptor_type descriptor, mapping_type mapping, accessor_type accessor)
        : descriptor_(std::move(descriptor)), mapping_(std::move(mapping)), accessor_(std::move(accessor))
    {}

    /// \brief Return the static rank.
    [[nodiscard]] static constexpr rank_type rank() noexcept { return extents_type::rank(); }

    /// \brief Return the number of dynamic extents.
    [[nodiscard]] static constexpr rank_type rank_dynamic() noexcept { return extents_type::rank_dynamic(); }

    /// \brief Return the compile-time extent for one axis.
    /// \param axis Axis whose static extent is requested.
    [[nodiscard]] static constexpr std::size_t static_extent(rank_type axis) noexcept
    {
      return extents_type::static_extent(axis);
    }

    /// \brief Report whether every mapping of this type is unique.
    [[nodiscard]] static constexpr bool is_always_unique() noexcept { return mapping_type::is_always_unique(); }

    /// \brief Report whether every mapping of this type is exhaustive.
    [[nodiscard]] static constexpr bool is_always_exhaustive() noexcept { return mapping_type::is_always_exhaustive(); }

    /// \brief Report whether every mapping of this type is strided.
    [[nodiscard]] static constexpr bool is_always_strided() noexcept { return mapping_type::is_always_strided(); }

    /// \brief Return the descriptor used to acquire a data handle.
    [[nodiscard]] constexpr auto data_descriptor() const noexcept -> data_descriptor_type const& { return descriptor_; }

    /// \brief Return the stored layout mapping.
    [[nodiscard]] constexpr auto mapping() const noexcept -> mapping_type const& { return mapping_; }

    /// \brief Return the stored accessor policy.
    [[nodiscard]] constexpr auto accessor() const noexcept -> accessor_type const& { return accessor_; }

    /// \brief Return the stored extents.
    [[nodiscard]] constexpr auto extents() const noexcept -> extents_type const& { return mapping_.extents(); }

    /// \brief Return the runtime extent of one axis.
    /// \param axis Axis whose extent is requested.
    [[nodiscard]] constexpr index_type extent(rank_type axis) const noexcept { return this->extents().extent(axis); }

    /// \brief Report whether this mapping is unique.
    [[nodiscard]] constexpr bool is_unique() const noexcept { return mapping_.is_unique(); }

    /// \brief Report whether this mapping is exhaustive.
    [[nodiscard]] constexpr bool is_exhaustive() const noexcept { return mapping_.is_exhaustive(); }

    /// \brief Report whether this mapping is strided.
    [[nodiscard]] constexpr bool is_strided() const noexcept { return mapping_.is_strided(); }

    /// \brief Return the stride for one axis when the mapping exposes strides.
    /// \param axis Axis whose stride is requested.
    [[nodiscard]] constexpr index_type stride(rank_type axis) const
      requires requires(mapping_type const& mapping) {
        { mapping.stride(axis) } -> std::convertible_to<index_type>;
      }
    {
      return mapping_.stride(axis);
    }

  private:
    [[no_unique_address]] data_descriptor_type descriptor_;
    [[no_unique_address]] mapping_type mapping_;
    [[no_unique_address]] accessor_type accessor_;
};

} // namespace uni20
