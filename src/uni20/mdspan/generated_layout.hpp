#pragma once

/**
 * \file generated_layout.hpp
 * \ingroup mdspan_ext
 * \brief Synthetic mdspan layout for values generated from logical indices.
 */

#include <uni20/core/compiler_attributes.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace uni20
{

/// \brief Synthetic non-strided layout used by generated tensor accessors.
/// \details The mapping encodes a logical index as an implementation-only
///          offset. It does not describe physical storage order and deliberately
///          does not model a strided mapping.
struct GeneratedLayout
{
    template <class Extents> class mapping {
      public:
        using layout_type = GeneratedLayout;
        using extents_type = Extents;
        using index_type = typename extents_type::index_type;
        using size_type = typename extents_type::size_type;
        using rank_type = typename extents_type::rank_type;

        UNI20_HOST_DEVICE constexpr mapping() = default;

        /// \brief Construct a synthetic mapping over the supplied logical extents.
        UNI20_HOST_DEVICE explicit constexpr mapping(extents_type const& extents) noexcept : extents_(extents) {}

        /// \brief Return the generated value's logical extents.
        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto extents() const noexcept -> extents_type const&
        {
          return extents_;
        }

        /// \brief Return the number of synthetic offsets in the logical index space.
        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto required_span_size() const noexcept -> index_type
        {
          index_type result = 1;
          for (rank_type axis = 0; axis < extents_type::rank(); ++axis)
          {
            if (extents_.extent(axis) == 0) return 0;
            result *= extents_.extent(axis);
          }
          return result;
        }

        /// \brief Encode a logical index as an accessor offset.
        template <class... Indices>
          requires(sizeof...(Indices) == extents_type::rank() && (std::is_convertible_v<Indices, index_type> && ...))
        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Indices... indices) const noexcept -> index_type
        {
          std::array<index_type, extents_type::rank()> logical{static_cast<index_type>(indices)...};
          index_type offset = 0;
          for (rank_type axis = 0; axis < extents_type::rank(); ++axis)
            offset = offset * extents_.extent(axis) + logical[axis];
          return offset;
        }

        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_unique() noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_exhaustive() noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_strided() noexcept { return false; }
        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_unique() const noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_exhaustive() const noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_strided() const noexcept { return false; }

        friend UNI20_HOST_DEVICE constexpr bool operator==(mapping const&, mapping const&) = default;

      private:
        [[no_unique_address]] extents_type extents_{};
    };
};

} // namespace uni20
