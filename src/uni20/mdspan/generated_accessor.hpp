#pragma once

/**
 * \file generated_accessor.hpp
 * \ingroup mdspan_ext
 * \brief Read-only mdspan accessor for values generated from logical indices.
 */

#include <uni20/mdspan/concepts.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Compact mdspan handle identifying a base offset in a generated value sequence.
struct generated_data_handle
{
    std::size_t base_offset = 0;

    friend constexpr bool operator==(generated_data_handle const&, generated_data_handle const&) = default;
};

/// \brief Accessor that evaluates a generator at the logical index encoded by an offset.
/// \tparam ElementType Scalar value returned by the generator.
/// \tparam Extents Mdspan extents used to decode linear offsets.
/// \tparam Generator Callable accepting an index array and returning an element value.
template <class ElementType, class Extents, class Generator> class generated_accessor {
  public:
    using value_type = std::remove_cv_t<ElementType>;
    using element_type = value_type const;
    using reference = value_type;
    using data_handle_type = generated_data_handle;
    using offset_policy = generated_accessor;
    using offset_type = std::size_t;
    using extents_type = Extents;
    using generator_type = Generator;
    using index_type = typename extents_type::index_type;

    constexpr generated_accessor()
      requires(std::default_initializable<extents_type> && std::default_initializable<generator_type>)
    = default;

    /// \brief Construct an accessor from the original extents and generator state.
    constexpr generated_accessor(extents_type extents, generator_type generator)
        : extents_(std::move(extents)), generator_(std::move(generator))
    {}

    /// \brief Generate the value at a handle-relative synthetic offset.
    [[nodiscard]] constexpr reference access(data_handle_type handle, offset_type offset) const
    {
      return generator_(this->decode(handle.base_offset + offset));
    }

    /// \brief Advance a generated handle while preserving its original index space.
    [[nodiscard]] constexpr data_handle_type offset(data_handle_type handle, offset_type offset) const noexcept
    {
      return data_handle_type{handle.base_offset + offset};
    }

    /// \brief Return the extents used to decode generated offsets.
    [[nodiscard]] constexpr auto extents() const noexcept -> extents_type const& { return extents_; }

    /// \brief Return the stored generator state.
    [[nodiscard]] constexpr auto generator() const noexcept -> generator_type const& { return generator_; }

  private:
    [[nodiscard]] constexpr auto decode(std::size_t linear) const -> std::array<index_type, extents_type::rank()>
    {
      std::array<index_type, extents_type::rank()> indices{};
      for (std::size_t axis = extents_type::rank(); axis > 0; --axis)
      {
        auto const extent = static_cast<std::size_t>(extents_.extent(axis - 1));
        indices[axis - 1] = static_cast<index_type>(linear % extent);
        linear /= extent;
      }
      return indices;
    }

    [[no_unique_address]] extents_type extents_{};
    [[no_unique_address]] generator_type generator_{};
};

} // namespace uni20
