/**
 * \file block_key.hpp
 * \ingroup symmetry
 * \brief Defines fixed-order logical block coordinates.
 */

#pragma once

#include <array>
#include <compare>
#include <cstddef>

namespace uni20
{

/// \brief Opaque logical block coordinate in domain-then-codomain factor order.
/// \tparam Order Number of tensor boundary factors.
template <std::size_t Order> class BlockKey {
  public:
    /// \brief Construct the all-zero coordinate.
    constexpr BlockKey() = default;

    /// \brief Construct from one coordinate per boundary factor.
    /// \param coordinates Domain coordinates followed by codomain coordinates.
    constexpr explicit BlockKey(std::array<std::size_t, Order> coordinates) : coordinates_(coordinates) {}

    /// \brief Return the compile-time coordinate count.
    static constexpr auto order() noexcept -> std::size_t { return Order; }

    /// \brief Return one factor coordinate.
    /// \param axis Boundary-factor position in domain-then-codomain order.
    constexpr auto coordinate(std::size_t axis) const -> std::size_t { return coordinates_[axis]; }

    /// \brief Compare block coordinates lexicographically.
    auto operator<=>(BlockKey const&) const = default;

  private:
    std::array<std::size_t, Order> coordinates_{};
};

} // namespace uni20
