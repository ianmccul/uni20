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

/// \brief Opaque logical block-selection coordinates in boundary order.
/// \details Only boundary spaces with a selectable structural choice contribute
///          a coordinate. Fixed `QNumSpace` and `DenseSpace` factors do not.
/// \tparam CoordinateCount Number of stored block-selection coordinates.
template <std::size_t CoordinateCount> class BlockKey {
  public:
    /// \brief Construct the all-zero coordinate.
    constexpr BlockKey() = default;

    /// \brief Construct from one coordinate per selectable boundary factor.
    /// \param coordinates Coordinate-bearing domain factors followed by
    ///                    coordinate-bearing codomain factors.
    constexpr explicit BlockKey(std::array<std::size_t, CoordinateCount> coordinates) : coordinates_(coordinates) {}

    /// \brief Return the compile-time number of stored coordinates.
    static constexpr auto size() noexcept -> std::size_t { return CoordinateCount; }

    /// \brief Return one factor coordinate.
    /// \param axis Position among coordinate-bearing factors in filtered
    ///             domain-then-codomain order.
    constexpr auto coordinate(std::size_t axis) const -> std::size_t { return coordinates_[axis]; }

    /// \brief Compare block coordinates lexicographically.
    auto operator<=>(BlockKey const&) const = default;

  private:
    std::array<std::size_t, CoordinateCount> coordinates_{};
};

} // namespace uni20
