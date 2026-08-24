/**
 * \file block_sector.hpp
 * \ingroup symmetry
 * \brief Defines one quantum-number sector and its degeneracy dimension.
 */

#pragma once

#include <uni20/symmetry/qnum.hpp>

#include <cstddef>

namespace uni20
{

/// \brief One symmetry sector inside a block-structured space.
struct BlockSector
{
    /// \brief Quantum number carried by the block.
    QNum q;

    /// \brief Degeneracy / number of states in the block.
    std::size_t dim;

    /// \brief Compare two block sectors for exact equality.
    /// \param other Other block sector.
    /// \return `true` if both quantum number and dimension match.
    auto operator==(BlockSector const& other) const -> bool = default;
};

} // namespace uni20
