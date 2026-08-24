/**
 * \file irregular_space.hpp
 * \ingroup symmetry
 * \brief Defines immutable ordered block spaces with repeated sectors.
 */

#pragma once

#include <uni20/symmetry/block_sector.hpp>
#include <uni20/symmetry/space.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Ordered block space whose quantum-number sectors may repeat.
/// \details `IrregularSpace` preserves construction order and segmentation.
///          Its structure is immutable, while its semantic leg label is mutable.
class IrregularSpace {
  public:
    /// \brief Construct an empty irregular space tied to one symmetry.
    /// \param sym Symmetry carried by every block.
    /// \param label Semantic leg label.
    explicit IrregularSpace(Symmetry sym, std::string label = {}) : sym_(sym), label_(std::move(label))
    {
      this->verify_symmetry();
    }

    /// \brief Construct an irregular space from an ordered block list.
    /// \param sym Symmetry carried by every block.
    /// \param sectors Ordered blocks; repeated quantum numbers are preserved.
    /// \param label Semantic leg label.
    IrregularSpace(Symmetry sym, std::initializer_list<BlockSector> sectors, std::string label = {})
        : IrregularSpace(sym, std::move(label))
    {
      this->append_sectors(sectors);
    }

    /// \brief Construct an irregular space from an input range.
    /// \tparam Sectors Input range whose elements construct `BlockSector`.
    /// \param sym Symmetry carried by every block.
    /// \param sectors Ordered blocks; repeated quantum numbers are preserved.
    /// \param label Semantic leg label.
    template <std::ranges::input_range Sectors>
      requires std::constructible_from<BlockSector, std::ranges::range_reference_t<Sectors>>
    IrregularSpace(Symmetry sym, Sectors&& sectors, std::string label = {}) : IrregularSpace(sym, std::move(label))
    {
      this->append_sectors(std::forward<Sectors>(sectors));
    }

    /// \brief Return the symmetry shared by all blocks.
    /// \return Irregular-space symmetry.
    auto symmetry() const -> Symmetry { return sym_; }

    /// \brief Return the number of stored blocks.
    /// \return Block count.
    auto size() const -> std::size_t { return sectors_.size(); }

    /// \brief Return whether the space contains no blocks.
    /// \return `true` when the ordered block list is empty.
    auto empty() const -> bool { return sectors_.empty(); }

    /// \brief Return the total dimension across all blocks.
    /// \return Sum of all block dimensions.
    auto total_dim() const -> std::size_t
    {
      return std::accumulate(sectors_.begin(), sectors_.end(), std::size_t{0},
                             [](std::size_t total, BlockSector const& sector) { return total + sector.dim; });
    }

    /// \brief Return whether any block carries the requested quantum number.
    /// \param q Quantum number to search for.
    /// \return `true` when at least one matching block exists.
    auto contains(QNum q) const -> bool
    {
      this->verify_symmetry(q);
      return std::ranges::find(sectors_, q, &BlockSector::q) != sectors_.end();
    }

    /// \brief Return indexed block access.
    /// \param index Zero-based block index.
    /// \return Block at the requested position.
    auto operator[](std::size_t index) const -> BlockSector const& { return sectors_[index]; }

    /// \brief Return the ordered block list.
    /// \return Read-only span over the blocks.
    auto sectors() const -> std::span<BlockSector const> { return sectors_; }

    /// \brief Return an iterator to the first block.
    /// \return Begin iterator.
    auto begin() const { return sectors_.begin(); }

    /// \brief Return an iterator past the last block.
    /// \return End iterator.
    auto end() const { return sectors_.end(); }

    /// \brief Return the semantic leg label.
    /// \return Label, which may be empty.
    auto label() const -> std::string const& { return label_; }

    /// \brief Replace the semantic leg label without changing block structure.
    /// \param label New label.
    void set_label(std::string label) { label_ = std::move(label); }

    /// \brief Compare symmetry, ordered segmented blocks, and label.
    /// \param other Other irregular space.
    /// \return `true` when structure and label are equal.
    auto operator==(IrregularSpace const& other) const -> bool = default;

  private:
    /// \brief Throw if the required symmetry handle is invalid.
    void verify_symmetry() const
    {
      if (!sym_.valid())
      {
        throw std::invalid_argument("IrregularSpace requires a valid symmetry");
      }
    }

    /// \brief Throw if a quantum number does not match this space's symmetry.
    /// \param q Quantum number to validate.
    void verify_symmetry(QNum const& q) const
    {
      if (!q.valid() || q.symmetry() != sym_)
      {
        throw std::invalid_argument("IrregularSpace block has the wrong symmetry");
      }
    }

    /// \brief Append construction-time blocks after validating them.
    /// \tparam Sectors Input range whose elements construct `BlockSector`.
    /// \param sectors Ordered block range.
    template <std::ranges::input_range Sectors> void append_sectors(Sectors&& sectors)
    {
      for (auto&& input : sectors)
      {
        BlockSector sector(std::forward<decltype(input)>(input));
        this->verify_symmetry(sector.q);
        if (sector.dim == 0)
        {
          throw std::invalid_argument("IrregularSpace blocks must have positive dimension");
        }
        sectors_.push_back(std::move(sector));
      }
    }

    Symmetry sym_;
    std::vector<BlockSector> sectors_;
    std::string label_;
};

static_assert(SymmetrySpace<IrregularSpace>);

} // namespace uni20
