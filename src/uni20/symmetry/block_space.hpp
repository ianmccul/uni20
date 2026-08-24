/**
 * \file block_space.hpp
 * \ingroup symmetry
 * \brief Defines immutable block-structured tensor spaces and regularization helpers.
 *
 * \details See `docs/symmetry/qnum.md` for the current `BlockSpace` and `QNumList` tensor-space API and
 *          how they fit into the planned DMRG integration work.
 */

#pragma once

#include <uni20/symmetry/block_sector.hpp>
#include <uni20/symmetry/irregular_space.hpp>
#include <uni20/symmetry/local_space.hpp>
#include <uni20/symmetry/qnum.hpp>
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Immutable coalesced tensor space represented as unique symmetry blocks.
/// \details Sector structure is canonicalized at construction. The semantic
///          leg label remains mutable and participates in exact equality.
class BlockSpace {
  public:
    /// \brief Construct an empty block space tied to one symmetry.
    /// \param sym Symmetry carried by every block.
    /// \param label Semantic leg label.
    explicit BlockSpace(Symmetry sym, std::string label = {}) : sym_(sym), label_(std::move(label))
    {
      this->verify_symmetry();
    }

    /// \brief Construct a block space from initial blocks.
    /// \param sym Symmetry carried by every block.
    /// \param sectors Initial block list.
    /// \param label Semantic leg label.
    BlockSpace(Symmetry sym, std::initializer_list<BlockSector> sectors, std::string label = {})
        : BlockSpace(sym, std::move(label))
    {
      this->append_sectors(sectors);
      this->canonicalize_sectors();
    }

    /// \brief Construct a block space from an input range.
    /// \tparam Sectors Input range whose elements construct `BlockSector`.
    /// \param sym Symmetry carried by every block.
    /// \param sectors Initial block range.
    /// \param label Semantic leg label.
    template <std::ranges::input_range Sectors>
      requires std::constructible_from<BlockSector, std::ranges::range_reference_t<Sectors>>
    BlockSpace(Symmetry sym, Sectors&& sectors, std::string label = {}) : BlockSpace(sym, std::move(label))
    {
      this->append_sectors(std::forward<Sectors>(sectors));
      this->canonicalize_sectors();
    }

    /// \brief Return the symmetry shared by all blocks.
    /// \return Block-space symmetry.
    auto symmetry() const -> Symmetry { return sym_; }

    /// \brief Return the number of stored blocks.
    /// \return Block count.
    auto size() const -> std::size_t { return sectors_.size(); }

    /// \brief Return whether the block space contains no blocks.
    /// \return `true` if there are no blocks.
    auto empty() const -> bool { return sectors_.empty(); }

    /// \brief Return the total dimension across all blocks.
    /// \return Sum of all block dimensions.
    auto total_dim() const -> std::size_t
    {
      return std::accumulate(sectors_.begin(), sectors_.end(), std::size_t{0},
                             [](std::size_t total, BlockSector const& sector) { return total + sector.dim; });
    }

    /// \brief Return whether a block with the given quantum number exists.
    /// \param q Quantum number to search for.
    /// \return `true` if a matching block exists.
    auto contains(QNum q) const -> bool
    {
      this->verify_symmetry(q);
      return std::find_if(sectors_.begin(), sectors_.end(), [&](BlockSector const& sector) { return sector.q == q; }) !=
             sectors_.end();
    }

    /// \brief Return indexed block access.
    /// \param index Zero-based block index.
    /// \return Block at the requested position.
    auto operator[](std::size_t index) const -> BlockSector const& { return sectors_[index]; }

    /// \brief Return the block list as a read-only span.
    /// \return Read-only block span.
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

    /// \brief Replace the semantic leg label without changing sector structure.
    /// \param label New label.
    void set_label(std::string label) { label_ = std::move(label); }

    /// \brief Compare two block spaces for exact equality.
    /// \param other Other block space.
    /// \return `true` if symmetry, canonical sectors, and label match.
    auto operator==(BlockSpace const& other) const -> bool = default;

  private:
    /// \brief Throw if the required symmetry handle is invalid.
    void verify_symmetry() const
    {
      if (!sym_.valid())
      {
        throw std::invalid_argument("BlockSpace requires a valid symmetry");
      }
    }

    /// \brief Throw if a quantum number does not match the block-space symmetry.
    /// \param q Quantum number to validate.
    void verify_symmetry(QNum const& q) const
    {
      if (!q.valid() || q.symmetry() != sym_)
      {
        throw std::invalid_argument("BlockSpace sector has the wrong symmetry");
      }
    }

    /// \brief Throw if a block is invalid for this space.
    /// \param sector Block to validate.
    void verify_sector(BlockSector const& sector) const
    {
      this->verify_symmetry(sector.q);
      if (sector.dim == 0)
      {
        throw std::invalid_argument("BlockSpace sectors must have positive dimension");
      }
    }

    /// \brief Append construction-time sectors after validating them.
    /// \tparam Sectors Input range whose elements construct `BlockSector`.
    /// \param sectors Sector range.
    template <std::ranges::input_range Sectors> void append_sectors(Sectors&& sectors)
    {
      for (auto&& input : sectors)
      {
        BlockSector sector(std::forward<decltype(input)>(input));
        this->verify_sector(sector);
        sectors_.push_back(std::move(sector));
      }
    }

    /// \brief Sort sectors canonically and reject repeated quantum numbers.
    void canonicalize_sectors()
    {
      std::sort(sectors_.begin(), sectors_.end(),
                [](BlockSector const& lhs, BlockSector const& rhs) { return lhs.q.raw_code() < rhs.q.raw_code(); });

      auto const duplicate =
          std::adjacent_find(sectors_.begin(), sectors_.end(),
                             [](BlockSector const& lhs, BlockSector const& rhs) { return lhs.q == rhs.q; });
      if (duplicate != sectors_.end())
      {
        throw std::invalid_argument("BlockSpace sectors must have unique quantum numbers");
      }
    }

    Symmetry sym_;
    std::vector<BlockSector> sectors_;
    std::string label_;
};

static_assert(SymmetrySpace<BlockSpace>);

/// \brief Result of regularizing a sparse `QNumList` into a coalesced `BlockSpace`.
struct QNumListRegularization
{
    /// \brief Original sparse space.
    QNumList original;

    /// \brief Coalesced block space with one block per quantum number.
    BlockSpace regular;

    /// \brief For each sparse-space entry, index of the containing block in `regular`.
    std::vector<std::size_t> block_index;

    /// \brief For each sparse-space entry, offset inside the containing block.
    std::vector<std::size_t> block_offset;
};

/// \brief Result of regularizing an immutable `LocalSpace`.
struct LocalSpaceRegularization
{
    /// \brief Original ordered local space.
    LocalSpace original;

    /// \brief Coalesced block space with one block per quantum number.
    BlockSpace regular;

    /// \brief For each local state, index of the containing block in `regular`.
    std::vector<std::size_t> block_index;

    /// \brief For each local state, offset inside the containing block.
    std::vector<std::size_t> block_offset;
};

/// \brief Destination range of one irregular block after regularization.
struct BlockRangeMapping
{
    /// \brief Index of the containing block in the regular `BlockSpace`.
    std::size_t block_index;

    /// \brief First offset occupied by the original block.
    std::size_t first;

    /// \brief One-past-the-end offset occupied by the original block.
    std::size_t last;

    /// \brief Return the number of mapped degeneracy states.
    /// \return Size of the half-open destination range.
    auto size() const -> std::size_t { return last - first; }

    /// \brief Compare destination block and range.
    /// \param other Other mapping.
    /// \return `true` when every field is equal.
    auto operator==(BlockRangeMapping const& other) const -> bool = default;
};

/// \brief Result of regularizing an ordered segmented `IrregularSpace`.
struct IrregularSpaceRegularization
{
    /// \brief Original irregular space.
    IrregularSpace original;

    /// \brief Canonical space with one block per quantum number.
    BlockSpace regular;

    /// \brief Destination block and range for every original block.
    std::vector<BlockRangeMapping> block_mapping;
};

namespace detail
{

/// \brief Compare two quantum numbers by canonical packed order.
/// \param lhs Left quantum number.
/// \param rhs Right quantum number.
/// \return `true` if `lhs` sorts before `rhs`.
inline auto qnum_code_less(QNum const& lhs, QNum const& rhs) -> bool { return lhs.raw_code() < rhs.raw_code(); }

/// \brief Return the canonical unique quantum numbers appearing in a container.
/// \tparam Range Range whose value type is `QNum`.
/// \param range Range of quantum numbers.
/// \return Sorted unique quantum numbers in canonical packed order.
template <typename Range> inline auto unique_qnums(Range const& range) -> std::vector<QNum>
{
  std::unordered_set<QNum> seen;
  std::vector<QNum> qnums;
  for (QNum const& q : range)
  {
    if (seen.insert(q).second)
    {
      qnums.push_back(q);
    }
  }
  std::sort(qnums.begin(), qnums.end(), qnum_code_less);
  return qnums;
}

} // namespace detail

/// \brief Regularize a sparse `QNumList` into one block per quantum number.
/// \param list Sparse-space quantum-number list.
/// \return Regularization result with per-entry block indices and offsets.
inline auto regularize(QNumList const& list) -> QNumListRegularization
{
  auto const qnums = detail::unique_qnums(list);

  std::unordered_map<QNum, std::size_t> count_by_q;
  for (QNum const& q : list)
  {
    count_by_q[q] += 1;
  }

  std::vector<BlockSector> sectors;
  sectors.reserve(qnums.size());
  for (QNum const& q : qnums)
  {
    sectors.push_back({q, count_by_q[q]});
  }
  BlockSpace regular(list.symmetry(), sectors);

  std::unordered_map<QNum, std::size_t> index_by_q;
  for (std::size_t index = 0; index < regular.size(); ++index)
  {
    index_by_q.emplace(regular[index].q, index);
  }

  std::unordered_map<QNum, std::size_t> offset_by_q;
  std::vector<std::size_t> block_index;
  std::vector<std::size_t> block_offset;
  block_index.reserve(list.size());
  block_offset.reserve(list.size());

  for (QNum const& q : list)
  {
    block_index.push_back(index_by_q.at(q));
    block_offset.push_back(offset_by_q[q]);
    offset_by_q[q] += 1;
  }

  return {list, std::move(regular), std::move(block_index), std::move(block_offset)};
}

/// \brief Regularize a `LocalSpace` into one block per quantum number.
/// \param space Ordered local space.
/// \return Regularization result with per-state block indices and offsets.
inline auto regularize(LocalSpace const& space) -> LocalSpaceRegularization
{
  auto const qnums = detail::unique_qnums(space);

  std::unordered_map<QNum, std::size_t> count_by_q;
  for (QNum const& q : space)
  {
    count_by_q[q] += 1;
  }

  std::vector<BlockSector> sectors;
  sectors.reserve(qnums.size());
  for (QNum const& q : qnums)
  {
    sectors.push_back({q, count_by_q[q]});
  }
  BlockSpace regular(space.symmetry(), sectors, space.label());

  std::unordered_map<QNum, std::size_t> index_by_q;
  for (std::size_t index = 0; index < regular.size(); ++index)
  {
    index_by_q.emplace(regular[index].q, index);
  }

  std::unordered_map<QNum, std::size_t> offset_by_q;
  std::vector<std::size_t> block_index;
  std::vector<std::size_t> block_offset;
  block_index.reserve(space.size());
  block_offset.reserve(space.size());

  for (QNum const& q : space)
  {
    block_index.push_back(index_by_q.at(q));
    block_offset.push_back(offset_by_q[q]);
    offset_by_q[q] += 1;
  }

  return {space, std::move(regular), std::move(block_index), std::move(block_offset)};
}

/// \brief Regularize an ordered segmented space into one block per quantum number.
/// \param space Irregular space whose block order and segmentation must be mapped.
/// \return Canonical space and one destination range for every original block.
inline auto regularize(IrregularSpace const& space) -> IrregularSpaceRegularization
{
  std::vector<QNum> original_qnums;
  original_qnums.reserve(space.size());
  for (BlockSector const& sector : space)
  {
    original_qnums.push_back(sector.q);
  }
  auto const qnums = detail::unique_qnums(original_qnums);

  std::unordered_map<QNum, std::size_t> dim_by_q;
  for (BlockSector const& sector : space)
  {
    dim_by_q[sector.q] += sector.dim;
  }

  std::vector<BlockSector> sectors;
  sectors.reserve(qnums.size());
  for (QNum const& q : qnums)
  {
    sectors.push_back({q, dim_by_q[q]});
  }
  BlockSpace regular(space.symmetry(), sectors, space.label());

  std::unordered_map<QNum, std::size_t> index_by_q;
  for (std::size_t index = 0; index < regular.size(); ++index)
  {
    index_by_q.emplace(regular[index].q, index);
  }

  std::unordered_map<QNum, std::size_t> offset_by_q;
  std::vector<BlockRangeMapping> block_mapping;
  block_mapping.reserve(space.size());
  for (BlockSector const& sector : space)
  {
    auto& offset = offset_by_q[sector.q];
    block_mapping.push_back({index_by_q.at(sector.q), offset, offset + sector.dim});
    offset += sector.dim;
  }

  return {space, std::move(regular), std::move(block_mapping)};
}

/// \brief Return the coalesced block-space representation of a sparse `QNumList`.
/// \param list Sparse-space quantum-number list.
/// \return Regularized block space.
inline auto to_block_space(QNumList const& list) -> BlockSpace { return regularize(list).regular; }

/// \brief Return the coalesced block-space representation of a `LocalSpace`.
/// \param space Ordered local space.
/// \return Regularized block space with the same leg label.
inline auto to_block_space(LocalSpace const& space) -> BlockSpace { return regularize(space).regular; }

/// \brief Return the canonical block-space representation of an `IrregularSpace`.
/// \param space Ordered segmented space.
/// \return Regularized block space with the same leg label.
inline auto to_block_space(IrregularSpace const& space) -> BlockSpace { return regularize(space).regular; }

} // namespace uni20
