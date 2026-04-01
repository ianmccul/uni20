/**
 * \file block_space.hpp
 * \brief Defines coalesced block-structured tensor spaces and regularization helpers.
 *
 * \details See `docs/qnum.md` for the current `BlockSpace` and `QNumList` tensor-space API and
 *          how they fit into the planned DMRG integration work.
 */

#pragma once

#include <uni20/symmetry/qnum.hpp>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace uni20
{

/// \brief One symmetry block inside a `BlockSpace`.
struct BlockSector
{
    /// \brief Quantum number carried by the block.
    QNum q;

    /// \brief Degeneracy / number of states in the block.
    std::size_t dim = 0;

    /// \brief Compare two block sectors for exact identity.
    /// \param other Other block sector.
    /// \return `true` if both quantum number and dimension match.
    auto operator==(BlockSector const& other) const -> bool = default;
};

/// \brief Half-open range inside one regularized block.
struct BlockRange
{
    /// \brief First offset in the target block.
    std::size_t first = 0;

    /// \brief One-past-the-end offset in the target block.
    std::size_t last = 0;

    /// \brief Return the size of the range.
    /// \return Number of covered entries.
    auto size() const -> std::size_t { return last - first; }
};

/// \brief Coalesced tensor space represented as symmetry blocks `(QNum, dim)`.
/// \details `BlockSpace` is the dense/coalesced counterpart to `QNumList`. See `docs/qnum.md`.
class BlockSpace {
  public:
    /// \brief Construct an empty block space tied to one symmetry.
    /// \param sym Symmetry carried by every block.
    explicit BlockSpace(Symmetry sym) : sym_(sym) {}

    /// \brief Construct a block space from initial blocks.
    /// \param sym Symmetry carried by every block.
    /// \param sectors Initial block list.
    BlockSpace(Symmetry sym, std::initializer_list<BlockSector> sectors) : sym_(sym), sectors_(sectors)
    {
        for (BlockSector const& sector : sectors_)
        {
            this->verify_sector(sector);
        }
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

    /// \brief Return whether each quantum number appears at most once.
    /// \return `true` if the block space is already regularized.
    auto is_regular() const -> bool
    {
        std::unordered_set<QNum> seen;
        for (BlockSector const& sector : sectors_)
        {
            auto const [_, inserted] = seen.insert(sector.q);
            if (!inserted)
            {
                return false;
            }
        }
        return true;
    }

    /// \brief Append a new block.
    /// \param sector Block to append.
    void push_back(BlockSector sector)
    {
        this->verify_sector(sector);
        sectors_.push_back(sector);
    }

    /// \brief Remove all blocks while keeping the symmetry.
    void clear() { sectors_.clear(); }

    /// \brief Return whether a block with the given quantum number exists.
    /// \param q Quantum number to search for.
    /// \return `true` if a matching block exists.
    auto contains(QNum q) const -> bool
    {
        this->verify_symmetry(q);
        return std::find_if(sectors_.begin(), sectors_.end(),
                            [&](BlockSector const& sector) { return sector.q == q; }) != sectors_.end();
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

    /// \brief Compare two block spaces for exact identity.
    /// \param other Other block space.
    /// \return `true` if both symmetry and block lists match.
    auto operator==(BlockSpace const& other) const -> bool = default;

  private:
    /// \brief Throw if a quantum number does not match the block-space symmetry.
    /// \param q Quantum number to validate.
    void verify_symmetry(QNum const& q) const
    {
        if (q.symmetry() != sym_)
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

    Symmetry sym_;
    std::vector<BlockSector> sectors_;
};

/// \brief Result of explicitly regularizing an irregular `BlockSpace`.
struct BlockSpaceRegularization
{
    /// \brief Original irregular block space.
    BlockSpace original;

    /// \brief Coalesced block space with unique quantum numbers in canonical order.
    BlockSpace regular;

    /// \brief For each original block, index of the corresponding block in `regular`.
    std::vector<std::size_t> block_index;

    /// \brief For each original block, covered range inside the target regular block.
    std::vector<BlockRange> block_range;
};

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

/// \brief Regularize an irregular `BlockSpace` by coalescing repeated quantum numbers.
/// \param space Block space to regularize.
/// \return Regularization result with block-to-block range mappings.
inline auto regularize(BlockSpace const& space) -> BlockSpaceRegularization
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

    BlockSpace regular(space.symmetry());
    std::unordered_map<QNum, std::size_t> index_by_q;
    for (QNum const& q : qnums)
    {
        index_by_q.emplace(q, regular.size());
        regular.push_back({q, dim_by_q[q]});
    }

    std::unordered_map<QNum, std::size_t> offset_by_q;
    std::vector<std::size_t> block_index;
    std::vector<BlockRange> block_range;
    block_index.reserve(space.size());
    block_range.reserve(space.size());

    for (BlockSector const& sector : space)
    {
        auto const index = index_by_q.at(sector.q);
        auto& offset = offset_by_q[sector.q];
        block_index.push_back(index);
        block_range.push_back({offset, offset + sector.dim});
        offset += sector.dim;
    }

    return {space, std::move(regular), std::move(block_index), std::move(block_range)};
}

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

    BlockSpace regular(list.symmetry());
    std::unordered_map<QNum, std::size_t> index_by_q;
    for (QNum const& q : qnums)
    {
        index_by_q.emplace(q, regular.size());
        regular.push_back({q, count_by_q[q]});
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

/// \brief Return the coalesced block-space representation of a sparse `QNumList`.
/// \param list Sparse-space quantum-number list.
/// \return Regularized block space.
inline auto to_block_space(QNumList const& list) -> BlockSpace { return regularize(list).regular; }

} // namespace uni20
