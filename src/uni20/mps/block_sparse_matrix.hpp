/**
 * \file block_sparse_matrix.hpp
 * \brief Block-sparse three-leg matrix storage for MPS tensors and environments.
 */

#pragma once

#include <uni20/operator/local_space.hpp>
#include <uni20/symmetry/block_space.hpp>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Logical key for a dense block in a three-leg block-sparse matrix.
/// \details The key represents `(row sector, local state, column sector)`.
struct ThreeLegBlockKey
{
    std::size_t row_sector = 0;
    std::size_t local = 0;
    std::size_t col_sector = 0;

    /// \brief Compare two keys for exact identity.
    /// \param other Other key.
    /// \return `true` if all coordinates match.
    auto operator==(ThreeLegBlockKey const& other) const -> bool = default;
};

/// \brief Hash functor for `ThreeLegBlockKey`.
struct ThreeLegBlockKeyHash
{
    /// \brief Hash a three-leg block key.
    /// \param key Key to hash.
    /// \return Combined hash value.
    auto operator()(ThreeLegBlockKey const& key) const noexcept -> std::size_t
    {
      auto seed = std::hash<std::size_t>{}(key.row_sector);
      auto combine = [&](std::size_t value) {
        seed ^= std::hash<std::size_t>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      };
      combine(key.local);
      combine(key.col_sector);
      return seed;
    }
};

/// \brief Dense payload descriptor for one three-leg block.
struct ThreeLegBlock
{
    ThreeLegBlockKey key;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t offset = 0;

    /// \brief Return the number of scalar coefficients in the dense payload.
    /// \return `rows * cols`.
    auto size() const -> std::size_t { return rows * cols; }
};

/// \brief Return whether one key satisfies the zero-flux three-leg selection rule.
/// \details The convention is `q_col = q_row + q_local`. This is the natural
///          MPS convention for a ket tensor and also the environment convention
///          used by the first U(1) DMRG prototype.
/// \param row_space Row block space.
/// \param local_space Local state space.
/// \param col_space Column block space.
/// \param key Block coordinate to test.
/// \return `true` when the block is symmetry-allowed.
inline auto three_leg_block_allowed(BlockSpace const& row_space, LocalSpace const& local_space,
                                    BlockSpace const& col_space, ThreeLegBlockKey key) -> bool
{
  return row_space[key.row_sector].q + local_space[key.local] == col_space[key.col_sector].q;
}

/// \brief Block-sparse matrix with one explicit `LocalSpace` leg.
/// \details This is the canonical first-pass storage for MPS site tensors and
///          MPO environments. Each stored block is a dense row-major matrix
///          whose row and column dimensions are inherited from the selected
///          row and column `BlockSpace` sectors.
class ThreeLegBlockMatrix {
  public:
    using key_type = ThreeLegBlockKey;
    using block_type = ThreeLegBlock;
    using index_type = std::size_t;

    /// \brief Construct an empty block-sparse matrix over fixed spaces.
    /// \param row_space Row block space.
    /// \param local_space Local state space.
    /// \param col_space Column block space.
    ThreeLegBlockMatrix(BlockSpace row_space, LocalSpace local_space, BlockSpace col_space)
        : row_space_(std::move(row_space)), local_space_(std::move(local_space)), col_space_(std::move(col_space)),
          blocks_by_row_(row_space_.size()), blocks_by_local_(local_space_.size()), blocks_by_col_(col_space_.size())
    {
      this->verify_spaces();
    }

    /// \brief Construct a matrix containing every symmetry-allowed block.
    /// \param row_space Row block space.
    /// \param local_space Local state space.
    /// \param col_space Column block space.
    /// \return Matrix with zero-filled allowed dense blocks.
    static auto with_allowed_blocks(BlockSpace row_space, LocalSpace local_space,
                                    BlockSpace col_space) -> ThreeLegBlockMatrix
    {
      ThreeLegBlockMatrix result(std::move(row_space), std::move(local_space), std::move(col_space));
      for (std::size_t row = 0; row < result.row_space_.size(); ++row)
      {
        for (std::size_t local = 0; local < result.local_space_.size(); ++local)
        {
          for (std::size_t col = 0; col < result.col_space_.size(); ++col)
          {
            ThreeLegBlockKey const key{.row_sector = row, .local = local, .col_sector = col};
            if (three_leg_block_allowed(result.row_space_, result.local_space_, result.col_space_, key))
            {
              static_cast<void>(result.insert_zero_block(key));
            }
          }
        }
      }
      return result;
    }

    /// \brief Return the row block space.
    /// \return Row block space.
    auto row_space() const -> BlockSpace const& { return row_space_; }

    /// \brief Return the local state space.
    /// \return Local state space.
    auto local_space() const -> LocalSpace const& { return local_space_; }

    /// \brief Return the column block space.
    /// \return Column block space.
    auto col_space() const -> BlockSpace const& { return col_space_; }

    /// \brief Return the number of stored blocks.
    /// \return Block count.
    auto block_count() const -> std::size_t { return blocks_.size(); }

    /// \brief Return whether no blocks are stored.
    /// \return `true` if the matrix has no blocks.
    auto empty() const -> bool { return blocks_.empty(); }

    /// \brief Return all dense block descriptors.
    /// \return Read-only block descriptor span.
    auto blocks() const -> std::span<block_type const> { return blocks_; }

    /// \brief Return read-only dense values for one block by index.
    /// \param index Block index.
    /// \return Dense row-major coefficient span.
    auto values(index_type index) const -> std::span<double const>
    {
      auto const& block = blocks_.at(index);
      return std::span<double const>(data_.data() + block.offset, block.size());
    }

    /// \brief Return mutable dense values for one block by index.
    /// \param index Block index.
    /// \return Dense row-major coefficient span.
    auto values(index_type index) -> std::span<double>
    {
      auto const& block = blocks_.at(index);
      return std::span<double>(data_.data() + block.offset, block.size());
    }

    /// \brief Return read-only dense values for one block by key.
    /// \throws std::out_of_range If no block exists at `key`.
    /// \param key Block coordinate.
    /// \return Dense row-major coefficient span.
    auto values(key_type key) const -> std::span<double const> { return this->values(this->block_index(key)); }

    /// \brief Return mutable dense values for one block by key.
    /// \throws std::out_of_range If no block exists at `key`.
    /// \param key Block coordinate.
    /// \return Dense row-major coefficient span.
    auto values(key_type key) -> std::span<double> { return this->values(this->block_index(key)); }

    /// \brief Return whether one block exists.
    /// \param key Block coordinate.
    /// \return `true` if a block is stored at `key`.
    auto contains(key_type key) const -> bool { return lookup_.contains(key); }

    /// \brief Return the descriptor for one block by key.
    /// \throws std::out_of_range If no block exists at `key`.
    /// \param key Block coordinate.
    /// \return Stored block descriptor.
    auto block(key_type key) const -> block_type const& { return blocks_.at(this->block_index(key)); }

    /// \brief Return block indexes with one row sector.
    /// \param row_sector Row sector index.
    /// \return Block indexes whose key uses `row_sector`.
    auto blocks_from_row(index_type row_sector) const -> std::span<index_type const>
    {
      return blocks_by_row_.at(row_sector);
    }

    /// \brief Return block indexes with one local state.
    /// \param local Local-state index.
    /// \return Block indexes whose key uses `local`.
    auto blocks_for_local(index_type local) const -> std::span<index_type const> { return blocks_by_local_.at(local); }

    /// \brief Return block indexes with one column sector.
    /// \param col_sector Column sector index.
    /// \return Block indexes whose key uses `col_sector`.
    auto blocks_to_col(index_type col_sector) const -> std::span<index_type const>
    {
      return blocks_by_col_.at(col_sector);
    }

    /// \brief Insert a zero-filled block.
    /// \throws std::invalid_argument If the key is invalid, forbidden, or already present.
    /// \param key Block coordinate.
    /// \return Mutable span over the inserted dense payload.
    auto insert_zero_block(key_type key) -> std::span<double>
    {
      this->validate_new_key(key);

      auto const index = blocks_.size();
      auto const rows = row_space_[key.row_sector].dim;
      auto const cols = col_space_[key.col_sector].dim;
      auto const offset = data_.size();
      blocks_.push_back(block_type{.key = key, .rows = rows, .cols = cols, .offset = offset});
      data_.resize(offset + rows * cols, 0.0);
      lookup_.emplace(key, index);
      blocks_by_row_[key.row_sector].push_back(index);
      blocks_by_local_[key.local].push_back(index);
      blocks_by_col_[key.col_sector].push_back(index);
      return this->values(index);
    }

    /// \brief Insert or overwrite one block payload.
    /// \throws std::invalid_argument If `values` has the wrong size.
    /// \param key Block coordinate.
    /// \param values New dense payload.
    void insert_or_assign_block(key_type key, std::span<double const> values)
    {
      if (!this->contains(key))
      {
        static_cast<void>(this->insert_zero_block(key));
      }
      this->assign_block(key, values);
    }

    /// \brief Overwrite one existing block payload.
    /// \throws std::invalid_argument If `values` has the wrong size.
    /// \throws std::out_of_range If no block exists at `key`.
    /// \param key Block coordinate.
    /// \param values New dense payload.
    void assign_block(key_type key, std::span<double const> values)
    {
      auto dst = this->values(key);
      if (dst.size() != values.size())
      {
        throw std::invalid_argument("three-leg block assignment has the wrong size");
      }
      std::copy(values.begin(), values.end(), dst.begin());
    }

  private:
    void verify_spaces() const
    {
      auto const sym = local_space_.symmetry();
      if (row_space_.symmetry() != sym || col_space_.symmetry() != sym)
      {
        throw std::invalid_argument("ThreeLegBlockMatrix spaces must share one symmetry");
      }
    }

    void validate_key_ranges(key_type key) const
    {
      if (key.row_sector >= row_space_.size() || key.local >= local_space_.size() ||
          key.col_sector >= col_space_.size())
      {
        throw std::out_of_range("three-leg block key is out of range");
      }
    }

    void validate_new_key(key_type key) const
    {
      this->validate_key_ranges(key);
      if (!three_leg_block_allowed(row_space_, local_space_, col_space_, key))
      {
        throw std::invalid_argument("three-leg block violates q_col = q_row + q_local");
      }
      if (this->contains(key))
      {
        throw std::invalid_argument("three-leg block already exists");
      }
    }

    auto block_index(key_type key) const -> index_type
    {
      this->validate_key_ranges(key);
      auto const it = lookup_.find(key);
      if (it == lookup_.end())
      {
        throw std::out_of_range("three-leg block is not present");
      }
      return it->second;
    }

    BlockSpace row_space_;
    LocalSpace local_space_;
    BlockSpace col_space_;
    std::vector<block_type> blocks_;
    std::vector<double> data_;
    std::unordered_map<key_type, index_type, ThreeLegBlockKeyHash> lookup_;
    std::vector<std::vector<index_type>> blocks_by_row_;
    std::vector<std::vector<index_type>> blocks_by_local_;
    std::vector<std::vector<index_type>> blocks_by_col_;
};

} // namespace uni20
