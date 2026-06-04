/**
 * \file device_block_sparse_matrix.hpp
 * \brief Device-resident block-sparse three-leg tensor storage.
 */

#pragma once

#include <uni20/mps/block_sparse_matrix.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Device-resident three-leg block-sparse matrix.
/// \details This prototype keeps symmetry metadata on the host and dense block
///          payloads in TensorContraction resident `MatrixFamily` storage. Host
///          payload materialization is explicit and should be used only at
///          cold-storage or test/debug boundaries.
class DeviceThreeLegBlockMatrix {
  public:
    using key_type = ThreeLegBlockKey;
    using block_type = ThreeLegBlock;
    using index_type = std::size_t;

    /// \brief Construct a device-resident block-sparse matrix from resident payloads.
    /// \throws std::invalid_argument If metadata and payload blocks disagree.
    /// \param row_space Row block space.
    /// \param local_space Local state space.
    /// \param col_space Column block space.
    /// \param blocks Dense block descriptors in resident payload order.
    /// \param values Resident dense block payloads.
    /// \param algebra Owner of the resident TensorContraction runtime.
    DeviceThreeLegBlockMatrix(BlockSpace row_space, LocalSpace local_space, BlockSpace col_space,
                              std::vector<block_type> blocks, tensorcontraction::MatrixFamily values,
                              std::shared_ptr<tensorcontraction::VectorAlgebraEngine> algebra)
        : row_space_(std::move(row_space)), local_space_(std::move(local_space)), col_space_(std::move(col_space)),
          blocks_(std::move(blocks)), values_(std::move(values)), algebra_(std::move(algebra)),
          blocks_by_row_(row_space_.size()), blocks_by_local_(local_space_.size()), blocks_by_col_(col_space_.size())
    {
      this->verify();
      this->build_indexes();
    }

    /// \brief Upload a host block-sparse matrix to resident device storage.
    /// \throws std::invalid_argument If the runtime is not resident.
    /// \param host Host matrix to upload.
    /// \param algebra Resident TensorContraction runtime.
    /// \return Device-resident matrix with matching metadata.
    static auto from_host(ThreeLegBlockMatrix const& host,
                          std::shared_ptr<tensorcontraction::VectorAlgebraEngine> algebra) -> DeviceThreeLegBlockMatrix
    {
      if (algebra == nullptr || algebra->uses_host_backend())
      {
        throw std::invalid_argument("DeviceThreeLegBlockMatrix requires a resident TensorContraction backend");
      }

      std::vector<tensorcontraction::MatrixFamily::Block> matrix_blocks;
      matrix_blocks.reserve(host.block_count());
      std::vector<block_type> blocks;
      blocks.reserve(host.block_count());
      for (auto const& block : host.blocks())
      {
        blocks.push_back(block);
        matrix_blocks.push_back(tensorcontraction::MatrixFamily::Block{.rows = block.rows, .cols = block.cols});
      }

      tensorcontraction::MatrixFamily values(matrix_blocks);
      for (std::size_t block = 0; block < host.block_count(); ++block)
      {
        values.assign(block, host.values(block));
      }
      algebra->upload(values);
      return DeviceThreeLegBlockMatrix(host.row_space(), host.local_space(), host.col_space(), std::move(blocks),
                                       std::move(values), std::move(algebra));
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

    /// \brief Return the number of resident dense blocks.
    /// \return Block count.
    auto block_count() const -> std::size_t { return blocks_.size(); }

    /// \brief Return whether no blocks are stored.
    /// \return `true` when the matrix is empty.
    auto empty() const -> bool { return blocks_.empty(); }

    /// \brief Return dense block descriptors.
    /// \return Read-only block descriptor span.
    auto blocks() const -> std::span<block_type const> { return blocks_; }

    /// \brief Return whether one block exists.
    /// \param key Block coordinate.
    /// \return `true` if a resident block exists at `key`.
    auto contains(key_type key) const -> bool { return lookup_.contains(key); }

    /// \brief Return the descriptor for one block by key.
    /// \throws std::out_of_range If no block exists at `key`.
    /// \param key Block coordinate.
    /// \return Stored block descriptor.
    auto block(key_type key) const -> block_type const& { return blocks_.at(this->block_index(key)); }

    /// \brief Return a block index for one key.
    /// \throws std::out_of_range If no block exists at `key`.
    /// \param key Block coordinate.
    /// \return Block index in resident payload order.
    auto block_index(key_type key) const -> index_type { return lookup_.at(key); }

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

    /// \brief Return mutable resident payload storage.
    /// \return Resident TensorContraction matrix family.
    auto values() -> tensorcontraction::MatrixFamily& { return values_; }

    /// \brief Return read-only resident payload storage.
    /// \return Resident TensorContraction matrix family.
    auto values() const -> tensorcontraction::MatrixFamily const& { return values_; }

    /// \brief Return the resident TensorContraction runtime.
    /// \return Shared runtime owner.
    auto algebra() const -> std::shared_ptr<tensorcontraction::VectorAlgebraEngine> const& { return algebra_; }

    /// \brief Explicitly materialize resident payloads into host storage.
    /// \details This is an authority-boundary operation. It copies every block
    ///          from the GPU into a host `ThreeLegBlockMatrix`.
    /// \throws std::logic_error If the resident runtime has expired.
    /// \return Host block-sparse matrix with synchronized payloads.
    auto materialize_to_host() -> ThreeLegBlockMatrix
    {
      if (algebra_ == nullptr)
      {
        throw std::logic_error("DeviceThreeLegBlockMatrix has no resident TensorContraction runtime");
      }
      algebra_->synchronize(values_);

      ThreeLegBlockMatrix host(row_space_, local_space_, col_space_);
      for (std::size_t block = 0; block < blocks_.size(); ++block)
      {
        auto values = host.insert_zero_block(blocks_[block].key);
        auto const source = values_.values(block);
        if (values.size() != source.size())
        {
          throw std::logic_error("device block materialization encountered incompatible block shapes");
        }
        std::copy(source.begin(), source.end(), values.begin());
      }
      return host;
    }

  private:
    BlockSpace row_space_;
    LocalSpace local_space_;
    BlockSpace col_space_;
    std::vector<block_type> blocks_;
    tensorcontraction::MatrixFamily values_;
    std::shared_ptr<tensorcontraction::VectorAlgebraEngine> algebra_;
    std::unordered_map<key_type, index_type, ThreeLegBlockKeyHash> lookup_;
    std::vector<std::vector<index_type>> blocks_by_row_;
    std::vector<std::vector<index_type>> blocks_by_local_;
    std::vector<std::vector<index_type>> blocks_by_col_;

    void verify() const
    {
      if (algebra_ == nullptr || algebra_->uses_host_backend())
      {
        throw std::invalid_argument("DeviceThreeLegBlockMatrix requires a resident TensorContraction backend");
      }
      if (values_.size() != blocks_.size())
      {
        throw std::invalid_argument("DeviceThreeLegBlockMatrix metadata and payload block counts differ");
      }

      std::unordered_map<key_type, index_type, ThreeLegBlockKeyHash> seen;
      for (std::size_t index = 0; index < blocks_.size(); ++index)
      {
        auto const& block = blocks_[index];
        if (block.key.row_sector >= row_space_.size() || block.key.local >= local_space_.size() ||
            block.key.col_sector >= col_space_.size())
        {
          throw std::invalid_argument("DeviceThreeLegBlockMatrix block key is out of range");
        }
        if (!three_leg_block_allowed(row_space_, local_space_, col_space_, block.key))
        {
          throw std::invalid_argument("DeviceThreeLegBlockMatrix block violates the symmetry selection rule");
        }
        if (!seen.emplace(block.key, index).second)
        {
          throw std::invalid_argument("DeviceThreeLegBlockMatrix contains duplicate block keys");
        }
        auto const expected_rows = row_space_[block.key.row_sector].dim;
        auto const expected_cols = col_space_[block.key.col_sector].dim;
        if (block.rows != expected_rows || block.cols != expected_cols)
        {
          throw std::invalid_argument("DeviceThreeLegBlockMatrix block dimensions do not match its spaces");
        }
        auto const expected_payload = tensorcontraction::MatrixFamily::Block{.rows = block.rows, .cols = block.cols};
        if (values_.block(index) != expected_payload)
        {
          throw std::invalid_argument("DeviceThreeLegBlockMatrix payload dimensions do not match metadata");
        }
      }
    }

    void build_indexes()
    {
      std::size_t offset = 0;
      for (std::size_t index = 0; index < blocks_.size(); ++index)
      {
        blocks_[index].offset = offset;
        offset += blocks_[index].size();
        lookup_.emplace(blocks_[index].key, index);
        blocks_by_row_[blocks_[index].key.row_sector].push_back(index);
        blocks_by_local_[blocks_[index].key.local].push_back(index);
        blocks_by_col_[blocks_[index].key.col_sector].push_back(index);
      }
    }
};

} // namespace uni20
