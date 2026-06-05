/**
 * \file block_sparse_dmrg.hpp
 * \brief Strict block-sparse U(1) helpers for the first Heisenberg DMRG path.
 */

#pragma once

#include <uni20/mps/block_sparse_mps.hpp>
#include <uni20/mps/device_block_sparse_matrix.hpp>
#include <uni20/mps/sparse_mpo_site.hpp>
#include <uni20/mps/two_site_split.hpp>
#include <uni20/operator/finite_triangular_mpo.hpp>
#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/lanczos.hpp>
#include <uni20/tensorcontraction/rabc_lanczos_fixture.hpp>
#include <uni20/tensorcontraction/svd.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <uni20/config.hpp>

#if UNI20_BACKEND_BLAS
#include <uni20/backend/blas/backend_blas.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Alias for a block-sparse MPO environment tensor.
/// \details The convention is `(bra bond, MPO virtual, ket bond)`.
using BlockSparseEnvironment = ThreeLegBlockMatrix;

/// \brief Chain of compiled sparse MPO sites.
class BlockSparseMpoChain {
  public:
    using container_type = std::vector<SparseMpoSite>;
    using size_type = container_type::size_type;

    /// \brief Construct an empty sparse MPO chain.
    BlockSparseMpoChain() = default;

    /// \brief Construct from compiled sparse MPO sites.
    /// \param sites Sites ordered from left to right.
    explicit BlockSparseMpoChain(container_type sites) : sites_(std::move(sites)) { this->check_structure(); }

    /// \brief Return the number of sites.
    /// \return Site count.
    auto size() const noexcept -> size_type { return sites_.size(); }

    /// \brief Return whether the chain has no sites.
    /// \return `true` if there are no sites.
    auto empty() const noexcept -> bool { return sites_.empty(); }

    /// \brief Return one sparse MPO site.
    /// \param index Site index.
    /// \return Sparse MPO site.
    auto operator[](size_type index) const -> SparseMpoSite const& { return sites_.at(index); }

    /// \brief Return the left boundary virtual space.
    /// \return Left boundary virtual space.
    auto left_boundary_virtual_space() const -> LocalSpace const& { return sites_.front().left_virtual_space(); }

    /// \brief Return the right boundary virtual space.
    /// \return Right boundary virtual space.
    auto right_boundary_virtual_space() const -> LocalSpace const& { return sites_.back().right_virtual_space(); }

    /// \brief Validate nearest-neighbor virtual spaces.
    /// \throws std::invalid_argument If the chain is inconsistent.
    void check_structure() const
    {
      for (size_type i = 1; i < sites_.size(); ++i)
      {
        if (sites_[i - 1].right_virtual_space() != sites_[i].left_virtual_space())
        {
          throw std::invalid_argument("BlockSparseMpoChain adjacent virtual spaces do not match");
        }
        if (sites_[i - 1].ket_space().symmetry() != sites_[i].ket_space().symmetry())
        {
          throw std::invalid_argument("BlockSparseMpoChain sites do not share one physical symmetry");
        }
      }
    }

  private:
    container_type sites_;
};

/// \brief Compile a finite triangular MPO into sparse U(1)-checked sites.
/// \param mpo Source finite MPO.
/// \return Block-sparse MPO chain.
inline auto make_block_sparse_mpo_chain(FiniteTriangularMPO const& mpo) -> BlockSparseMpoChain
{
  BlockSparseMpoChain::container_type sites;
  sites.reserve(mpo.size());
  for (std::size_t site = 0; site < mpo.size(); ++site)
  {
    sites.emplace_back(mpo[site]);
  }
  return BlockSparseMpoChain(std::move(sites));
}

namespace detail
{

inline void validate_block_sparse_site_component(ThreeLegBlockMatrix const& site, SparseMpoSite const& mpo_site)
{
  if (site.local_space() != mpo_site.bra_space() || site.local_space() != mpo_site.ket_space())
  {
    throw std::invalid_argument("block-sparse MPO local spaces do not match the MPS site");
  }
}

inline void validate_block_sparse_environment(BlockSparseEnvironment const& env, BlockSpace const& expected_bond,
                                              LocalSpace const& expected_virtual, char const* side)
{
  if (env.row_space() != expected_bond || env.col_space() != expected_bond)
  {
    throw std::invalid_argument(std::string("block-sparse ") + side + " environment bond spaces do not match");
  }
  if (env.local_space() != expected_virtual)
  {
    throw std::invalid_argument(std::string("block-sparse ") + side + " environment virtual space does not match");
  }
}

inline void add_dense_gemm(std::span<double const> lhs, std::size_t rows, std::size_t shared,
                           std::span<double const> rhs, std::size_t cols, std::span<double> out, double coefficient)
{
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t mid = 0; mid < shared; ++mid)
    {
      double const lhs_value = coefficient * lhs[row * shared + mid];
      if (lhs_value == 0.0)
      {
        continue;
      }
      for (std::size_t col = 0; col < cols; ++col)
      {
        out[row * cols + col] += lhs_value * rhs[mid * cols + col];
      }
    }
  }
}

/// \brief Convert a matrix extent to the configured BLAS integer type.
/// \throws std::length_error If `value` does not fit in `blas_int`.
/// \param value Size value to convert.
/// \param name Human-readable diagnostic name.
/// \return BLAS-compatible integer value.
inline auto checked_blas_size(std::size_t value, char const* name) -> blas_int
{
  if (value > static_cast<std::size_t>(std::numeric_limits<blas_int>::max()))
  {
    throw std::length_error(std::string(name) + " exceeds BLAS integer range");
  }
  return static_cast<blas_int>(value);
}

/// \brief Read one value from a row-major matrix with optional logical transpose.
/// \param values Row-major dense matrix values.
/// \param transpose `'N'` for normal access, otherwise transposed access.
/// \param row Logical row.
/// \param col Logical column.
/// \param rows Logical row count after applying `transpose`.
/// \param cols Logical column count after applying `transpose`.
/// \return Matrix coefficient.
inline auto row_major_value(std::span<double const> values, char transpose, std::size_t row, std::size_t col,
                            std::size_t rows, std::size_t cols) -> double
{
  if (transpose == 'N')
  {
    return values[row * cols + col];
  }
  return values[col * rows + row];
}

/// \brief Accumulate a row-major matrix product, using BLAS when configured.
/// \details Column-major BLAS receives the equivalent operation `C^T = B^T A^T`.
/// \param transpose_lhs `'N'` for `lhs`, otherwise `lhs^T`.
/// \param transpose_rhs `'N'` for `rhs`, otherwise `rhs^T`.
/// \param rows Logical output row count.
/// \param cols Logical output column count.
/// \param shared Contracted dimension.
/// \param alpha Product coefficient.
/// \param lhs Left matrix payload.
/// \param rhs Right matrix payload.
/// \param beta Existing-output coefficient.
/// \param out Output matrix payload.
inline void row_major_gemm(char transpose_lhs, char transpose_rhs, std::size_t rows, std::size_t cols,
                           std::size_t shared, double alpha, std::span<double const> lhs, std::span<double const> rhs,
                           double beta, std::span<double> out)
{
#if UNI20_BACKEND_BLAS
  auto const lhs_leading_dim = checked_blas_size(transpose_lhs == 'N' ? shared : rows, "BLAS lhs leading dimension");
  auto const rhs_leading_dim = checked_blas_size(transpose_rhs == 'N' ? cols : shared, "BLAS rhs leading dimension");
  blas::gemm(transpose_rhs, transpose_lhs, checked_blas_size(cols, "BLAS result columns"),
             checked_blas_size(rows, "BLAS result rows"), checked_blas_size(shared, "BLAS shared dimension"), alpha,
             rhs.data(), rhs_leading_dim, lhs.data(), lhs_leading_dim, beta, out.data(),
             checked_blas_size(cols, "BLAS output leading dimension"));
#else
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      double value = 0.0;
      for (std::size_t inner = 0; inner < shared; ++inner)
      {
        value += row_major_value(lhs, transpose_lhs, row, inner, rows, shared) *
                 row_major_value(rhs, transpose_rhs, inner, col, shared, cols);
      }
      out[row * cols + col] = beta * out[row * cols + col] + alpha * value;
    }
  }
#endif
}

/// \brief Accumulate one left-environment block contribution.
/// \param bra_site Bra MPS block.
/// \param env Incoming environment block.
/// \param ket_site Ket MPS block.
/// \param left_bra_dim Incoming bra-bond dimension.
/// \param left_ket_dim Incoming ket-bond dimension.
/// \param right_bra_dim Outgoing bra-bond dimension.
/// \param right_ket_dim Outgoing ket-bond dimension.
/// \param out Output environment block.
/// \param work Reusable temporary buffer.
/// \param coefficient MPO entry coefficient.
inline void add_environment_left(std::span<double const> bra_site, std::span<double const> env,
                                 std::span<double const> ket_site, std::size_t left_bra_dim, std::size_t left_ket_dim,
                                 std::size_t right_bra_dim, std::size_t right_ket_dim, std::span<double> out,
                                 std::vector<double>& work, double coefficient)
{
  work.resize(left_bra_dim * right_ket_dim);
  row_major_gemm('N', 'N', left_bra_dim, right_ket_dim, left_ket_dim, 1.0, env, ket_site, 0.0, work);
  row_major_gemm('T', 'N', right_bra_dim, right_ket_dim, left_bra_dim, coefficient, bra_site, work, 1.0, out);
}

/// \brief Accumulate one right-environment block contribution.
/// \param bra_site Bra MPS block.
/// \param env Incoming environment block.
/// \param ket_site Ket MPS block.
/// \param left_bra_dim Outgoing bra-bond dimension.
/// \param left_ket_dim Outgoing ket-bond dimension.
/// \param right_bra_dim Incoming bra-bond dimension.
/// \param right_ket_dim Incoming ket-bond dimension.
/// \param out Output environment block.
/// \param work Reusable temporary buffer.
/// \param coefficient MPO entry coefficient.
inline void add_environment_right(std::span<double const> bra_site, std::span<double const> env,
                                  std::span<double const> ket_site, std::size_t left_bra_dim, std::size_t left_ket_dim,
                                  std::size_t right_bra_dim, std::size_t right_ket_dim, std::span<double> out,
                                  std::vector<double>& work, double coefficient)
{
  work.resize(left_bra_dim * right_ket_dim);
  row_major_gemm('N', 'N', left_bra_dim, right_ket_dim, right_bra_dim, 1.0, bra_site, env, 0.0, work);
  row_major_gemm('N', 'T', left_bra_dim, left_ket_dim, right_ket_dim, coefficient, work, ket_site, 1.0, out);
}

} // namespace detail

/// \brief Insert identity blocks into an environment at one scalar virtual index.
/// \throws std::invalid_argument If the virtual index does not carry scalar charge.
/// \param env Environment to modify.
/// \param virtual_index MPO virtual index.
inline void set_identity_blocks(BlockSparseEnvironment& env, std::size_t virtual_index)
{
  if (!is_scalar(env.local_space()[virtual_index]))
  {
    throw std::invalid_argument("identity environment requires a scalar MPO virtual index");
  }
  for (std::size_t sector = 0; sector < env.row_space().size(); ++sector)
  {
    if (env.row_space()[sector].q != env.col_space()[sector].q ||
        env.row_space()[sector].dim != env.col_space()[sector].dim)
    {
      throw std::invalid_argument("identity environment requires identical row and column bond sectors");
    }
    ThreeLegBlockKey const key{.row_sector = sector, .local = virtual_index, .col_sector = sector};
    auto values = env.insert_zero_block(key);
    auto const dim = env.row_space()[sector].dim;
    for (std::size_t i = 0; i < dim; ++i)
    {
      values[i * dim + i] = 1.0;
    }
  }
}

/// \brief Build the left boundary environment for a block-sparse MPS/MPO pair.
/// \param psi MPS state.
/// \param mpo Sparse MPO chain.
/// \return Left boundary environment.
inline auto make_left_boundary_environment(BlockSparseFiniteMPS const& psi,
                                           BlockSparseMpoChain const& mpo) -> BlockSparseEnvironment
{
  if (psi.empty() || mpo.empty() || psi.size() != mpo.size())
  {
    throw std::invalid_argument("block-sparse left boundary requires non-empty equal-size MPS and MPO chains");
  }
  BlockSparseEnvironment env(psi[0].row_space(), mpo.left_boundary_virtual_space(), psi[0].row_space());
  set_identity_blocks(env, 0);
  return env;
}

/// \brief Build the right boundary environment for a block-sparse MPS/MPO pair.
/// \param psi MPS state.
/// \param mpo Sparse MPO chain.
/// \return Right boundary environment.
inline auto make_right_boundary_environment(BlockSparseFiniteMPS const& psi,
                                            BlockSparseMpoChain const& mpo) -> BlockSparseEnvironment
{
  if (psi.empty() || mpo.empty() || psi.size() != mpo.size())
  {
    throw std::invalid_argument("block-sparse right boundary requires non-empty equal-size MPS and MPO chains");
  }
  auto const& right_space = psi[psi.size() - 1].col_space();
  BlockSparseEnvironment env(right_space, mpo.right_boundary_virtual_space(), right_space);
  set_identity_blocks(env, env.local_space().size() - 1);
  return env;
}

/// \brief Extend a left block-sparse environment across one site.
/// \param left_env Environment to the left of `site`.
/// \param site MPS site tensor.
/// \param mpo_site Sparse MPO site.
/// \return Environment to the right of `site`.
inline auto extend_left_environment(BlockSparseEnvironment const& left_env, ThreeLegBlockMatrix const& site,
                                    SparseMpoSite const& mpo_site) -> BlockSparseEnvironment
{
  detail::validate_block_sparse_site_component(site, mpo_site);
  detail::validate_block_sparse_environment(left_env, site.row_space(), mpo_site.left_virtual_space(), "left");

  BlockSparseEnvironment next(site.col_space(), mpo_site.right_virtual_space(), site.col_space());
  std::vector<double> work;
  for (auto const& env_block : left_env.blocks())
  {
    auto const env_values = left_env.values(env_block.key);
    for (auto entry_index : mpo_site.entries_from_left_virtual(env_block.key.local))
    {
      auto const& entry = mpo_site.entries()[entry_index];
      for (auto bra_index : site.blocks_from_row(env_block.key.row_sector))
      {
        auto const& bra_block = site.blocks()[bra_index];
        if (bra_block.key.local != entry.key.bra)
        {
          continue;
        }
        for (auto ket_index : site.blocks_from_row(env_block.key.col_sector))
        {
          auto const& ket_block = site.blocks()[ket_index];
          if (ket_block.key.local != entry.key.ket)
          {
            continue;
          }

          ThreeLegBlockKey const out_key{.row_sector = bra_block.key.col_sector,
                                         .local = entry.key.right_virtual,
                                         .col_sector = ket_block.key.col_sector};
          if (!next.contains(out_key))
          {
            static_cast<void>(next.insert_zero_block(out_key));
          }
          detail::add_environment_left(site.values(bra_index), env_values, site.values(ket_index), bra_block.rows,
                                       ket_block.rows, bra_block.cols, ket_block.cols, next.values(out_key), work,
                                       entry.value);
        }
      }
    }
  }
  return next;
}

/// \brief Extend a right block-sparse environment across one site.
/// \param right_env Environment to the right of `site`.
/// \param site MPS site tensor.
/// \param mpo_site Sparse MPO site.
/// \return Environment to the left of `site`.
inline auto extend_right_environment(BlockSparseEnvironment const& right_env, ThreeLegBlockMatrix const& site,
                                     SparseMpoSite const& mpo_site) -> BlockSparseEnvironment
{
  detail::validate_block_sparse_site_component(site, mpo_site);
  detail::validate_block_sparse_environment(right_env, site.col_space(), mpo_site.right_virtual_space(), "right");

  BlockSparseEnvironment previous(site.row_space(), mpo_site.left_virtual_space(), site.row_space());
  std::vector<double> work;
  for (auto const& env_block : right_env.blocks())
  {
    auto const env_values = right_env.values(env_block.key);
    for (auto entry_index : mpo_site.entries_to_right_virtual(env_block.key.local))
    {
      auto const& entry = mpo_site.entries()[entry_index];
      for (auto bra_index : site.blocks_to_col(env_block.key.row_sector))
      {
        auto const& bra_block = site.blocks()[bra_index];
        if (bra_block.key.local != entry.key.bra)
        {
          continue;
        }
        for (auto ket_index : site.blocks_to_col(env_block.key.col_sector))
        {
          auto const& ket_block = site.blocks()[ket_index];
          if (ket_block.key.local != entry.key.ket)
          {
            continue;
          }

          ThreeLegBlockKey const out_key{.row_sector = bra_block.key.row_sector,
                                         .local = entry.key.left_virtual,
                                         .col_sector = ket_block.key.row_sector};
          if (!previous.contains(out_key))
          {
            static_cast<void>(previous.insert_zero_block(out_key));
          }
          detail::add_environment_right(site.values(bra_index), env_values, site.values(ket_index), bra_block.rows,
                                        ket_block.rows, bra_block.cols, ket_block.cols, previous.values(out_key), work,
                                        entry.value);
        }
      }
    }
  }
  return previous;
}

/// \brief Build all left block-sparse environments.
/// \param psi MPS state.
/// \param mpo Sparse MPO chain.
/// \return Environment cache indexed by lattice cut.
inline auto build_left_environments(BlockSparseFiniteMPS const& psi,
                                    BlockSparseMpoChain const& mpo) -> std::vector<BlockSparseEnvironment>
{
  std::vector<BlockSparseEnvironment> environments;
  environments.reserve(psi.size() + 1);
  environments.push_back(make_left_boundary_environment(psi, mpo));
  for (std::size_t site = 0; site < psi.size(); ++site)
  {
    environments.push_back(extend_left_environment(environments.back(), psi[site], mpo[site]));
  }
  return environments;
}

/// \brief Build all right block-sparse environments.
/// \param psi MPS state.
/// \param mpo Sparse MPO chain.
/// \return Environment cache indexed by lattice cut.
inline auto build_right_environments(BlockSparseFiniteMPS const& psi,
                                     BlockSparseMpoChain const& mpo) -> std::vector<BlockSparseEnvironment>
{
  std::vector<BlockSparseEnvironment> reversed;
  reversed.reserve(psi.size() + 1);
  reversed.push_back(make_right_boundary_environment(psi, mpo));
  for (std::size_t offset = 0; offset < psi.size(); ++offset)
  {
    auto const site = psi.size() - 1 - offset;
    reversed.push_back(extend_right_environment(reversed.back(), psi[site], mpo[site]));
  }
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

/// \brief Return the fused local index for two physical states.
/// \param right_physical_dim Right local dimension.
/// \param left_physical Left physical index.
/// \param right_physical Right physical index.
/// \return Fused local index.
inline auto two_site_pair_index(std::size_t right_physical_dim, std::size_t left_physical,
                                std::size_t right_physical) -> std::size_t
{
  return left_physical * right_physical_dim + right_physical;
}

/// \brief Build the fused two-site physical space.
/// \param left_physical_space Left site physical space.
/// \param right_physical_space Right site physical space.
/// \return Local space with charges `q_left + q_right`.
inline auto make_two_site_pair_space(LocalSpace const& left_physical_space,
                                     LocalSpace const& right_physical_space) -> LocalSpace
{
  auto const sym = left_physical_space.symmetry();
  if (right_physical_space.symmetry() != sym)
  {
    throw std::invalid_argument("two-site pair space requires matching physical symmetries");
  }

  QNumList qnums(sym);
  for (std::size_t left = 0; left < left_physical_space.size(); ++left)
  {
    for (std::size_t right = 0; right < right_physical_space.size(); ++right)
    {
      qnums.push_back(left_physical_space[left] + right_physical_space[right]);
    }
  }
  return LocalSpace(std::move(qnums));
}

/// \brief Logical layout for a block-sparse two-site center vector.
class BlockSparseTwoSiteLayout {
  public:
    /// \brief Construct the legal block layout for one two-site center.
    /// \param left_physical_space Left physical space.
    /// \param right_physical_space Right physical space.
    /// \param left_bond_space Left external MPS bond space.
    /// \param right_bond_space Right external MPS bond space.
    BlockSparseTwoSiteLayout(LocalSpace left_physical_space, LocalSpace right_physical_space,
                             BlockSpace left_bond_space, BlockSpace right_bond_space)
        : left_physical_space_(std::move(left_physical_space)), right_physical_space_(std::move(right_physical_space)),
          pair_space_(make_two_site_pair_space(left_physical_space_, right_physical_space_)),
          left_bond_space_(std::move(left_bond_space)), right_bond_space_(std::move(right_bond_space))
    {
      this->build_allowed_blocks();
    }

    /// \brief Return the left physical space.
    /// \return Left physical space.
    auto left_physical_space() const -> LocalSpace const& { return left_physical_space_; }

    /// \brief Return the right physical space.
    /// \return Right physical space.
    auto right_physical_space() const -> LocalSpace const& { return right_physical_space_; }

    /// \brief Return the fused two-site local space.
    /// \return Fused physical space.
    auto pair_space() const -> LocalSpace const& { return pair_space_; }

    /// \brief Return the left external bond space.
    /// \return Left bond space.
    auto left_bond_space() const -> BlockSpace const& { return left_bond_space_; }

    /// \brief Return the right external bond space.
    /// \return Right bond space.
    auto right_bond_space() const -> BlockSpace const& { return right_bond_space_; }

    /// \brief Return legal block keys in MatrixFamily order.
    /// \return Block key span.
    auto keys() const -> std::span<ThreeLegBlockKey const> { return keys_; }

    /// \brief Return the number of legal two-site blocks.
    /// \return Block count.
    auto block_count() const -> std::size_t { return keys_.size(); }

    /// \brief Return whether a block key is present.
    /// \param key Block key.
    /// \return `true` if the key belongs to this layout.
    auto contains(ThreeLegBlockKey key) const -> bool { return lookup_.contains(key); }

    /// \brief Return the MatrixFamily block index for a key.
    /// \throws std::out_of_range If the key is absent.
    /// \param key Block key.
    /// \return Block index.
    auto block_index(ThreeLegBlockKey key) const -> std::size_t
    {
      auto const it = lookup_.find(key);
      if (it == lookup_.end())
      {
        throw std::out_of_range("two-site block key is not in the layout");
      }
      return it->second;
    }

    /// \brief Return the dense shape for one block.
    /// \param key Block key.
    /// \return MatrixFamily block shape.
    auto block_shape(ThreeLegBlockKey key) const -> tensorcontraction::MatrixFamily::Block
    {
      return tensorcontraction::MatrixFamily::Block{left_bond_space_[key.row_sector].dim,
                                                    right_bond_space_[key.col_sector].dim};
    }

    /// \brief Return MatrixFamily block shapes in layout order.
    /// \return Dense block shapes.
    auto matrix_family_blocks() const -> std::vector<tensorcontraction::MatrixFamily::Block>
    {
      std::vector<tensorcontraction::MatrixFamily::Block> blocks;
      blocks.reserve(keys_.size());
      for (auto key : keys_)
      {
        blocks.push_back(this->block_shape(key));
      }
      return blocks;
    }

  private:
    void build_allowed_blocks()
    {
      for (std::size_t row = 0; row < left_bond_space_.size(); ++row)
      {
        for (std::size_t local = 0; local < pair_space_.size(); ++local)
        {
          for (std::size_t col = 0; col < right_bond_space_.size(); ++col)
          {
            ThreeLegBlockKey const key{.row_sector = row, .local = local, .col_sector = col};
            if (three_leg_block_allowed(left_bond_space_, pair_space_, right_bond_space_, key))
            {
              lookup_.emplace(key, keys_.size());
              keys_.push_back(key);
            }
          }
        }
      }
      if (keys_.empty())
      {
        throw std::invalid_argument("two-site block layout has no symmetry-allowed blocks");
      }
    }

    LocalSpace left_physical_space_;
    LocalSpace right_physical_space_;
    LocalSpace pair_space_;
    BlockSpace left_bond_space_;
    BlockSpace right_bond_space_;
    std::vector<ThreeLegBlockKey> keys_;
    std::unordered_map<ThreeLegBlockKey, std::size_t, ThreeLegBlockKeyHash> lookup_;
};

/// \brief Construct an empty MatrixFamily matching a block-sparse layout.
/// \param layout Two-site block layout.
/// \return Zero-filled MatrixFamily.
inline auto make_zero_matrix_family(BlockSparseTwoSiteLayout const& layout) -> tensorcontraction::MatrixFamily
{
  return tensorcontraction::MatrixFamily(layout.matrix_family_blocks());
}

/// \brief Validate that a MatrixFamily matches a block-sparse two-site layout.
/// \throws std::invalid_argument If the shape differs.
/// \param family MatrixFamily to validate.
/// \param layout Expected layout.
inline void validate_matrix_family_layout(tensorcontraction::MatrixFamily const& family,
                                          BlockSparseTwoSiteLayout const& layout)
{
  if (family.size() != layout.block_count())
  {
    throw std::invalid_argument("MatrixFamily block count does not match the two-site block layout");
  }
  for (std::size_t i = 0; i < family.size(); ++i)
  {
    if (family.block(i) != layout.block_shape(layout.keys()[i]))
    {
      throw std::invalid_argument("MatrixFamily block shape does not match the two-site block layout");
    }
  }
}

/// \brief Build the block-sparse two-site center vector from adjacent MPS sites.
/// \param psi Block-sparse MPS.
/// \param left_site Left site index.
/// \param layout Two-site layout.
/// \return MatrixFamily containing only legal U(1) blocks.
inline auto make_two_site_vector(BlockSparseFiniteMPS const& psi, std::size_t left_site,
                                 BlockSparseTwoSiteLayout const& layout) -> tensorcontraction::MatrixFamily
{
  if (left_site + 1 >= psi.size())
  {
    throw std::out_of_range("block-sparse two-site vector requires two adjacent MPS sites");
  }
  auto const& left = psi[left_site];
  auto const& right = psi[left_site + 1];
  if (left.col_space() != right.row_space())
  {
    throw std::invalid_argument("block-sparse two-site vector adjacent bond spaces do not match");
  }
  if (layout.left_physical_space() != left.local_space() || layout.right_physical_space() != right.local_space() ||
      layout.left_bond_space() != left.row_space() || layout.right_bond_space() != right.col_space())
  {
    throw std::invalid_argument("block-sparse two-site vector layout does not match MPS sites");
  }

  auto result = make_zero_matrix_family(layout);
  for (auto const& left_block : left.blocks())
  {
    for (auto right_index : right.blocks_from_row(left_block.key.col_sector))
    {
      auto const& right_block = right.blocks()[right_index];
      auto const pair = two_site_pair_index(right.local_space().size(), left_block.key.local, right_block.key.local);
      ThreeLegBlockKey const out_key{
          .row_sector = left_block.key.row_sector, .local = pair, .col_sector = right_block.key.col_sector};
      if (!layout.contains(out_key))
      {
        throw std::logic_error("MPS block product produced a forbidden two-site block");
      }
      auto out = result.values(layout.block_index(out_key));
      detail::add_dense_gemm(left.values(left_block.key), left_block.rows, left_block.cols, right.values(right_index),
                             right_block.cols, out, 1.0);
    }
  }
  return result;
}

/// \brief Build the block-sparse two-site center vector and keep it resident.
/// \param psi Block-sparse MPS.
/// \param left_site Left site index.
/// \param layout Two-site layout.
/// \param algebra Resident TensorContraction vector algebra engine.
/// \return MatrixFamily whose current values live in the engine's resident buffers.
inline auto
make_two_site_vector_resident(BlockSparseFiniteMPS const& psi, std::size_t left_site,
                              BlockSparseTwoSiteLayout const& layout,
                              tensorcontraction::VectorAlgebraEngine& algebra) -> tensorcontraction::MatrixFamily
{
  if (left_site + 1 >= psi.size())
  {
    throw std::out_of_range("block-sparse two-site vector requires two adjacent MPS sites");
  }
  auto const& left = psi[left_site];
  auto const& right = psi[left_site + 1];
  if (left.col_space() != right.row_space())
  {
    throw std::invalid_argument("block-sparse two-site vector adjacent bond spaces do not match");
  }
  if (layout.left_physical_space() != left.local_space() || layout.right_physical_space() != right.local_space() ||
      layout.left_bond_space() != left.row_space() || layout.right_bond_space() != right.col_space())
  {
    throw std::invalid_argument("block-sparse two-site vector layout does not match MPS sites");
  }

  std::vector<tensorcontraction::MatrixFamily::Block> left_blocks;
  left_blocks.reserve(left.block_count());
  for (auto const& block : left.blocks())
  {
    left_blocks.push_back(tensorcontraction::MatrixFamily::Block{block.rows, block.cols});
  }

  std::vector<tensorcontraction::MatrixFamily::Block> right_blocks;
  right_blocks.reserve(right.block_count());
  for (auto const& block : right.blocks())
  {
    right_blocks.push_back(tensorcontraction::MatrixFamily::Block{block.rows, block.cols});
  }

  tensorcontraction::MatrixFamily left_operands(left_blocks);
  tensorcontraction::MatrixFamily right_operands(right_blocks);
  for (std::size_t block = 0; block < left.block_count(); ++block)
  {
    left_operands.assign(block, left.values(block));
  }
  for (std::size_t block = 0; block < right.block_count(); ++block)
  {
    right_operands.assign(block, right.values(block));
  }

  std::vector<std::size_t> left_block_for_product;
  std::vector<std::size_t> right_block_for_product;
  std::vector<std::size_t> result_block_for_product;
  for (std::size_t left_index = 0; left_index < left.block_count(); ++left_index)
  {
    auto const& left_block = left.blocks()[left_index];
    for (auto right_index : right.blocks_from_row(left_block.key.col_sector))
    {
      auto const& right_block = right.blocks()[right_index];
      auto const pair = two_site_pair_index(right.local_space().size(), left_block.key.local, right_block.key.local);
      ThreeLegBlockKey const out_key{
          .row_sector = left_block.key.row_sector, .local = pair, .col_sector = right_block.key.col_sector};
      if (!layout.contains(out_key))
      {
        throw std::logic_error("MPS block product produced a forbidden two-site block");
      }
      left_block_for_product.push_back(left_index);
      right_block_for_product.push_back(right_index);
      result_block_for_product.push_back(layout.block_index(out_key));
    }
  }

  auto result = make_zero_matrix_family(layout);
  algebra.gemm_sparse_selected_to_resident(left_operands, right_operands, result, left_block_for_product,
                                           right_block_for_product, result_block_for_product);
  return result;
}

namespace detail
{

inline void add_effective_two_site_block(std::span<double const> left_env, std::span<double const> input,
                                         std::span<double const> right_env, std::span<double> output,
                                         std::size_t out_left_dim, std::size_t in_left_dim, std::size_t out_right_dim,
                                         std::size_t in_right_dim, double coefficient)
{
  for (std::size_t out_left = 0; out_left < out_left_dim; ++out_left)
  {
    for (std::size_t in_left = 0; in_left < in_left_dim; ++in_left)
    {
      double const left_value = coefficient * left_env[out_left * in_left_dim + in_left];
      if (left_value == 0.0)
      {
        continue;
      }
      for (std::size_t out_right = 0; out_right < out_right_dim; ++out_right)
      {
        double value = 0.0;
        for (std::size_t in_right = 0; in_right < in_right_dim; ++in_right)
        {
          value += input[in_left * in_right_dim + in_right] * right_env[out_right * in_right_dim + in_right];
        }
        output[out_left * out_right_dim + out_right] += left_value * value;
      }
    }
  }
}

} // namespace detail

/// \brief Apply the strict U(1) two-site effective Hamiltonian.
/// \param left_env Environment left of the two-site center.
/// \param left_mpo Sparse MPO site at the left center site.
/// \param right_mpo Sparse MPO site at the right center site.
/// \param right_env Environment right of the two-site center.
/// \param layout Two-site block layout for input and output.
/// \param input Input block vector.
/// \param output Output block vector.
inline void apply_two_site_effective_hamiltonian(BlockSparseEnvironment const& left_env, SparseMpoSite const& left_mpo,
                                                 SparseMpoSite const& right_mpo,
                                                 BlockSparseEnvironment const& right_env,
                                                 BlockSparseTwoSiteLayout const& layout,
                                                 tensorcontraction::MatrixFamily const& input,
                                                 tensorcontraction::MatrixFamily& output)
{
  validate_matrix_family_layout(input, layout);
  validate_matrix_family_layout(output, layout);
  if (left_env.local_space() != left_mpo.left_virtual_space() ||
      left_mpo.right_virtual_space() != right_mpo.left_virtual_space() ||
      right_env.local_space() != right_mpo.right_virtual_space())
  {
    throw std::invalid_argument("block-sparse effective Hamiltonian virtual spaces do not match");
  }

  tensorcontraction::zero(output);
  for (auto const& left_env_block : left_env.blocks())
  {
    auto const left_env_values = left_env.values(left_env_block.key);
    for (auto left_entry_index : left_mpo.entries_from_left_virtual(left_env_block.key.local))
    {
      auto const& left_entry = left_mpo.entries()[left_entry_index];
      for (auto right_entry_index : right_mpo.entries_from_left_virtual(left_entry.key.right_virtual))
      {
        auto const& right_entry = right_mpo.entries()[right_entry_index];
        for (auto right_env_index : right_env.blocks_for_local(right_entry.key.right_virtual))
        {
          auto const& right_env_block = right_env.blocks()[right_env_index];
          auto const ket_pair =
              two_site_pair_index(layout.right_physical_space().size(), left_entry.key.ket, right_entry.key.ket);
          auto const bra_pair =
              two_site_pair_index(layout.right_physical_space().size(), left_entry.key.bra, right_entry.key.bra);
          ThreeLegBlockKey const input_key{.row_sector = left_env_block.key.col_sector,
                                           .local = ket_pair,
                                           .col_sector = right_env_block.key.col_sector};
          ThreeLegBlockKey const output_key{.row_sector = left_env_block.key.row_sector,
                                            .local = bra_pair,
                                            .col_sector = right_env_block.key.row_sector};
          if (!layout.contains(input_key) || !layout.contains(output_key))
          {
            continue;
          }

          auto const coefficient = left_entry.value * right_entry.value;
          detail::add_effective_two_site_block(
              left_env_values, input.values(layout.block_index(input_key)), right_env.values(right_env_index),
              output.values(layout.block_index(output_key)), left_env_block.rows, left_env_block.cols,
              right_env_block.rows, right_env_block.cols, coefficient);
        }
      }
    }
  }
}

/// \brief Resident TensorContraction representation of a strict U(1) two-site Hamiltonian.
struct BlockSparseTwoSiteEffectiveHamiltonian
{
    tensorcontraction::EffectiveHamiltonianOperator op;
    BlockSparseTwoSiteLayout layout;
};

/// \brief Wall-clock and process-CPU timing for one profiled DMRG substage.
struct BlockSparseStageTiming
{
    double wall_seconds = 0.0;
    double cpu_seconds = 0.0;
};

/// \brief Fine-grained timing data for one block-sparse two-site solve.
struct BlockSparseTwoSiteSolveTimings
{
    BlockSparseStageTiming layout;
    BlockSparseStageTiming effective_hamiltonian;
    BlockSparseStageTiming engine;
    BlockSparseStageTiming initial_vector;
    BlockSparseStageTiming lanczos;
};

/// \brief Fine-grained timing data for one block-sparse two-site split.
struct BlockSparseTwoSiteSplitTimings
{
    BlockSparseStageTiming sectors;
    BlockSparseStageTiming plan;
    BlockSparseStageTiming svd;
    BlockSparseStageTiming metadata;
    BlockSparseStageTiming materialize;
};

namespace detail
{

/// \brief Wall-clock and process-CPU checkpoint for fine-grained profiling.
struct ProfileCheckpoint
{
    std::chrono::steady_clock::time_point wall_time;
    double process_cpu_seconds = 0.0;
};

/// \brief Return process CPU seconds consumed by the current benchmark process.
/// \return CPU seconds accumulated by this process.
inline auto profile_cpu_seconds() -> double
{
  return static_cast<double>(std::clock()) / static_cast<double>(CLOCKS_PER_SEC);
}

/// \brief Capture a fine-grained profile timing checkpoint.
/// \return Fine-grained profiling checkpoint.
inline auto profile_checkpoint() -> ProfileCheckpoint
{
  return ProfileCheckpoint{.wall_time = std::chrono::steady_clock::now(), .process_cpu_seconds = profile_cpu_seconds()};
}

/// \brief Return elapsed profile timing between checkpoints.
/// \param start Initial checkpoint.
/// \param stop Final checkpoint.
/// \return Wall-clock and process-CPU elapsed times.
inline auto profile_elapsed(ProfileCheckpoint const& start, ProfileCheckpoint const& stop) -> BlockSparseStageTiming
{
  return BlockSparseStageTiming{
      .wall_seconds = std::chrono::duration<double>(stop.wall_time - start.wall_time).count(),
      .cpu_seconds = stop.process_cpu_seconds - start.process_cpu_seconds,
  };
}

inline auto make_environment_matrix_family(BlockSparseEnvironment const& env) -> tensorcontraction::MatrixFamily
{
  std::vector<tensorcontraction::MatrixFamily::Block> blocks;
  blocks.reserve(env.block_count());
  for (auto const& block : env.blocks())
  {
    blocks.push_back(tensorcontraction::MatrixFamily::Block{block.rows, block.cols});
  }

  tensorcontraction::MatrixFamily family(blocks);
  auto source = env.coalesced_values();
  auto target = family.coalesced_values();
  if (source.size() != target.size())
  {
    throw std::logic_error("environment MatrixFamily staging encountered incompatible coalesced storage");
  }
  std::copy(source.begin(), source.end(), target.begin());
  return family;
}

inline auto
make_transposed_environment_matrix_family(BlockSparseEnvironment const& env) -> tensorcontraction::MatrixFamily
{
  std::vector<tensorcontraction::MatrixFamily::Block> blocks;
  blocks.reserve(env.block_count());
  for (auto const& block : env.blocks())
  {
    blocks.push_back(tensorcontraction::MatrixFamily::Block{block.cols, block.rows});
  }

  tensorcontraction::MatrixFamily family(blocks);
  for (std::size_t block = 0; block < env.block_count(); ++block)
  {
    auto const source_block = env.blocks()[block];
    auto const source = env.values(block);
    auto target = family.values(block);
    for (std::size_t row = 0; row < source_block.rows; ++row)
    {
      for (std::size_t col = 0; col < source_block.cols; ++col)
      {
        target[col * source_block.rows + row] = source[row * source_block.cols + col];
      }
    }
  }
  return family;
}

inline auto optional_env_size(char const* name) -> std::optional<std::size_t>
{
  char const* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0')
  {
    return std::nullopt;
  }

  std::string const text(raw);
  std::size_t consumed = 0;
  auto const value = std::stoull(text, &consumed);
  if (consumed != text.size())
  {
    throw std::invalid_argument(std::string("invalid integer value for ") + name + ": " + text);
  }
  return static_cast<std::size_t>(value);
}

inline auto env_flag_enabled(char const* name) -> bool
{
  char const* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0')
  {
    return false;
  }

  std::string const value(raw);
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

inline auto rabc_fixture_match_counter() -> std::size_t&
{
  static std::size_t counter = 0;
  return counter;
}

inline void maybe_dump_rabc_fixture(BlockSparseFiniteMPS const& psi, std::size_t left_site,
                                    BlockSparseTwoSiteLayout const& layout,
                                    tensorcontraction::EffectiveHamiltonianOperator const& op)
{
  char const* path = std::getenv("UNI20_RABC_DUMP_PATH");
  if (path == nullptr || *path == '\0')
  {
    return;
  }

  if (auto const target_left_site = optional_env_size("UNI20_RABC_DUMP_LEFT_SITE");
      target_left_site.has_value() && left_site != *target_left_site)
  {
    return;
  }

  if (auto const min_bond_dim = optional_env_size("UNI20_RABC_DUMP_MIN_BOND_DIM"); min_bond_dim.has_value())
  {
    if (layout.left_bond_space().total_dim() < *min_bond_dim || layout.right_bond_space().total_dim() < *min_bond_dim)
    {
      return;
    }
  }

  auto& match_counter = rabc_fixture_match_counter();
  auto const match_index = match_counter++;
  if (auto const target_match = optional_env_size("UNI20_RABC_DUMP_MATCH_INDEX");
      target_match.has_value() && match_index != *target_match)
  {
    return;
  }

  auto host_input = make_two_site_vector(psi, left_site, layout);
  tensorcontraction::write_variable_middle_rabc_fixture(path, op, host_input);
  std::fprintf(stderr, "[UNI20][RABC_FIXTURE] wrote %s left_site=%zu match=%zu left_dim=%zu right_dim=%zu blocks=%zu\n",
               path, left_site, match_index, layout.left_bond_space().total_dim(),
               layout.right_bond_space().total_dim(), layout.block_count());
  if (env_flag_enabled("UNI20_RABC_DUMP_EXIT"))
  {
    std::fflush(stderr);
    std::exit(0);
  }
}

inline void validate_block_sparse_effective_hamiltonian_inputs(BlockSparseEnvironment const& left_env,
                                                               SparseMpoSite const& left_mpo,
                                                               SparseMpoSite const& right_mpo,
                                                               BlockSparseEnvironment const& right_env)
{
  if (left_env.local_space() != left_mpo.left_virtual_space() ||
      left_mpo.right_virtual_space() != right_mpo.left_virtual_space() ||
      right_env.local_space() != right_mpo.right_virtual_space())
  {
    throw std::invalid_argument("block-sparse effective Hamiltonian virtual spaces do not match");
  }
}

} // namespace detail

/// \brief Compile a strict U(1) two-site Hamiltonian into TensorContraction terms.
/// \param left_env Environment left of the two-site center.
/// \param left_mpo Sparse MPO site at the left center site.
/// \param right_mpo Sparse MPO site at the right center site.
/// \param right_env Environment right of the two-site center.
/// \param layout Legal two-site input/output layout.
/// \return Resident-capable TensorContraction operator and layout metadata.
inline auto
make_two_site_effective_hamiltonian(BlockSparseEnvironment const& left_env, SparseMpoSite const& left_mpo,
                                    SparseMpoSite const& right_mpo, BlockSparseEnvironment const& right_env,
                                    BlockSparseTwoSiteLayout layout) -> BlockSparseTwoSiteEffectiveHamiltonian
{
  detail::validate_block_sparse_effective_hamiltonian_inputs(left_env, left_mpo, right_mpo, right_env);

  auto a_mats = detail::make_environment_matrix_family(left_env);
  auto c_mats = detail::make_transposed_environment_matrix_family(right_env);
  auto const input_blocks = layout.matrix_family_blocks();
  auto const output_blocks = input_blocks;
  std::vector<tensorcontraction::EffectiveHamiltonianOperator::Term> terms;

  for (std::size_t left_env_index = 0; left_env_index < left_env.block_count(); ++left_env_index)
  {
    auto const& left_env_block = left_env.blocks()[left_env_index];
    for (auto left_entry_index : left_mpo.entries_from_left_virtual(left_env_block.key.local))
    {
      auto const& left_entry = left_mpo.entries()[left_entry_index];
      for (auto right_entry_index : right_mpo.entries_from_left_virtual(left_entry.key.right_virtual))
      {
        auto const& right_entry = right_mpo.entries()[right_entry_index];
        for (auto right_env_index : right_env.blocks_for_local(right_entry.key.right_virtual))
        {
          auto const& right_env_block = right_env.blocks()[right_env_index];
          auto const ket_pair =
              two_site_pair_index(layout.right_physical_space().size(), left_entry.key.ket, right_entry.key.ket);
          auto const bra_pair =
              two_site_pair_index(layout.right_physical_space().size(), left_entry.key.bra, right_entry.key.bra);
          ThreeLegBlockKey const input_key{.row_sector = left_env_block.key.col_sector,
                                           .local = ket_pair,
                                           .col_sector = right_env_block.key.col_sector};
          ThreeLegBlockKey const output_key{.row_sector = left_env_block.key.row_sector,
                                            .local = bra_pair,
                                            .col_sector = right_env_block.key.row_sector};
          if (!layout.contains(input_key) || !layout.contains(output_key))
          {
            continue;
          }

          terms.push_back(tensorcontraction::EffectiveHamiltonianOperator::Term{.r = layout.block_index(output_key),
                                                                                .a = left_env_index,
                                                                                .b = layout.block_index(input_key),
                                                                                .c = right_env_index,
                                                                                .coefficient = left_entry.value *
                                                                                               right_entry.value});
        }
      }
    }
  }

  if (terms.empty())
  {
    throw std::invalid_argument("block-sparse effective Hamiltonian has no legal TensorContraction terms");
  }

  return BlockSparseTwoSiteEffectiveHamiltonian{
      .op = tensorcontraction::EffectiveHamiltonianOperator::variable_middle(std::move(a_mats), std::move(c_mats),
                                                                             input_blocks, output_blocks, terms),
      .layout = std::move(layout)};
}

/// \brief Result of splitting a U(1)-block-sparse two-site center.
struct BlockSparseTwoSiteSplitResult
{
    ThreeLegBlockMatrix left;
    ThreeLegBlockMatrix right;
    tensorcontraction::SvdSpectrum spectrum;
    std::vector<std::size_t> sector_ranks;
    BlockSparseTwoSiteSplitTimings timings;
};

/// \brief Device-resident result of splitting a U(1)-block-sparse two-site center.
struct DeviceBlockSparseTwoSiteSplitResult
{
    DeviceThreeLegBlockMatrix left;
    DeviceThreeLegBlockMatrix right;
    tensorcontraction::SvdSpectrum spectrum;
    std::vector<std::size_t> sector_ranks;
    BlockSparseTwoSiteSplitTimings timings;

    /// \brief Explicitly materialize the split tensors into host storage.
    /// \return Host split result with synchronized block payloads.
    auto materialize_to_host() -> BlockSparseTwoSiteSplitResult
    {
      return BlockSparseTwoSiteSplitResult{.left = left.materialize_to_host(),
                                           .right = right.materialize_to_host(),
                                           .spectrum = spectrum,
                                           .sector_ranks = sector_ranks,
                                           .timings = timings};
    }
};

namespace detail
{

struct SectorSingularValue
{
    std::size_t sector = 0;
    std::size_t rank = 0;
    double value = 0.0;
};

inline auto make_single_block_matrix(std::size_t rows, std::size_t cols) -> tensorcontraction::MatrixFamily
{
  std::vector<tensorcontraction::MatrixFamily::Block> blocks{tensorcontraction::MatrixFamily::Block{rows, cols}};
  return tensorcontraction::MatrixFamily(blocks);
}

inline auto assemble_svd_sector_matrix(BlockSparseTwoSiteLayout const& layout,
                                       tensorcontraction::MatrixFamily const& center,
                                       TwoSiteSvdSector const& sector) -> tensorcontraction::MatrixFamily
{
  auto matrix = make_single_block_matrix(sector.row_dim, sector.col_dim);
  auto values = matrix.values(0);
  for (auto const& row_term : sector.rows)
  {
    for (auto const& col_term : sector.cols)
    {
      auto const pair =
          two_site_pair_index(layout.right_physical_space().size(), row_term.left_physical, col_term.right_physical);
      ThreeLegBlockKey const key{
          .row_sector = row_term.left_sector, .local = pair, .col_sector = col_term.right_sector};
      if (!layout.contains(key))
      {
        continue;
      }
      auto const source = center.values(layout.block_index(key));
      for (std::size_t row = 0; row < row_term.dim; ++row)
      {
        for (std::size_t col = 0; col < col_term.dim; ++col)
        {
          values[(row_term.offset + row) * sector.col_dim + col_term.offset + col] = source[row * col_term.dim + col];
        }
      }
    }
  }
  return matrix;
}

inline auto select_sector_ranks(std::span<tensorcontraction::SingleBlockSvd const> svds,
                                tensorcontraction::SvdOptions options)
    -> std::pair<std::vector<std::size_t>, tensorcontraction::SvdSpectrum>
{
  if (options.max_rank == 0)
  {
    throw std::invalid_argument("block-sparse SVD requires a positive max_rank");
  }
  if (options.cutoff < 0.0 || std::isnan(options.cutoff))
  {
    throw std::invalid_argument("block-sparse SVD requires a finite non-negative cutoff");
  }

  std::vector<SectorSingularValue> candidates;
  std::vector<SectorSingularValue> positive_values;
  std::size_t full_rank = 0;
  for (std::size_t sector = 0; sector < svds.size(); ++sector)
  {
    for (std::size_t rank = 0; rank < svds[sector].singular_values.size(); ++rank)
    {
      double const value = svds[sector].singular_values[rank];
      if (value > 0.0)
      {
        ++full_rank;
        positive_values.push_back(SectorSingularValue{.sector = sector, .rank = rank, .value = value});
      }
      if (value > options.cutoff)
      {
        candidates.push_back(SectorSingularValue{.sector = sector, .rank = rank, .value = value});
      }
    }
  }
  if (candidates.empty() && !positive_values.empty())
  {
    candidates.push_back(*std::max_element(positive_values.begin(), positive_values.end(),
                                           [](auto const& lhs, auto const& rhs) { return lhs.value < rhs.value; }));
  }
  std::sort(candidates.begin(), candidates.end(), [](auto const& lhs, auto const& rhs) {
    if (lhs.value != rhs.value)
    {
      return lhs.value > rhs.value;
    }
    if (lhs.sector != rhs.sector)
    {
      return lhs.sector < rhs.sector;
    }
    return lhs.rank < rhs.rank;
  });
  if (candidates.size() > options.max_rank)
  {
    candidates.resize(options.max_rank);
  }

  std::vector<std::size_t> ranks(svds.size(), 0);
  for (auto const& singular : candidates)
  {
    ranks[singular.sector] = std::max(ranks[singular.sector], singular.rank + 1);
  }

  tensorcontraction::SvdSpectrum spectrum;
  spectrum.full_rank = full_rank;
  std::vector<double> kept_values;
  for (std::size_t sector = 0; sector < svds.size(); ++sector)
  {
    for (std::size_t rank = 0; rank < ranks[sector]; ++rank)
    {
      kept_values.push_back(svds[sector].singular_values[rank]);
    }
    for (std::size_t rank = ranks[sector]; rank < svds[sector].singular_values.size(); ++rank)
    {
      double const value = svds[sector].singular_values[rank];
      spectrum.discarded_weight += value * value;
    }
  }
  std::sort(kept_values.begin(), kept_values.end(), std::greater<>{});
  spectrum.singular_values = std::move(kept_values);
  return {std::move(ranks), std::move(spectrum)};
}

inline auto shared_sector_indexes(std::span<std::size_t const> ranks) -> std::vector<std::size_t>
{
  std::vector<std::size_t> indexes(ranks.size(), std::numeric_limits<std::size_t>::max());
  std::size_t next = 0;
  for (std::size_t i = 0; i < ranks.size(); ++i)
  {
    if (ranks[i] != 0)
    {
      indexes[i] = next++;
    }
  }
  return indexes;
}

inline auto make_resident_block_sparse_svd_plan(BlockSparseTwoSiteLayout const& layout,
                                                std::span<TwoSiteSvdSector const> sectors)
    -> tensorcontraction::ResidentBlockSparseSvdPlan
{
  tensorcontraction::ResidentBlockSparseSvdPlan plan;
  plan.sectors.reserve(sectors.size());
  for (auto const& sector : sectors)
  {
    tensorcontraction::ResidentBlockSvdSector resident_sector{
        .row_dim = sector.row_dim,
        .col_dim = sector.col_dim,
        .source_terms = {},
        .left_terms = {},
        .right_terms = {},
    };
    resident_sector.left_terms.reserve(sector.rows.size());
    for (auto const& row_term : sector.rows)
    {
      resident_sector.left_terms.push_back(
          tensorcontraction::ResidentBlockSvdTerm{.offset = row_term.offset, .extent = row_term.dim});
    }
    resident_sector.right_terms.reserve(sector.cols.size());
    for (auto const& col_term : sector.cols)
    {
      resident_sector.right_terms.push_back(
          tensorcontraction::ResidentBlockSvdTerm{.offset = col_term.offset, .extent = col_term.dim});
    }

    for (auto const& row_term : sector.rows)
    {
      for (auto const& col_term : sector.cols)
      {
        auto const pair =
            two_site_pair_index(layout.right_physical_space().size(), row_term.left_physical, col_term.right_physical);
        ThreeLegBlockKey const key{
            .row_sector = row_term.left_sector, .local = pair, .col_sector = col_term.right_sector};
        if (!layout.contains(key))
        {
          continue;
        }
        resident_sector.source_terms.push_back(tensorcontraction::ResidentBlockSvdSourceTerm{
            .source_block = layout.block_index(key),
            .row_offset = row_term.offset,
            .col_offset = col_term.offset,
        });
      }
    }

    plan.sectors.push_back(std::move(resident_sector));
  }
  return plan;
}

inline auto make_device_split_blocks(std::span<TwoSiteSvdSector const> sectors, std::span<std::size_t const> ranks,
                                     std::span<std::size_t const> shared_indexes,
                                     bool left_side) -> std::vector<ThreeLegBlock>
{
  std::vector<ThreeLegBlock> blocks;
  for (std::size_t sector_index = 0; sector_index < sectors.size(); ++sector_index)
  {
    auto const rank = ranks[sector_index];
    if (rank == 0)
    {
      continue;
    }
    auto const shared_index = shared_indexes[sector_index];
    auto const& sector = sectors[sector_index];
    if (left_side)
    {
      for (auto const& row_term : sector.rows)
      {
        blocks.push_back(ThreeLegBlock{
            .key = ThreeLegBlockKey{.row_sector = row_term.left_sector,
                                    .local = row_term.left_physical,
                                    .col_sector = shared_index},
            .rows = row_term.dim,
            .cols = rank,
            .offset = 0,
        });
      }
      continue;
    }
    for (auto const& col_term : sector.cols)
    {
      blocks.push_back(ThreeLegBlock{
          .key = ThreeLegBlockKey{.row_sector = shared_index,
                                  .local = col_term.right_physical,
                                  .col_sector = col_term.right_sector},
          .rows = rank,
          .cols = col_term.dim,
          .offset = 0,
      });
    }
  }
  return blocks;
}

} // namespace detail

/// \brief Split a legal U(1) two-site center back into neighboring MPS tensors.
/// \param center MatrixFamily center in `layout` order.
/// \param layout Two-site block layout.
/// \param direction Direction that determines which side absorbs singular values.
/// \param options Truncation options.
/// \return Block-sparse replacement sites and spectrum metadata.
inline auto split_two_site_center(BlockSparseTwoSiteLayout const& layout, tensorcontraction::MatrixFamily const& center,
                                  TwoSiteSplitDirection direction,
                                  tensorcontraction::SvdOptions options = {}) -> BlockSparseTwoSiteSplitResult
{
  validate_matrix_family_layout(center, layout);
  auto const sectors = make_two_site_svd_sectors(layout.left_physical_space(), layout.right_physical_space(),
                                                 layout.left_bond_space(), layout.right_bond_space());
  std::vector<tensorcontraction::SingleBlockSvd> svds;
  svds.reserve(sectors.size());
  for (auto const& sector : sectors)
  {
    auto matrix = detail::assemble_svd_sector_matrix(layout, center, sector);
    svds.push_back(tensorcontraction::single_block_svd_cusolver_required(matrix));
  }

  auto [ranks, spectrum] = detail::select_sector_ranks(svds, options);
  auto shared_space = make_shared_bond_space_from_sector_ranks(sectors, ranks);
  auto const shared_indexes = detail::shared_sector_indexes(ranks);
  ThreeLegBlockMatrix left(layout.left_bond_space(), layout.left_physical_space(), shared_space);
  ThreeLegBlockMatrix right(shared_space, layout.right_physical_space(), layout.right_bond_space());
  bool const absorb_left = direction == TwoSiteSplitDirection::RightToLeft;

  for (std::size_t sector_index = 0; sector_index < sectors.size(); ++sector_index)
  {
    auto const rank = ranks[sector_index];
    if (rank == 0)
    {
      continue;
    }
    auto const shared_index = shared_indexes[sector_index];
    auto const& sector = sectors[sector_index];
    auto const& svd = svds[sector_index];
    auto const u_values = svd.u.values(0);
    auto const vt_values = svd.vt.values(0);
    auto const svd_rank = svd.singular_values.size();

    for (auto const& row_term : sector.rows)
    {
      ThreeLegBlockKey const key{
          .row_sector = row_term.left_sector, .local = row_term.left_physical, .col_sector = shared_index};
      auto values = left.insert_zero_block(key);
      for (std::size_t row = 0; row < row_term.dim; ++row)
      {
        for (std::size_t bond = 0; bond < rank; ++bond)
        {
          double value = u_values[(row_term.offset + row) * svd_rank + bond];
          if (absorb_left)
          {
            value *= svd.singular_values[bond];
          }
          values[row * rank + bond] = value;
        }
      }
    }

    for (auto const& col_term : sector.cols)
    {
      ThreeLegBlockKey const key{
          .row_sector = shared_index, .local = col_term.right_physical, .col_sector = col_term.right_sector};
      auto values = right.insert_zero_block(key);
      for (std::size_t bond = 0; bond < rank; ++bond)
      {
        for (std::size_t col = 0; col < col_term.dim; ++col)
        {
          double value = vt_values[bond * sector.col_dim + col_term.offset + col];
          if (!absorb_left)
          {
            value *= svd.singular_values[bond];
          }
          values[bond * col_term.dim + col] = value;
        }
      }
    }
  }

  return BlockSparseTwoSiteSplitResult{.left = std::move(left),
                                       .right = std::move(right),
                                       .spectrum = std::move(spectrum),
                                       .sector_ranks = std::move(ranks),
                                       .timings = {}};
}

/// \brief Result of one block-sparse two-site solve.
struct BlockSparseTwoSiteSolveResult
{
    tensorcontraction::LanczosResult lanczos;
    BlockSparseTwoSiteLayout layout;
    tensorcontraction::MatrixFamily optimized_vector;
    std::shared_ptr<tensorcontraction::VectorAlgebraEngine> resident_algebra;
    BlockSparseTwoSiteSolveTimings timings;
};

/// \brief Solve one strict U(1) two-site effective Hamiltonian.
/// \param psi Current MPS.
/// \param mpo Sparse MPO chain.
/// \param left_site Left site of the two-site center.
/// \param left_env Environment left of the center.
/// \param right_env Environment right of the center.
/// \param options Lanczos options.
/// \return Optimized local state and Lanczos metadata.
inline auto solve_two_site(BlockSparseFiniteMPS const& psi, BlockSparseMpoChain const& mpo, std::size_t left_site,
                           BlockSparseEnvironment const& left_env, BlockSparseEnvironment const& right_env,
                           tensorcontraction::LanczosOptions options = {}) -> BlockSparseTwoSiteSolveResult
{
  if (left_site + 1 >= psi.size() || left_site + 1 >= mpo.size())
  {
    throw std::out_of_range("block-sparse solve_two_site requires two adjacent MPS and MPO sites");
  }

  BlockSparseTwoSiteSolveTimings timings;
  auto stage_start = detail::profile_checkpoint();
  BlockSparseTwoSiteLayout layout(psi[left_site].local_space(), psi[left_site + 1].local_space(),
                                  psi[left_site].row_space(), psi[left_site + 1].col_space());
  auto stage_stop = detail::profile_checkpoint();
  timings.layout = detail::profile_elapsed(stage_start, stage_stop);

  stage_start = stage_stop;
  auto effective_hamiltonian =
      make_two_site_effective_hamiltonian(left_env, mpo[left_site], mpo[left_site + 1], right_env, std::move(layout));
  detail::maybe_dump_rabc_fixture(psi, left_site, effective_hamiltonian.layout, effective_hamiltonian.op);
  stage_stop = detail::profile_checkpoint();
  timings.effective_hamiltonian = detail::profile_elapsed(stage_start, stage_stop);

  stage_start = stage_stop;
  auto algebra = std::make_shared<tensorcontraction::VectorAlgebraEngine>();
  if (algebra->uses_host_backend())
  {
    throw std::runtime_error("block-sparse U(1) DMRG requires the TensorContraction resident CUDA/MPI backend");
  }
  stage_stop = detail::profile_checkpoint();
  timings.engine = detail::profile_elapsed(stage_start, stage_stop);

  stage_start = stage_stop;
  auto optimized_vector = make_two_site_vector_resident(psi, left_site, effective_hamiltonian.layout, *algebra);
  algebra->set_host_synchronization(false);
  stage_stop = detail::profile_checkpoint();
  timings.initial_vector = detail::profile_elapsed(stage_start, stage_stop);

  auto apply = [&](tensorcontraction::MatrixFamily const& x, tensorcontraction::MatrixFamily& y) {
    effective_hamiltonian.op.apply_resident(x, y, *algebra);
  };
  stage_start = stage_stop;
  auto lanczos = tensorcontraction::lanczos_lowest_with_engine(optimized_vector, apply, *algebra, options);
  stage_stop = detail::profile_checkpoint();
  timings.lanczos = detail::profile_elapsed(stage_start, stage_stop);

  return BlockSparseTwoSiteSolveResult{.lanczos = lanczos,
                                       .layout = std::move(effective_hamiltonian.layout),
                                       .optimized_vector = std::move(optimized_vector),
                                       .resident_algebra = std::move(algebra),
                                       .timings = timings};
}

/// \brief Split a solved resident block-sparse center without copying SVD factors to host.
/// \param solution Resident local solve result.
/// \param direction Sweep direction determining singular-value absorption.
/// \param options Truncation options.
/// \return Device-resident split replacement sites.
inline auto
split_two_site_solution_resident(BlockSparseTwoSiteSolveResult& solution, TwoSiteSplitDirection direction,
                                 tensorcontraction::SvdOptions options = {}) -> DeviceBlockSparseTwoSiteSplitResult
{
  if (solution.resident_algebra == nullptr)
  {
    throw std::invalid_argument("resident block-sparse split requires a resident algebra engine");
  }
  validate_matrix_family_layout(solution.optimized_vector, solution.layout);
  BlockSparseTwoSiteSplitTimings timings;
  auto stage_start = detail::profile_checkpoint();
  auto const sectors =
      make_two_site_svd_sectors(solution.layout.left_physical_space(), solution.layout.right_physical_space(),
                                solution.layout.left_bond_space(), solution.layout.right_bond_space());
  auto stage_stop = detail::profile_checkpoint();
  timings.sectors = detail::profile_elapsed(stage_start, stage_stop);

  stage_start = stage_stop;
  auto plan = detail::make_resident_block_sparse_svd_plan(solution.layout, sectors);
  stage_stop = detail::profile_checkpoint();
  timings.plan = detail::profile_elapsed(stage_start, stage_stop);

  stage_start = stage_stop;
  auto resident_split = tensorcontraction::block_sparse_svd_split_resident_required(
      solution.optimized_vector, plan,
      direction == TwoSiteSplitDirection::RightToLeft ? tensorcontraction::SvdAbsorbSingularValues::Left
                                                      : tensorcontraction::SvdAbsorbSingularValues::Right,
      options, *solution.resident_algebra);
  stage_stop = detail::profile_checkpoint();
  timings.svd = detail::profile_elapsed(stage_start, stage_stop);

  stage_start = stage_stop;
  auto shared_space = make_shared_bond_space_from_sector_ranks(sectors, resident_split.sector_ranks);
  auto const shared_indexes = detail::shared_sector_indexes(resident_split.sector_ranks);
  auto left_blocks = detail::make_device_split_blocks(sectors, resident_split.sector_ranks, shared_indexes, true);
  auto right_blocks = detail::make_device_split_blocks(sectors, resident_split.sector_ranks, shared_indexes, false);

  DeviceThreeLegBlockMatrix left(solution.layout.left_bond_space(), solution.layout.left_physical_space(), shared_space,
                                 std::move(left_blocks), std::move(resident_split.left), solution.resident_algebra);
  DeviceThreeLegBlockMatrix right(shared_space, solution.layout.right_physical_space(),
                                  solution.layout.right_bond_space(), std::move(right_blocks),
                                  std::move(resident_split.right), solution.resident_algebra);
  stage_stop = detail::profile_checkpoint();
  timings.metadata = detail::profile_elapsed(stage_start, stage_stop);

  return DeviceBlockSparseTwoSiteSplitResult{.left = std::move(left),
                                             .right = std::move(right),
                                             .spectrum = std::move(resident_split.spectrum),
                                             .sector_ranks = std::move(resident_split.sector_ranks),
                                             .timings = timings};
}

/// \brief Split a solved resident block-sparse center after explicit host synchronization.
/// \param solution Resident local solve result.
/// \param direction Sweep direction determining singular-value absorption.
/// \param options Truncation options.
/// \return Split replacement sites.
inline auto split_two_site_solution(BlockSparseTwoSiteSolveResult& solution, TwoSiteSplitDirection direction,
                                    tensorcontraction::SvdOptions options = {}) -> BlockSparseTwoSiteSplitResult
{
  auto resident_split = split_two_site_solution_resident(solution, direction, options);
  auto const stage_start = detail::profile_checkpoint();
  auto split = resident_split.materialize_to_host();
  auto const stage_stop = detail::profile_checkpoint();
  split.timings.materialize = detail::profile_elapsed(stage_start, stage_stop);
  return split;
}

/// \brief Replace the optimized two-site center in a block-sparse MPS.
/// \param psi MPS to update.
/// \param left_site Left site index.
/// \param split Split replacement sites.
inline void replace_two_site_solution(BlockSparseFiniteMPS& psi, std::size_t left_site,
                                      BlockSparseTwoSiteSplitResult split)
{
  psi.replace_adjacent(left_site, std::move(split.left), std::move(split.right));
}

/// \brief Return the scalar energy from a completed block-sparse environment.
/// \throws std::invalid_argument If the requested environment is not scalar.
/// \param env Completed environment.
/// \param virtual_index MPO virtual index to trace.
/// \return Scalar trace value.
inline auto environment_scalar(BlockSparseEnvironment const& env, std::size_t virtual_index) -> double
{
  if (!is_scalar(env.local_space()[virtual_index]))
  {
    throw std::invalid_argument("block-sparse environment scalar requires a scalar virtual index");
  }
  double value = 0.0;
  for (std::size_t sector = 0; sector < env.row_space().size(); ++sector)
  {
    ThreeLegBlockKey const key{.row_sector = sector, .local = virtual_index, .col_sector = sector};
    if (!env.contains(key))
    {
      continue;
    }
    auto const values = env.values(key);
    auto const dim = env.row_space()[sector].dim;
    for (std::size_t i = 0; i < dim; ++i)
    {
      value += values[i * dim + i];
    }
  }
  return value;
}

/// \brief Compute an MPS/MPO expectation value through block-sparse environments.
/// \param psi MPS state.
/// \param mpo Sparse MPO chain.
/// \return Scalar expectation value.
inline auto mps_expectation_value(BlockSparseFiniteMPS const& psi, BlockSparseMpoChain const& mpo) -> double
{
  auto const left_envs = build_left_environments(psi, mpo);
  auto const& final_env = left_envs.back();
  return environment_scalar(final_env, final_env.local_space().size() - 1);
}

/// \brief Direction of a block-sparse two-site sweep.
enum class BlockSparseTwoSiteSweepDirection
{
  LeftToRight,
  RightToLeft,
};

/// \brief Metadata for one block-sparse two-site bond update.
struct BlockSparseTwoSiteBondUpdate
{
    std::size_t left_site = 0;
    double energy = 0.0;
    tensorcontraction::LanczosResult lanczos;
    double discarded_weight = 0.0;
    std::size_t kept_rank = 0;
    std::size_t full_rank = 0;
    std::optional<BlockSpace> shared_bond_space;
    double solve_seconds = 0.0;
    double split_seconds = 0.0;
    double replace_seconds = 0.0;
    double environment_seconds = 0.0;
    double solve_cpu_seconds = 0.0;
    double split_cpu_seconds = 0.0;
    double replace_cpu_seconds = 0.0;
    double environment_cpu_seconds = 0.0;
    BlockSparseTwoSiteSolveTimings solve_timings;
    BlockSparseTwoSiteSplitTimings split_timings;
};

/// \brief Observer invoked after each block-sparse bond update.
using BlockSparseTwoSiteSweepObserver =
    std::function<void(BlockSparseTwoSiteSweepDirection, BlockSparseTwoSiteBondUpdate const&)>;

/// \brief Options for one block-sparse two-site sweep.
struct BlockSparseTwoSiteSweepOptions
{
    tensorcontraction::LanczosOptions lanczos{};
    tensorcontraction::SvdOptions svd{};
    BlockSparseTwoSiteSweepObserver observer{};
};

/// \brief Result of one block-sparse two-site sweep.
struct BlockSparseTwoSiteSweepResult
{
    BlockSparseTwoSiteSweepDirection direction = BlockSparseTwoSiteSweepDirection::LeftToRight;
    std::vector<BlockSparseTwoSiteBondUpdate> updates;
};

namespace detail
{

/// \brief Wall-clock and process-CPU checkpoint for one benchmarked sweep stage.
struct SweepStageCheckpoint
{
    std::chrono::steady_clock::time_point wall_time;
    double process_cpu_seconds = 0.0;
};

/// \brief Return process CPU seconds consumed by the current benchmark process.
/// \return CPU seconds accumulated by this process.
inline auto process_cpu_seconds() -> double
{
  return static_cast<double>(std::clock()) / static_cast<double>(CLOCKS_PER_SEC);
}

/// \brief Capture wall-clock and process-CPU timing at a sweep stage boundary.
/// \return Stage timing checkpoint.
inline auto sweep_stage_checkpoint() -> SweepStageCheckpoint
{
  return SweepStageCheckpoint{.wall_time = std::chrono::steady_clock::now(),
                              .process_cpu_seconds = process_cpu_seconds()};
}

inline auto elapsed_seconds(std::chrono::steady_clock::time_point start,
                            std::chrono::steady_clock::time_point stop) -> double
{
  return std::chrono::duration<double>(stop - start).count();
}

/// \brief Return wall-clock seconds between two stage checkpoints.
/// \param start Initial checkpoint.
/// \param stop Final checkpoint.
/// \return Wall-clock elapsed seconds.
inline auto elapsed_wall_seconds(SweepStageCheckpoint const& start, SweepStageCheckpoint const& stop) -> double
{
  return elapsed_seconds(start.wall_time, stop.wall_time);
}

/// \brief Return process CPU seconds between two stage checkpoints.
/// \param start Initial checkpoint.
/// \param stop Final checkpoint.
/// \return Process CPU elapsed seconds.
inline auto elapsed_cpu_seconds(SweepStageCheckpoint const& start, SweepStageCheckpoint const& stop) -> double
{
  return stop.process_cpu_seconds - start.process_cpu_seconds;
}

inline void validate_block_sparse_sweep_inputs(BlockSparseFiniteMPS const& psi, BlockSparseMpoChain const& mpo)
{
  if (psi.size() != mpo.size())
  {
    throw std::invalid_argument("block-sparse sweep requires MPS and MPO chains of equal length");
  }
  if (psi.size() < 2)
  {
    throw std::invalid_argument("block-sparse sweep requires at least two sites");
  }
}

inline auto make_block_sparse_bond_update(std::size_t left_site, BlockSparseTwoSiteSolveResult const& solution,
                                          BlockSparseTwoSiteSplitResult const& split, double solve_seconds,
                                          double split_seconds, double replace_seconds, double environment_seconds,
                                          double solve_cpu_seconds, double split_cpu_seconds,
                                          double replace_cpu_seconds,
                                          double environment_cpu_seconds) -> BlockSparseTwoSiteBondUpdate
{
  return BlockSparseTwoSiteBondUpdate{.left_site = left_site,
                                      .energy = solution.lanczos.eigenvalue,
                                      .lanczos = solution.lanczos,
                                      .discarded_weight = split.spectrum.discarded_weight,
                                      .kept_rank = split.spectrum.singular_values.size(),
                                      .full_rank = split.spectrum.full_rank,
                                      .shared_bond_space = split.left.col_space(),
                                      .solve_seconds = solve_seconds,
                                      .split_seconds = split_seconds,
                                      .replace_seconds = replace_seconds,
                                      .environment_seconds = environment_seconds,
                                      .solve_cpu_seconds = solve_cpu_seconds,
                                      .split_cpu_seconds = split_cpu_seconds,
                                      .replace_cpu_seconds = replace_cpu_seconds,
                                      .environment_cpu_seconds = environment_cpu_seconds,
                                      .solve_timings = solution.timings,
                                      .split_timings = split.timings};
}

inline void assign_environment(std::vector<BlockSparseEnvironment>& environments, std::size_t index,
                               BlockSparseEnvironment environment)
{
  if (index < environments.size())
  {
    environments[index] = std::move(environment);
    return;
  }
  if (index == environments.size())
  {
    environments.push_back(std::move(environment));
    return;
  }
  throw std::logic_error("cannot assign a block-sparse environment past the next cache slot");
}

inline void validate_left_to_right_environment_cache(BlockSparseFiniteMPS const& psi,
                                                     std::vector<BlockSparseEnvironment> const& left_envs,
                                                     std::vector<BlockSparseEnvironment> const& right_envs)
{
  if (left_envs.empty())
  {
    throw std::invalid_argument("block-sparse left-to-right sweep requires a left boundary environment");
  }
  if (right_envs.size() != psi.size() + 1)
  {
    throw std::invalid_argument("block-sparse left-to-right sweep requires a complete right environment cache");
  }
}

inline void validate_right_to_left_environment_cache(BlockSparseFiniteMPS const& psi,
                                                     std::vector<BlockSparseEnvironment> const& left_envs,
                                                     std::vector<BlockSparseEnvironment> const& right_envs)
{
  if (left_envs.size() < psi.size())
  {
    throw std::invalid_argument("block-sparse right-to-left sweep requires left environments through the final bond");
  }
  if (right_envs.size() != psi.size() + 1)
  {
    throw std::invalid_argument("block-sparse right-to-left sweep requires a complete right environment cache");
  }
}

} // namespace detail

/// \brief Sweep left-to-right using the strict block-sparse U(1) path.
/// \param psi MPS state to update.
/// \param mpo Sparse MPO chain.
/// \param left_envs Mutable left environment cache.
/// \param right_envs Right environment cache.
/// \param options Sweep options.
/// \return Sweep result.
inline auto sweep_two_site_left_to_right(BlockSparseFiniteMPS& psi, BlockSparseMpoChain const& mpo,
                                         std::vector<BlockSparseEnvironment>& left_envs,
                                         std::vector<BlockSparseEnvironment> const& right_envs,
                                         BlockSparseTwoSiteSweepOptions options = {}) -> BlockSparseTwoSiteSweepResult
{
  detail::validate_block_sparse_sweep_inputs(psi, mpo);
  detail::validate_left_to_right_environment_cache(psi, left_envs, right_envs);

  BlockSparseTwoSiteSweepResult result{.direction = BlockSparseTwoSiteSweepDirection::LeftToRight, .updates = {}};
  result.updates.reserve(psi.size() - 1);
  for (std::size_t left_site = 0; left_site + 1 < psi.size(); ++left_site)
  {
    auto const solve_start = detail::sweep_stage_checkpoint();
    auto solution =
        solve_two_site(psi, mpo, left_site, left_envs[left_site], right_envs[left_site + 2], options.lanczos);
    auto const split_start = detail::sweep_stage_checkpoint();
    auto split = split_two_site_solution(solution, TwoSiteSplitDirection::LeftToRight, options.svd);
    auto const replace_start = detail::sweep_stage_checkpoint();
    auto update = detail::make_block_sparse_bond_update(
        left_site, solution, split, detail::elapsed_wall_seconds(solve_start, split_start),
        detail::elapsed_wall_seconds(split_start, replace_start), 0.0, 0.0,
        detail::elapsed_cpu_seconds(solve_start, split_start), detail::elapsed_cpu_seconds(split_start, replace_start),
        0.0, 0.0);
    replace_two_site_solution(psi, left_site, std::move(split));
    auto const env_start = detail::sweep_stage_checkpoint();
    update.replace_seconds = detail::elapsed_wall_seconds(replace_start, env_start);
    update.replace_cpu_seconds = detail::elapsed_cpu_seconds(replace_start, env_start);
    detail::assign_environment(left_envs, left_site + 1,
                               extend_left_environment(left_envs[left_site], psi[left_site], mpo[left_site]));
    auto const env_stop = detail::sweep_stage_checkpoint();
    update.environment_seconds = detail::elapsed_wall_seconds(env_start, env_stop);
    update.environment_cpu_seconds = detail::elapsed_cpu_seconds(env_start, env_stop);
    if (options.observer)
    {
      options.observer(BlockSparseTwoSiteSweepDirection::LeftToRight, update);
    }
    result.updates.push_back(update);
  }
  return result;
}

/// \brief Sweep right-to-left using the strict block-sparse U(1) path.
/// \param psi MPS state to update.
/// \param mpo Sparse MPO chain.
/// \param left_envs Left environment cache.
/// \param right_envs Mutable right environment cache.
/// \param options Sweep options.
/// \return Sweep result.
inline auto sweep_two_site_right_to_left(BlockSparseFiniteMPS& psi, BlockSparseMpoChain const& mpo,
                                         std::vector<BlockSparseEnvironment> const& left_envs,
                                         std::vector<BlockSparseEnvironment>& right_envs,
                                         BlockSparseTwoSiteSweepOptions options = {}) -> BlockSparseTwoSiteSweepResult
{
  detail::validate_block_sparse_sweep_inputs(psi, mpo);
  detail::validate_right_to_left_environment_cache(psi, left_envs, right_envs);

  BlockSparseTwoSiteSweepResult result{.direction = BlockSparseTwoSiteSweepDirection::RightToLeft, .updates = {}};
  result.updates.reserve(psi.size() - 1);
  for (std::size_t offset = 0; offset + 1 < psi.size(); ++offset)
  {
    auto const left_site = psi.size() - 2 - offset;
    auto const solve_start = detail::sweep_stage_checkpoint();
    auto solution =
        solve_two_site(psi, mpo, left_site, left_envs[left_site], right_envs[left_site + 2], options.lanczos);
    auto const split_start = detail::sweep_stage_checkpoint();
    auto split = split_two_site_solution(solution, TwoSiteSplitDirection::RightToLeft, options.svd);
    auto const replace_start = detail::sweep_stage_checkpoint();
    auto update = detail::make_block_sparse_bond_update(
        left_site, solution, split, detail::elapsed_wall_seconds(solve_start, split_start),
        detail::elapsed_wall_seconds(split_start, replace_start), 0.0, 0.0,
        detail::elapsed_cpu_seconds(solve_start, split_start), detail::elapsed_cpu_seconds(split_start, replace_start),
        0.0, 0.0);
    replace_two_site_solution(psi, left_site, std::move(split));
    auto const env_start = detail::sweep_stage_checkpoint();
    update.replace_seconds = detail::elapsed_wall_seconds(replace_start, env_start);
    update.replace_cpu_seconds = detail::elapsed_cpu_seconds(replace_start, env_start);
    detail::assign_environment(
        right_envs, left_site + 1,
        extend_right_environment(right_envs[left_site + 2], psi[left_site + 1], mpo[left_site + 1]));
    auto const env_stop = detail::sweep_stage_checkpoint();
    update.environment_seconds = detail::elapsed_wall_seconds(env_start, env_stop);
    update.environment_cpu_seconds = detail::elapsed_cpu_seconds(env_start, env_stop);
    if (options.observer)
    {
      options.observer(BlockSparseTwoSiteSweepDirection::RightToLeft, update);
    }
    result.updates.push_back(update);
  }
  return result;
}

/// \brief Options for the block-sparse two-site DMRG driver.
struct BlockSparseTwoSiteDmrgOptions
{
    std::size_t sweeps = 1;
    BlockSparseTwoSiteSweepOptions sweep;
};

/// \brief Completed pair of left-to-right and right-to-left block-sparse sweeps.
struct BlockSparseTwoSiteDmrgSweepPair
{
    std::size_t sweep = 0;
    BlockSparseTwoSiteSweepResult left_to_right;
    BlockSparseTwoSiteSweepResult right_to_left;
};

/// \brief Result of a block-sparse two-site DMRG run.
struct BlockSparseTwoSiteDmrgResult
{
    std::vector<BlockSparseTwoSiteDmrgSweepPair> sweeps;
};

/// \brief Return the final edge local energy from a block-sparse DMRG result.
/// \throws std::logic_error If the result is incomplete.
/// \param result DMRG result.
/// \return Final right-to-left edge solve energy.
inline auto final_two_site_energy(BlockSparseTwoSiteDmrgResult const& result) -> double
{
  if (result.sweeps.empty() || result.sweeps.back().right_to_left.updates.empty())
  {
    throw std::logic_error("block-sparse DMRG result does not contain any completed right-to-left updates");
  }
  return result.sweeps.back().right_to_left.updates.back().energy;
}

/// \brief Run the strict block-sparse U(1) two-site DMRG front-end.
/// \param psi MPS state to update.
/// \param mpo Sparse MPO chain.
/// \param options DMRG options.
/// \return Sweep history.
inline auto run_two_site_dmrg(BlockSparseFiniteMPS& psi, BlockSparseMpoChain const& mpo,
                              BlockSparseTwoSiteDmrgOptions options = {}) -> BlockSparseTwoSiteDmrgResult
{
  detail::validate_block_sparse_sweep_inputs(psi, mpo);
  if (options.sweeps == 0)
  {
    throw std::invalid_argument("block-sparse DMRG requires at least one sweep");
  }

  BlockSparseTwoSiteDmrgResult result;
  result.sweeps.reserve(options.sweeps);
  std::vector<BlockSparseEnvironment> left_envs;
  left_envs.reserve(psi.size() + 1);
  left_envs.push_back(make_left_boundary_environment(psi, mpo));
  auto right_envs = build_right_environments(psi, mpo);

  for (std::size_t sweep = 0; sweep < options.sweeps; ++sweep)
  {
    auto left_to_right = sweep_two_site_left_to_right(psi, mpo, left_envs, right_envs, options.sweep);
    auto right_to_left = sweep_two_site_right_to_left(psi, mpo, left_envs, right_envs, options.sweep);
    result.sweeps.push_back(BlockSparseTwoSiteDmrgSweepPair{
        .sweep = sweep, .left_to_right = std::move(left_to_right), .right_to_left = std::move(right_to_left)});
  }
  return result;
}

} // namespace uni20
