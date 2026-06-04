/**
 * \file block_sparse_mps.hpp
 * \brief Block-sparse finite MPS helpers for symmetry-aware DMRG prototypes.
 */

#pragma once

#include <uni20/mps/block_sparse_matrix.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Minimal finite MPS container using `ThreeLegBlockMatrix` site tensors.
class BlockSparseFiniteMPS {
  public:
    using site_type = ThreeLegBlockMatrix;
    using container_type = std::vector<site_type>;
    using size_type = container_type::size_type;

    /// \brief Construct an empty finite MPS.
    BlockSparseFiniteMPS() = default;

    /// \brief Construct from a sequence of block-sparse site tensors.
    /// \param sites Site tensors ordered from left to right.
    explicit BlockSparseFiniteMPS(container_type sites) : sites_(std::move(sites)) { this->check_structure(); }

    /// \brief Return the number of sites.
    /// \return Site count.
    auto size() const noexcept -> size_type { return sites_.size(); }

    /// \brief Return whether the chain has no sites.
    /// \return `true` if the chain is empty.
    auto empty() const noexcept -> bool { return sites_.empty(); }

    /// \brief Return one site tensor.
    /// \param index Site index.
    /// \return Site tensor.
    auto operator[](size_type index) const -> site_type const& { return sites_.at(index); }

    /// \brief Return one mutable site tensor.
    /// \param index Site index.
    /// \return Mutable site tensor.
    auto operator[](size_type index) -> site_type& { return sites_.at(index); }

    /// \brief Return an iterator to the first site.
    /// \return Begin iterator.
    auto begin() const { return sites_.begin(); }

    /// \brief Return an iterator past the last site.
    /// \return End iterator.
    auto end() const { return sites_.end(); }

    /// \brief Validate adjacent bond spaces and shared physical symmetry.
    /// \throws std::invalid_argument If the site sequence is inconsistent.
    void check_structure() const
    {
      for (size_type i = 1; i < sites_.size(); ++i)
      {
        if (sites_[i - 1].col_space() != sites_[i].row_space())
        {
          throw std::invalid_argument("BlockSparseFiniteMPS adjacent bond spaces do not match");
        }
        if (sites_[i - 1].local_space().symmetry() != sites_[i].local_space().symmetry())
        {
          throw std::invalid_argument("BlockSparseFiniteMPS sites do not share one physical symmetry");
        }
      }
    }

  private:
    container_type sites_;
};

/// \brief Build one-dimensional cumulative bond spaces for a product state.
/// \details Starting from the identity boundary charge, each site updates the
///          right bond charge as `q_right = q_left + q_physical`.
/// \param physical_space Physical local space.
/// \param physical_indices Product-state physical index at each site.
/// \return `length + 1` one-sector bond spaces.
inline auto make_product_state_bond_spaces(LocalSpace const& physical_space,
                                           std::span<std::size_t const> physical_indices) -> std::vector<BlockSpace>
{
  auto const sym = physical_space.symmetry();
  std::vector<BlockSpace> bonds;
  bonds.reserve(physical_indices.size() + 1);

  QNum charge = QNum::identity(sym);
  bonds.push_back(BlockSpace(sym, {BlockSector{.q = charge, .dim = 1}}));
  for (std::size_t physical : physical_indices)
  {
    if (physical >= physical_space.size())
    {
      throw std::out_of_range("product-state physical index is out of range");
    }
    charge = charge + physical_space[physical];
    bonds.push_back(BlockSpace(sym, {BlockSector{.q = charge, .dim = 1}}));
  }
  return bonds;
}

/// \brief Construct a normalized block-sparse product-state MPS.
/// \param physical_space Physical local space.
/// \param physical_indices Product-state physical index at each site.
/// \return Block-sparse finite MPS with one unit block per site.
inline auto make_block_sparse_product_state(LocalSpace const& physical_space,
                                            std::span<std::size_t const> physical_indices) -> BlockSparseFiniteMPS
{
  auto bonds = make_product_state_bond_spaces(physical_space, physical_indices);
  BlockSparseFiniteMPS::container_type sites;
  sites.reserve(physical_indices.size());

  for (std::size_t site = 0; site < physical_indices.size(); ++site)
  {
    ThreeLegBlockMatrix tensor(bonds[site], physical_space, bonds[site + 1]);
    ThreeLegBlockKey const key{.row_sector = 0, .local = physical_indices[site], .col_sector = 0};
    auto values = tensor.insert_zero_block(key);
    values[0] = 1.0;
    sites.push_back(std::move(tensor));
  }

  return BlockSparseFiniteMPS(std::move(sites));
}

/// \brief Return alternating `0, 1, 0, 1, ...` spin-half product-state indices.
/// \param length Number of sites.
/// \return Physical index sequence.
inline auto make_alternating_spin_half_indices(std::size_t length) -> std::vector<std::size_t>
{
  std::vector<std::size_t> indices;
  indices.reserve(length);
  for (std::size_t site = 0; site < length; ++site)
  {
    indices.push_back(site % 2);
  }
  return indices;
}

/// \brief One row-side term in a block SVD sector.
struct TwoSiteSvdRowTerm
{
    std::size_t left_sector = 0;
    std::size_t left_physical = 0;
    std::size_t offset = 0;
    std::size_t dim = 0;
};

/// \brief One column-side term in a block SVD sector.
struct TwoSiteSvdColTerm
{
    std::size_t right_physical = 0;
    std::size_t right_sector = 0;
    std::size_t offset = 0;
    std::size_t dim = 0;
};

/// \brief Fused matrix sector for a two-site block SVD.
/// \details Rows are `(left bond, left physical)` terms with charge
///          `q_left + q_left_physical`. Columns are `(right physical, right bond)`
///          terms with matching charge `q_right + dual(q_right_physical)`.
struct TwoSiteSvdSector
{
    QNum shared_q;
    std::size_t row_dim = 0;
    std::size_t col_dim = 0;
    std::vector<TwoSiteSvdRowTerm> rows;
    std::vector<TwoSiteSvdColTerm> cols;
};

namespace detail
{

inline auto find_or_add_svd_sector(std::vector<TwoSiteSvdSector>& sectors, QNum q) -> TwoSiteSvdSector&
{
  auto const it = std::find_if(sectors.begin(), sectors.end(),
                               [&](TwoSiteSvdSector const& sector) { return sector.shared_q == q; });
  if (it != sectors.end())
  {
    return *it;
  }
  TwoSiteSvdSector sector;
  sector.shared_q = q;
  sectors.push_back(std::move(sector));
  return sectors.back();
}

} // namespace detail

/// \brief Build the fused sectors needed for a symmetry-blocked two-site SVD.
/// \param left_physical_space Left physical local space.
/// \param right_physical_space Right physical local space.
/// \param left_bond_space External left MPS bond space.
/// \param right_bond_space External right MPS bond space.
/// \return Fused SVD sectors with nonzero row and column dimensions.
inline auto make_two_site_svd_sectors(LocalSpace const& left_physical_space, LocalSpace const& right_physical_space,
                                      BlockSpace const& left_bond_space,
                                      BlockSpace const& right_bond_space) -> std::vector<TwoSiteSvdSector>
{
  auto const sym = left_physical_space.symmetry();
  if (right_physical_space.symmetry() != sym || left_bond_space.symmetry() != sym || right_bond_space.symmetry() != sym)
  {
    throw std::invalid_argument("two-site SVD spaces must share one symmetry");
  }

  std::vector<TwoSiteSvdSector> sectors;
  for (std::size_t left_sector = 0; left_sector < left_bond_space.size(); ++left_sector)
  {
    for (std::size_t left_physical = 0; left_physical < left_physical_space.size(); ++left_physical)
    {
      auto& sector =
          detail::find_or_add_svd_sector(sectors, left_bond_space[left_sector].q + left_physical_space[left_physical]);
      sector.rows.push_back(TwoSiteSvdRowTerm{.left_sector = left_sector,
                                              .left_physical = left_physical,
                                              .offset = sector.row_dim,
                                              .dim = left_bond_space[left_sector].dim});
      sector.row_dim += left_bond_space[left_sector].dim;
    }
  }

  for (std::size_t right_physical = 0; right_physical < right_physical_space.size(); ++right_physical)
  {
    for (std::size_t right_sector = 0; right_sector < right_bond_space.size(); ++right_sector)
    {
      auto& sector = detail::find_or_add_svd_sector(sectors, right_bond_space[right_sector].q +
                                                                 dual(right_physical_space[right_physical]));
      sector.cols.push_back(TwoSiteSvdColTerm{.right_physical = right_physical,
                                              .right_sector = right_sector,
                                              .offset = sector.col_dim,
                                              .dim = right_bond_space[right_sector].dim});
      sector.col_dim += right_bond_space[right_sector].dim;
    }
  }

  sectors.erase(
      std::remove_if(sectors.begin(), sectors.end(),
                     [](TwoSiteSvdSector const& sector) { return sector.row_dim == 0 || sector.col_dim == 0; }),
      sectors.end());
  std::sort(sectors.begin(), sectors.end(), [](TwoSiteSvdSector const& lhs, TwoSiteSvdSector const& rhs) {
    return lhs.shared_q.raw_code() < rhs.shared_q.raw_code();
  });
  return sectors;
}

/// \brief Build a shared bond space from per-sector SVD ranks.
/// \throws std::invalid_argument If all requested ranks are zero or sizes mismatch.
/// \param sectors Fused SVD sectors.
/// \param ranks Kept rank for each sector.
/// \return Shared MPS bond space.
inline auto make_shared_bond_space_from_sector_ranks(std::span<TwoSiteSvdSector const> sectors,
                                                     std::span<std::size_t const> ranks) -> BlockSpace
{
  if (sectors.size() != ranks.size())
  {
    throw std::invalid_argument("shared bond sector ranks do not match SVD sectors");
  }
  if (sectors.empty())
  {
    throw std::invalid_argument("cannot build a shared bond space from no SVD sectors");
  }

  auto const sym = sectors.front().shared_q.symmetry();
  BlockSpace result(sym);
  for (std::size_t i = 0; i < sectors.size(); ++i)
  {
    if (sectors[i].shared_q.symmetry() != sym)
    {
      throw std::invalid_argument("SVD sectors must share one symmetry");
    }
    if (ranks[i] != 0)
    {
      result.push_back(BlockSector{.q = sectors[i].shared_q, .dim = ranks[i]});
    }
  }
  if (result.empty())
  {
    throw std::invalid_argument("shared bond space would be empty");
  }
  return result;
}

} // namespace uni20
