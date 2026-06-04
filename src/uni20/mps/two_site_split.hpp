/**
 * \file two_site_split.hpp
 * \brief Split an optimized two-site center back into neighboring MPS tensors.
 */

#pragma once

#include <uni20/mps/two_site_solve.hpp>
#include <uni20/tensorcontraction/svd.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace uni20
{

enum class TwoSiteSplitDirection
{
  LeftToRight,
  RightToLeft,
};

struct TwoSiteSplitResult
{
    MpsSiteTensor left;
    MpsSiteTensor right;
    tensorcontraction::SvdSpectrum spectrum;
};

namespace detail
{

inline auto mpi_has_multiple_ranks() -> bool
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0)
  {
    return false;
  }

  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized != 0)
  {
    return false;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size > 1;
}

inline void broadcast_doubles_from_rank_zero(std::span<double> values)
{
  std::size_t offset = 0;
  while (offset < values.size())
  {
    auto const remaining = values.size() - offset;
    auto const count = std::min<std::size_t>(remaining, static_cast<std::size_t>(std::numeric_limits<int>::max()));
    MPI_Bcast(values.data() + offset, static_cast<int>(count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    offset += count;
  }
}

inline void broadcast_size_from_rank_zero(std::size_t& value)
{
  auto wire_value = static_cast<unsigned long long>(value);
  MPI_Bcast(&wire_value, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  value = static_cast<std::size_t>(wire_value);
}

inline void require_matching_rank_zero_size(std::size_t local_value, char const* name)
{
  auto rank_zero_value = local_value;
  broadcast_size_from_rank_zero(rank_zero_value);

  int local_mismatch = local_value == rank_zero_value ? 0 : 1;
  int global_mismatch = 0;
  MPI_Allreduce(&local_mismatch, &global_mismatch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (global_mismatch != 0)
  {
    throw std::runtime_error(std::string("MPI ranks computed different two-site SVD ") + name);
  }
}

inline void broadcast_svd_spectrum_from_rank_zero(tensorcontraction::SvdSpectrum& spectrum)
{
  if (!mpi_has_multiple_ranks())
  {
    return;
  }

  require_matching_rank_zero_size(spectrum.singular_values.size(), "kept rank");
  broadcast_doubles_from_rank_zero(spectrum.singular_values);
  MPI_Bcast(&spectrum.discarded_weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  broadcast_size_from_rank_zero(spectrum.full_rank);
}

inline void broadcast_site_tensor_from_rank_zero(MpsSiteTensor& site)
{
  if (!mpi_has_multiple_ranks())
  {
    return;
  }

  for (std::size_t physical = 0; physical < site.physical_dim(); ++physical)
  {
    broadcast_doubles_from_rank_zero(site.values(physical));
  }
}

} // namespace detail

inline auto make_dense_shared_bond_space(Symmetry sym, std::size_t dim) -> BlockSpace
{
  if (dim == 0)
  {
    throw std::invalid_argument("two-site split produced an empty shared bond");
  }
  return BlockSpace(sym, {BlockSector{QNum::identity(sym), dim}});
}

inline auto split_two_site_center(tensorcontraction::MatrixFamily const& center,
                                  TwoSiteEffectiveHamiltonianLayout const& layout,
                                  LocalSpace const& left_physical_space, LocalSpace const& right_physical_space,
                                  BlockSpace const& left_bond_space, BlockSpace const& right_bond_space,
                                  TwoSiteSplitDirection direction,
                                  tensorcontraction::SvdOptions options = {}) -> TwoSiteSplitResult
{
  if (center.size() != 1)
  {
    throw std::invalid_argument("two-site split requires a single-block center");
  }
  if (center.block(0) != tensorcontraction::MatrixFamily::Block{layout.left_bond_dim * layout.left_physical_dim,
                                                                layout.right_physical_dim * layout.right_bond_dim})
  {
    throw std::invalid_argument("two-site split center shape does not match the layout");
  }
  if (left_physical_space.size() != layout.left_physical_dim ||
      right_physical_space.size() != layout.right_physical_dim || left_bond_space.total_dim() != layout.left_bond_dim ||
      right_bond_space.total_dim() != layout.right_bond_dim)
  {
    throw std::invalid_argument("two-site split metadata does not match the layout");
  }
  if (left_physical_space.symmetry() != right_physical_space.symmetry())
  {
    throw std::invalid_argument("two-site split physical spaces must share one symmetry");
  }

  auto const absorb = direction == TwoSiteSplitDirection::RightToLeft
                          ? tensorcontraction::SvdAbsorbSingularValues::Left
                          : tensorcontraction::SvdAbsorbSingularValues::Right;
  auto split_blocks = tensorcontraction::single_block_svd_split(
      center,
      tensorcontraction::SingleBlockSvdSplitLayout{.left_bond_dim = layout.left_bond_dim,
                                                   .left_physical_dim = layout.left_physical_dim,
                                                   .right_physical_dim = layout.right_physical_dim,
                                                   .right_bond_dim = layout.right_bond_dim},
      absorb, options);
  // Keep the prototype MPI ranks on one host-state trajectory.  This is also
  // robust against valid rank-local SVD sign or degenerate-subspace choices.
  detail::broadcast_svd_spectrum_from_rank_zero(split_blocks.spectrum);
  auto const rank = split_blocks.spectrum.singular_values.size();
  auto shared_bond_space = make_dense_shared_bond_space(left_physical_space.symmetry(), rank);
  MpsSiteTensor left(left_physical_space, left_bond_space, shared_bond_space);
  MpsSiteTensor right(right_physical_space, shared_bond_space, right_bond_space);

  for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
  {
    left.assign(left_phys, split_blocks.left.values(left_phys));
  }

  for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
  {
    right.assign(right_phys, split_blocks.right.values(right_phys));
  }

  detail::broadcast_site_tensor_from_rank_zero(left);
  detail::broadcast_site_tensor_from_rank_zero(right);

  return TwoSiteSplitResult{
      .left = std::move(left), .right = std::move(right), .spectrum = std::move(split_blocks.spectrum)};
}

inline auto split_two_site_solution(TwoSiteSolveResult const& solution, LocalSpace const& left_physical_space,
                                    LocalSpace const& right_physical_space, BlockSpace const& left_bond_space,
                                    BlockSpace const& right_bond_space, TwoSiteSplitDirection direction,
                                    tensorcontraction::SvdOptions options = {}) -> TwoSiteSplitResult
{
  return split_two_site_center(solution.optimized_matrix, solution.layout, left_physical_space, right_physical_space,
                               left_bond_space, right_bond_space, direction, options);
}

inline auto split_two_site_solution(TwoSiteSolveResult const& solution, FiniteMPS const& psi, std::size_t left_site,
                                    TwoSiteSplitDirection direction,
                                    tensorcontraction::SvdOptions options = {}) -> TwoSiteSplitResult
{
  if (left_site + 1 >= psi.size())
  {
    throw std::out_of_range("split_two_site_solution requires two adjacent MPS sites");
  }
  return split_two_site_solution(solution, psi[left_site].physical_space(), psi[left_site + 1].physical_space(),
                                 psi[left_site].left_bond_space(), psi[left_site + 1].right_bond_space(), direction,
                                 options);
}

inline void replace_two_site_solution(FiniteMPS& psi, std::size_t left_site, TwoSiteSplitResult split)
{
  psi.replace_adjacent(left_site, std::move(split.left), std::move(split.right));
}

} // namespace uni20
