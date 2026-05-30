/**
 * \file two_site_split.hpp
 * \brief Split an optimized two-site center back into neighboring MPS tensors.
 */

#pragma once

#include <uni20/mps/two_site_solve.hpp>
#include <uni20/tensorcontraction/svd.hpp>

#include <cstddef>
#include <stdexcept>
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
    tensorcontraction::SingleBlockSvd svd;
};

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

  auto svd = tensorcontraction::single_block_svd(center, options);
  auto const rank = svd.singular_values.size();
  auto shared_bond_space = make_dense_shared_bond_space(left_physical_space.symmetry(), rank);
  MpsSiteTensor left(left_physical_space, left_bond_space, shared_bond_space);
  MpsSiteTensor right(right_physical_space, shared_bond_space, right_bond_space);

  auto const u_values = svd.u.values(0);
  auto const vt_values = svd.vt.values(0);
  for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
  {
    auto dst = left.values(left_phys);
    for (std::size_t left_bond = 0; left_bond < layout.left_bond_dim; ++left_bond)
    {
      auto const row = left_bond * layout.left_physical_dim + left_phys;
      for (std::size_t bond = 0; bond < rank; ++bond)
      {
        double value = u_values[row * rank + bond];
        if (direction == TwoSiteSplitDirection::RightToLeft)
        {
          value *= svd.singular_values[bond];
        }
        dst[left_bond * rank + bond] = value;
      }
    }
  }

  for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
  {
    auto dst = right.values(right_phys);
    for (std::size_t bond = 0; bond < rank; ++bond)
    {
      for (std::size_t right_bond = 0; right_bond < layout.right_bond_dim; ++right_bond)
      {
        auto const col = right_phys * layout.right_bond_dim + right_bond;
        double value = vt_values[bond * (layout.right_physical_dim * layout.right_bond_dim) + col];
        if (direction == TwoSiteSplitDirection::LeftToRight)
        {
          value *= svd.singular_values[bond];
        }
        dst[bond * layout.right_bond_dim + right_bond] = value;
      }
    }
  }

  return TwoSiteSplitResult{.left = std::move(left), .right = std::move(right), .svd = std::move(svd)};
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
