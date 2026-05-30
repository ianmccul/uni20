/**
 * \file two_site_sweep.hpp
 * \brief First dense two-site DMRG sweep prototype.
 */

#pragma once

#include <uni20/mps/two_site_split.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace uni20
{

enum class TwoSiteSweepDirection
{
  LeftToRight,
  RightToLeft,
};

struct TwoSiteSweepOptions
{
    tensorcontraction::LanczosOptions lanczos;
    tensorcontraction::SvdOptions svd;
};

struct TwoSiteBondUpdate
{
    std::size_t left_site = 0;
    double energy = 0.0;
    tensorcontraction::LanczosResult lanczos;
    double discarded_weight = 0.0;
    std::size_t kept_rank = 0;
    std::size_t full_rank = 0;
};

struct TwoSiteSweepResult
{
    TwoSiteSweepDirection direction = TwoSiteSweepDirection::LeftToRight;
    std::vector<TwoSiteBondUpdate> updates;
};

inline void validate_two_site_sweep_inputs(FiniteMPS const& psi, FiniteTriangularMPO const& mpo)
{
  if (psi.size() != mpo.size())
  {
    throw std::invalid_argument("two-site sweep requires MPS and MPO chains of equal length");
  }
  if (psi.size() < 2)
  {
    throw std::invalid_argument("two-site sweep requires at least two sites");
  }
}

inline auto make_bond_update(std::size_t left_site, TwoSiteSolveResult const& solution,
                             TwoSiteSplitResult const& split) -> TwoSiteBondUpdate
{
  return TwoSiteBondUpdate{.left_site = left_site,
                           .energy = solution.lanczos.eigenvalue,
                           .lanczos = solution.lanczos,
                           .discarded_weight = split.svd.discarded_weight,
                           .kept_rank = split.svd.singular_values.size(),
                           .full_rank = split.svd.full_rank};
}

inline auto sweep_two_site_left_to_right(FiniteMPS& psi, FiniteTriangularMPO const& mpo,
                                         TwoSiteSweepOptions options = {}) -> TwoSiteSweepResult
{
  validate_two_site_sweep_inputs(psi, mpo);

  auto left_envs = build_left_environments(psi, mpo);
  auto right_envs = build_right_environments(psi, mpo);

  TwoSiteSweepResult result{.direction = TwoSiteSweepDirection::LeftToRight, .updates = {}};
  result.updates.reserve(psi.size() - 1);

  for (std::size_t left_site = 0; left_site + 1 < psi.size(); ++left_site)
  {
    auto solution =
        solve_two_site(psi, mpo, left_site, left_envs[left_site], right_envs[left_site + 2], options.lanczos);
    auto split = split_two_site_solution(solution, psi, left_site, TwoSiteSplitDirection::LeftToRight, options.svd);
    result.updates.push_back(make_bond_update(left_site, solution, split));
    replace_two_site_solution(psi, left_site, std::move(split));
    left_envs[left_site + 1] = extend_left_environment(left_envs[left_site], psi[left_site], mpo[left_site]);
  }

  return result;
}

inline auto sweep_two_site_right_to_left(FiniteMPS& psi, FiniteTriangularMPO const& mpo,
                                         TwoSiteSweepOptions options = {}) -> TwoSiteSweepResult
{
  validate_two_site_sweep_inputs(psi, mpo);

  auto left_envs = build_left_environments(psi, mpo);
  auto right_envs = build_right_environments(psi, mpo);

  TwoSiteSweepResult result{.direction = TwoSiteSweepDirection::RightToLeft, .updates = {}};
  result.updates.reserve(psi.size() - 1);

  for (std::size_t offset = 0; offset + 1 < psi.size(); ++offset)
  {
    auto const left_site = psi.size() - 2 - offset;
    auto solution =
        solve_two_site(psi, mpo, left_site, left_envs[left_site], right_envs[left_site + 2], options.lanczos);
    auto split = split_two_site_solution(solution, psi, left_site, TwoSiteSplitDirection::RightToLeft, options.svd);
    result.updates.push_back(make_bond_update(left_site, solution, split));
    replace_two_site_solution(psi, left_site, std::move(split));
    right_envs[left_site + 1] =
        extend_right_environment(right_envs[left_site + 2], psi[left_site + 1], mpo[left_site + 1]);
  }

  return result;
}

inline auto sweep_two_site(FiniteMPS& psi, FiniteTriangularMPO const& mpo, TwoSiteSweepDirection direction,
                           TwoSiteSweepOptions options = {}) -> TwoSiteSweepResult
{
  if (direction == TwoSiteSweepDirection::LeftToRight)
  {
    return sweep_two_site_left_to_right(psi, mpo, options);
  }
  return sweep_two_site_right_to_left(psi, mpo, options);
}

} // namespace uni20
