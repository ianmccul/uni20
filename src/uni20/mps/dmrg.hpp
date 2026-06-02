/**
 * \file dmrg.hpp
 * \brief Minimal dense two-site DMRG front-end for the first prototype.
 */

#pragma once

#include <uni20/mps/two_site_sweep.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace uni20
{

struct TwoSiteDmrgOptions
{
    std::size_t sweeps = 1;
    TwoSiteSweepOptions sweep;
};

struct TwoSiteDmrgSweepPair
{
    std::size_t sweep = 0;
    TwoSiteSweepResult left_to_right;
    TwoSiteSweepResult right_to_left;
};

struct TwoSiteDmrgResult
{
    std::vector<TwoSiteDmrgSweepPair> sweeps;
};

inline auto final_two_site_energy(TwoSiteDmrgResult const& result) -> double
{
  if (result.sweeps.empty() || result.sweeps.back().right_to_left.updates.empty())
  {
    throw std::logic_error("two-site DMRG result does not contain any completed right-to-left updates");
  }
  return result.sweeps.back().right_to_left.updates.back().energy;
}

inline auto run_two_site_dmrg(FiniteMPS& psi, FiniteTriangularMPO const& mpo,
                              TwoSiteDmrgOptions options = {}) -> TwoSiteDmrgResult
{
  validate_two_site_sweep_inputs(psi, mpo);
  if (options.sweeps == 0)
  {
    throw std::invalid_argument("two-site DMRG requires at least one sweep");
  }

  TwoSiteDmrgResult result;
  result.sweeps.reserve(options.sweeps);

  std::vector<MpoEnvironment> left_envs;
  left_envs.reserve(psi.size() + 1);
  left_envs.push_back(make_left_boundary_environment(psi, mpo));
  auto right_envs = build_right_environments(psi, mpo);

  for (std::size_t sweep = 0; sweep < options.sweeps; ++sweep)
  {
    auto left_to_right = sweep_two_site_left_to_right(psi, mpo, left_envs, right_envs, options.sweep);
    auto right_to_left = sweep_two_site_right_to_left(psi, mpo, left_envs, right_envs, options.sweep);
    result.sweeps.push_back(TwoSiteDmrgSweepPair{
        .sweep = sweep, .left_to_right = std::move(left_to_right), .right_to_left = std::move(right_to_left)});
  }
  return result;
}

} // namespace uni20
