/**
 * \file two_site_sweep.hpp
 * \brief First dense two-site DMRG sweep prototype.
 */

#pragma once

#include <uni20/mps/two_site_split.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20
{

enum class TwoSiteSweepDirection
{
  LeftToRight,
  RightToLeft,
};

struct TwoSiteBondUpdate;

using TwoSiteSweepObserver = std::function<void(TwoSiteSweepDirection, TwoSiteBondUpdate const&)>;

struct TwoSiteSweepOptions
{
    tensorcontraction::LanczosOptions lanczos{};
    tensorcontraction::SvdOptions svd{};
    TwoSiteSweepObserver observer{};
};

struct TwoSiteBondUpdate
{
    std::size_t left_site = 0;
    double energy = 0.0;
    tensorcontraction::LanczosResult lanczos;
    double discarded_weight = 0.0;
    std::size_t kept_rank = 0;
    std::size_t full_rank = 0;
    double solve_seconds = 0.0;
    double split_seconds = 0.0;
    double replace_seconds = 0.0;
    double environment_seconds = 0.0;
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

inline auto make_bond_update(std::size_t left_site, TwoSiteSolveResult const& solution, TwoSiteSplitResult const& split,
                             double solve_seconds, double split_seconds, double replace_seconds,
                             double environment_seconds) -> TwoSiteBondUpdate
{
  return TwoSiteBondUpdate{.left_site = left_site,
                           .energy = solution.lanczos.eigenvalue,
                           .lanczos = solution.lanczos,
                           .discarded_weight = split.spectrum.discarded_weight,
                           .kept_rank = split.spectrum.singular_values.size(),
                           .full_rank = split.spectrum.full_rank,
                           .solve_seconds = solve_seconds,
                           .split_seconds = split_seconds,
                           .replace_seconds = replace_seconds,
                           .environment_seconds = environment_seconds};
}

inline auto two_site_elapsed_seconds(std::chrono::steady_clock::time_point start,
                                     std::chrono::steady_clock::time_point stop) -> double
{
  return std::chrono::duration<double>(stop - start).count();
}

inline void assign_environment(std::vector<MpoEnvironment>& environments, std::size_t index, MpoEnvironment environment)
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
  throw std::logic_error("cannot assign an MPO environment past the next cache slot");
}

inline void validate_left_to_right_environment_cache(FiniteMPS const& psi, std::vector<MpoEnvironment> const& left_envs,
                                                     std::vector<MpoEnvironment> const& right_envs)
{
  if (left_envs.empty())
  {
    throw std::invalid_argument("left-to-right sweep requires a left boundary environment");
  }
  if (right_envs.size() != psi.size() + 1)
  {
    throw std::invalid_argument("left-to-right sweep requires a complete right environment cache");
  }
}

inline void validate_right_to_left_environment_cache(FiniteMPS const& psi, std::vector<MpoEnvironment> const& left_envs,
                                                     std::vector<MpoEnvironment> const& right_envs)
{
  if (left_envs.size() < psi.size())
  {
    throw std::invalid_argument("right-to-left sweep requires left environments through the final updated bond");
  }
  if (right_envs.size() != psi.size() + 1)
  {
    throw std::invalid_argument("right-to-left sweep requires a complete right environment cache");
  }
}

inline auto sweep_two_site_left_to_right(FiniteMPS& psi, FiniteTriangularMPO const& mpo,
                                         std::vector<MpoEnvironment>& left_envs,
                                         std::vector<MpoEnvironment> const& right_envs,
                                         TwoSiteSweepOptions options = {}) -> TwoSiteSweepResult
{
  validate_two_site_sweep_inputs(psi, mpo);
  validate_left_to_right_environment_cache(psi, left_envs, right_envs);

  TwoSiteSweepResult result{.direction = TwoSiteSweepDirection::LeftToRight, .updates = {}};
  result.updates.reserve(psi.size() - 1);

  for (std::size_t left_site = 0; left_site + 1 < psi.size(); ++left_site)
  {
    auto const solve_start = std::chrono::steady_clock::now();
    auto solution =
        solve_two_site(psi, mpo, left_site, left_envs[left_site], right_envs[left_site + 2], options.lanczos);
    auto const split_start = std::chrono::steady_clock::now();
    auto split = split_two_site_solution(solution, psi, left_site, TwoSiteSplitDirection::LeftToRight, options.svd);
    auto const replace_start = std::chrono::steady_clock::now();
    auto update = make_bond_update(left_site, solution, split, two_site_elapsed_seconds(solve_start, split_start),
                                   two_site_elapsed_seconds(split_start, replace_start), 0.0, 0.0);
    replace_two_site_solution(psi, left_site, std::move(split));
    auto const env_start = std::chrono::steady_clock::now();
    update.replace_seconds = two_site_elapsed_seconds(replace_start, env_start);
    assign_environment(left_envs, left_site + 1,
                       extend_left_environment(left_envs[left_site], psi[left_site], mpo[left_site]));
    update.environment_seconds = two_site_elapsed_seconds(env_start, std::chrono::steady_clock::now());
    if (options.observer)
    {
      options.observer(TwoSiteSweepDirection::LeftToRight, update);
    }
    result.updates.push_back(update);
  }

  return result;
}

inline auto sweep_two_site_right_to_left(FiniteMPS& psi, FiniteTriangularMPO const& mpo,
                                         std::vector<MpoEnvironment> const& left_envs,
                                         std::vector<MpoEnvironment>& right_envs,
                                         TwoSiteSweepOptions options = {}) -> TwoSiteSweepResult
{
  validate_two_site_sweep_inputs(psi, mpo);
  validate_right_to_left_environment_cache(psi, left_envs, right_envs);

  TwoSiteSweepResult result{.direction = TwoSiteSweepDirection::RightToLeft, .updates = {}};
  result.updates.reserve(psi.size() - 1);

  for (std::size_t offset = 0; offset + 1 < psi.size(); ++offset)
  {
    auto const left_site = psi.size() - 2 - offset;
    auto const solve_start = std::chrono::steady_clock::now();
    auto solution =
        solve_two_site(psi, mpo, left_site, left_envs[left_site], right_envs[left_site + 2], options.lanczos);
    auto const split_start = std::chrono::steady_clock::now();
    auto split = split_two_site_solution(solution, psi, left_site, TwoSiteSplitDirection::RightToLeft, options.svd);
    auto const replace_start = std::chrono::steady_clock::now();
    auto update = make_bond_update(left_site, solution, split, two_site_elapsed_seconds(solve_start, split_start),
                                   two_site_elapsed_seconds(split_start, replace_start), 0.0, 0.0);
    replace_two_site_solution(psi, left_site, std::move(split));
    auto const env_start = std::chrono::steady_clock::now();
    update.replace_seconds = two_site_elapsed_seconds(replace_start, env_start);
    assign_environment(right_envs, left_site + 1,
                       extend_right_environment(right_envs[left_site + 2], psi[left_site + 1], mpo[left_site + 1]));
    update.environment_seconds = two_site_elapsed_seconds(env_start, std::chrono::steady_clock::now());
    if (options.observer)
    {
      options.observer(TwoSiteSweepDirection::RightToLeft, update);
    }
    result.updates.push_back(update);
  }

  return result;
}

inline auto sweep_two_site_left_to_right(FiniteMPS& psi, FiniteTriangularMPO const& mpo,
                                         TwoSiteSweepOptions options = {}) -> TwoSiteSweepResult
{
  validate_two_site_sweep_inputs(psi, mpo);
  auto left_envs = build_left_environments(psi, mpo);
  auto right_envs = build_right_environments(psi, mpo);
  return sweep_two_site_left_to_right(psi, mpo, left_envs, right_envs, std::move(options));
}

inline auto sweep_two_site_right_to_left(FiniteMPS& psi, FiniteTriangularMPO const& mpo,
                                         TwoSiteSweepOptions options = {}) -> TwoSiteSweepResult
{
  validate_two_site_sweep_inputs(psi, mpo);
  auto left_envs = build_left_environments(psi, mpo);
  auto right_envs = build_right_environments(psi, mpo);
  return sweep_two_site_right_to_left(psi, mpo, left_envs, right_envs, std::move(options));
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
