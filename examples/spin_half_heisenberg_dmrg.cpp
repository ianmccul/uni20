#include <uni20/models/heisenberg.hpp>
#include <uni20/mps/dmrg.hpp>

#include <fmt/core.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace uni20;

namespace
{

auto bond_space(Symmetry sym, std::size_t dim) -> BlockSpace
{
  return BlockSpace(sym, {BlockSector{QNum::identity(sym), dim}});
}

auto make_spin_half_dense_site() -> SpinHalfSite
{
  Symmetry const symmetry{"Trivial:U(1)"};
  QNum const scalar = QNum::identity(symmetry);
  LocalSpace const space(symmetry, {scalar, scalar});

  LocalOperator identity(space, space, scalar);
  identity.insert_or_assign(0, 0, 1.0);
  identity.insert_or_assign(1, 1, 1.0);

  LocalOperator sz(space, space, scalar);
  sz.insert_or_assign(0, 0, 0.5);
  sz.insert_or_assign(1, 1, -0.5);

  LocalOperator sp(space, space, scalar);
  sp.insert_or_assign(0, 1, 1.0);

  LocalOperator sm(space, space, scalar);
  sm.insert_or_assign(1, 0, 1.0);

  LocalOperator sigma_z(space, space, scalar);
  sigma_z.insert_or_assign(0, 0, 1.0);
  sigma_z.insert_or_assign(1, 1, -1.0);

  return SpinHalfSite{
      .symmetry = symmetry,
      .space = space,
      .up = scalar,
      .down = scalar,
      .identity = std::move(identity),
      .sz = std::move(sz),
      .sp = std::move(sp),
      .sm = std::move(sm),
      .sigma_z = std::move(sigma_z),
  };
}

void ensure_mpi_initialized()
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0)
  {
    return;
  }

  MPI_Init(nullptr, nullptr);
  std::atexit([] {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0)
    {
      MPI_Finalize();
    }
  });
}

auto alternating_product_state(SpinHalfSite const& spin, std::size_t length) -> FiniteMPS
{
  std::vector<MpsSiteTensor> sites;
  sites.reserve(length);
  for (std::size_t site = 0; site < length; ++site)
  {
    MpsSiteTensor tensor(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
    if (site % 2 == 0)
    {
      tensor.assign(0, std::array{1.0});
      tensor.assign(1, std::array{0.0});
    }
    else
    {
      tensor.assign(0, std::array{0.0});
      tensor.assign(1, std::array{1.0});
    }
    sites.push_back(std::move(tensor));
  }
  return FiniteMPS(std::move(sites));
}

auto dense_index(std::size_t n, std::size_t row, std::size_t col) -> std::size_t { return row * n + col; }

auto exact_open_heisenberg_energy(std::size_t length) -> double
{
  if (length < 2)
  {
    throw std::invalid_argument("exact Heisenberg reference requires at least two sites");
  }

  auto const dim = std::size_t{1} << length;
  std::vector<double> hamiltonian(dim * dim, 0.0);
  for (std::size_t state = 0; state < dim; ++state)
  {
    for (std::size_t site = 0; site + 1 < length; ++site)
    {
      bool const left_down = ((state >> site) & 1U) != 0U;
      bool const right_down = ((state >> (site + 1)) & 1U) != 0U;
      double const left_sz = left_down ? -0.5 : 0.5;
      double const right_sz = right_down ? -0.5 : 0.5;
      hamiltonian[dense_index(dim, state, state)] += left_sz * right_sz;
      if (left_down != right_down)
      {
        auto const flipped = state ^ (std::size_t{1} << site) ^ (std::size_t{1} << (site + 1));
        hamiltonian[dense_index(dim, flipped, state)] += 0.5;
      }
    }
  }

  std::vector<double> eigenvectors(dim * dim, 0.0);
  for (std::size_t i = 0; i < dim; ++i)
  {
    eigenvectors[dense_index(dim, i, i)] = 1.0;
  }

  auto const max_sweeps = std::max<std::size_t>(64, 32 * dim * dim);
  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep)
  {
    std::size_t p = 0;
    std::size_t q = 0;
    double max_offdiag = 0.0;
    for (std::size_t row = 0; row < dim; ++row)
    {
      for (std::size_t col = row + 1; col < dim; ++col)
      {
        double const value = std::abs(hamiltonian[dense_index(dim, row, col)]);
        if (value > max_offdiag)
        {
          max_offdiag = value;
          p = row;
          q = col;
        }
      }
    }

    if (max_offdiag <= 100.0 * std::numeric_limits<double>::epsilon())
    {
      break;
    }

    double const app = hamiltonian[dense_index(dim, p, p)];
    double const aqq = hamiltonian[dense_index(dim, q, q)];
    double const apq = hamiltonian[dense_index(dim, p, q)];
    double const tau = (aqq - app) / (2.0 * apq);
    double const sign = tau < 0.0 ? -1.0 : 1.0;
    double const t = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    double const c = 1.0 / std::sqrt(1.0 + t * t);
    double const s = t * c;

    for (std::size_t k = 0; k < dim; ++k)
    {
      if (k == p || k == q)
      {
        continue;
      }
      double const akp = hamiltonian[dense_index(dim, k, p)];
      double const akq = hamiltonian[dense_index(dim, k, q)];
      hamiltonian[dense_index(dim, k, p)] = hamiltonian[dense_index(dim, p, k)] = c * akp - s * akq;
      hamiltonian[dense_index(dim, k, q)] = hamiltonian[dense_index(dim, q, k)] = s * akp + c * akq;
    }
    hamiltonian[dense_index(dim, p, p)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    hamiltonian[dense_index(dim, q, q)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    hamiltonian[dense_index(dim, p, q)] = 0.0;
    hamiltonian[dense_index(dim, q, p)] = 0.0;
  }

  double lowest = hamiltonian[dense_index(dim, 0, 0)];
  for (std::size_t i = 1; i < dim; ++i)
  {
    lowest = std::min(lowest, hamiltonian[dense_index(dim, i, i)]);
  }
  return lowest;
}

auto mps_expectation_value(FiniteMPS const& psi, FiniteTriangularMPO const& mpo) -> double
{
  auto left_envs = build_left_environments(psi, mpo);
  auto const& final_env = left_envs.back();
  if (final_env.bond_dim() != 1)
  {
    throw std::logic_error("MPS expectation helper requires scalar boundary MPS bonds");
  }
  return final_env.values(final_env.virtual_dim() - 1)[0];
}

auto dmrg_options(std::size_t sweeps, std::size_t max_rank) -> TwoSiteDmrgOptions
{
  return TwoSiteDmrgOptions{
      .sweeps = sweeps,
      .sweep =
          TwoSiteSweepOptions{
              .lanczos =
                  tensorcontraction::LanczosOptions{.max_iterations = 24, .min_iterations = 2, .tolerance = 1.0e-12},
              .svd = tensorcontraction::SvdOptions{.max_rank = max_rank},
          },
  };
}

void run_exact_small_chain_check()
{
  auto const length = std::size_t{4};
  auto const sweep_count = std::size_t{4};
  auto const spin = make_spin_half_dense_site();
  auto psi = alternating_product_state(spin, length);
  auto mpo = make_spin_half_heisenberg_mpo(length, spin, 1.0, 0.0);

  auto const initial_energy = mps_expectation_value(psi, mpo);
  auto const exact_energy = exact_open_heisenberg_energy(length);
  auto result = run_two_site_dmrg(psi, mpo, dmrg_options(sweep_count, 16));
  auto const final_energy = mps_expectation_value(psi, mpo);
  auto const error = std::abs(final_energy - exact_energy);

  fmt::print("\nlength-4 exact check\n");
  fmt::print("initial <H>: {:.16g}\n", initial_energy);
  for (auto const& sweep : result.sweeps)
  {
    auto const& lr = sweep.left_to_right.updates.back();
    auto const& rl = sweep.right_to_left.updates.back();
    fmt::print("sweep {} local energies: L->R {:.16g}, R->L {:.16g}; kept ranks {}, {}\n", sweep.sweep, lr.energy,
               rl.energy, lr.kept_rank, rl.kept_rank);
  }
  fmt::print("final <H>: {:.16g}\n", final_energy);
  fmt::print("exact <H>: {:.16g}\n", exact_energy);
  fmt::print("absolute error: {:.16g}\n", error);

  if (final_energy >= initial_energy)
  {
    throw std::runtime_error("length-4 DMRG did not lower the initial product-state energy");
  }
  if (error > 1.0e-8)
  {
    throw std::runtime_error("length-4 DMRG final energy differs from exact diagonalization by more than tolerance");
  }
}

void run_large_chain_sweep_check()
{
  auto const length = std::size_t{20};
  auto const sweep_count = std::size_t{3};
  auto const max_rank = std::size_t{16};
  auto const spin = make_spin_half_dense_site();
  auto psi = alternating_product_state(spin, length);
  auto mpo = make_spin_half_heisenberg_mpo(length, spin, 1.0, 0.0);
  auto options = dmrg_options(1, max_rank);

  fmt::print("\nlength-20 sweep check\n");
  fmt::print("max rank: {}\n", max_rank);
  double previous_energy = mps_expectation_value(psi, mpo);
  fmt::print("initial <H>: {:.16g}\n", previous_energy);
  std::fflush(stdout);
  for (std::size_t sweep = 0; sweep < sweep_count; ++sweep)
  {
    auto result = run_two_site_dmrg(psi, mpo, options);
    auto const energy = mps_expectation_value(psi, mpo);
    auto const& lr = result.sweeps.front().left_to_right.updates.back();
    auto const& rl = result.sweeps.front().right_to_left.updates.back();
    fmt::print("sweep {} <H>: {:.16g}; delta {:.16g}; edge local energies L->R {:.16g}, R->L {:.16g}; kept ranks {}, "
               "{}\n",
               sweep, energy, energy - previous_energy, lr.energy, rl.energy, lr.kept_rank, rl.kept_rank);
    std::fflush(stdout);
    if (energy > previous_energy + 1.0e-8)
    {
      throw std::runtime_error("length-20 sweep increased the global MPS energy beyond tolerance");
    }
    previous_energy = energy;
  }

  auto const final_energy_per_site = previous_energy / static_cast<double>(length);
  auto const final_energy_per_bond = previous_energy / static_cast<double>(length - 1);
  fmt::print("final energy per site: {:.16g}\n", final_energy_per_site);
  fmt::print("final energy per bond: {:.16g}\n", final_energy_per_bond);

  if (previous_energy > -8.0)
  {
    throw std::runtime_error("length-20 DMRG energy is too high for the open Heisenberg chain smoke threshold");
  }
}

} // namespace

auto main() -> int
{
  try
  {
    ensure_mpi_initialized();

    fmt::print("spin-1/2 open Heisenberg DMRG smoke test\n");
    fmt::print("using dense placeholder symmetry: all states carry the identity charge\n");
    run_exact_small_chain_check();
    run_large_chain_sweep_check();
  }
  catch (std::exception const& ex)
  {
    fmt::print(stderr, "spin_half_heisenberg_dmrg failed: {}\n", ex.what());
    return 1;
  }

  return 0;
}
