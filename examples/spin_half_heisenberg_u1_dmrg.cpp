#include <uni20/models/heisenberg.hpp>
#include <uni20/mps/block_sparse_dmrg.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <uni20/common/terminal.hpp>

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <utility>

using namespace uni20;

namespace
{

auto alternating_product_state(SpinHalfSite const& spin, std::size_t length) -> BlockSparseFiniteMPS
{
  auto const indices = make_alternating_spin_half_indices(length);
  return make_block_sparse_product_state(spin.space, indices);
}

auto sweep_options(std::size_t max_rank,
                   BlockSparseTwoSiteSweepObserver observer = {}) -> BlockSparseTwoSiteSweepOptions
{
  return BlockSparseTwoSiteSweepOptions{
      .lanczos = tensorcontraction::LanczosOptions{.max_iterations = 24, .min_iterations = 2, .tolerance = 1.0e-12},
      .svd = tensorcontraction::SvdOptions{.max_rank = max_rank},
      .observer = std::move(observer),
  };
}

auto reported_site(BlockSparseTwoSiteSweepDirection direction,
                   BlockSparseTwoSiteBondUpdate const& update) -> std::size_t
{
  if (direction == BlockSparseTwoSiteSweepDirection::LeftToRight)
  {
    return update.left_site;
  }
  return update.left_site + 1;
}

auto truncation_sum(BlockSparseTwoSiteSweepResult const& result) -> double
{
  double total = 0.0;
  for (auto const& update : result.updates)
  {
    total += update.discarded_weight;
  }
  return total;
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

class BenchFile {
  public:
    explicit BenchFile(bool include_debug_global_energy) : include_debug_global_energy_(include_debug_global_energy)
    {
      char const* path = std::getenv("MP_BENCHFILE");
      if (path == nullptr || *path == '\0')
      {
        return;
      }

      file_.open(path);
      if (file_)
      {
        fmt::print(file_, "#Time #SweepNum #Site #States #Energy #Trunc #Residual #Iter #Tol");
        if (include_debug_global_energy_)
        {
          fmt::print(file_, " #DebugGlobalEnergy");
        }
        fmt::print(file_, " #SolveS #SplitS #ReplaceS #EnvS\n");
      }
    }

    [[nodiscard]] auto enabled() const noexcept -> bool { return file_.good(); }

    void write(std::size_t half_sweep, std::size_t site, BlockSparseTwoSiteBondUpdate const& update,
               std::optional<double> global_energy)
    {
      if (!this->enabled())
      {
        return;
      }

      using seconds = std::chrono::duration<double>;
      auto const elapsed = seconds(std::chrono::steady_clock::now() - start_).count();
      fmt::print(file_, "{:.9g} {} {} {} {:.16g} {:.16g} {:.9g} {} {:.9g}", elapsed, half_sweep, site, update.kept_rank,
                 update.energy, update.discarded_weight, update.lanczos.residual_norm, update.lanczos.iterations,
                 update.lanczos.tolerance);
      if (include_debug_global_energy_ && global_energy.has_value())
      {
        fmt::print(file_, " {:.16g}", *global_energy);
      }
      fmt::print(file_, " {:.9g} {:.9g} {:.9g} {:.9g}", update.solve_seconds, update.split_seconds,
                 update.replace_seconds, update.environment_seconds);
      fmt::print(file_, "\n");
    }

    void flush()
    {
      if (this->enabled())
      {
        file_.flush();
      }
    }

  private:
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    std::ofstream file_;
    bool include_debug_global_energy_ = false;
};

void run_exact_small_chain_check()
{
  auto const length = std::size_t{4};
  auto const sweep_count = std::size_t{4};
  auto const spin = make_spin_half_u1_site();
  auto psi = alternating_product_state(spin, length);
  auto const mpo = make_block_sparse_mpo_chain(make_spin_half_heisenberg_mpo(length, spin, 1.0, 0.0));

  auto const initial_energy = mps_expectation_value(psi, mpo);
  auto result = run_two_site_dmrg(psi, mpo,
                                  BlockSparseTwoSiteDmrgOptions{
                                      .sweeps = sweep_count,
                                      .sweep = sweep_options(16),
                                  });
  auto const final_energy = mps_expectation_value(psi, mpo);
  auto const exact_energy = -1.6160254037844386;

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
  fmt::print("absolute error: {:.16g}\n", std::abs(final_energy - exact_energy));

  if (std::abs(final_energy - exact_energy) > 1.0e-10)
  {
    throw std::runtime_error("strict U(1) length-4 DMRG final energy differs from exact energy");
  }
}

void run_large_chain_sweep_check()
{
  auto const length = terminal::getenv_or_default<std::size_t>("UNI20_HEISENBERG_LENGTH", 20);
  auto const sweep_count = terminal::getenv_or_default<std::size_t>("UNI20_HEISENBERG_SWEEPS", 3);
  auto const max_rank = terminal::getenv_or_default<std::size_t>("UNI20_HEISENBERG_MAX_RANK", 16);
  if (length < 2)
  {
    throw std::invalid_argument("strict U(1) DMRG check requires at least two sites");
  }

  auto const spin = make_spin_half_u1_site();
  auto psi = alternating_product_state(spin, length);
  auto const mpo = make_block_sparse_mpo_chain(make_spin_half_heisenberg_mpo(length, spin, 1.0, 0.0));
  bool const debug_global_energy = std::getenv("UNI20_DMRG_DEBUG_GLOBAL_ENERGY") != nullptr;
  bool const check_sweep_global_energy =
      debug_global_energy || std::getenv("UNI20_HEISENBERG_CHECK_GLOBAL_ENERGY") != nullptr;

  fmt::print("\nlength-{} strict U(1) sweep check\n", length);
  fmt::print("max rank: {}\n", max_rank);
  fmt::print("sweeps: {}\n", sweep_count);
  double previous_energy = mps_expectation_value(psi, mpo);
  fmt::print("initial <H>: {:.16g}\n", previous_energy);
  std::fflush(stdout);

  std::vector<BlockSparseEnvironment> left_envs;
  left_envs.reserve(length + 1);
  left_envs.push_back(make_left_boundary_environment(psi, mpo));
  auto right_envs = build_right_environments(psi, mpo);

  BenchFile bench(debug_global_energy);
  std::size_t half_sweep = 0;
  auto observer = [&](BlockSparseTwoSiteSweepDirection direction, BlockSparseTwoSiteBondUpdate const& update) {
    auto const site = reported_site(direction, update);
    std::optional<double> global_energy;
    if (debug_global_energy)
    {
      global_energy = mps_expectation_value(psi, mpo);
    }
    fmt::print("Sweep={} Site={} Energy={:.16g} States={} TruncError={:.16g} Residual={:.9g} Iter={} Tol={:.9g}",
               half_sweep, site, update.energy, update.kept_rank, update.discarded_weight, update.lanczos.residual_norm,
               update.lanczos.iterations, update.lanczos.tolerance);
    if (global_energy.has_value())
    {
      fmt::print(" DebugGlobalEnergy={:.16g}", *global_energy);
    }
    fmt::print(" SolveS={:.6g} SplitS={:.6g} ReplaceS={:.6g} EnvS={:.6g}", update.solve_seconds, update.split_seconds,
               update.replace_seconds, update.environment_seconds);
    fmt::print("\n");
    bench.write(half_sweep, site, update, global_energy);
  };
  auto options = sweep_options(max_rank, observer);

  for (std::size_t sweep = 0; sweep < sweep_count; ++sweep)
  {
    half_sweep = 2 * sweep;
    auto left_to_right = sweep_two_site_left_to_right(psi, mpo, left_envs, right_envs, options);
    fmt::print("Cumulative truncation error for sweep: {:.16g}\n", truncation_sum(left_to_right));
    bench.flush();

    half_sweep = 2 * sweep + 1;
    auto right_to_left = sweep_two_site_right_to_left(psi, mpo, left_envs, right_envs, options);
    fmt::print("Cumulative truncation error for sweep: {:.16g}\n", truncation_sum(right_to_left));
    bench.flush();

    auto const& lr = left_to_right.updates.back();
    auto const& rl = right_to_left.updates.back();
    auto const edge_energy = rl.energy;
    auto const energy = check_sweep_global_energy ? mps_expectation_value(psi, mpo) : edge_energy;
    auto const label = check_sweep_global_energy ? "<H>" : "edge energy";
    fmt::print("sweep {} {}: {:.16g}; delta {:.16g}; edge local energies L->R {:.16g}, R->L {:.16g}; kept ranks {}, "
               "{}\n",
               sweep, label, energy, energy - previous_energy, lr.energy, rl.energy, lr.kept_rank, rl.kept_rank);
    std::fflush(stdout);
    if (energy > previous_energy + 1.0e-8)
    {
      throw std::runtime_error("strict U(1) DMRG sweep increased the energy beyond tolerance");
    }
    previous_energy = energy;
  }

  fmt::print("final energy per site: {:.16g}\n", previous_energy / static_cast<double>(length));
  fmt::print("final energy per bond: {:.16g}\n", previous_energy / static_cast<double>(length - 1));
}

} // namespace

auto main() -> int
{
  try
  {
    ensure_mpi_initialized();

    fmt::print("spin-1/2 open Heisenberg DMRG smoke test\n");
    fmt::print("using strict U(1) block-sparse path\n");
    run_exact_small_chain_check();
    run_large_chain_sweep_check();
  }
  catch (std::exception const& ex)
  {
    fmt::print(stderr, "spin-half strict U(1) Heisenberg DMRG example failed: {}\n", ex.what());
    return 1;
  }

  return 0;
}
