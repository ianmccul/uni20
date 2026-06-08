#include <uni20/models/fermi_hubbard.hpp>
#include <uni20/mps/block_sparse_dmrg.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <uni20/common/terminal.hpp>

#include <mpi.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace uni20;

namespace
{

auto alternating_half_filled_indices(std::size_t length) -> std::vector<std::size_t>
{
  if (length % 2 != 0)
  {
    throw std::invalid_argument("half-filled spin-zero Hubbard sector requires an even chain length");
  }

  std::vector<std::size_t> indices;
  indices.reserve(length);
  for (std::size_t site = 0; site < length; ++site)
  {
    indices.push_back(site % 2 == 0 ? 3 : 2);
  }
  return indices;
}

auto product_state_total_charge(LocalSpace const& physical_space, std::span<std::size_t const> physical_indices) -> QNum
{
  QNum charge = QNum::identity(physical_space.symmetry());
  for (auto const physical : physical_indices)
  {
    charge = charge + physical_space[physical];
  }
  return charge;
}

void validate_product_state_sector(BlockSparseFiniteMPS const& psi, QNum const& expected_total_charge)
{
  if (psi.empty())
  {
    throw std::invalid_argument("Hubbard DMRG product-state sector validation requires a nonempty MPS");
  }

  auto const& left_boundary = psi[0].row_space();
  if (left_boundary.size() != 1 || left_boundary[0].dim != 1 ||
      left_boundary[0].q != QNum::identity(expected_total_charge.symmetry()))
  {
    throw std::invalid_argument("Hubbard DMRG product state has an invalid left boundary sector");
  }

  auto const& right_boundary = psi[psi.size() - 1].col_space();
  if (right_boundary.size() != 1 || right_boundary[0].dim != 1 || right_boundary[0].q != expected_total_charge)
  {
    throw std::invalid_argument("Hubbard DMRG product state is not in the requested total symmetry sector");
  }
}

auto sweep_options(std::size_t max_rank,
                   BlockSparseTwoSiteSweepObserver observer = {}) -> BlockSparseTwoSiteSweepOptions
{
  return BlockSparseTwoSiteSweepOptions{
      .lanczos = tensorcontraction::LanczosOptions{
          .max_iterations = static_cast<int>(terminal::getenv_or_default<long>("UNI20_HUBBARD_LANCZOS_MAX_ITERS", 24)),
          .min_iterations = static_cast<int>(terminal::getenv_or_default<long>("UNI20_HUBBARD_LANCZOS_MIN_ITERS", 2)),
          .tolerance = 1.0e-12},
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

auto format_bond_sectors(BlockSpace const& bond_space) -> std::string
{
  std::string text = "[";
  for (std::size_t index = 0; index < bond_space.size(); ++index)
  {
    if (index != 0)
    {
      text += ",";
    }
    auto const& sector = bond_space[index];
    text += "(";
    text += uni20::to_string(sector.q);
    text += ",";
    text += std::to_string(sector.dim);
    text += ")";
  }
  text += "]";
  return text;
}

auto format_bond_sectors(std::optional<BlockSpace> const& bond_space) -> std::string
{
  if (!bond_space.has_value())
  {
    return "[]";
  }
  return format_bond_sectors(*bond_space);
}

auto profile_solver_steps() -> bool { return std::getenv("UNI20_DMRG_PROFILE_SOLVER") != nullptr; }

auto boundary_sector_string(BlockSparseFiniteMPS const& psi) -> std::string
{
  if (psi.empty())
  {
    return "[]";
  }
  return format_bond_sectors(psi[psi.size() - 1].col_space());
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
    explicit BenchFile(bool include_debug_global_energy)
        : include_debug_global_energy_(include_debug_global_energy), include_solver_profile_(profile_solver_steps())
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
        fmt::print(file_, " #SolveS #SplitS #ReplaceS #EnvS #SolveCpuS #SplitCpuS #ReplaceCpuS #EnvCpuS "
                          "#BondSectors #RabcOutputBlocks #RabcOutputShape");
        if (include_solver_profile_)
        {
          fmt::print(file_, " #SolveLayoutS #SolveEffHS #SolveEngineS #SolveVectorS #SolveLanczosS"
                            " #SplitSectorsS #SplitPlanS #SplitSvdS #SplitMetadataS #SplitMaterializeS"
                            " #SolveLayoutCpuS #SolveEffHCpuS #SolveEngineCpuS #SolveVectorCpuS #SolveLanczosCpuS"
                            " #SplitSectorsCpuS #SplitPlanCpuS #SplitSvdCpuS #SplitMetadataCpuS "
                            "#SplitMaterializeCpuS"
                            " #LanczosWorkspaceS #LanczosBasisS #LanczosMatvecS #LanczosStoreHvS #LanczosOrthoS"
                            " #LanczosReduceS #LanczosRitzDiagS #LanczosRitzVectorS #LanczosResidualVectorS"
                            " #LanczosResidualNormS #LanczosFinishS"
                            " #LanczosWorkspaceCpuS #LanczosBasisCpuS #LanczosMatvecCpuS #LanczosStoreHvCpuS"
                            " #LanczosOrthoCpuS #LanczosReduceCpuS #LanczosRitzDiagCpuS #LanczosRitzVectorCpuS"
                            " #LanczosResidualVectorCpuS #LanczosResidualNormCpuS #LanczosFinishCpuS"
                            " #LanczosMatvecN #LanczosRitzDiagN #LanczosRitzVectorN #LanczosResidualVectorN");
        }
        fmt::print(file_, "\n");
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
      fmt::print(file_, " {:.9g} {:.9g} {:.9g} {:.9g}", update.solve_cpu_seconds, update.split_cpu_seconds,
                 update.replace_cpu_seconds, update.environment_cpu_seconds);
      fmt::print(file_, " {} {} {}", format_bond_sectors(update.shared_bond_space), update.rabc_output_blocks,
                 update.rabc_output_shape_signature);
      if (include_solver_profile_)
      {
        auto const& solve = update.solve_timings;
        auto const& split = update.split_timings;
        fmt::print(file_, " {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}", solve.layout.wall_seconds,
                   solve.effective_hamiltonian.wall_seconds, solve.engine.wall_seconds,
                   solve.initial_vector.wall_seconds, solve.lanczos.wall_seconds);
        fmt::print(file_, " {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}", split.sectors.wall_seconds, split.plan.wall_seconds,
                   split.svd.wall_seconds, split.metadata.wall_seconds, split.materialize.wall_seconds);
        fmt::print(file_, " {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}", solve.layout.cpu_seconds,
                   solve.effective_hamiltonian.cpu_seconds, solve.engine.cpu_seconds, solve.initial_vector.cpu_seconds,
                   solve.lanczos.cpu_seconds);
        fmt::print(file_, " {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}", split.sectors.cpu_seconds, split.plan.cpu_seconds,
                   split.svd.cpu_seconds, split.metadata.cpu_seconds, split.materialize.cpu_seconds);
        auto const& lanczos = update.lanczos.timings;
        fmt::print(file_, " {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}",
                   lanczos.workspace.wall_seconds, lanczos.basis_setup.wall_seconds, lanczos.matvec.wall_seconds,
                   lanczos.store_hamiltonian_vector.wall_seconds, lanczos.orthogonalization.wall_seconds,
                   lanczos.reductions.wall_seconds, lanczos.ritz_diagonalization.wall_seconds,
                   lanczos.ritz_vector.wall_seconds, lanczos.residual_vector.wall_seconds,
                   lanczos.residual_norm.wall_seconds, lanczos.finish.wall_seconds);
        fmt::print(file_, " {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}",
                   lanczos.workspace.cpu_seconds, lanczos.basis_setup.cpu_seconds, lanczos.matvec.cpu_seconds,
                   lanczos.store_hamiltonian_vector.cpu_seconds, lanczos.orthogonalization.cpu_seconds,
                   lanczos.reductions.cpu_seconds, lanczos.ritz_diagonalization.cpu_seconds,
                   lanczos.ritz_vector.cpu_seconds, lanczos.residual_vector.cpu_seconds,
                   lanczos.residual_norm.cpu_seconds, lanczos.finish.cpu_seconds);
        fmt::print(file_, " {} {} {} {}", lanczos.matvec_count, lanczos.ritz_diagonalization_count,
                   lanczos.ritz_vector_count, lanczos.residual_vector_count);
      }
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
    bool include_solver_profile_ = false;
};

void run_hubbard_sweep_check()
{
  auto const length = terminal::getenv_or_default<std::size_t>("UNI20_HUBBARD_LENGTH", 20);
  auto const sweep_count = terminal::getenv_or_default<std::size_t>("UNI20_HUBBARD_SWEEPS", 3);
  auto const max_rank = terminal::getenv_or_default<std::size_t>("UNI20_HUBBARD_MAX_RANK", 16);
  auto const hopping = terminal::getenv_or_default<double>("UNI20_HUBBARD_T", 1.0);
  auto const onsite_u = terminal::getenv_or_default<double>("UNI20_HUBBARD_U", 4.0);
  if (length < 2)
  {
    throw std::invalid_argument("strict U(1)xU(1) Hubbard DMRG check requires at least two sites");
  }

  auto const site = make_fermi_hubbard_u1u1_site();
  auto const physical_indices = alternating_half_filled_indices(length);
  auto const target_sector = product_state_total_charge(site.space, physical_indices);
  auto psi = make_block_sparse_product_state(site.space, physical_indices);
  validate_product_state_sector(psi, target_sector);
  auto const mpo = make_block_sparse_mpo_chain(make_fermi_hubbard_mpo(length, site, hopping, onsite_u));
  bool const debug_global_energy = std::getenv("UNI20_DMRG_DEBUG_GLOBAL_ENERGY") != nullptr;
  bool const check_sweep_global_energy =
      debug_global_energy || std::getenv("UNI20_HUBBARD_CHECK_GLOBAL_ENERGY") != nullptr;

  fmt::print("\nlength-{} strict U(1)xU(1) Hubbard sweep check\n", length);
  fmt::print("hopping t: {:.16g}\n", hopping);
  fmt::print("onsite U: {:.16g}\n", onsite_u);
  fmt::print("target sector: {} (half-filled, spin zero)\n", uni20::to_string(target_sector));
  fmt::print("initial right boundary sector: {}\n", boundary_sector_string(psi));
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
    auto const site_index = reported_site(direction, update);
    std::optional<double> global_energy;
    if (debug_global_energy)
    {
      global_energy = mps_expectation_value(psi, mpo);
    }
    fmt::print("Sweep={} Site={} Energy={:.16g} States={} TruncError={:.16g} Residual={:.9g} Iter={} Tol={:.9g}",
               half_sweep, site_index, update.energy, update.kept_rank, update.discarded_weight,
               update.lanczos.residual_norm, update.lanczos.iterations, update.lanczos.tolerance);
    if (global_energy.has_value())
    {
      fmt::print(" DebugGlobalEnergy={:.16g}", *global_energy);
    }
    fmt::print(" SolveS={:.6g} SplitS={:.6g} ReplaceS={:.6g} EnvS={:.6g}", update.solve_seconds, update.split_seconds,
               update.replace_seconds, update.environment_seconds);
    fmt::print("\n");
    bench.write(half_sweep, site_index, update, global_energy);
  };
  auto options = sweep_options(max_rank, observer);

  // Optional bond-dimension ramp: hold each max_rank for one full sweep (2 half
  // sweeps) while doubling 2,4,8,... up to UNI20_HUBBARD_MAX_RANK, giving a clean
  // scaling series of central-bond structures (M x M, plus M x M/2 transitions)
  // from a single run.
  bool const ramp_max_rank = std::getenv("UNI20_HUBBARD_RAMP") != nullptr;
  std::vector<std::size_t> rank_schedule;
  if (ramp_max_rank)
  {
    for (std::size_t rank = 2; rank < max_rank; rank *= 2)
    {
      rank_schedule.push_back(rank);
    }
    rank_schedule.push_back(max_rank);
    fmt::print("max-rank ramp: {} stages (2..{})\n", rank_schedule.size(), max_rank);
  }
  std::size_t const effective_sweeps = ramp_max_rank ? rank_schedule.size() : sweep_count;

  for (std::size_t sweep = 0; sweep < effective_sweeps; ++sweep)
  {
    if (ramp_max_rank)
    {
      options.svd.max_rank = rank_schedule[sweep];
    }
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
    if (check_sweep_global_energy && energy > previous_energy + 1.0e-8)
    {
      throw std::runtime_error("strict U(1)xU(1) Hubbard DMRG sweep increased the energy beyond tolerance");
    }
    if (!check_sweep_global_energy && energy > previous_energy + 1.0e-8)
    {
      fmt::print(stderr,
                 "warning: edge local energy increased by {:.16g}; enable UNI20_HUBBARD_CHECK_GLOBAL_ENERGY=1 for "
                 "the slower global diagnostic\n",
                 energy - previous_energy);
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

    fmt::print("Fermi-Hubbard DMRG smoke benchmark\n");
    fmt::print("using strict U(1)xU(1) block-sparse path\n");
    run_hubbard_sweep_check();
  }
  catch (std::exception const& ex)
  {
    fmt::print(stderr, "strict U(1)xU(1) Fermi-Hubbard DMRG example failed: {}\n", ex.what());
    return 1;
  }

  return 0;
}
