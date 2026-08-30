#include <uni20/config.hpp>

#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#if UNI20_BACKEND_CUBLAS
#include <uni20/backend/cublas/execution.hpp>
#endif
#if UNI20_BACKEND_CUDA
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/storage/cuda_storage.hpp>
#endif
#if UNI20_BACKEND_CUSOLVER
#include <uni20/backend/cusolver/execution.hpp>
#endif
#include <uni20/core/scalar_io.hpp>
#include <uni20/core/scalar_precision.hpp>
#include <uni20/core/types.hpp>
#include <uni20/models/spin_half_heisenberg.hpp>
#include <uni20/symmetry/block_tensor_storage.hpp>
#include <uni20/tensor_network/environment_cache.hpp>
#include <uni20/tensor_network/two_site_dmrg.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

enum class MeasurementMode
{
  off,
  coarse,
  detailed,
};

enum class ExecutionMode
{
  cpu,
  cuda,
};

enum class MpsStorageMode
{
  packed,
  parallel_packed,
  parallel_aligned_packed,
  parallel_separate,
};

struct Options
{
    std::size_t sites = 4;
    std::size_t maximum_states = 16;
    std::size_t maximum_sweeps = 8;
    std::string energy_tolerance = "0";
    std::size_t local_matvecs = 4;
    std::size_t block_threads = 1;
    std::size_t steady_sweeps = 0;
    std::size_t cuda_streams = 4;
    std::size_t cuda_submitters = 1;
    std::optional<std::size_t> cuda_cublas_handles{};
    std::optional<std::size_t> cuda_cusolver_handles{};
    int cuda_device = 0;
    MeasurementMode measurements = MeasurementMode::off;
    uni20::ScalarPrecision precision = uni20::ScalarPrecision::fp64;
    MpsStorageMode mps_storage = MpsStorageMode::packed;
    ExecutionMode execution = ExecutionMode::cpu;
    bool complex_scalar = false;
    bool check = false;
};

struct ReferenceEnergy
{
    std::string_view value;
    std::string_view float32_tolerance;
    std::string_view tolerance;
    std::string_view float128_tolerance;
    std::string_view source;
};

[[nodiscard]] auto option_value(std::string_view argument, std::string_view prefix) -> std::string_view
{
  if (!argument.starts_with(prefix)) return {};
  return argument.substr(prefix.size());
}

[[nodiscard]] auto parse_size(std::string_view value, std::string_view option) -> std::size_t
{
  std::size_t result = 0;
  auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() || result == 0)
    throw std::invalid_argument("invalid positive value for " + std::string(option));
  return result;
}

[[nodiscard]] auto parse_nonnegative_int(std::string_view value, std::string_view option) -> int
{
  int result = 0;
  auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() || result < 0)
    throw std::invalid_argument("invalid nonnegative value for " + std::string(option));
  return result;
}
[[nodiscard]] auto parse_options(int argc, char** argv) -> Options
{
  Options options;
  for (int index = 1; index < argc; ++index)
  {
    std::string_view const argument(argv[index]);
    if (auto const value = option_value(argument, "--sites="); !value.empty())
      options.sites = parse_size(value, "--sites");
    else if (auto const value = option_value(argument, "--max-states="); !value.empty())
      options.maximum_states = parse_size(value, "--max-states");
    else if (auto const value = option_value(argument, "--max-sweeps="); !value.empty())
      options.maximum_sweeps = parse_size(value, "--max-sweeps");
    else if (auto const value = option_value(argument, "--energy-tol="); !value.empty())
      options.energy_tolerance = value;
    else if (auto const value = option_value(argument, "--local-matvecs="); !value.empty())
      options.local_matvecs = parse_size(value, "--local-matvecs");
    else if (auto const value = option_value(argument, "--block-threads="); !value.empty())
      options.block_threads = parse_size(value, "--block-threads");
    else if (auto const value = option_value(argument, "--steady-sweeps="); !value.empty())
      options.steady_sweeps = parse_size(value, "--steady-sweeps");
    else if (auto const value = option_value(argument, "--cuda-streams="); !value.empty())
      options.cuda_streams = parse_size(value, "--cuda-streams");
    else if (auto const value = option_value(argument, "--cuda-submitters="); !value.empty())
      options.cuda_submitters = parse_size(value, "--cuda-submitters");
    else if (auto const value = option_value(argument, "--cuda-cublas-handles="); !value.empty())
      options.cuda_cublas_handles = parse_size(value, "--cuda-cublas-handles");
    else if (auto const value = option_value(argument, "--cuda-cusolver-handles="); !value.empty())
      options.cuda_cusolver_handles = parse_size(value, "--cuda-cusolver-handles");
    else if (auto const value = option_value(argument, "--cuda-device="); !value.empty())
      options.cuda_device = parse_nonnegative_int(value, "--cuda-device");
    else if (auto const value = option_value(argument, "--measurements="); value == "off")
      options.measurements = MeasurementMode::off;
    else if (auto const value = option_value(argument, "--measurements="); value == "coarse")
      options.measurements = MeasurementMode::coarse;
    else if (auto const value = option_value(argument, "--measurements="); value == "detailed")
      options.measurements = MeasurementMode::detailed;
    else if (auto const value = option_value(argument, "--precision="); !value.empty())
    {
      auto const precision = uni20::parse_scalar_precision(value);
      if (!precision) throw std::invalid_argument("unknown precision: " + std::string(value));
      options.precision = *precision;
    }
    else if (auto const value = option_value(argument, "--mps-storage="); value == "packed")
      options.mps_storage = MpsStorageMode::packed;
    else if (auto const value = option_value(argument, "--mps-storage="); value == "parallel-packed")
      options.mps_storage = MpsStorageMode::parallel_packed;
    else if (auto const value = option_value(argument, "--mps-storage="); value == "parallel-aligned-packed")
      options.mps_storage = MpsStorageMode::parallel_aligned_packed;
    else if (auto const value = option_value(argument, "--mps-storage="); value == "parallel-separate")
      options.mps_storage = MpsStorageMode::parallel_separate;
    else if (auto const value = option_value(argument, "--execution="); value == "cpu")
      options.execution = ExecutionMode::cpu;
    else if (auto const value = option_value(argument, "--execution="); value == "cuda")
      options.execution = ExecutionMode::cuda;
    else if (auto const value = option_value(argument, "--scalar="); value == "real")
      options.complex_scalar = false;
    else if (auto const value = option_value(argument, "--scalar="); value == "complex")
      options.complex_scalar = true;
    else if (argument == "--check")
      options.check = true;
    else
      throw std::invalid_argument("unknown option: " + std::string(argument));
  }
  if (options.block_threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("--block-threads exceeds the scheduler concurrency range");
  if (options.check && options.steady_sweeps != 0)
    throw std::invalid_argument("--check cannot be combined with --steady-sweeps");
  if (options.execution == ExecutionMode::cuda && options.complex_scalar)
    throw std::invalid_argument("CUDA DMRG currently supports --scalar=real only");
  if (options.execution == ExecutionMode::cuda && options.precision != uni20::ScalarPrecision::fp64)
    throw std::invalid_argument("CUDA DMRG currently supports --precision=fp64 only");
  if (options.cuda_cublas_handles && *options.cuda_cublas_handles > options.cuda_streams)
    throw std::invalid_argument("--cuda-cublas-handles cannot exceed --cuda-streams");
  if (options.cuda_cusolver_handles && *options.cuda_cusolver_handles > options.cuda_streams)
    throw std::invalid_argument("--cuda-cusolver-handles cannot exceed --cuda-streams");
  return options;
}

[[nodiscard]] auto reference_energy(std::size_t sites) -> std::optional<ReferenceEnergy>
{
  if (sites == 4)
    return ReferenceEnergy{.value = "-1.6160254037844386467637231707529361834714026269051903140279034897259665084544",
                           .float32_tolerance = "1e-6",
                           .tolerance = "1e-12",
                           .float128_tolerance = "1e-24",
                           .source = "exact analytic value"};
  if (sites == 20)
    return ReferenceEnergy{.value = "-8.682473334398985",
                           .float32_tolerance = "1e-5",
                           .tolerance = "1e-10",
                           .float128_tolerance = "1e-10",
                           .source = "Matrix Product Toolkit mp-dmrg-2site, m=128"};
  return std::nullopt;
}

template <class Measurements> void print_performance_measurements(Measurements const& measurements)
{
  using uni20::performance::MeasurementLevel;
  using event = uni20::tensor_network::TwoSiteDmrgPerformanceEvent;
  for (std::size_t index = 0; index < static_cast<std::size_t>(event::count); ++index)
  {
    auto const phase = static_cast<event>(index);
    auto const& statistics = measurements[phase];
    if (statistics.count == 0) continue;
    std::cout << "timing " << uni20::tensor_network::two_site_dmrg_performance_event_name(phase)
              << "  count=" << statistics.count << "  total=" << std::chrono::duration<double>(statistics.total).count()
              << "  mean=" << std::chrono::duration<double>(statistics.mean()).count() << " seconds\n";
  }

  if constexpr (uni20::performance::measurement_level_v<Measurements> == MeasurementLevel::detailed)
  {
    auto const batches = measurements.batches(event::svd_sector_batch);
    std::size_t requested_items = 0;
    std::size_t started_items = 0;
    std::size_t completed_items = 0;
    std::size_t peak_concurrency = 0;
    uni20::performance::Duration total_item_duration = uni20::performance::Duration::zero();
    uni20::performance::Duration maximum_item_duration = uni20::performance::Duration::zero();
    uni20::performance::Duration maximum_finish_spread = uni20::performance::Duration::zero();
    uni20::performance::Duration total_return_tail = uni20::performance::Duration::zero();
    for (auto const& batch : batches)
    {
      requested_items += batch.requested_items;
      started_items += batch.started_items;
      completed_items += batch.completed_items;
      peak_concurrency = std::max(peak_concurrency, batch.peak_concurrency);
      total_item_duration += batch.total_item_duration;
      maximum_item_duration = std::max(maximum_item_duration, batch.maximum_item_duration);
      maximum_finish_spread = std::max(maximum_finish_spread, batch.item_finish_spread);
      total_return_tail += batch.return_after_last_finish;
    }
    std::cout << "SVD batch detail" << "  batches=" << batches.size() << "  requested=" << requested_items
              << "  started=" << started_items << "  completed=" << completed_items
              << "  peak-overlap=" << peak_concurrency
              << "  item-total=" << std::chrono::duration<double>(total_item_duration).count()
              << "  max-item=" << std::chrono::duration<double>(maximum_item_duration).count()
              << "  max-finish-spread=" << std::chrono::duration<double>(maximum_finish_spread).count()
              << "  return-tail-total=" << std::chrono::duration<double>(total_return_tail).count() << " seconds\n";
  }
}

[[nodiscard]] constexpr auto direction_name(uni20::tensor_network::MpsSweepDirection direction) -> std::string_view
{
  return direction == uni20::tensor_network::MpsSweepDirection::left_to_right ? "left-to-right" : "right-to-left";
}

[[nodiscard]] constexpr auto
opposite_direction(uni20::tensor_network::MpsSweepDirection direction) -> uni20::tensor_network::MpsSweepDirection
{
  return direction == uni20::tensor_network::MpsSweepDirection::left_to_right
             ? uni20::tensor_network::MpsSweepDirection::right_to_left
             : uni20::tensor_network::MpsSweepDirection::left_to_right;
}

struct SweepObservation
{
    double terminal_energy;
    double maximum_discarded_weight;
    std::size_t maximum_bond_dimension;
};

template <class Steps> [[nodiscard]] auto observe_sweep(Steps const& steps) -> SweepObservation
{
  SweepObservation result{
      .terminal_energy = static_cast<double>(steps.back().local_energy),
      .maximum_discarded_weight = 0.0,
      .maximum_bond_dimension = 0,
  };
  for (auto const& step : steps)
  {
    result.maximum_discarded_weight =
        std::max(result.maximum_discarded_weight, static_cast<double>(step.bond.truncation.discarded_weight));
    result.maximum_bond_dimension = std::max(result.maximum_bond_dimension, step.bond.truncation.retained_rank);
  }
  return result;
}

template <class Output, uni20::BlockTensorView Input>
[[nodiscard]] auto materialize_sparse_block_tensor(Input const& input) -> Output
{
  static_assert(uni20::SparseBlockStorage<typename Output::storage_policy>);
  using key_type = typename Output::key_type;
  Output output(input.symmetry(), input.domain(), input.codomain(),
                std::vector<key_type>(input.stored_keys().begin(), input.stored_keys().end()));
  for (std::size_t ordinal = 0; ordinal < output.stored_block_count(); ++ordinal)
  {
    auto output_block = output.block_by_ordinal(ordinal);
    auto input_block = input.block_by_ordinal(ordinal);
    uni20::copy(output_block, input_block);
  }
  return output;
}

} // namespace

template <class MpsStorage> constexpr auto mps_storage_name() -> std::string_view
{
#if UNI20_BACKEND_CUDA
  if constexpr (std::same_as<MpsStorage, uni20::ParallelPackedSparseBlockStorage<uni20::CudaStorage>>)
    return "parallel-packed-cuda";
#endif
  if constexpr (std::same_as<MpsStorage, uni20::PackedSparseBlockStorage<>>)
    return "packed";
  else if constexpr (std::same_as<MpsStorage, uni20::ParallelPackedSparseBlockStorage<>>)
    return "parallel-packed";
  else if constexpr (std::same_as<MpsStorage, uni20::ParallelPackedSparseBlockStorage<uni20::HostStorage, 64>>)
    return "parallel-aligned-packed";
  else
    return "parallel-separate";
}

template <class MpsStorage> void synchronize_execution()
{
#if UNI20_BACKEND_CUDA
  if constexpr (std::same_as<typename MpsStorage::leaf_storage_policy, uni20::CudaStorage>)
    uni20::cuda::device_resources().streams().synchronize();
#endif
}

template <uni20::RealOrComplex Scalar, uni20::BlockTensorStorage MpsStorage>
[[nodiscard]] auto make_initial_mps(Options const& run, uni20::models::SpinHalfU1Site const& local)
{
  if constexpr (uni20::ImmediateLocalBlockStorageFor<MpsStorage, Scalar, 3, 2>)
  {
    return uni20::models::make_neel_product_mps<Scalar, MpsStorage>(run.sites, local);
  }
  else
  {
    auto host = uni20::models::make_neel_product_mps<Scalar>(run.sites, local);
    using result_type = uni20::models::SpinHalfU1Mps<Scalar, MpsStorage>;
    using site_type = typename result_type::site_type;
    std::vector<site_type> sites;
    sites.reserve(host.size());
    for (std::size_t site = 0; site < host.size(); ++site)
      sites.push_back(materialize_sparse_block_tensor<site_type>(host.site(site)));
    return result_type(std::move(sites));
  }
}

template <uni20::RealOrComplex Scalar, uni20::BlockTensorStorage MpsStorage> int run_example(Options const& run)
{
  using real_type = uni20::make_real_t<Scalar>;
  using std::abs;
  using leaf_storage = typename MpsStorage::leaf_storage_policy;
  static constexpr std::size_t block_alignment = std::same_as<leaf_storage, uni20::HostStorage> ? 64 : 1;
  using environment_storage = uni20::ParallelPackedSparseBlockStorage<leaf_storage, block_alignment>;
  using center_storage = uni20::ParallelPackedCompleteBlockStorage<leaf_storage, block_alignment>;
  std::optional<std::size_t> const batch_concurrency = [&]() -> std::optional<std::size_t> {
#if UNI20_BACKEND_CUDA
    if constexpr (std::same_as<leaf_storage, uni20::CudaStorage>) return run.cuda_submitters;
#endif
    return std::nullopt;
  }();
  uni20::async::TbbScheduler scheduler{
      static_cast<int>(run.block_threads), {}, {.maximum_concurrency = batch_concurrency}};
  uni20::async::ScopedScheduler use_scheduler(&scheduler);
  auto const local = uni20::models::make_spin_half_u1_site();
  auto mps = make_initial_mps<Scalar, MpsStorage>(run, local);
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo<Scalar>(run.sites, local);
  using mps_type = std::remove_cvref_t<decltype(mps)>;
  using mpo_type = std::remove_cvref_t<decltype(mpo)>;
  uni20::tensor_network::MpoEnvironmentCache<mps_type, mpo_type, environment_storage> environments(mps, mpo, 0, 0);
  synchronize_execution<MpsStorage>();

  uni20::tensor_network::TwoSiteDmrgRunOptions<real_type> options;
  options.maximum_sweeps = run.maximum_sweeps;
  options.energy_tolerance = uni20::parse_real<real_type>(run.energy_tolerance);
  options.bond_options.local_solver.matvec_iterations = run.local_matvecs;
  options.bond_options.truncation.maximum_retained_extent = run.maximum_states;
  auto execute = [&]<class Measurements>(Measurements& measurements) {
    std::cout << "scalar=" << (uni20::Complex<Scalar> ? "complex" : "real") << '\n';
    std::cout << "precision=" << uni20::scalar_precision_name(run.precision) << '\n';
    std::cout << "execution=" << (run.execution == ExecutionMode::cpu ? "cpu" : "cuda") << '\n';
    std::cout << "mps-storage=" << mps_storage_name<MpsStorage>() << '\n';
    std::cout << "block-threads=" << run.block_threads << '\n';
    if (run.execution == ExecutionMode::cuda)
    {
      std::cout << "cuda-device=" << run.cuda_device << "\ncuda-streams=" << run.cuda_streams
                << "\ncuda-submitters=" << run.cuda_submitters;
#if UNI20_BACKEND_CUDA && UNI20_BACKEND_CUBLAS && UNI20_BACKEND_CUSOLVER
      auto& resources = uni20::cuda::device_resources(run.cuda_device);
      std::cout << "\ncuda-cublas-handles=" << uni20::cublas::execution_pool(resources).handle_count()
                << "\ncuda-cusolver-handles=" << uni20::cusolver::execution_pool(resources).handle_count();
#endif
      std::cout << '\n';
    }

    if (run.steady_sweeps != 0)
    {
      using direction_type = uni20::tensor_network::MpsSweepDirection;
      direction_type direction = direction_type::left_to_right;
      std::size_t const expected_growth_sweeps = std::max<std::size_t>(1, std::bit_width(run.maximum_states - 1));
      std::size_t const growth_sweep_limit = expected_growth_sweeps + 2;
      std::size_t growth_sweeps = 0;
      double growth_elapsed = 0.0;
      double final_energy = 0.0;
      bool reached_requested_dimension = false;
      uni20::performance::NoMeasurements growth_measurements;

      while (growth_sweeps < growth_sweep_limit)
      {
        auto const start = std::chrono::steady_clock::now();
        auto steps = uni20::tensor_network::sweep_two_site_dmrg(mps, mpo, environments, direction, options.bond_options,
                                                                growth_measurements, center_storage{});
        synchronize_execution<MpsStorage>();
        double const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        growth_elapsed += elapsed;
        auto const observation = observe_sweep(steps);
        final_energy = observation.terminal_energy;
        std::cout << "growth-sweep " << growth_sweeps << "  " << direction_name(direction)
                  << "  energy=" << observation.terminal_energy << "  max-m=" << observation.maximum_bond_dimension
                  << "  max-discarded=" << observation.maximum_discarded_weight << "  elapsed=" << elapsed
                  << " seconds\n";
        ++growth_sweeps;
        direction = opposite_direction(direction);
        if (observation.maximum_bond_dimension >= run.maximum_states)
        {
          reached_requested_dimension = true;
          break;
        }
      }
      if (!reached_requested_dimension)
        throw std::runtime_error("steady-state benchmark did not reach --max-states within the growth sweep limit");

      double steady_elapsed = 0.0;
      for (std::size_t sweep = 0; sweep < run.steady_sweeps; ++sweep)
      {
        auto const start = std::chrono::steady_clock::now();
        auto steps = [&] {
          if constexpr (uni20::performance::measurement_level_v<Measurements> ==
                        uni20::performance::MeasurementLevel::none)
            return uni20::tensor_network::sweep_two_site_dmrg(mps, mpo, environments, direction, options.bond_options,
                                                              center_storage{});
          else
            return uni20::tensor_network::sweep_two_site_dmrg(mps, mpo, environments, direction, options.bond_options,
                                                              measurements, center_storage{});
        }();
        synchronize_execution<MpsStorage>();
        double const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        steady_elapsed += elapsed;
        auto const observation = observe_sweep(steps);
        final_energy = observation.terminal_energy;
        std::cout << "steady-sweep " << sweep << "  " << direction_name(direction)
                  << "  energy=" << observation.terminal_energy << "  max-m=" << observation.maximum_bond_dimension
                  << "  max-discarded=" << observation.maximum_discarded_weight << "  elapsed=" << elapsed
                  << " seconds\n";
        direction = opposite_direction(direction);
      }

      std::cout << "growth-sweeps=" << growth_sweeps << "  growth-elapsed=" << growth_elapsed << " seconds\n";
      std::cout << "steady-sweeps=" << run.steady_sweeps << "  steady-elapsed=" << steady_elapsed
                << " seconds  steady-mean=" << steady_elapsed / static_cast<double>(run.steady_sweeps) << " seconds\n";
      std::cout << "elapsed=" << growth_elapsed + steady_elapsed << " seconds\n";
      if constexpr (uni20::performance::measurement_level_v<Measurements> != uni20::performance::MeasurementLevel::none)
        print_performance_measurements(measurements);

      auto const reference = reference_energy(run.sites);
      if (reference)
      {
        double const reference_value = static_cast<double>(uni20::parse_real<real_type>(reference->value));
        std::cout << "reference=" << reference->value << "  difference=" << final_energy - reference_value
                  << "  source=" << reference->source << '\n';
      }
      return 0;
    }

    auto const start = std::chrono::steady_clock::now();
    auto result = [&] {
      if constexpr (uni20::performance::measurement_level_v<Measurements> == uni20::performance::MeasurementLevel::none)
        return uni20::tensor_network::run_two_site_dmrg(mps, mpo, environments, options, center_storage{});
      else
        return uni20::tensor_network::run_two_site_dmrg(mps, mpo, environments, options, measurements,
                                                        center_storage{});
    }();
    synchronize_execution<MpsStorage>();
    auto const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    for (auto const& sweep : result.sweeps)
    {
      std::cout << "sweep " << sweep.sweep_index << "  " << direction_name(sweep.direction)
                << "  energy=" << uni20::format_real(sweep.terminal_local_energy)
                << "  max-m=" << sweep.maximum_bond_dimension
                << "  max-discarded=" << uni20::format_real(sweep.maximum_discarded_weight) << '\n';
    }
    std::cout << "converged=" << std::boolalpha << result.converged << "  elapsed=" << elapsed << " seconds\n";
    if constexpr (uni20::performance::measurement_level_v<Measurements> != uni20::performance::MeasurementLevel::none)
      print_performance_measurements(measurements);

    auto const reference = reference_energy(run.sites);
    if (reference)
    {
      real_type const reference_value = uni20::parse_real<real_type>(reference->value);
      real_type const difference = result.sweeps.back().terminal_local_energy - reference_value;
      std::cout << "reference=" << uni20::format_real(reference_value)
                << "  difference=" << uni20::format_real(difference) << "  source=" << reference->source << '\n';
    }

    if (run.check)
    {
      if (!reference) throw std::runtime_error("no reference energy is registered for the requested chain length");
      real_type const reference_value = uni20::parse_real<real_type>(reference->value);
      std::string_view tolerance = reference->tolerance;
      if constexpr (std::same_as<real_type, uni20::float32>) tolerance = reference->float32_tolerance;
#if UNI20_HAS_FLOAT128
      if constexpr (std::same_as<real_type, uni20::float128>) tolerance = reference->float128_tolerance;
#endif
      if (!result.converged || !result.sweeps.back().terminal_energy_is_global ||
          abs(result.sweeps.back().terminal_local_energy - reference_value) > uni20::parse_real<real_type>(tolerance))
        throw std::runtime_error("Heisenberg DMRG did not reach the registered reference energy");
    }
    return 0;
  };

  switch (run.measurements)
  {
    case MeasurementMode::off:
    {
      uni20::performance::NoMeasurements measurements;
      return execute(measurements);
    }
    case MeasurementMode::coarse:
    {
      uni20::tensor_network::TwoSiteDmrgPerformanceMeasurements measurements;
      return execute(measurements);
    }
    case MeasurementMode::detailed:
    {
      uni20::tensor_network::DetailedTwoSiteDmrgPerformanceMeasurements measurements;
      return execute(measurements);
    }
  }
  std::unreachable();
}

int main(int argc, char** argv)
{
  Options const options = parse_options(argc, argv);
  if (options.execution == ExecutionMode::cuda)
  {
#if UNI20_BACKEND_CUDA && UNI20_BACKEND_CUBLAS && UNI20_BACKEND_CUSOLVER
    auto cuda_lifetime = uni20::cuda::initialize({.device_ordinals = {options.cuda_device},
                                                  .default_device = options.cuda_device,
                                                  .streams_per_device = options.cuda_streams});
    auto& resources = cuda_lifetime.device_resources(options.cuda_device);
    if (options.cuda_cublas_handles)
      static_cast<void>(uni20::cublas::execution_pool(resources, *options.cuda_cublas_handles));
    if (options.cuda_cusolver_handles)
      static_cast<void>(uni20::cusolver::execution_pool(resources, *options.cuda_cusolver_handles));
    uni20::cuda::ScopedDevice selected_device(options.cuda_device);
    return run_example<double, uni20::ParallelPackedSparseBlockStorage<uni20::CudaStorage>>(options);
#else
    throw std::runtime_error("--execution=cuda requires CUDA, cuBLAS, and cuSOLVER backends");
#endif
  }

  return uni20::visit_scalar_precision(options.precision, [&]<uni20::Real Real>() {
    auto run = [&]<class MpsStorage>() {
      if (options.complex_scalar) return run_example<uni20::complex<Real>, MpsStorage>(options);
      return run_example<Real, MpsStorage>(options);
    };

    switch (options.mps_storage)
    {
      case MpsStorageMode::packed:
        return run.template operator()<uni20::PackedSparseBlockStorage<>>();
      case MpsStorageMode::parallel_packed:
        return run.template operator()<uni20::ParallelPackedSparseBlockStorage<>>();
      case MpsStorageMode::parallel_aligned_packed:
        return run.template operator()<uni20::ParallelPackedSparseBlockStorage<uni20::HostStorage, 64>>();
      case MpsStorageMode::parallel_separate:
        return run.template operator()<uni20::ParallelSeparateSparseBlockStorage<>>();
    }
    std::unreachable();
  });
}
