#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/core/scalar_io.hpp>
#include <uni20/core/scalar_precision.hpp>
#include <uni20/core/types.hpp>
#include <uni20/models/spin_half_heisenberg.hpp>
#include <uni20/symmetry/block_tensor_storage.hpp>
#include <uni20/tensor_network/environment_cache.hpp>
#include <uni20/tensor_network/two_site_dmrg.hpp>

#include <algorithm>
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

namespace
{

enum class MeasurementMode
{
  off,
  coarse,
  detailed,
};

struct Options
{
    std::size_t sites = 4;
    std::size_t maximum_states = 16;
    std::size_t maximum_sweeps = 8;
    std::string energy_tolerance = "1e-12";
    std::size_t local_matvecs = 4;
    std::size_t block_threads = 1;
    MeasurementMode measurements = MeasurementMode::off;
    uni20::ScalarPrecision precision = uni20::ScalarPrecision::fp64;
    bool complex_scalar = false;
    bool check = false;
};

struct ReferenceEnergy
{
    std::string_view value;
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
  return options;
}

[[nodiscard]] auto reference_energy(std::size_t sites) -> std::optional<ReferenceEnergy>
{
  if (sites == 4)
    return ReferenceEnergy{.value = "-1.6160254037844386467637231707529361834714026269051903140279034897259665084544",
                           .tolerance = "1e-12",
                           .float128_tolerance = "1e-24",
                           .source = "exact analytic value"};
  if (sites == 20)
    return ReferenceEnergy{.value = "-8.682473334398985",
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

} // namespace

template <uni20::RealOrComplex Scalar> int run_example(Options const& run)
{
  using real_type = uni20::make_real_t<Scalar>;
  using std::abs;
  using parallel_storage = uni20::ParallelPackedSparseBlockStorage<>;
  auto const local = uni20::models::make_spin_half_u1_site();
  auto mps = uni20::models::make_neel_product_mps<Scalar>(run.sites, local);
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo<Scalar>(run.sites, local);
  using mps_type = std::remove_cvref_t<decltype(mps)>;
  using mpo_type = std::remove_cvref_t<decltype(mpo)>;
  uni20::tensor_network::MpoEnvironmentCache<mps_type, mpo_type, parallel_storage> environments(mps, mpo, 0, 0);
  uni20::async::TbbScheduler scheduler{static_cast<int>(run.block_threads)};
  uni20::async::ScopedScheduler use_scheduler(&scheduler);

  uni20::tensor_network::TwoSiteDmrgRunOptions<real_type> options;
  options.maximum_sweeps = run.maximum_sweeps;
  options.energy_tolerance = uni20::parse_real<real_type>(run.energy_tolerance);
  options.bond_options.local_solver.matvec_iterations = run.local_matvecs;
  options.bond_options.truncation.maximum_retained_extent = run.maximum_states;
  auto execute = [&]<class Measurements>(Measurements& measurements) {
    auto const start = std::chrono::steady_clock::now();
    auto result = [&] {
      if constexpr (uni20::performance::measurement_level_v<Measurements> == uni20::performance::MeasurementLevel::none)
        return uni20::tensor_network::run_two_site_dmrg(mps, mpo, environments, options, parallel_storage{});
      else
        return uni20::tensor_network::run_two_site_dmrg(mps, mpo, environments, options, measurements,
                                                        parallel_storage{});
    }();
    auto const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "scalar=" << (uni20::Complex<Scalar> ? "complex" : "real") << '\n';
    std::cout << "precision=" << uni20::scalar_precision_name(run.precision) << '\n';
    std::cout << "block-threads=" << run.block_threads << '\n';
    for (auto const& sweep : result.sweeps)
    {
      std::cout << "sweep " << sweep.sweep_index << "  "
                << (sweep.direction == uni20::tensor_network::MpsSweepDirection::left_to_right ? "left-to-right"
                                                                                               : "right-to-left")
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
  return uni20::visit_scalar_precision(options.precision, [&]<uni20::Real Real>() {
    if (options.complex_scalar) return run_example<uni20::complex<Real>>(options);
    return run_example<Real>(options);
  });
}
