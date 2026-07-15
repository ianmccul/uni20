/**
 * \file async_ad_stress_example.cpp
 * \brief Exercises reverse-mode pruning and failure propagation on a configurable TBB graph.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/async/var.hpp>
#include <uni20/async/var_toys.hpp>
#include <uni20/common/display.hpp>
#include <uni20/common/presentation.hpp>
#include <uni20/core/numeric_limits.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fmt/core.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace
{

using uni20::async::AsyncTask;
using uni20::async::ReadBuffer;
using uni20::async::ScopedScheduler;
using uni20::async::TbbScheduler;
using uni20::async::Var;
using uni20::async::WriteBuffer;

struct Options
{
    std::size_t threads = 4;
    std::size_t iterations = 10;
    std::size_t depth = 16;
    std::size_t dead_branches = 32;
    std::size_t failing_branches = 8;
    std::uint64_t seed = 20;
    bool show_help = false;
};

[[nodiscard]] std::size_t parse_size(std::string_view value, std::string_view option, bool allow_zero)
{
  std::size_t result = 0;
  auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() || (!allow_zero && result == 0))
  {
    throw std::invalid_argument(std::string(option) +
                                (allow_zero ? " requires a nonnegative integer" : " requires a positive integer"));
  }
  return result;
}

[[nodiscard]] std::uint64_t parse_seed(std::string_view value)
{
  std::uint64_t result = 0;
  auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size())
    throw std::invalid_argument("--seed requires a nonnegative integer");
  return result;
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index)
  {
    std::string_view const argument(argv[index]);
    if (argument == "--help" || argument == "-h")
      options.show_help = true;
    else if (argument.starts_with("--threads="))
      options.threads = parse_size(argument.substr(std::string_view("--threads=").size()), "--threads", false);
    else if (argument.starts_with("--iterations="))
      options.iterations = parse_size(argument.substr(std::string_view("--iterations=").size()), "--iterations", false);
    else if (argument.starts_with("--depth="))
      options.depth = parse_size(argument.substr(std::string_view("--depth=").size()), "--depth", false);
    else if (argument.starts_with("--dead-branches="))
      options.dead_branches =
          parse_size(argument.substr(std::string_view("--dead-branches=").size()), "--dead-branches", true);
    else if (argument.starts_with("--failing-branches="))
      options.failing_branches =
          parse_size(argument.substr(std::string_view("--failing-branches=").size()), "--failing-branches", true);
    else if (argument.starts_with("--seed="))
      options.seed = parse_seed(argument.substr(std::string_view("--seed=").size()));
    else
      throw std::invalid_argument("unknown option '" + std::string(argument) + "'");
  }

  if (options.failing_branches > options.dead_branches)
    throw std::invalid_argument("--failing-branches cannot exceed --dead-branches");
  if (options.threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("--threads exceeds TBB's concurrency range");
  return options;
}

void print_usage(char const* program)
{
  fmt::print("Usage: {} [--threads=N] [--iterations=N] [--depth=N] [--dead-branches=N] "
             "[--failing-branches=N] [--seed=N]\n",
             program);
  fmt::print("  --threads           TBB arena concurrency; default 4\n");
  fmt::print("  --iterations        independently constructed graphs; default 10\n");
  fmt::print("  --depth             live nonlinear chain depth; default 16\n");
  fmt::print("  --dead-branches     discarded branches per live level; default 32\n");
  fmt::print("  --failing-branches  discarded branches that fail per level; default 8\n");
  fmt::print("  --seed              deterministic topology seed; default 20\n");
}

[[nodiscard]] std::uint64_t mix(std::uint64_t value) noexcept
{
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

[[nodiscard]] double unit_interval(std::uint64_t value) noexcept
{
  constexpr double kScale = 1.0 / 9'007'199'254'740'992.0;
  return static_cast<double>(mix(value) >> 11) * kScale;
}

struct AdStressFailure
{
    std::size_t branch;
};

struct AdStressCounters
{
    std::atomic<std::size_t> successful_forwards{0};
    std::atomic<std::size_t> failed_forwards{0};
    std::atomic<std::size_t> reverse_bodies{0};
};

Var<double> tracked_branch(Var<double> input, bool fail, std::size_t branch, AdStressCounters* counters)
{
  Var<double> result;

  uni20::async::schedule([](ReadBuffer<double> input, WriteBuffer<double> output, bool fail, std::size_t branch,
                            AdStressCounters* counters) static -> AsyncTask {
    auto input_buffer = co_await input.transfer();
    double const value = input_buffer.get();
    input_buffer.release();

    if (fail)
    {
      counters->failed_forwards.fetch_add(1, std::memory_order_relaxed);
      throw AdStressFailure{branch};
    }

    counters->successful_forwards.fetch_add(1, std::memory_order_relaxed);
    co_await output = value;
  }(input.value.read(), result.value.write(), fail, branch, counters));

  uni20::async::schedule([](ReadBuffer<double> input_gradient, WriteBuffer<double> output_gradient,
                            AdStressCounters* counters) static -> AsyncTask {
    auto gradient = co_await input_gradient.transfer().or_cancel();
    double const value = gradient.get();
    gradient.release();
    counters->reverse_bodies.fetch_add(1, std::memory_order_relaxed);
    co_await output_gradient += value;
  }(result.grad.input(), input.grad.output(), counters));

  return result;
}

Var<double> make_dead_expression(Var<double>& input, std::uint64_t key)
{
  double const scale = 0.2 + 0.1 * unit_interval(key + 1);
  double const bias = 0.06 * (unit_interval(key + 2) - 0.5);
  switch (mix(key) % 6)
  {
    case 0:
      return sin(input);
    case 1:
      return cos(input);
    case 2:
      return scale * input;
    case 3:
      return input * scale;
    case 4:
      return input + bias;
    default:
      return input * input;
  }
}

struct StressResult
{
    double elapsed = 0.0;
    double maximum_value_error = 0.0;
    double maximum_gradient_error = 0.0;
    std::size_t successful_forwards = 0;
    std::size_t failed_forwards = 0;
    std::size_t reverse_bodies = 0;
    bool observed_failure_preserved = false;
};

StressResult run_stress(Options const& options)
{
  auto const start = std::chrono::steady_clock::now();
  TbbScheduler scheduler(static_cast<int>(options.threads));
  ScopedScheduler scoped_scheduler(&scheduler);
  AdStressCounters counters;
  StressResult result;

  for (std::size_t iteration = 0; iteration < options.iterations; ++iteration)
  {
    scheduler.pause();
    {
      std::uint64_t const iteration_key = mix(options.seed + iteration);
      double const initial_value = 0.1 + 0.1 * unit_interval(iteration_key);
      double expected_value = initial_value;
      double expected_gradient = 1.0;
      std::vector<Var<double>> chain;
      chain.reserve(options.depth + 1);
      chain.emplace_back(initial_value);

      for (std::size_t level = 0; level < options.depth; ++level)
      {
        std::uint64_t const level_key = mix(iteration_key + level);
        double const scale = 0.65 + 0.15 * unit_interval(level_key + 1);
        double const bias = 0.16 * (unit_interval(level_key + 2) - 0.5);
        double const argument = scale * expected_value + bias;
        expected_value = std::sin(argument);
        expected_gradient *= scale * std::cos(argument);
        chain.emplace_back(sin(scale * chain.back() + bias));

        std::size_t const failure_phase =
            options.dead_branches == 0 ? 0 : static_cast<std::size_t>(mix(level_key + 3) % options.dead_branches);
        for (std::size_t branch = 0; branch < options.dead_branches; ++branch)
        {
          bool const fail = ((branch + failure_phase) % options.dead_branches) < options.failing_branches;
          std::size_t const branch_id = (iteration * options.depth + level) * options.dead_branches + branch;
          std::uint64_t const branch_key = mix(level_key + branch);
          auto unused = tracked_branch(make_dead_expression(chain.back(), branch_key), fail, branch_id, &counters);
          static_cast<void>(unused);
        }
      }

      chain.back().grad = 1.0;
      for (std::size_t level = 1; level + 1 < chain.size(); ++level)
        chain[level].grad.finalize();
      auto value = chain.back().value.read();
      auto gradient = chain.front().grad.backprop().read();
      scheduler.run_all();

      result.maximum_value_error =
          std::max(result.maximum_value_error, std::abs(value.get_wait(scheduler) - expected_value));
      result.maximum_gradient_error =
          std::max(result.maximum_gradient_error, std::abs(gradient.get_wait(scheduler) - expected_gradient));
    }
    scheduler.run_all();
  }

  AdStressCounters probe_counters;
  constexpr std::size_t kObservedBranch = std::numeric_limits<std::size_t>::max();
  scheduler.pause();
  {
    Var<double> input = 0.25;
    auto observed = tracked_branch(cos(input), true, kObservedBranch, &probe_counters);
    auto observed_value = observed.value.read();
    scheduler.run_all();
    try
    {
      static_cast<void>(observed_value.get_wait(scheduler));
    }
    catch (AdStressFailure const& failure)
    {
      result.observed_failure_preserved = failure.branch == kObservedBranch;
    }
  }
  scheduler.run_all();

  result.successful_forwards = counters.successful_forwards.load(std::memory_order_relaxed);
  result.failed_forwards = counters.failed_forwards.load(std::memory_order_relaxed);
  result.reverse_bodies = counters.reverse_bodies.load(std::memory_order_relaxed);
  result.observed_failure_preserved = result.observed_failure_preserved &&
                                      probe_counters.failed_forwards.load(std::memory_order_relaxed) == 1 &&
                                      probe_counters.reverse_bodies.load(std::memory_order_relaxed) == 0;
  result.elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}

int report_result(Options const& options, StressResult const& result)
{
  std::size_t const expected_failed = options.iterations * options.depth * options.failing_branches;
  std::size_t const expected_successful =
      options.iterations * options.depth * (options.dead_branches - options.failing_branches);
  double const tolerance = 256.0 * uni20::numeric_limits<double>::epsilon() * static_cast<double>(options.depth);
  bool const correct = result.maximum_value_error <= tolerance && result.maximum_gradient_error <= tolerance &&
                       result.successful_forwards == expected_successful && result.failed_forwards == expected_failed &&
                       result.reverse_bodies == 0 && result.observed_failure_preserved;

  uni20::presentation::report_builder report("Async reverse-mode AD stress");
  report
      .status(correct ? uni20::presentation::semantic_glyph::success : uni20::presentation::semantic_glyph::failure,
              correct ? "validation passed" : "validation failed")
      .field("TBB concurrency", options.threads)
      .field("iterations", options.iterations)
      .field("live depth", options.depth)
      .field("dead branches per level", options.dead_branches)
      .field("failing branches per level", options.failing_branches)
      .field("seed", options.seed)
      .field("observed failure preserved", result.observed_failure_preserved ? "yes" : "no");

  report.table("Execution and validation")
      .grid()
      .column("quantity", uni20::presentation::table_alignment::left)
      .column("value", uni20::presentation::table_alignment::decimal)
      .column("expected", uni20::presentation::table_alignment::decimal)
      .row("elapsed", fmt::format("{:.6f} s", result.elapsed), "")
      .row("successful discarded forwards", result.successful_forwards, expected_successful)
      .row("failed discarded forwards", result.failed_forwards, expected_failed)
      .row("discarded reverse bodies entered", result.reverse_bodies, 0)
      .row("maximum value error", fmt::format("{:.17g}", result.maximum_value_error), "")
      .row("maximum gradient error", fmt::format("{:.17g}", result.maximum_gradient_error), "")
      .row("tolerance", fmt::format("{:.17g}", tolerance), "");

  uni20::display::emit(std::move(report), uni20::display::stream::out);
  return correct ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    Options const options = parse_options(argc, argv);
    if (options.show_help)
    {
      print_usage(argv[0]);
      return 0;
    }
    return report_result(options, run_stress(options));
  }
  catch (std::exception const& error)
  {
    uni20::presentation::report_builder report("Async reverse-mode AD stress");
    report.status(uni20::presentation::semantic_glyph::failure, "configuration failed").field("error", error.what());
    uni20::display::emit(std::move(report), uni20::display::stream::err);
    return 2;
  }
}
