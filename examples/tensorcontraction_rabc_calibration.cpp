#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <cuda_runtime_api.h>
#include <fmt/core.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace utc = uni20::tensorcontraction;

namespace
{

using Block = utc::MatrixFamily::Block;
using Term = utc::EffectiveHamiltonianOperator::Term;

struct CalibrationCase
{
    std::string family;
    std::string name;
    std::vector<Block> a_blocks;
    std::vector<Block> c_blocks;
    std::vector<Block> center_blocks;
    std::vector<Term> terms;
    std::vector<int> layout;
    int inner_iterations = 1;
    int min_devices = 1;
};

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

auto optional_env_string(char const* name) -> std::string
{
  if (auto const* raw = std::getenv(name); raw != nullptr)
  {
    return raw;
  }
  return {};
}

auto env_int(char const* name, int fallback) -> int
{
  auto text = optional_env_string(name);
  if (text.empty())
  {
    return fallback;
  }
  std::size_t consumed = 0;
  auto const value = std::stoi(text, &consumed);
  if (consumed != text.size())
  {
    throw std::invalid_argument(std::string("invalid integer value for ") + name + ": " + text);
  }
  return value;
}

void check_cuda(cudaError_t error, char const* expression)
{
  if (error != cudaSuccess)
  {
    throw std::runtime_error(fmt::format("{} failed: {}", expression, cudaGetErrorString(error)));
  }
}

#define CHECK_CUDA(EXPR) check_cuda((EXPR), #EXPR)

auto visible_cuda_devices() -> int
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess)
  {
    (void)cudaGetLastError();
    return 0;
  }
  return device_count;
}

auto active_cuda_devices(int visible_count) -> int
{
  if (visible_count <= 0)
  {
    return 0;
  }

  auto const configured = optional_env_string("UNI20_TENSORCONTRACTION_DEVICES");
  if (configured.empty())
  {
    return visible_count;
  }
  if (configured == "all" || configured == "ALL")
  {
    return visible_count;
  }

  try
  {
    return std::clamp(std::stoi(configured), 1, visible_count);
  }
  catch (...)
  {
    return 1;
  }
}

auto case_devices(std::span<int const> layout) -> std::vector<int>
{
  std::vector<int> devices(layout.begin(), layout.end());
  std::ranges::sort(devices);
  auto const last = std::ranges::unique(devices).begin();
  devices.erase(last, devices.end());
  return devices;
}

void synchronize_devices(std::span<int const> devices)
{
  for (int device : devices)
  {
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

auto layout_string(std::span<int const> layout) -> std::string
{
  std::string result;
  for (std::size_t i = 0; i < layout.size(); ++i)
  {
    if (i != 0)
    {
      result += ',';
    }
    result += std::to_string(layout[i]);
  }
  return result;
}

auto all_on_device(std::size_t block_count, int device) -> std::vector<int>
{
  return std::vector<int>(block_count, device);
}

auto split_layout(std::size_t half_blocks) -> std::vector<int>
{
  std::vector<int> layout(2 * half_blocks, 0);
  std::fill(layout.begin() + static_cast<std::ptrdiff_t>(half_blocks), layout.end(), 1);
  return layout;
}

void fill_family(utc::MatrixFamily& family, double scale)
{
  auto values = family.coalesced_values();
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    values[i] = scale * static_cast<double>((i % 97) + 1);
  }
}

auto make_family(std::span<Block const> blocks, double scale) -> utc::MatrixFamily
{
  utc::MatrixFamily family(blocks);
  fill_family(family, scale);
  return family;
}

auto square_blocks(std::size_t count, std::size_t dimension) -> std::vector<Block>
{
  return std::vector<Block>(count, Block{.rows = dimension, .cols = dimension});
}

auto local_square_case(std::string family, std::string name, std::size_t dimension, std::size_t block_count,
                       int inner_iterations) -> CalibrationCase
{
  std::vector<Term> terms;
  terms.reserve(block_count);
  for (std::size_t block = 0; block < block_count; ++block)
  {
    terms.push_back(Term{.r = block, .a = block, .b = block, .c = block, .coefficient = 1.0});
  }
  return CalibrationCase{.family = std::move(family),
                         .name = std::move(name),
                         .a_blocks = square_blocks(block_count, dimension),
                         .c_blocks = square_blocks(block_count, dimension),
                         .center_blocks = square_blocks(block_count, dimension),
                         .terms = std::move(terms),
                         .layout = all_on_device(block_count, 0),
                         .inner_iterations = inner_iterations};
}

auto peer_pair_case(std::string family, std::string name, std::size_t dimension, std::size_t pair_count,
                    int inner_iterations) -> CalibrationCase
{
  auto const block_count = 2 * pair_count;
  std::vector<Term> terms;
  terms.reserve(block_count);
  for (std::size_t pair = 0; pair < pair_count; ++pair)
  {
    terms.push_back(Term{.r = pair + pair_count, .a = 0, .b = pair, .c = 0, .coefficient = 1.0});
    terms.push_back(Term{.r = pair, .a = 0, .b = pair + pair_count, .c = 0, .coefficient = 1.0});
  }
  return CalibrationCase{.family = std::move(family),
                         .name = std::move(name),
                         .a_blocks = square_blocks(1, dimension),
                         .c_blocks = square_blocks(1, dimension),
                         .center_blocks = square_blocks(block_count, dimension),
                         .terms = std::move(terms),
                         .layout = split_layout(pair_count),
                         .inner_iterations = inner_iterations,
                         .min_devices = 2};
}

auto source_accumulation_case(std::string name, std::size_t dimension, std::size_t fan_in,
                              int inner_iterations) -> CalibrationCase
{
  std::vector<Term> terms;
  terms.reserve(fan_in * fan_in);
  for (std::size_t r = 0; r < fan_in; ++r)
  {
    for (std::size_t b = 0; b < fan_in; ++b)
    {
      terms.push_back(Term{.r = r, .a = 0, .b = b, .c = 0, .coefficient = 1.0});
    }
  }
  return CalibrationCase{.family = "source_accumulation",
                         .name = std::move(name),
                         .a_blocks = square_blocks(1, dimension),
                         .c_blocks = square_blocks(1, dimension),
                         .center_blocks = square_blocks(fan_in, dimension),
                         .terms = std::move(terms),
                         .layout = all_on_device(fan_in, 0),
                         .inner_iterations = inner_iterations};
}

auto output_accumulation_case(std::string name, std::size_t dimension, std::size_t group_count,
                              int inner_iterations) -> CalibrationCase
{
  auto const block_count = 2 * group_count;
  std::vector<Term> terms;
  terms.reserve(2 * block_count);
  for (std::size_t r = 0; r < block_count; ++r)
  {
    auto const local_b = r;
    auto const remote_b = r < group_count ? r + group_count : r - group_count;
    terms.push_back(Term{.r = r, .a = 0, .b = local_b, .c = 0, .coefficient = 1.0});
    terms.push_back(Term{.r = r, .a = 0, .b = remote_b, .c = 0, .coefficient = -1.0});
  }
  return CalibrationCase{.family = "output_accumulation",
                         .name = std::move(name),
                         .a_blocks = square_blocks(1, dimension),
                         .c_blocks = square_blocks(1, dimension),
                         .center_blocks = square_blocks(block_count, dimension),
                         .terms = std::move(terms),
                         .layout = split_layout(group_count),
                         .inner_iterations = inner_iterations,
                         .min_devices = 2};
}

auto calibration_cases() -> std::vector<CalibrationCase>
{
  std::vector<CalibrationCase> cases;
  cases.push_back(local_square_case("gemm_throughput", "square64", 64, 1, 16));
  cases.push_back(local_square_case("gemm_throughput", "square128", 128, 1, 8));
  cases.push_back(local_square_case("gemm_throughput", "square256", 256, 1, 4));

  cases.push_back(local_square_case("gemm_launch", "tiny8_blocks1", 8, 1, 32));
  cases.push_back(local_square_case("gemm_launch", "tiny8_blocks8", 8, 8, 24));
  cases.push_back(local_square_case("gemm_launch", "tiny8_blocks32", 8, 32, 16));
  cases.push_back(local_square_case("gemm_launch", "tiny8_blocks128", 8, 128, 8));

  cases.push_back(peer_pair_case("peer_latency", "tiny8_pairs1", 8, 1, 32));
  cases.push_back(peer_pair_case("peer_latency", "tiny8_pairs8", 8, 8, 24));
  cases.push_back(peer_pair_case("peer_latency", "tiny8_pairs32", 8, 32, 16));
  cases.push_back(peer_pair_case("peer_latency", "tiny8_pairs128", 8, 128, 8));

  cases.push_back(peer_pair_case("peer_bandwidth", "square64_pairs1", 64, 1, 16));
  cases.push_back(peer_pair_case("peer_bandwidth", "square128_pairs1", 128, 1, 8));
  cases.push_back(peer_pair_case("peer_bandwidth", "square256_pairs1", 256, 1, 4));

  cases.push_back(source_accumulation_case("tiny8_fanin2", 8, 2, 32));
  cases.push_back(source_accumulation_case("tiny8_fanin4", 8, 4, 24));
  cases.push_back(source_accumulation_case("tiny8_fanin8", 8, 8, 16));
  cases.push_back(source_accumulation_case("tiny8_fanin16", 8, 16, 8));
  cases.push_back(source_accumulation_case("square64_fanin2", 64, 2, 16));
  cases.push_back(source_accumulation_case("square64_fanin4", 64, 4, 8));
  cases.push_back(source_accumulation_case("square64_fanin8", 64, 8, 4));
  cases.push_back(source_accumulation_case("square128_fanin2", 128, 2, 8));
  cases.push_back(source_accumulation_case("square128_fanin4", 128, 4, 4));

  cases.push_back(output_accumulation_case("tiny8_groups1", 8, 1, 32));
  cases.push_back(output_accumulation_case("tiny8_groups8", 8, 8, 24));
  cases.push_back(output_accumulation_case("tiny8_groups32", 8, 32, 16));
  cases.push_back(output_accumulation_case("tiny8_groups128", 8, 128, 8));
  cases.push_back(output_accumulation_case("square64_groups1", 64, 1, 16));
  cases.push_back(output_accumulation_case("square64_groups8", 64, 8, 8));
  cases.push_back(output_accumulation_case("square64_groups32", 64, 32, 4));
  cases.push_back(output_accumulation_case("square128_groups1", 128, 1, 8));
  cases.push_back(output_accumulation_case("square128_groups8", 128, 8, 4));
  return cases;
}

auto selected_families() -> std::vector<std::string>
{
  auto text = optional_env_string("UNI20_RABC_CALIBRATION_FAMILIES");
  if (text.empty() || text == "all")
  {
    return {};
  }

  std::vector<std::string> families;
  std::size_t begin = 0;
  while (begin <= text.size())
  {
    auto const comma = text.find(',', begin);
    auto const end = comma == std::string::npos ? text.size() : comma;
    if (begin != end)
    {
      families.emplace_back(text.substr(begin, end - begin));
    }
    if (comma == std::string::npos)
    {
      break;
    }
    begin = comma + 1;
  }
  return families;
}

auto family_selected(std::string_view family, std::span<std::string const> selected) -> bool
{
  return selected.empty() || std::find(selected.begin(), selected.end(), family) != selected.end();
}

void set_case_layout(CalibrationCase const& item)
{
  setenv("UNI20_TENSORCONTRACTION_RABC_PLACEMENT", "manual", 1);
  auto layout = layout_string(item.layout);
  setenv("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT", layout.c_str(), 1);
  unsetenv("UNI20_TENSORCONTRACTION_RABC_B_LAYOUT");
  unsetenv("UNI20_TENSORCONTRACTION_RABC_R_LAYOUT");
}

auto run_case(CalibrationCase const& item, int repeats, int warmup) -> std::vector<double>
{
  set_case_layout(item);
  auto const devices = case_devices(item.layout);

  auto a = make_family(item.a_blocks, 1.0e-3);
  auto c = make_family(item.c_blocks, 2.0e-3);
  auto input = make_family(item.center_blocks, 3.0e-3);
  utc::MatrixFamily output(item.center_blocks);
  output.fill(0.0);

  auto op = utc::EffectiveHamiltonianOperator::variable_middle(std::move(a), std::move(c), input.blocks(),
                                                               item.center_blocks, item.terms);
  utc::VectorAlgebraEngine algebra;
  if (algebra.uses_host_backend())
  {
    throw std::runtime_error("R/A/B/C calibration requires the resident CUDA TensorContraction backend");
  }

  algebra.upload(input);
  algebra.set_host_synchronization(false);

  for (int pass = 0; pass < warmup; ++pass)
  {
    for (int inner = 0; inner < item.inner_iterations; ++inner)
    {
      op.apply_resident(input, output, algebra);
      synchronize_devices(devices);
    }
  }

  std::vector<double> timings;
  timings.reserve(static_cast<std::size_t>(repeats));
  for (int repeat = 0; repeat < repeats; ++repeat)
  {
    auto const begin = std::chrono::steady_clock::now();
    for (int inner = 0; inner < item.inner_iterations; ++inner)
    {
      op.apply_resident(input, output, algebra);
      synchronize_devices(devices);
    }
    auto const end = std::chrono::steady_clock::now();
    timings.push_back(std::chrono::duration<double>(end - begin).count() / item.inner_iterations);
  }
  return timings;
}

auto mean(std::span<double const> values) -> double
{
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

auto min_value(std::span<double const> values) -> double { return *std::min_element(values.begin(), values.end()); }

auto max_value(std::span<double const> values) -> double { return *std::max_element(values.begin(), values.end()); }

auto value_count(std::span<Block const> blocks) -> std::size_t
{
  std::size_t total = 0;
  for (auto const& block : blocks)
  {
    total += block.rows * block.cols;
  }
  return total;
}

void print_result(CalibrationCase const& item, std::span<double const> timings, int active_devices)
{
  auto const devices = case_devices(item.layout);
  fmt::print("{{\"kind\":\"rabc_calibration\",\"family\":\"{}\",\"name\":\"{}\",\"active_devices\":{},"
             "\"used_devices\":{},\"block_count\":{},\"a_blocks\":{},\"c_blocks\":{},\"term_count\":{},"
             "\"inner_iterations\":{},\"center_values\":{},\"center_bytes\":{},\"a_values\":{},\"a_bytes\":{},"
             "\"c_values\":{},\"c_bytes\":{},\"mean_apply_s\":{:.17g},\"min_apply_s\":{:.17g},"
             "\"max_apply_s\":{:.17g},\"layout\":\"{}\"}}\n",
             item.family, item.name, active_devices, devices.size(), item.center_blocks.size(), item.a_blocks.size(),
             item.c_blocks.size(), item.terms.size(), item.inner_iterations, value_count(item.center_blocks),
             value_count(item.center_blocks) * sizeof(double), value_count(item.a_blocks),
             value_count(item.a_blocks) * sizeof(double), value_count(item.c_blocks),
             value_count(item.c_blocks) * sizeof(double), mean(timings), min_value(timings), max_value(timings),
             layout_string(item.layout));
}

} // namespace

auto main() -> int
{
  try
  {
    ensure_mpi_initialized();
    auto const visible_devices = visible_cuda_devices();
    if (visible_devices <= 0)
    {
      throw std::runtime_error("R/A/B/C calibration requires at least one visible CUDA device");
    }
    if (optional_env_string("UNI20_TENSORCONTRACTION_DEVICES").empty())
    {
      setenv("UNI20_TENSORCONTRACTION_DEVICES", "all", 1);
    }
    auto const active_devices = active_cuda_devices(visible_devices);
    auto const repeats = env_int("UNI20_RABC_CALIBRATION_REPEATS", 3);
    auto const warmup = env_int("UNI20_RABC_CALIBRATION_WARMUP", 1);
    auto const selected = selected_families();

    for (auto const& item : calibration_cases())
    {
      if (!family_selected(item.family, selected))
      {
        continue;
      }
      if (active_devices < item.min_devices)
      {
        fmt::print("{{\"kind\":\"rabc_calibration_skip\",\"family\":\"{}\",\"name\":\"{}\","
                   "\"reason\":\"requires_more_devices\",\"active_devices\":{},\"min_devices\":{}}}\n",
                   item.family, item.name, active_devices, item.min_devices);
        continue;
      }
      auto timings = run_case(item, repeats, warmup);
      print_result(item, timings, active_devices);
    }
  }
  catch (std::exception const& ex)
  {
    fmt::print(stderr, "R/A/B/C calibration failed: {}\n", ex.what());
    return 1;
  }
  return 0;
}
