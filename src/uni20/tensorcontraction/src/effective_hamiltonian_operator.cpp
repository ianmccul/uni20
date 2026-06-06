#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/rabc_lanczos_fixture.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"
#include "Utils.h"

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction
{

enum class VariableFamily
{
  Middle,
  Right,
};

struct ResidentOutputPlacementRange
{
    int device = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
    double model_seconds = 0.0;
};

struct ResidentOutputPlacementCache
{
    std::string policy;
    int device_count = 0;
    std::size_t output_count = 0;
    std::size_t input_count = 0;
    std::size_t term_count = 0;
    std::string layout_key;
    bool use_default_localization = false;
    bool coalesced_ranges = false;
    bool trace_plan_emitted = false;
    std::vector<ResidentOutputPlacementRange> ranges;
    std::vector<int> devices;
};

struct EffectiveHamiltonianOperator::Impl
{
    MatrixFamily r_mats;
    MatrixFamily a_mats;
    MatrixFamily b_mats;
    MatrixFamily c_mats;
    std::vector<MatrixFamily::Block> input_blocks;
    std::vector<MatrixFamily::Block> output_blocks;
    std::vector<Term> terms;
    VariableFamily variable_family = VariableFamily::Right;
    std::unique_ptr<tensor::Swapper> swapper;
    std::unique_ptr<tensor::Arranger> arranger;
    ResidentOutputPlacementCache output_placement_cache;
    bool is_compiled = false;

    Impl(MatrixFamily a, MatrixFamily b, std::span<MatrixFamily::Block const> input,
         std::span<MatrixFamily::Block const> output, std::span<Term const> input_terms)
        : r_mats(output), a_mats(std::move(a)), b_mats(std::move(b)), c_mats(input),
          input_blocks(input.begin(), input.end()), output_blocks(output.begin(), output.end()),
          terms(input_terms.begin(), input_terms.end()), variable_family(VariableFamily::Right)
    {
      initialize_runtime();
    }

    Impl(VariableFamily variable, MatrixFamily a, MatrixFamily c, std::span<MatrixFamily::Block const> input,
         std::span<MatrixFamily::Block const> output, std::span<Term const> input_terms)
        : r_mats(output), a_mats(std::move(a)), b_mats(input), c_mats(std::move(c)),
          input_blocks(input.begin(), input.end()), output_blocks(output.begin(), output.end()),
          terms(input_terms.begin(), input_terms.end()), variable_family(variable)
    {
      initialize_runtime();
    }

    void initialize_runtime();
    [[nodiscard]] bool host_backend() const { return arranger == nullptr; }
};

namespace
{

std::vector<ResidentOutputPlacementRange> default_byte_balanced_ranges(std::span<tensor::Matrix const> mats,
                                                                       int device_count);

int checked_index(std::size_t value)
{
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    throw std::length_error("TensorContraction term index must fit in int");
  }
  return static_cast<int>(value);
}

void validate_index(std::size_t index, std::size_t size, char family)
{
  if (index >= size)
  {
    throw std::out_of_range(std::string("TensorContraction term references missing ") + family + " block");
  }
}

std::vector<tensor::TermTy> convert_terms(std::span<EffectiveHamiltonianOperator::Term const> terms)
{
  std::vector<tensor::TermTy> converted;
  converted.reserve(terms.size());
  for (auto const& term : terms)
  {
    converted.emplace_back(checked_index(term.r), checked_index(term.a), checked_index(term.b), checked_index(term.c),
                           term.coefficient);
  }
  return converted;
}

void validate_term_shapes(MatrixFamily const& r_mats, MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                          MatrixFamily const& c_mats, std::span<EffectiveHamiltonianOperator::Term const> terms)
{
  for (auto const& term : terms)
  {
    validate_index(term.r, r_mats.size(), 'R');
    validate_index(term.a, a_mats.size(), 'A');
    validate_index(term.b, b_mats.size(), 'B');
    validate_index(term.c, c_mats.size(), 'C');

    auto const r = r_mats.block(term.r);
    auto const a = a_mats.block(term.a);
    auto const b = b_mats.block(term.b);
    auto const c = c_mats.block(term.c);

    if (b.cols != c.rows)
    {
      throw std::invalid_argument("TensorContraction term has incompatible B/C dimensions");
    }
    if (a.cols != b.rows)
    {
      throw std::invalid_argument("TensorContraction term has incompatible A/(B*C) dimensions");
    }
    if (r.rows != a.rows || r.cols != c.cols)
    {
      throw std::invalid_argument("TensorContraction term result dimensions do not match R block");
    }
  }
}

void validate_family_shape(MatrixFamily const& actual, std::span<MatrixFamily::Block const> expected,
                           char const* family_name)
{
  if (actual.blocks().size() != expected.size())
  {
    throw std::invalid_argument(std::string("TensorContraction ") + family_name + " vector has the wrong block count");
  }
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    if (actual.block(i) != expected[i])
    {
      throw std::invalid_argument(std::string("TensorContraction ") + family_name +
                                  " vector has incompatible block shapes");
    }
  }
}

auto clone_family(MatrixFamily const& source) -> MatrixFamily
{
  MatrixFamily clone(source.blocks());
  clone.assign(source);
  return clone;
}

bool use_host_effective_hamiltonian_backend()
{
  auto const* backend = std::getenv("UNI20_TENSORCONTRACTION_BACKEND");
  if (backend == nullptr)
  {
    return false;
  }
  return std::string(backend) == "host" || std::string(backend) == "cpu";
}

bool use_legacy_arranger_rabc_planner()
{
  auto const* planner = std::getenv("UNI20_TENSORCONTRACTION_RABC_PLANNER");
  if (planner == nullptr)
  {
    return false;
  }
  auto const value = std::string(planner);
  return value == "arranger" || value == "legacy";
}

std::string rabc_placement_policy()
{
  auto const* planner = std::getenv("UNI20_TENSORCONTRACTION_RABC_PLACEMENT");
  if (planner == nullptr)
  {
    return {};
  }
  return std::string(planner);
}

bool use_cost_based_rabc_placement(std::string const& policy)
{
  return policy == "cost" || policy == "greedy" || policy == "cost-greedy";
}

bool use_block_cost_based_rabc_placement(std::string const& policy)
{
  return policy == "cost-block" || policy == "block" || policy == "greedy-block";
}

bool use_manual_rabc_placement(std::string const& policy) { return policy == "manual" || policy == "layout"; }

bool use_striped_rabc_placement(std::string const& policy)
{
  return policy == "stripe" || policy == "striped" || policy == "round-robin" || policy == "alternating";
}

bool use_empirical_contiguous_rabc_placement(std::string const& policy)
{
  return policy == "empirical" || policy == "empirical-contiguous" || policy == "bench-contiguous";
}

bool log_cost_based_rabc_placement()
{
  auto const* raw = std::getenv("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LOG");
  return raw != nullptr && (std::string(raw) == "1" || std::string(raw) == "true" || std::string(raw) == "on");
}

double env_double_or(char const* name, double fallback)
{
  auto const* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0')
  {
    return fallback;
  }
  std::string const text(raw);
  std::size_t consumed = 0;
  double const value = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value) || !(value > 0.0))
  {
    throw std::invalid_argument(std::string("invalid positive floating-point value for ") + name + ": " + text);
  }
  return value;
}

std::optional<std::string> optional_env_string(char const* name)
{
  auto const* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0')
  {
    return std::nullopt;
  }
  return std::string(raw);
}

std::string trim_ascii(std::string_view text)
{
  auto begin = text.begin();
  auto end = text.end();
  while (begin != end && std::isspace(static_cast<unsigned char>(*begin)) != 0)
  {
    ++begin;
  }
  while (begin != end && std::isspace(static_cast<unsigned char>(*(end - 1))) != 0)
  {
    --end;
  }
  return std::string(begin, end);
}

std::string empirical_coefficients_line_from_file(std::string const& path)
{
  std::ifstream input(path);
  if (!input.good())
  {
    throw std::runtime_error("failed to open UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS_FILE: " + path);
  }

  std::string line;
  while (std::getline(input, line))
  {
    auto trimmed = trim_ascii(line);
    if (trimmed.empty() || trimmed.front() == '#')
    {
      continue;
    }

    constexpr std::string_view runtime_prefix = "runtime_coefficients=";
    if (trimmed.starts_with(runtime_prefix))
    {
      return trim_ascii(std::string_view(trimmed).substr(runtime_prefix.size()));
    }

    constexpr std::string_view env_prefix = "UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS=";
    auto const env_position = trimmed.find(env_prefix);
    if (env_position != std::string::npos)
    {
      return trim_ascii(std::string_view(trimmed).substr(env_position + env_prefix.size()));
    }

    return trimmed;
  }

  throw std::invalid_argument("UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS_FILE has no coefficient line: " +
                              path);
}

std::optional<std::string> empirical_coefficients_text()
{
  if (auto const text = optional_env_string("UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS"); text.has_value())
  {
    return text;
  }
  if (auto const path = optional_env_string("UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS_FILE");
      path.has_value())
  {
    return empirical_coefficients_line_from_file(*path);
  }
  return std::nullopt;
}

std::vector<int> parse_manual_rabc_layout(std::size_t block_count, int device_count)
{
  auto const layout_text = optional_env_string("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT");
  if (!layout_text.has_value())
  {
    throw std::invalid_argument(
        "UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual requires UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT");
  }

  std::vector<int> devices;
  devices.reserve(block_count);
  std::size_t begin = 0;
  while (begin <= layout_text->size())
  {
    std::size_t const end = layout_text->find(',', begin);
    auto const token = layout_text->substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (token.empty())
    {
      throw std::invalid_argument("manual R/A/B/C placement layout contains an empty device token");
    }

    std::size_t consumed = 0;
    int const device = std::stoi(token, &consumed);
    if (consumed != token.size() || device < 0 || device >= device_count)
    {
      throw std::invalid_argument("manual R/A/B/C placement layout contains an invalid CUDA device id: " + token);
    }
    devices.push_back(device);

    if (end == std::string::npos)
    {
      break;
    }
    begin = end + 1;
  }

  if (devices.size() != block_count)
  {
    throw std::invalid_argument("manual R/A/B/C placement layout block count does not match the center vector");
  }
  return devices;
}

struct MatrixUseKey
{
    char family = '\0';
    std::size_t index = 0;

    auto operator<=>(MatrixUseKey const&) const = default;
};

struct BcUseKey
{
    std::size_t b = 0;
    std::size_t c = 0;

    auto operator<=>(BcUseKey const&) const = default;
};

struct FirstStageGroupKey
{
    char side = 'R';
    std::size_t first = 0;
    std::size_t second = 0;

    auto operator<=>(FirstStageGroupKey const&) const = default;
};

struct RabcPlacementModel
{
    double gflops = 1000.0;
    double central_gbps = 32.0;
    bool count_environment_bytes = false;
    double env_gbps = 32.0;
    double contiguous_min_speedup = 1.05;
    double arbitrary_min_speedup = 1.25;
};

double gemm_flops(MatrixFamily::Block lhs, MatrixFamily::Block rhs)
{
  return 2.0 * static_cast<double>(lhs.rows) * static_cast<double>(lhs.cols) * static_cast<double>(rhs.cols);
}

std::optional<int> pre_store_device_for(tensor::Swapper& swapper, tensor::Matrix mat)
{
  auto [device, buffer] = swapper.getPreStoreBufferOrNone(mat);
  if (buffer == nullptr)
  {
    return std::nullopt;
  }
  return device;
}

RabcPlacementModel rabc_placement_model()
{
  double const central_gbps = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_CENTRAL_GBPS", 32.0);
  return RabcPlacementModel{
      .gflops = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_GFLOPS", 1000.0),
      .central_gbps = central_gbps,
      .count_environment_bytes = std::getenv("UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_BYTES") != nullptr,
      .env_gbps = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_GBPS", central_gbps),
      .contiguous_min_speedup = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_CONTIGUOUS_MIN_SPEEDUP", 1.05),
      .arbitrary_min_speedup = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_ARBITRARY_MIN_SPEEDUP", 1.25)};
}

double block_transfer_seconds(MatrixFamily::Block block, double gbps)
{
  return static_cast<double>(block.rows) * static_cast<double>(block.cols) * sizeof(double) / (gbps * 1.0e9);
}

double rabc_range_model_seconds(MatrixFamily const& a_mats, MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                                std::span<std::vector<EffectiveHamiltonianOperator::Term const*> const> terms_by_r,
                                std::span<std::optional<int> const> b_source_device, std::size_t begin, std::size_t end,
                                int device, RabcPlacementModel model)
{
  double seconds = 0.0;
  std::set<BcUseKey> staged_bc;
  std::set<MatrixUseKey> staged_mats;
  for (std::size_t r = begin; r < end; ++r)
  {
    for (auto const* term : terms_by_r[r])
    {
      BcUseKey const bc{.b = term->b, .c = term->c};
      if (staged_bc.insert(bc).second)
      {
        seconds += gemm_flops(b_mats.block(term->b), c_mats.block(term->c)) / (model.gflops * 1.0e9);
      }

      auto const a = a_mats.block(term->a);
      auto const b = b_mats.block(term->b);
      auto const c = c_mats.block(term->c);
      auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
      seconds += gemm_flops(a, intermediate) / (model.gflops * 1.0e9);

      MatrixUseKey const b_key{.family = 'B', .index = term->b};
      if (staged_mats.insert(b_key).second &&
          (!b_source_device[term->b].has_value() || *b_source_device[term->b] != device))
      {
        seconds += block_transfer_seconds(b, model.central_gbps);
      }

      if (model.count_environment_bytes)
      {
        MatrixUseKey const a_key{.family = 'A', .index = term->a};
        MatrixUseKey const c_key{.family = 'C', .index = term->c};
        if (staged_mats.insert(a_key).second)
        {
          seconds += block_transfer_seconds(a, model.env_gbps);
        }
        if (staged_mats.insert(c_key).second)
        {
          seconds += block_transfer_seconds(c, model.env_gbps);
        }
      }
    }
  }
  return seconds;
}

std::vector<ResidentOutputPlacementRange>
cost_based_rabc_output_ranges(MatrixFamily const& a_mats, MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                              std::span<tensor::Matrix const> r_raw_mats, std::span<tensor::Matrix const> b_raw_mats,
                              std::span<EffectiveHamiltonianOperator::Term const> terms, std::size_t output_count,
                              tensor::Swapper& swapper)
{
  int const device_count = swapper.getDeviceCount();
  if (device_count <= 1 || output_count == 0)
  {
    return {};
  }

  int const range_count = std::min(device_count, static_cast<int>(output_count));
  std::vector<std::vector<EffectiveHamiltonianOperator::Term const*>> terms_by_r(output_count);
  for (auto const& term : terms)
  {
    if (term.r < output_count)
    {
      terms_by_r[term.r].push_back(&term);
    }
  }

  std::vector<std::optional<int>> b_source_device(b_mats.size());
  for (std::size_t b = 0; b < b_mats.size(); ++b)
  {
    b_source_device[b] = pre_store_device_for(swapper, b_raw_mats[b]);
  }

  auto const model = rabc_placement_model();
  std::vector<std::vector<std::vector<double>>> range_cost(
      static_cast<std::size_t>(range_count),
      std::vector<std::vector<double>>(output_count + 1, std::vector<double>(output_count + 1, 0.0)));
  for (int device = 0; device < range_count; ++device)
  {
    for (std::size_t begin = 0; begin < output_count; ++begin)
    {
      for (std::size_t end = begin + 1; end <= output_count; ++end)
      {
        range_cost[static_cast<std::size_t>(device)][begin][end] =
            rabc_range_model_seconds(a_mats, b_mats, c_mats, terms_by_r, b_source_device, begin, end, device, model);
      }
    }
  }

  double const infinity = std::numeric_limits<double>::infinity();
  std::vector<std::vector<double>> dp(static_cast<std::size_t>(range_count + 1),
                                      std::vector<double>(output_count + 1, infinity));
  std::vector<std::vector<std::size_t>> split(static_cast<std::size_t>(range_count + 1),
                                              std::vector<std::size_t>(output_count + 1, 0));
  dp[0][0] = 0.0;
  for (int used = 1; used <= range_count; ++used)
  {
    for (std::size_t end = static_cast<std::size_t>(used); end <= output_count; ++end)
    {
      for (std::size_t begin = static_cast<std::size_t>(used - 1); begin < end; ++begin)
      {
        double const previous = dp[static_cast<std::size_t>(used - 1)][begin];
        if (!std::isfinite(previous))
        {
          continue;
        }
        double const candidate = std::max(previous, range_cost[static_cast<std::size_t>(used - 1)][begin][end]);
        if (candidate < dp[static_cast<std::size_t>(used)][end])
        {
          dp[static_cast<std::size_t>(used)][end] = candidate;
          split[static_cast<std::size_t>(used)][end] = begin;
        }
      }
    }
  }

  std::vector<ResidentOutputPlacementRange> ranges(static_cast<std::size_t>(range_count));
  std::size_t end = output_count;
  for (int used = range_count; used >= 1; --used)
  {
    std::size_t const begin = split[static_cast<std::size_t>(used)][end];
    ranges[static_cast<std::size_t>(used - 1)] =
        ResidentOutputPlacementRange{.device = used - 1,
                                     .begin = begin,
                                     .end = end,
                                     .model_seconds = range_cost[static_cast<std::size_t>(used - 1)][begin][end]};
    end = begin;
  }

  auto annotate_ranges = [&range_cost](std::vector<ResidentOutputPlacementRange>& placement_ranges) {
    for (auto& range : placement_ranges)
    {
      range.model_seconds = range_cost[static_cast<std::size_t>(range.device)][range.begin][range.end];
    }
  };
  auto max_model_seconds = [](std::span<ResidentOutputPlacementRange const> placement_ranges) {
    double seconds = 0.0;
    for (auto const range : placement_ranges)
    {
      seconds = std::max(seconds, range.model_seconds);
    }
    return seconds;
  };

  auto default_ranges = default_byte_balanced_ranges(r_raw_mats, device_count);
  annotate_ranges(default_ranges);
  double const selected_seconds = max_model_seconds(ranges);
  double const default_seconds = max_model_seconds(default_ranges);
  bool const fallback_to_default = selected_seconds > 0.0 && default_seconds > 0.0 &&
                                   default_seconds / selected_seconds < model.contiguous_min_speedup;
  if (fallback_to_default)
  {
    ranges = std::move(default_ranges);
  }

  if (log_cost_based_rabc_placement())
  {
    fmt::print(stderr, "[TENSORCONTRACTION][RABC_PLACEMENT] policy=cost-contiguous devices={} outputs={}", device_count,
               output_count);
    for (auto const range : ranges)
    {
      fmt::print(stderr, " device{}={{blocks=[{},{}),model_s={:.6g}}}", range.device, range.begin, range.end,
                 range.model_seconds);
    }
    if (fallback_to_default)
    {
      fmt::print(stderr,
                 " fallback=default-byte-balanced selected_model_s={:.6g} default_model_s={:.6g}"
                 " min_speedup={:.6g}",
                 selected_seconds, default_seconds, model.contiguous_min_speedup);
    }
    fmt::print(stderr, "\n");
  }

  return ranges;
}

std::vector<tensor::Matrix> matrix_subrange(std::span<tensor::Matrix const> mats, std::size_t begin, std::size_t end)
{
  return {mats.begin() + static_cast<std::ptrdiff_t>(begin), mats.begin() + static_cast<std::ptrdiff_t>(end)};
}

std::vector<ResidentOutputPlacementRange> default_byte_balanced_ranges(std::span<tensor::Matrix const> mats,
                                                                       int device_count)
{
  if (mats.empty() || device_count <= 0)
  {
    return {};
  }

  std::size_t total_bytes = 0;
  for (auto const& mat : mats)
  {
    total_bytes += mat.sizeInByte();
  }

  int const range_count = std::max(1, std::min(device_count, static_cast<int>(mats.size())));
  std::vector<ResidentOutputPlacementRange> ranges;
  ranges.reserve(static_cast<std::size_t>(range_count));

  std::size_t begin = 0;
  for (int range_index = 0; range_index < range_count; ++range_index)
  {
    std::size_t end = begin;
    if (range_index == range_count - 1)
    {
      end = mats.size();
    }
    else
    {
      std::size_t const target_bytes =
          (total_bytes * static_cast<std::size_t>(range_index + 1)) / static_cast<std::size_t>(range_count);
      std::size_t prefix_bytes = 0;
      for (std::size_t index = 0; index < begin; ++index)
      {
        prefix_bytes += mats[index].sizeInByte();
      }
      while (end < mats.size() && (end == begin || prefix_bytes + mats[end].sizeInByte() <= target_bytes))
      {
        prefix_bytes += mats[end].sizeInByte();
        ++end;
      }
      std::size_t const remaining_blocks = mats.size() - end;
      std::size_t const remaining_ranges = static_cast<std::size_t>(range_count - range_index - 1);
      if (remaining_blocks < remaining_ranges)
      {
        end -= remaining_ranges - remaining_blocks;
      }
    }
    ranges.push_back(
        ResidentOutputPlacementRange{.device = range_index, .begin = begin, .end = end, .model_seconds = 0.0});
    begin = end;
  }
  return ranges;
}

bool placement_ranges_match_default(std::span<ResidentOutputPlacementRange const> ranges,
                                    std::span<tensor::Matrix const> mats, int device_count)
{
  auto const defaults = default_byte_balanced_ranges(mats, device_count);
  if (ranges.size() != defaults.size())
  {
    return false;
  }
  for (std::size_t index = 0; index < ranges.size(); ++index)
  {
    if (ranges[index].device != defaults[index].device || ranges[index].begin != defaults[index].begin ||
        ranges[index].end != defaults[index].end)
    {
      return false;
    }
  }
  return true;
}

std::vector<int> default_byte_balanced_devices(std::span<tensor::Matrix const> mats, int device_count)
{
  std::vector<int> devices(mats.size(), 0);
  for (auto const range : default_byte_balanced_ranges(mats, device_count))
  {
    for (std::size_t index = range.begin; index < range.end; ++index)
    {
      devices[index] = range.device;
    }
  }
  return devices;
}

std::vector<int> striped_devices(std::size_t block_count, int device_count)
{
  std::vector<int> devices(block_count, 0);
  for (std::size_t block = 0; block < block_count; ++block)
  {
    devices[block] = static_cast<int>(block % static_cast<std::size_t>(device_count));
  }
  return devices;
}

std::vector<double> parse_double_list(std::string_view text, std::string_view name)
{
  std::vector<double> values;
  std::size_t begin = 0;
  while (begin <= text.size())
  {
    std::size_t const end = text.find(',', begin);
    auto const token =
        std::string{text.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin)};
    if (token.empty())
    {
      throw std::invalid_argument(std::string(name) + " contains an empty coefficient token");
    }

    std::size_t consumed = 0;
    double const value = std::stod(token, &consumed);
    if (consumed != token.size() || !std::isfinite(value))
    {
      throw std::invalid_argument(std::string(name) + " contains an invalid finite coefficient: " + token);
    }
    values.push_back(value);

    if (end == std::string_view::npos)
    {
      break;
    }
    begin = end + 1;
  }
  return values;
}

bool placement_devices_match_default(std::span<int const> devices, std::span<tensor::Matrix const> mats,
                                     int device_count)
{
  auto const defaults = default_byte_balanced_devices(mats, device_count);
  return devices.size() == defaults.size() && std::equal(devices.begin(), devices.end(), defaults.begin());
}

bool center_block_layout_supported(MatrixFamily const& b_mats, std::span<tensor::Matrix const> r_raw_mats,
                                   std::span<EffectiveHamiltonianOperator::Term const> terms)
{
  if (b_mats.size() != r_raw_mats.size())
  {
    return false;
  }
  for (std::size_t block = 0; block < b_mats.size(); ++block)
  {
    auto const b = b_mats.block(block);
    if (b.rows != r_raw_mats[block].getFirstDim() || b.cols != r_raw_mats[block].getSecondDim())
    {
      return false;
    }
  }
  for (auto const& term : terms)
  {
    if (term.b >= b_mats.size() || term.r >= r_raw_mats.size())
    {
      return false;
    }
  }
  return true;
}

std::vector<double> rabc_layout_load_seconds(MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                                             MatrixFamily const& c_mats,
                                             std::span<EffectiveHamiltonianOperator::Term const> terms,
                                             std::span<int const> center_devices, int device_count,
                                             RabcPlacementModel model)
{
  std::vector<double> load(static_cast<std::size_t>(device_count), 0.0);
  std::vector<std::set<BcUseKey>> staged_bc(static_cast<std::size_t>(device_count));
  std::vector<std::set<MatrixUseKey>> staged_mats(static_cast<std::size_t>(device_count));
  for (auto const& term : terms)
  {
    int const device = center_devices[term.r];
    auto const device_index = static_cast<std::size_t>(device);
    BcUseKey const bc{.b = term.b, .c = term.c};
    if (staged_bc[device_index].insert(bc).second)
    {
      load[device_index] += gemm_flops(b_mats.block(term.b), c_mats.block(term.c)) / (model.gflops * 1.0e9);
    }

    auto const a = a_mats.block(term.a);
    auto const b = b_mats.block(term.b);
    auto const c = c_mats.block(term.c);
    auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
    load[device_index] += gemm_flops(a, intermediate) / (model.gflops * 1.0e9);

    MatrixUseKey const b_key{.family = 'B', .index = term.b};
    if (staged_mats[device_index].insert(b_key).second && center_devices[term.b] != device)
    {
      load[device_index] += block_transfer_seconds(b, model.central_gbps);
    }

    if (model.count_environment_bytes)
    {
      MatrixUseKey const a_key{.family = 'A', .index = term.a};
      MatrixUseKey const c_key{.family = 'C', .index = term.c};
      if (staged_mats[device_index].insert(a_key).second)
      {
        load[device_index] += block_transfer_seconds(a, model.env_gbps);
      }
      if (staged_mats[device_index].insert(c_key).second)
      {
        load[device_index] += block_transfer_seconds(c, model.env_gbps);
      }
    }
  }
  return load;
}

double max_load(std::span<double const> load)
{
  return load.empty() ? 0.0 : *std::max_element(load.begin(), load.end());
}

struct RabcEmpiricalDeviceFeatures
{
    double right_flops = 0.0;
    double b_peer_bytes = 0.0;
    double terms = 0.0;
    double unique_bc = 0.0;
    double output_bytes = 0.0;
    double b_cut_terms = 0.0;
    double b_peer_blocks = 0.0;
    double right_duplicate_groups = 0.0;
    double mixed_duplicate_groups = 0.0;
    double mixed_left_groups = 0.0;
    double mixed_right_groups = 0.0;
};

struct RabcEmpiricalLayoutFeatures
{
    RabcEmpiricalDeviceFeatures device0;
    RabcEmpiricalDeviceFeatures device1;
    double layout_transitions = 1.0;
    double layout_segments = 2.0;
    double active_devices = 2.0;
    double max_output_block_fraction = 0.0;
    double max_output_byte_fraction = 0.0;
};

struct RabcEmpiricalCoefficients
{
    double intercept = 0.0;
    std::array<RabcEmpiricalDeviceFeatures, 2> devices{};
    double layout_transitions = 0.0;
    double layout_segments = 0.0;
    double active_devices = 0.0;
    double max_output_block_fraction = 0.0;
    double max_output_byte_fraction = 0.0;
    bool graph_features = false;
};

RabcEmpiricalCoefficients empirical_rabc_coefficients()
{
  auto const text = empirical_coefficients_text();
  if (!text.has_value())
  {
    throw std::invalid_argument("UNI20_TENSORCONTRACTION_RABC_PLACEMENT=empirical-contiguous requires "
                                "UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS or "
                                "UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS_FILE");
  }
  auto const values = parse_double_list(*text, "UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS(_FILE)");
  constexpr std::size_t base_device_features = 5;
  constexpr std::size_t graph_device_features = 6;
  constexpr std::size_t layout_features = 5;
  constexpr std::size_t expected_base_size = 1 + 2 * base_device_features + layout_features;
  constexpr std::size_t expected_graph_size = 1 + 2 * (base_device_features + graph_device_features) + layout_features;
  if (values.size() != expected_base_size && values.size() != expected_graph_size)
  {
    throw std::invalid_argument(
        fmt::format("UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS(_FILE) expected {} or {} values, got {}",
                    expected_base_size, expected_graph_size, values.size()));
  }

  RabcEmpiricalCoefficients coefficients;
  coefficients.graph_features = values.size() == expected_graph_size;
  std::size_t index = 0;
  coefficients.intercept = values[index++];
  for (auto& device : coefficients.devices)
  {
    device.right_flops = values[index++];
    device.b_peer_bytes = values[index++];
    device.terms = values[index++];
    device.unique_bc = values[index++];
    device.output_bytes = values[index++];
    if (coefficients.graph_features)
    {
      device.b_cut_terms = values[index++];
      device.b_peer_blocks = values[index++];
      device.right_duplicate_groups = values[index++];
      device.mixed_duplicate_groups = values[index++];
      device.mixed_left_groups = values[index++];
      device.mixed_right_groups = values[index++];
    }
  }
  coefficients.layout_transitions = values[index++];
  coefficients.layout_segments = values[index++];
  coefficients.active_devices = values[index++];
  coefficients.max_output_block_fraction = values[index++];
  coefficients.max_output_byte_fraction = values[index++];
  return coefficients;
}

int empirical_contiguous_device_for(std::size_t block, std::size_t cut) { return block < cut ? 0 : 1; }

struct EmpiricalMixedOrderStats
{
    std::set<std::size_t> right_c;
    std::set<std::size_t> left_a;
    double right_flops = 0.0;
    double left_flops = 0.0;
};

std::map<std::pair<int, std::size_t>, bool>
empirical_left_first_by_device_b(MatrixFamily const& a_mats, MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                                 std::span<EffectiveHamiltonianOperator::Term const> terms, std::size_t cut)
{
  std::map<std::pair<int, std::size_t>, EmpiricalMixedOrderStats> stats;
  for (auto const& term : terms)
  {
    int const device = empirical_contiguous_device_for(term.r, cut);
    auto& row = stats[{device, term.b}];
    auto const a = a_mats.block(term.a);
    auto const b = b_mats.block(term.b);
    auto const c = c_mats.block(term.c);

    if (row.right_c.insert(term.c).second)
    {
      row.right_flops += gemm_flops(b, c);
    }
    auto const right_intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
    row.right_flops += gemm_flops(a, right_intermediate);

    if (row.left_a.insert(term.a).second)
    {
      row.left_flops += gemm_flops(a, b);
    }
    auto const left_intermediate = MatrixFamily::Block{.rows = a.rows, .cols = b.cols};
    row.left_flops += gemm_flops(left_intermediate, c);
  }

  std::map<std::pair<int, std::size_t>, bool> left_first;
  for (auto const& [key, value] : stats)
  {
    left_first[key] = value.left_flops < value.right_flops;
  }
  return left_first;
}

std::array<std::size_t, 2> duplicate_group_counts(std::array<std::set<FirstStageGroupKey>, 2> const& groups)
{
  std::size_t duplicates = 0;
  for (auto const& group : groups[0])
  {
    if (groups[1].contains(group))
    {
      ++duplicates;
    }
  }
  return {duplicates, duplicates};
}

RabcEmpiricalLayoutFeatures empirical_contiguous_features(MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                                                          MatrixFamily const& c_mats,
                                                          std::span<tensor::Matrix const> r_raw_mats,
                                                          std::span<EffectiveHamiltonianOperator::Term const> terms,
                                                          std::size_t cut)
{
  RabcEmpiricalLayoutFeatures features;
  std::array<std::set<BcUseKey>, 2> staged_bc;
  std::array<std::set<std::size_t>, 2> staged_b;
  std::array<std::set<std::size_t>, 2> peer_b;
  std::array<std::set<FirstStageGroupKey>, 2> right_first_groups;
  std::array<std::set<FirstStageGroupKey>, 2> mixed_first_groups;
  std::array<std::size_t, 2> output_bytes{0, 0};
  auto const mixed_left_first = empirical_left_first_by_device_b(a_mats, b_mats, c_mats, terms, cut);

  for (std::size_t r = 0; r < r_raw_mats.size(); ++r)
  {
    int const device = empirical_contiguous_device_for(r, cut);
    output_bytes[static_cast<std::size_t>(device)] += r_raw_mats[r].sizeInByte();
  }

  features.device0.output_bytes = static_cast<double>(output_bytes[0]);
  features.device1.output_bytes = static_cast<double>(output_bytes[1]);
  for (auto const& term : terms)
  {
    int const device = empirical_contiguous_device_for(term.r, cut);
    auto const device_index = static_cast<std::size_t>(device);
    auto& device_features = device == 0 ? features.device0 : features.device1;
    device_features.terms += 1.0;

    BcUseKey const bc{.b = term.b, .c = term.c};
    if (staged_bc[device_index].insert(bc).second)
    {
      device_features.right_flops += gemm_flops(b_mats.block(term.b), c_mats.block(term.c));
    }
    right_first_groups[device_index].insert(FirstStageGroupKey{.side = 'R', .first = term.b, .second = term.c});

    auto const a = a_mats.block(term.a);
    auto const b = b_mats.block(term.b);
    auto const c = c_mats.block(term.c);
    auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
    device_features.right_flops += gemm_flops(a, intermediate);

    bool const peer_b_block = empirical_contiguous_device_for(term.b, cut) != device;
    if (peer_b_block)
    {
      device_features.b_cut_terms += 1.0;
      peer_b[device_index].insert(term.b);
    }
    if (staged_b[device_index].insert(term.b).second && peer_b_block)
    {
      device_features.b_peer_bytes += static_cast<double>(b.rows) * static_cast<double>(b.cols) * sizeof(double);
    }

    if (mixed_left_first.at({device, term.b}))
    {
      mixed_first_groups[device_index].insert(FirstStageGroupKey{.side = 'L', .first = term.a, .second = term.b});
    }
    else
    {
      mixed_first_groups[device_index].insert(FirstStageGroupKey{.side = 'R', .first = term.b, .second = term.c});
    }
  }

  features.device0.unique_bc = static_cast<double>(staged_bc[0].size());
  features.device1.unique_bc = static_cast<double>(staged_bc[1].size());
  features.device0.b_peer_blocks = static_cast<double>(peer_b[0].size());
  features.device1.b_peer_blocks = static_cast<double>(peer_b[1].size());
  for (auto const& [key, left_first] : mixed_left_first)
  {
    auto& device_features = key.first == 0 ? features.device0 : features.device1;
    if (left_first)
    {
      device_features.mixed_left_groups += 1.0;
    }
    else
    {
      device_features.mixed_right_groups += 1.0;
    }
  }
  auto const right_duplicates = duplicate_group_counts(right_first_groups);
  auto const mixed_duplicates = duplicate_group_counts(mixed_first_groups);
  features.device0.right_duplicate_groups = static_cast<double>(right_duplicates[0]);
  features.device1.right_duplicate_groups = static_cast<double>(right_duplicates[1]);
  features.device0.mixed_duplicate_groups = static_cast<double>(mixed_duplicates[0]);
  features.device1.mixed_duplicate_groups = static_cast<double>(mixed_duplicates[1]);
  features.max_output_block_fraction =
      std::max(static_cast<double>(cut), static_cast<double>(r_raw_mats.size() - cut)) /
      static_cast<double>(r_raw_mats.size());
  std::size_t const total_bytes = output_bytes[0] + output_bytes[1];
  features.max_output_byte_fraction =
      total_bytes == 0
          ? 0.0
          : static_cast<double>(std::max(output_bytes[0], output_bytes[1])) / static_cast<double>(total_bytes);
  return features;
}

double empirical_contiguous_score(RabcEmpiricalLayoutFeatures const& features,
                                  RabcEmpiricalCoefficients const& coefficients)
{
  auto score_device = [](RabcEmpiricalDeviceFeatures const& values,
                         RabcEmpiricalDeviceFeatures const& weights) -> double {
    return values.right_flops * weights.right_flops + values.b_peer_bytes * weights.b_peer_bytes +
           values.terms * weights.terms + values.unique_bc * weights.unique_bc +
           values.output_bytes * weights.output_bytes + values.b_cut_terms * weights.b_cut_terms +
           values.b_peer_blocks * weights.b_peer_blocks +
           values.right_duplicate_groups * weights.right_duplicate_groups +
           values.mixed_duplicate_groups * weights.mixed_duplicate_groups +
           values.mixed_left_groups * weights.mixed_left_groups +
           values.mixed_right_groups * weights.mixed_right_groups;
  };

  return coefficients.intercept + score_device(features.device0, coefficients.devices[0]) +
         score_device(features.device1, coefficients.devices[1]) +
         features.layout_transitions * coefficients.layout_transitions +
         features.layout_segments * coefficients.layout_segments +
         features.active_devices * coefficients.active_devices +
         features.max_output_block_fraction * coefficients.max_output_block_fraction +
         features.max_output_byte_fraction * coefficients.max_output_byte_fraction;
}

std::vector<ResidentOutputPlacementRange>
empirical_contiguous_rabc_output_ranges(MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                                        MatrixFamily const& c_mats, std::span<tensor::Matrix const> r_raw_mats,
                                        std::span<EffectiveHamiltonianOperator::Term const> terms,
                                        tensor::Swapper& swapper)
{
  int const device_count = swapper.getDeviceCount();
  if (device_count != 2)
  {
    throw std::invalid_argument("empirical-contiguous R/A/B/C placement currently requires exactly two CUDA devices");
  }
  if (r_raw_mats.size() < 2)
  {
    return {};
  }

  auto const coefficients = empirical_rabc_coefficients();
  std::size_t best_cut = 1;
  double best_score = std::numeric_limits<double>::infinity();
  for (std::size_t cut = 1; cut < r_raw_mats.size(); ++cut)
  {
    double const score = empirical_contiguous_score(
        empirical_contiguous_features(a_mats, b_mats, c_mats, r_raw_mats, terms, cut), coefficients);
    if (score < best_score)
    {
      best_score = score;
      best_cut = cut;
    }
  }

  auto default_ranges = default_byte_balanced_ranges(r_raw_mats, device_count);
  std::size_t const default_cut = default_ranges.empty() ? 0 : default_ranges.front().end;
  double const default_score =
      default_cut == 0
          ? std::numeric_limits<double>::infinity()
          : empirical_contiguous_score(
                empirical_contiguous_features(a_mats, b_mats, c_mats, r_raw_mats, terms, default_cut), coefficients);
  double const min_speedup = env_double_or("UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_MIN_SPEEDUP", 1.0);
  bool const fallback_to_default =
      std::isfinite(default_score) && std::isfinite(best_score) && default_score / best_score < min_speedup;
  if (fallback_to_default)
  {
    for (auto& range : default_ranges)
    {
      range.model_seconds = default_score;
    }
    if (log_cost_based_rabc_placement())
    {
      fmt::print(stderr,
                 "[TENSORCONTRACTION][RABC_PLACEMENT] policy=empirical-contiguous devices=2 outputs={} "
                 "fallback=default-byte-balanced selected_cut={} selected_score={:.6g} default_cut={} "
                 "default_score={:.6g} min_speedup={:.6g}\n",
                 r_raw_mats.size(), best_cut, best_score, default_cut, default_score, min_speedup);
    }
    return default_ranges;
  }

  if (log_cost_based_rabc_placement())
  {
    fmt::print(stderr,
               "[TENSORCONTRACTION][RABC_PLACEMENT] policy=empirical-contiguous devices=2 outputs={} cut={} "
               "score={:.6g} default_cut={} default_score={:.6g}\n",
               r_raw_mats.size(), best_cut, best_score, default_cut, default_score);
  }

  return {ResidentOutputPlacementRange{.device = 0, .begin = 0, .end = best_cut, .model_seconds = best_score},
          ResidentOutputPlacementRange{
              .device = 1, .begin = best_cut, .end = r_raw_mats.size(), .model_seconds = best_score}};
}

struct RabcDeviceCostFeatures
{
    int device = 0;
    std::size_t input_blocks = 0;
    std::size_t output_blocks = 0;
    std::size_t terms = 0;
    std::size_t unique_bc = 0;
    std::size_t unique_a = 0;
    std::size_t unique_b = 0;
    std::size_t unique_c = 0;
    double bc_flops = 0.0;
    double accumulate_flops = 0.0;
    std::size_t b_local_bytes = 0;
    std::size_t b_peer_bytes = 0;
    std::size_t a_bytes = 0;
    std::size_t c_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t intermediate_bytes = 0;
};

struct RabcCostFeatures
{
    std::vector<int> input_devices;
    std::vector<int> output_devices;
    std::vector<RabcDeviceCostFeatures> devices;
};

std::vector<int> resident_devices_for(tensor::Swapper& swapper, std::span<tensor::Matrix const> mats)
{
  std::vector<int> devices;
  devices.reserve(mats.size());
  for (auto const& mat : mats)
  {
    auto device = pre_store_device_for(swapper, mat);
    devices.push_back(device.value_or(0));
  }
  return devices;
}

RabcCostFeatures rabc_cost_features(MatrixFamily const& a_mats, MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                                    std::span<tensor::Matrix const> r_raw_mats,
                                    std::span<tensor::Matrix const> b_raw_mats,
                                    std::span<EffectiveHamiltonianOperator::Term const> terms, tensor::Swapper& swapper)
{
  int const device_count = swapper.getDeviceCount();
  RabcCostFeatures features;
  features.input_devices = resident_devices_for(swapper, b_raw_mats);
  features.output_devices = resident_devices_for(swapper, r_raw_mats);
  features.devices.resize(static_cast<std::size_t>(device_count));
  std::vector<std::set<BcUseKey>> staged_bc(static_cast<std::size_t>(device_count));
  std::vector<std::set<MatrixUseKey>> staged_mats(static_cast<std::size_t>(device_count));

  for (int device = 0; device < device_count; ++device)
  {
    features.devices[static_cast<std::size_t>(device)].device = device;
  }
  for (std::size_t block = 0; block < features.input_devices.size(); ++block)
  {
    auto& device_features = features.devices[static_cast<std::size_t>(features.input_devices[block])];
    ++device_features.input_blocks;
  }
  for (std::size_t block = 0; block < features.output_devices.size(); ++block)
  {
    auto& device_features = features.devices[static_cast<std::size_t>(features.output_devices[block])];
    ++device_features.output_blocks;
    device_features.output_bytes += r_raw_mats[block].sizeInByte();
  }

  for (auto const& term : terms)
  {
    int const device = features.output_devices[term.r];
    auto const device_index = static_cast<std::size_t>(device);
    auto& device_features = features.devices[device_index];
    ++device_features.terms;

    BcUseKey const bc{.b = term.b, .c = term.c};
    if (staged_bc[device_index].insert(bc).second)
    {
      auto const b = b_mats.block(term.b);
      auto const c = c_mats.block(term.c);
      device_features.bc_flops += gemm_flops(b, c);
      device_features.intermediate_bytes += b.rows * c.cols * sizeof(double);
    }

    auto const a = a_mats.block(term.a);
    auto const b = b_mats.block(term.b);
    auto const c = c_mats.block(term.c);
    auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
    device_features.accumulate_flops += gemm_flops(a, intermediate);

    MatrixUseKey const b_key{.family = 'B', .index = term.b};
    if (staged_mats[device_index].insert(b_key).second)
    {
      ++device_features.unique_b;
      if (features.input_devices[term.b] == device)
      {
        device_features.b_local_bytes += b_raw_mats[term.b].sizeInByte();
      }
      else
      {
        device_features.b_peer_bytes += b_raw_mats[term.b].sizeInByte();
      }
    }

    MatrixUseKey const a_key{.family = 'A', .index = term.a};
    if (staged_mats[device_index].insert(a_key).second)
    {
      ++device_features.unique_a;
      device_features.a_bytes += a.rows * a.cols * sizeof(double);
    }

    MatrixUseKey const c_key{.family = 'C', .index = term.c};
    if (staged_mats[device_index].insert(c_key).second)
    {
      ++device_features.unique_c;
      device_features.c_bytes += c.rows * c.cols * sizeof(double);
    }
  }

  for (int device = 0; device < device_count; ++device)
  {
    features.devices[static_cast<std::size_t>(device)].unique_bc = staged_bc[static_cast<std::size_t>(device)].size();
  }
  return features;
}

std::string json_int_array(std::span<int const> values)
{
  std::string result = "[";
  for (std::size_t index = 0; index < values.size(); ++index)
  {
    if (index != 0)
    {
      result += ',';
    }
    result += std::to_string(values[index]);
  }
  result += ']';
  return result;
}

std::string json_device_features(std::span<RabcDeviceCostFeatures const> features)
{
  std::string result = "[";
  for (std::size_t index = 0; index < features.size(); ++index)
  {
    auto const& item = features[index];
    if (index != 0)
    {
      result += ',';
    }
    result += fmt::format("{{\"device\":{},\"input_blocks\":{},\"output_blocks\":{},\"terms\":{},"
                          "\"unique_bc\":{},\"unique_a\":{},\"unique_b\":{},\"unique_c\":{},"
                          "\"bc_flops\":{:.17g},\"accumulate_flops\":{:.17g},"
                          "\"b_local_bytes\":{},\"b_peer_bytes\":{},\"a_bytes\":{},\"c_bytes\":{},"
                          "\"output_bytes\":{},\"intermediate_bytes\":{}}}",
                          item.device, item.input_blocks, item.output_blocks, item.terms, item.unique_bc, item.unique_a,
                          item.unique_b, item.unique_c, item.bc_flops, item.accumulate_flops, item.b_local_bytes,
                          item.b_peer_bytes, item.a_bytes, item.c_bytes, item.output_bytes, item.intermediate_bytes);
  }
  result += ']';
  return result;
}

bool rabc_trace_enabled() { return optional_env_string("UNI20_TENSORCONTRACTION_RABC_TRACE_PATH").has_value(); }

bool rabc_trace_terms_enabled() { return tensor::envFlagEnabled("UNI20_TENSORCONTRACTION_RABC_TRACE_TERMS"); }

std::uint64_t next_rabc_trace_index()
{
  static std::uint64_t next = 0;
  return next++;
}

struct RabcTraceDeviceTiming
{
    int device = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    double gpu_seconds = 0.0;
};

std::vector<RabcTraceDeviceTiming> start_rabc_trace_timing(tensor::Swapper& swapper)
{
  std::vector<RabcTraceDeviceTiming> timings;
  timings.reserve(static_cast<std::size_t>(swapper.getDeviceCount()));
  for (int device = 0; device < swapper.getDeviceCount(); ++device)
  {
    CUDA_CALL(cudaSetDevice(device));
    RabcTraceDeviceTiming timing{.device = device};
    CUDA_CALL(cudaEventCreate(&timing.start));
    CUDA_CALL(cudaEventCreate(&timing.stop));
    CUDA_CALL(cudaEventRecord(timing.start, cudaStreamLegacy));
    timings.push_back(timing);
  }
  return timings;
}

void stop_rabc_trace_timing(std::span<RabcTraceDeviceTiming> timings)
{
  for (auto& timing : timings)
  {
    CUDA_CALL(cudaSetDevice(timing.device));
    CUDA_CALL(cudaEventRecord(timing.stop, cudaStreamLegacy));
  }
}

void finish_rabc_trace_timing(std::span<RabcTraceDeviceTiming> timings)
{
  for (auto& timing : timings)
  {
    CUDA_CALL(cudaSetDevice(timing.device));
    CUDA_CALL(cudaEventSynchronize(timing.stop));
    float milliseconds = 0.0F;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, timing.start, timing.stop));
    timing.gpu_seconds = static_cast<double>(milliseconds) * 1.0e-3;
  }
}

void destroy_rabc_trace_timing(std::span<RabcTraceDeviceTiming> timings)
{
  for (auto& timing : timings)
  {
    CUDA_CALL(cudaSetDevice(timing.device));
    if (timing.start != nullptr)
    {
      CUDA_CALL(cudaEventDestroy(timing.start));
      timing.start = nullptr;
    }
    if (timing.stop != nullptr)
    {
      CUDA_CALL(cudaEventDestroy(timing.stop));
      timing.stop = nullptr;
    }
  }
}

double max_gpu_seconds(std::span<RabcTraceDeviceTiming const> timings)
{
  double result = 0.0;
  for (auto const& timing : timings)
  {
    result = std::max(result, timing.gpu_seconds);
  }
  return result;
}

std::string json_device_timings(std::span<RabcTraceDeviceTiming const> timings)
{
  std::string result = "[";
  for (std::size_t index = 0; index < timings.size(); ++index)
  {
    if (index != 0)
    {
      result += ',';
    }
    result += fmt::format("{{\"device\":{},\"gpu_s\":{:.17g}}}", timings[index].device, timings[index].gpu_seconds);
  }
  result += ']';
  return result;
}

void write_rabc_trace(std::uint64_t index, std::string const& policy, RabcCostFeatures const& features,
                      MatrixFamily const& a_mats, MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                      std::span<tensor::Matrix const> r_raw_mats,
                      std::span<EffectiveHamiltonianOperator::Term const> terms, double enqueue_seconds,
                      double sync_seconds, double wall_seconds, std::span<RabcTraceDeviceTiming const> timings)
{
  auto const path = optional_env_string("UNI20_TENSORCONTRACTION_RABC_TRACE_PATH");
  if (!path.has_value())
  {
    return;
  }

  auto* file = std::fopen(path->c_str(), "a");
  if (file == nullptr)
  {
    throw std::runtime_error("failed to open R/A/B/C trace file: " + *path);
  }

  fmt::print(file,
             "{{\"kind\":\"rabc_matvec\",\"index\":{},\"policy\":\"{}\",\"device_count\":{},"
             "\"block_count\":{},\"term_count\":{},\"enqueue_s\":{:.17g},\"sync_s\":{:.17g},"
             "\"wall_s\":{:.17g},\"gpu_s\":{:.17g},\"input_layout\":{},\"output_layout\":{},"
             "\"device_timings\":{},\"devices\":{}",
             index, policy, features.devices.size(), features.output_devices.size(), terms.size(), enqueue_seconds,
             sync_seconds, wall_seconds, max_gpu_seconds(timings), json_int_array(features.input_devices),
             json_int_array(features.output_devices), json_device_timings(timings),
             json_device_features(features.devices));
  if (rabc_trace_terms_enabled())
  {
    fmt::print(file, ",\"terms\":[");
    for (std::size_t term_index = 0; term_index < terms.size(); ++term_index)
    {
      auto const& term = terms[term_index];
      if (term_index != 0)
      {
        fmt::print(file, ",");
      }
      auto const a = a_mats.block(term.a);
      auto const b = b_mats.block(term.b);
      auto const c = c_mats.block(term.c);
      auto const r = r_raw_mats[term.r];
      auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
      fmt::print(file,
                 "{{\"r\":{},\"a\":{},\"b\":{},\"c\":{},\"coefficient\":{:.17g},\"device\":{},"
                 "\"r_rows\":{},\"r_cols\":{},\"a_rows\":{},\"a_cols\":{},"
                 "\"b_rows\":{},\"b_cols\":{},\"c_rows\":{},\"c_cols\":{},"
                 "\"bc_flops\":{:.17g},\"accumulate_flops\":{:.17g},\"intermediate_bytes\":{}}}",
                 term.r, term.a, term.b, term.c, term.coefficient, features.output_devices[term.r], r.getFirstDim(),
                 r.getSecondDim(), a.rows, a.cols, b.rows, b.cols, c.rows, c.cols, gemm_flops(b, c),
                 gemm_flops(a, intermediate), intermediate.rows * intermediate.cols * sizeof(double));
    }
    fmt::print(file, "]");
  }
  fmt::print(file, "}}\n");
  std::fclose(file);
}

std::vector<int> cost_based_rabc_output_devices(MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                                                MatrixFamily const& c_mats, std::span<tensor::Matrix const> r_raw_mats,
                                                std::span<EffectiveHamiltonianOperator::Term const> terms,
                                                tensor::Swapper& swapper)
{
  int const device_count = swapper.getDeviceCount();
  auto devices = default_byte_balanced_devices(r_raw_mats, device_count);
  auto const model = rabc_placement_model();
  auto const default_devices = devices;
  double const default_best =
      max_load(rabc_layout_load_seconds(a_mats, b_mats, c_mats, terms, devices, device_count, model));
  double best = default_best;

  bool changed = true;
  while (changed)
  {
    changed = false;
    for (std::size_t block = 0; block < devices.size(); ++block)
    {
      int const original_device = devices[block];
      int best_device = original_device;
      double best_candidate = best;
      for (int device = 0; device < device_count; ++device)
      {
        if (device == original_device)
        {
          continue;
        }
        devices[block] = device;
        double const candidate =
            max_load(rabc_layout_load_seconds(a_mats, b_mats, c_mats, terms, devices, device_count, model));
        if (candidate < best_candidate)
        {
          best_candidate = candidate;
          best_device = device;
        }
      }
      devices[block] = best_device;
      if (best_device != original_device)
      {
        best = best_candidate;
        changed = true;
      }
    }
  }

  if (best > 0.0 && default_best / best < model.arbitrary_min_speedup)
  {
    devices = default_devices;
  }

  if (log_cost_based_rabc_placement())
  {
    auto const load = rabc_layout_load_seconds(a_mats, b_mats, c_mats, terms, devices, device_count, model);
    fmt::print(stderr, "[TENSORCONTRACTION][RABC_PLACEMENT] policy=cost-block-stable devices={} outputs={}",
               device_count, devices.size());
    for (int device = 0; device < device_count; ++device)
    {
      std::size_t count = 0;
      for (int assigned : devices)
      {
        if (assigned == device)
        {
          ++count;
        }
      }
      fmt::print(stderr, " device{}={{blocks={},model_s={:.6g}}}", device, count,
                 load[static_cast<std::size_t>(device)]);
    }
    fmt::print(stderr, "\n");
  }

  return devices;
}

bool ensure_rabc_output_placement_cache(ResidentOutputPlacementCache& cache, MatrixFamily const& a_mats,
                                        MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                                        std::span<tensor::Matrix const> r_raw_mats,
                                        std::span<tensor::Matrix const> b_raw_mats,
                                        std::span<EffectiveHamiltonianOperator::Term const> terms,
                                        std::size_t output_count, tensor::Swapper& swapper)
{
  auto const policy = rabc_placement_policy();
  bool const use_contiguous_cost = use_cost_based_rabc_placement(policy);
  bool const use_block_cost = use_block_cost_based_rabc_placement(policy);
  bool const use_manual = use_manual_rabc_placement(policy);
  bool const use_striped = use_striped_rabc_placement(policy);
  bool const use_empirical_contiguous = use_empirical_contiguous_rabc_placement(policy);
  int const device_count = swapper.getDeviceCount();
  auto layout_key = std::string{};
  if (use_manual)
  {
    layout_key = optional_env_string("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT").value_or("");
  }
  else if (use_empirical_contiguous)
  {
    if (auto coefficients = optional_env_string("UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS");
        coefficients.has_value())
    {
      layout_key = "coefficients:" + *coefficients;
    }
    else
    {
      layout_key = "coefficients_file:" +
                   optional_env_string("UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS_FILE").value_or("");
    }
    if (auto min_speedup = optional_env_string("UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_MIN_SPEEDUP");
        min_speedup.has_value())
    {
      layout_key += ";min_speedup=" + *min_speedup;
    }
  }
  if ((!use_contiguous_cost && !use_block_cost && !use_manual && !use_striped && !use_empirical_contiguous) ||
      device_count <= 1)
  {
    cache = ResidentOutputPlacementCache{};
    return false;
  }

  if (cache.policy == policy && cache.device_count == device_count && cache.output_count == output_count &&
      cache.input_count == b_raw_mats.size() && cache.term_count == terms.size() && cache.layout_key == layout_key)
  {
    return !cache.ranges.empty() || !cache.devices.empty();
  }

  ResidentOutputPlacementCache next;
  next.policy = policy;
  next.device_count = device_count;
  next.output_count = output_count;
  next.input_count = b_raw_mats.size();
  next.term_count = terms.size();
  next.layout_key = layout_key;
  if (use_contiguous_cost)
  {
    next.coalesced_ranges = true;
    next.ranges =
        cost_based_rabc_output_ranges(a_mats, b_mats, c_mats, r_raw_mats, b_raw_mats, terms, output_count, swapper);
    next.use_default_localization = placement_ranges_match_default(next.ranges, r_raw_mats, device_count);
    if (next.use_default_localization && log_cost_based_rabc_placement())
    {
      fmt::print(stderr, "[TENSORCONTRACTION][RABC_PLACEMENT] policy={} uses default byte-balanced ranges\n", policy);
    }
  }
  else if (use_empirical_contiguous)
  {
    next.coalesced_ranges = true;
    if (!center_block_layout_supported(b_mats, r_raw_mats, terms))
    {
      throw std::invalid_argument("empirical R/A/B/C placement requires matching B/R center block spaces");
    }
    next.ranges = empirical_contiguous_rabc_output_ranges(a_mats, b_mats, c_mats, r_raw_mats, terms, swapper);
    next.use_default_localization = placement_ranges_match_default(next.ranges, r_raw_mats, device_count);
    if (next.use_default_localization && log_cost_based_rabc_placement())
    {
      fmt::print(stderr, "[TENSORCONTRACTION][RABC_PLACEMENT] policy={} uses default byte-balanced ranges\n", policy);
    }
  }
  else if (use_manual)
  {
    next.coalesced_ranges = false;
    if (!center_block_layout_supported(b_mats, r_raw_mats, terms))
    {
      throw std::invalid_argument("manual R/A/B/C placement requires matching B/R center block spaces");
    }
    next.devices = parse_manual_rabc_layout(output_count, device_count);
    next.use_default_localization = placement_devices_match_default(next.devices, r_raw_mats, device_count);
    if (log_cost_based_rabc_placement())
    {
      fmt::print(stderr, "[TENSORCONTRACTION][RABC_PLACEMENT] policy=manual devices={} outputs={} layout={}\n",
                 device_count, output_count, json_int_array(next.devices));
    }
  }
  else if (use_striped)
  {
    next.coalesced_ranges = false;
    bool default_reason_logged = false;
    if (!center_block_layout_supported(b_mats, r_raw_mats, terms))
    {
      next.devices = default_byte_balanced_devices(r_raw_mats, device_count);
      next.use_default_localization = true;
      if (log_cost_based_rabc_placement())
      {
        fmt::print(stderr,
                   "[TENSORCONTRACTION][RABC_PLACEMENT] policy={} falls back to default byte-balanced ranges: "
                   "B/R block spaces differ\n",
                   policy);
        default_reason_logged = true;
      }
    }
    else
    {
      next.devices = striped_devices(output_count, device_count);
      next.use_default_localization = placement_devices_match_default(next.devices, r_raw_mats, device_count);
    }
    if (log_cost_based_rabc_placement())
    {
      fmt::print(stderr, "[TENSORCONTRACTION][RABC_PLACEMENT] policy={} devices={} outputs={} layout={}\n", policy,
                 device_count, output_count, json_int_array(next.devices));
    }
    if (next.use_default_localization && !default_reason_logged && log_cost_based_rabc_placement())
    {
      fmt::print(stderr, "[TENSORCONTRACTION][RABC_PLACEMENT] policy={} uses default byte-balanced ranges\n", policy);
    }
  }
  else
  {
    next.coalesced_ranges = false;
    bool default_reason_logged = false;
    if (!center_block_layout_supported(b_mats, r_raw_mats, terms))
    {
      next.devices = default_byte_balanced_devices(r_raw_mats, device_count);
      next.use_default_localization = true;
      if (log_cost_based_rabc_placement())
      {
        fmt::print(stderr,
                   "[TENSORCONTRACTION][RABC_PLACEMENT] policy={} falls back to default byte-balanced ranges: "
                   "B/R block spaces differ\n",
                   policy);
        default_reason_logged = true;
      }
    }
    else
    {
      next.devices = cost_based_rabc_output_devices(a_mats, b_mats, c_mats, r_raw_mats, terms, swapper);
      next.use_default_localization = placement_devices_match_default(next.devices, r_raw_mats, device_count);
    }
    if (next.use_default_localization && !default_reason_logged && log_cost_based_rabc_placement())
    {
      fmt::print(stderr, "[TENSORCONTRACTION][RABC_PLACEMENT] policy={} uses default byte-balanced ranges\n", policy);
    }
  }
  cache = std::move(next);
  return !cache.ranges.empty() || !cache.devices.empty();
}

bool maybe_place_rabc_center_vectors(std::span<tensor::Matrix const> input_raw_mats,
                                     std::span<tensor::Matrix const> output_raw_mats, MatrixFamily const& a_mats,
                                     MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                                     std::span<EffectiveHamiltonianOperator::Term const> terms,
                                     tensor::Swapper& swapper, ResidentOutputPlacementCache& cache)
{
  if (input_raw_mats.size() != output_raw_mats.size())
  {
    return false;
  }

  if (!ensure_rabc_output_placement_cache(cache, a_mats, b_mats, c_mats, output_raw_mats, input_raw_mats, terms,
                                          output_raw_mats.size(), swapper))
  {
    return false;
  }
  if (cache.use_default_localization)
  {
    return false;
  }

  if (cache.coalesced_ranges)
  {
    for (auto const range : cache.ranges)
    {
      swapper.ensurePreStoreCoalescedOnDevice(matrix_subrange(input_raw_mats, range.begin, range.end), range.device,
                                              /*preserveExistingContent=*/true);
      swapper.ensurePreStoreCoalescedOnDevice(matrix_subrange(output_raw_mats, range.begin, range.end), range.device,
                                              /*preserveExistingContent=*/false);
    }
    return !cache.ranges.empty();
  }

  int const device_count = swapper.getDeviceCount();
  for (int device = 0; device < device_count; ++device)
  {
    std::vector<tensor::Matrix> input_group;
    std::vector<tensor::Matrix> output_group;
    for (std::size_t block = 0; block < output_raw_mats.size(); ++block)
    {
      if (cache.devices[block] == device)
      {
        input_group.push_back(input_raw_mats[block]);
        output_group.push_back(output_raw_mats[block]);
      }
    }
    if (!input_group.empty())
    {
      swapper.ensurePreStoreCoalescedOnDevice(input_group, device, /*preserveExistingContent=*/true);
      swapper.ensurePreStoreCoalescedOnDevice(output_group, device, /*preserveExistingContent=*/false);
    }
  }
  return true;
}

void host_apply(MatrixFamily const& r_mats, MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                MatrixFamily const& c_mats, std::span<EffectiveHamiltonianOperator::Term const> terms,
                MatrixFamily& out)
{
  out.fill(0.0);
  for (auto const& term : terms)
  {
    auto const r_block = r_mats.block(term.r);
    auto const a_block = a_mats.block(term.a);
    auto const b_block = b_mats.block(term.b);
    auto const c_block = c_mats.block(term.c);
    auto const a = a_mats.values(term.a);
    auto const b = b_mats.values(term.b);
    auto const c = c_mats.values(term.c);
    auto r = out.values(term.r);

    // This is the host mirror of TensorContraction's A * B * C path.  It is
    // intentionally simple and exists so small MPS unit tests do not need to
    // construct CUDA/NCCL runtimes and their large virtual-address mappings.
    std::vector<double> bc(b_block.rows * c_block.cols, 0.0);
    for (std::size_t row = 0; row < b_block.rows; ++row)
    {
      for (std::size_t inner = 0; inner < b_block.cols; ++inner)
      {
        auto const b_value = b[row * b_block.cols + inner];
        for (std::size_t col = 0; col < c_block.cols; ++col)
        {
          bc[row * c_block.cols + col] += b_value * c[inner * c_block.cols + col];
        }
      }
    }

    for (std::size_t row = 0; row < r_block.rows; ++row)
    {
      for (std::size_t inner = 0; inner < a_block.cols; ++inner)
      {
        auto const a_value = a[row * a_block.cols + inner];
        for (std::size_t col = 0; col < r_block.cols; ++col)
        {
          r[row * r_block.cols + col] += term.coefficient * a_value * bc[inner * c_block.cols + col];
        }
      }
    }
  }
}

auto output_device_for(tensor::Swapper& swapper, tensor::Matrix r_mat) -> int
{
  auto [device_id, buffer] = swapper.getPreStoreBufferOrNone(r_mat);
  if (buffer != nullptr)
  {
    return device_id;
  }
  constexpr int fallback_device = 0;
  swapper.registerGpuAllocation(r_mat, fallback_device);
  return fallback_device;
}

auto require_buffer_on(tensor::Swapper& swapper, tensor::Matrix mat,
                       int device_id) -> std::shared_ptr<tensor::GpuBuffer>
{
  auto buffer = swapper.getGpuBufferOrNone(mat, device_id);
  if (buffer != nullptr)
  {
    return buffer;
  }
  swapper.registerGpuAllocation(mat, device_id);
  buffer = swapper.getGpuBufferOrNone(mat, device_id);
  if (buffer == nullptr)
  {
    throw std::logic_error("TensorContraction deterministic RABC executor failed to allocate an output buffer");
  }
  return buffer;
}

void zero_device_matrix(tensor::Swapper& swapper, tensor::Matrix mat, int device_id)
{
  auto buffer = require_buffer_on(swapper, mat, device_id);
  auto access = swapper.createAccessPlan({}, {buffer}, device_id);
  CUDA_CALL(cudaMemsetAsync(buffer->getPtr(), 0, mat.sizeInByte(), access.stream()));
}

void gemm_device_matrix(tensor::Swapper& swapper, tensor::Matrix result, tensor::Matrix lhs, tensor::Matrix rhs,
                        double alpha, double beta, int device_id)
{
  auto lhs_buffer = swapper.ensureLocalCopy(lhs, device_id);
  auto rhs_buffer = swapper.ensureLocalCopy(rhs, device_id);
  auto result_buffer = require_buffer_on(swapper, result, device_id);
  auto access = swapper.createBlasAccessPlan({lhs_buffer, rhs_buffer}, {result_buffer}, device_id);

  CUBLAS_CALL(cublasDgemm(access.handle(), CUBLAS_OP_N, CUBLAS_OP_N, rhs.getSecondDim(), lhs.getFirstDim(),
                          lhs.getSecondDim(), &alpha, rhs_buffer->getPtr(), rhs.getSecondDim(), lhs_buffer->getPtr(),
                          lhs.getSecondDim(), &beta, result_buffer->getPtr(), result.getSecondDim()));
}

struct RightFirstIntermediateKey
{
    int device_id = 0;
    int b = 0;
    int c = 0;

    auto operator<=>(RightFirstIntermediateKey const&) const = default;
};

struct RightFirstIntermediate
{
    tensor::Matrix matrix;
};

void deterministic_right_first_apply(std::span<tensor::Matrix const> r_mats, std::span<tensor::Matrix const> a_mats,
                                     std::span<tensor::Matrix const> b_mats, std::span<tensor::Matrix const> c_mats,
                                     std::span<EffectiveHamiltonianOperator::Term const> terms,
                                     tensor::Swapper& swapper)
{
  std::vector<int> r_devices(r_mats.size(), 0);
  std::vector<bool> r_written(r_mats.size(), false);
  for (std::size_t r = 0; r < r_mats.size(); ++r)
  {
    r_devices[r] = output_device_for(swapper, r_mats[r]);
  }

  std::map<RightFirstIntermediateKey, RightFirstIntermediate> intermediates;
  for (auto const& term : terms)
  {
    auto const target_device = r_devices.at(term.r);
    auto const& b_mat = b_mats[term.b];
    auto const& c_mat = c_mats[term.c];
    RightFirstIntermediateKey const key{
        .device_id = target_device, .b = checked_index(term.b), .c = checked_index(term.c)};
    auto [intermediate_it, inserted] = intermediates.try_emplace(key);
    if (inserted)
    {
      auto intermediate =
          tensor::Matrix(nullptr, checked_index(b_mat.getFirstDim()), checked_index(c_mat.getSecondDim()));
      swapper.registerGpuAllocation(intermediate, target_device);
      gemm_device_matrix(swapper, intermediate, b_mat, c_mat, 1.0, 0.0, target_device);
      intermediate_it->second.matrix = intermediate;
    }

    // This is the initial deterministic planner: right-first only.  The seam is
    // deliberately narrow so a later cost model can choose left-first when the
    // left basis is smaller or when communication costs favor it.
    double const beta = r_written[static_cast<std::size_t>(term.r)] ? 1.0 : 0.0;
    gemm_device_matrix(swapper, r_mats[term.r], a_mats[term.a], intermediate_it->second.matrix, term.coefficient, beta,
                       target_device);
    r_written[static_cast<std::size_t>(term.r)] = true;
  }

  for (std::size_t r = 0; r < r_mats.size(); ++r)
  {
    if (!r_written[r])
    {
      fmt::print(stderr,
                 "[TENSORCONTRACTION][RABC_WARNING] Output R block id={} shape={}x{} device={} received no "
                 "Hamiltonian terms; zeroing it explicitly.\n",
                 r_mats[r].getId(), r_mats[r].getFirstDim(), r_mats[r].getSecondDim(), r_devices[r]);
      zero_device_matrix(swapper, r_mats[r], r_devices[r]);
    }
  }

  for (auto const& [_, intermediate] : intermediates)
  {
    swapper.clear(intermediate.matrix);
  }
}

} // namespace

void EffectiveHamiltonianOperator::Impl::initialize_runtime()
{
  if (use_host_effective_hamiltonian_backend())
  {
    return;
  }
  swapper = std::make_unique<tensor::Swapper>();
  arranger = std::make_unique<tensor::Arranger>(*swapper);
}

EffectiveHamiltonianOperator::EffectiveHamiltonianOperator(MatrixFamily a_mats, MatrixFamily b_mats,
                                                           std::span<MatrixFamily::Block const> input_blocks,
                                                           std::span<MatrixFamily::Block const> output_blocks,
                                                           std::span<Term const> terms)
    : impl_(nullptr)
{
  MatrixFamily r_mats(output_blocks);
  MatrixFamily c_mats(input_blocks);
  validate_term_shapes(r_mats, a_mats, b_mats, c_mats, terms);
  impl_ = std::make_unique<Impl>(std::move(a_mats), std::move(b_mats), input_blocks, output_blocks, terms);
}

EffectiveHamiltonianOperator::EffectiveHamiltonianOperator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

auto EffectiveHamiltonianOperator::variable_middle(MatrixFamily a_mats, MatrixFamily c_mats,
                                                   std::span<MatrixFamily::Block const> input_blocks,
                                                   std::span<MatrixFamily::Block const> output_blocks,
                                                   std::span<Term const> terms) -> EffectiveHamiltonianOperator
{
  MatrixFamily r_mats(output_blocks);
  MatrixFamily b_mats(input_blocks);
  validate_term_shapes(r_mats, a_mats, b_mats, c_mats, terms);
  return EffectiveHamiltonianOperator(std::make_unique<Impl>(VariableFamily::Middle, std::move(a_mats),
                                                             std::move(c_mats), input_blocks, output_blocks, terms));
}

EffectiveHamiltonianOperator::EffectiveHamiltonianOperator(EffectiveHamiltonianOperator&&) noexcept = default;
EffectiveHamiltonianOperator&
EffectiveHamiltonianOperator::operator=(EffectiveHamiltonianOperator&&) noexcept = default;
EffectiveHamiltonianOperator::~EffectiveHamiltonianOperator() = default;

std::size_t EffectiveHamiltonianOperator::term_count() const noexcept { return impl_->terms.size(); }

bool EffectiveHamiltonianOperator::compiled() const noexcept { return impl_->is_compiled; }

MatrixFamily EffectiveHamiltonianOperator::make_input_vector() const { return MatrixFamily(impl_->input_blocks); }

MatrixFamily EffectiveHamiltonianOperator::make_output_vector() const { return MatrixFamily(impl_->output_blocks); }

void EffectiveHamiltonianOperator::compile()
{
  if (impl_->is_compiled)
  {
    return;
  }

  validate_term_shapes(impl_->r_mats, impl_->a_mats, impl_->b_mats, impl_->c_mats, impl_->terms);
  if (impl_->host_backend())
  {
    impl_->is_compiled = true;
    return;
  }

  auto terms = convert_terms(impl_->terms);
  auto const& r = raw_matrices(impl_->r_mats);
  auto const& a = raw_matrices(impl_->a_mats);
  auto const& b = raw_matrices(impl_->b_mats);
  auto const& c = raw_matrices(impl_->c_mats);

  impl_->arranger->resetWork();
  impl_->arranger->analyzeComputation(r, a, b, c, terms);
  impl_->arranger->compileWorklists(r, a, b, c, /*syncResultsToHost=*/true);
  impl_->is_compiled = true;
}

void EffectiveHamiltonianOperator::apply(MatrixFamily const& x, MatrixFamily& y)
{
  validate_family_shape(x, impl_->input_blocks, "input");
  validate_family_shape(y, impl_->output_blocks, "output");

  if (!impl_->is_compiled)
  {
    compile();
  }

  if (impl_->variable_family == VariableFamily::Middle)
  {
    impl_->b_mats.assign(x);
  }
  else
  {
    impl_->c_mats.assign(x);
  }
  impl_->r_mats.fill(0.0);
  if (impl_->host_backend())
  {
    host_apply(impl_->r_mats, impl_->a_mats, impl_->b_mats, impl_->c_mats, impl_->terms, impl_->r_mats);
  }
  else
  {
    impl_->arranger->doContraction(raw_matrices(impl_->r_mats), raw_matrices(impl_->a_mats),
                                   raw_matrices(impl_->b_mats), raw_matrices(impl_->c_mats));
  }
  y.assign(impl_->r_mats);
}

void EffectiveHamiltonianOperator::apply_resident(MatrixFamily const& x, MatrixFamily& y, VectorAlgebraEngine& algebra)
{
  validate_family_shape(x, impl_->input_blocks, "input");
  validate_family_shape(y, impl_->output_blocks, "output");

  if (impl_->host_backend() || algebra.uses_host_backend())
  {
    this->apply(x, y);
    return;
  }

  auto& arranger = algebra.resident_arranger();
  auto terms = convert_terms(impl_->terms);
  auto const& r = raw_matrices(y);
  auto const& a = raw_matrices(impl_->a_mats);
  auto const& b = impl_->variable_family == VariableFamily::Middle ? raw_matrices(x) : raw_matrices(impl_->b_mats);
  auto const& c = impl_->variable_family == VariableFamily::Middle ? raw_matrices(impl_->c_mats) : raw_matrices(x);

  validate_term_shapes(y, impl_->a_mats, impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
                       impl_->variable_family == VariableFamily::Middle ? impl_->c_mats : x, impl_->terms);

  // Static environments are host-authored but should not be refreshed during
  // every Krylov matvec.  The active Lanczos input/output vectors are already
  // resident in this same runtime.
  arranger.localizeCoalescedForLinearAlgebra(a, impl_->a_mats.coalesced_values(), /*uploadFromHost=*/true,
                                             /*refreshExisting=*/false);
  if (impl_->variable_family == VariableFamily::Middle)
  {
    arranger.localizeCoalescedForLinearAlgebra(c, impl_->c_mats.coalesced_values(), /*uploadFromHost=*/true,
                                               /*refreshExisting=*/false);
  }
  else
  {
    arranger.localizeCoalescedForLinearAlgebra(b, impl_->b_mats.coalesced_values(), /*uploadFromHost=*/true,
                                               /*refreshExisting=*/false);
  }
  arranger.localizeCoalescedForLinearAlgebra(raw_matrices(x), x.coalesced_values(), /*uploadFromHost=*/false);
  auto& swapper = arranger.residentSwapper();
  bool const use_legacy_planner = use_legacy_arranger_rabc_planner();
  bool const custom_center_placement_available = impl_->variable_family == VariableFamily::Middle;
  bool const output_placement_selected =
      custom_center_placement_available && !use_legacy_planner &&
      maybe_place_rabc_center_vectors(raw_matrices(x), r, impl_->a_mats, x, impl_->c_mats, impl_->terms, swapper,
                                      impl_->output_placement_cache);
  if (!output_placement_selected)
  {
    arranger.localizeCoalescedForLinearAlgebra(raw_matrices(y), y.coalesced_values(), /*uploadFromHost=*/false);
  }

  if (!use_legacy_planner)
  {
    bool const trace_enabled = rabc_trace_enabled();
    auto trace_timings = trace_enabled ? start_rabc_trace_timing(swapper) : std::vector<RabcTraceDeviceTiming>{};
    auto const trace_index = trace_enabled ? next_rabc_trace_index() : 0;
    auto const enqueue_start = std::chrono::steady_clock::now();
    deterministic_right_first_apply(r, a, b, c, impl_->terms, swapper);
    auto const enqueue_stop = std::chrono::steady_clock::now();
    if (trace_enabled)
    {
      stop_rabc_trace_timing(trace_timings);
      finish_rabc_trace_timing(trace_timings);
      auto const sync_stop = std::chrono::steady_clock::now();
      auto const features = rabc_cost_features(
          impl_->a_mats, impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
          impl_->variable_family == VariableFamily::Middle ? impl_->c_mats : x, r, b, impl_->terms, swapper);
      write_rabc_trace(trace_index, rabc_placement_policy(), features, impl_->a_mats,
                       impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
                       impl_->variable_family == VariableFamily::Middle ? impl_->c_mats : x, r, impl_->terms,
                       std::chrono::duration<double>(enqueue_stop - enqueue_start).count(),
                       std::chrono::duration<double>(sync_stop - enqueue_stop).count(),
                       std::chrono::duration<double>(sync_stop - enqueue_start).count(), trace_timings);
      destroy_rabc_trace_timing(trace_timings);
    }
    return;
  }

  arranger.resetWork();
  arranger.analyzeComputation(r, a, b, c, terms);
  arranger.compileWorklists(r, a, b, c, /*syncResultsToHost=*/false);
  arranger.doContraction(r, a, b, c);
}

auto capture_variable_middle_rabc_fixture(EffectiveHamiltonianOperator const& op,
                                          MatrixFamily const& input_vector) -> RabcLanczosFixture
{
  if (op.impl_->variable_family != VariableFamily::Middle)
  {
    throw std::invalid_argument("R/A/B/C Lanczos fixture capture requires a variable-middle operator");
  }
  validate_family_shape(input_vector, op.impl_->input_blocks, "input");

  return RabcLanczosFixture{.a_mats = clone_family(op.impl_->a_mats),
                            .c_mats = clone_family(op.impl_->c_mats),
                            .input_vector = clone_family(input_vector),
                            .output_blocks = op.impl_->output_blocks,
                            .terms = op.impl_->terms};
}

} // namespace uni20::tensorcontraction
