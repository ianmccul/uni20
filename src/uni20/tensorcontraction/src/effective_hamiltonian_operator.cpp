#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/rabc_lanczos_fixture.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"
#include "Utils.h"

#include <fmt/core.h>

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction
{

enum class VariableFamily
{
  Middle,
  Right,
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

bool use_cost_based_rabc_placement()
{
  auto const* planner = std::getenv("UNI20_TENSORCONTRACTION_RABC_PLACEMENT");
  if (planner == nullptr)
  {
    return false;
  }
  auto const value = std::string(planner);
  return value == "cost" || value == "greedy" || value == "cost-greedy";
}

bool use_block_cost_based_rabc_placement()
{
  auto const* planner = std::getenv("UNI20_TENSORCONTRACTION_RABC_PLACEMENT");
  if (planner == nullptr)
  {
    return false;
  }
  auto const value = std::string(planner);
  return value == "cost-block" || value == "block" || value == "greedy-block";
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

struct RabcPlacementModel
{
    double gflops = 1000.0;
    double central_gbps = 32.0;
    bool count_environment_bytes = false;
    double env_gbps = 32.0;
};

struct RabcPlacementRange
{
    int device = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
    double model_seconds = 0.0;
};

double gemm_flops(MatrixFamily::Block lhs, MatrixFamily::Block rhs)
{
  return 2.0 * static_cast<double>(lhs.rows) * static_cast<double>(lhs.cols) * static_cast<double>(rhs.cols);
}

double term_right_first_flops(MatrixFamily const& a_mats, MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                              EffectiveHamiltonianOperator::Term const& term)
{
  auto const a = a_mats.block(term.a);
  auto const b = b_mats.block(term.b);
  auto const c = c_mats.block(term.c);
  auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
  return gemm_flops(b, c) + gemm_flops(a, intermediate);
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
  return RabcPlacementModel{.gflops = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_GFLOPS", 1000.0),
                            .central_gbps = central_gbps,
                            .count_environment_bytes =
                                std::getenv("UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_BYTES") != nullptr,
                            .env_gbps = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_GBPS", central_gbps)};
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

std::vector<int> greedy_rabc_output_devices(MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                                            MatrixFamily const& c_mats, std::span<tensor::Matrix const> b_raw_mats,
                                            std::span<EffectiveHamiltonianOperator::Term const> terms,
                                            std::size_t output_count, tensor::Swapper& swapper)
{
  int const device_count = swapper.getDeviceCount();
  std::vector<std::vector<EffectiveHamiltonianOperator::Term const*>> terms_by_r(output_count);
  for (auto const& term : terms)
  {
    if (term.r < output_count)
    {
      terms_by_r[term.r].push_back(&term);
    }
  }

  double const gflops = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_GFLOPS", 1000.0);
  double const central_gbps = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_CENTRAL_GBPS", 32.0);
  bool const count_environment_bytes = std::getenv("UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_BYTES") != nullptr;
  double const env_gbps = env_double_or("UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_GBPS", central_gbps);

  std::vector<std::optional<int>> b_source_device(b_mats.size());
  for (std::size_t b = 0; b < b_mats.size(); ++b)
  {
    b_source_device[b] = pre_store_device_for(swapper, b_raw_mats[b]);
  }

  std::vector<std::size_t> order(output_count);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    double lhs_flops = 0.0;
    double rhs_flops = 0.0;
    for (auto const* term : terms_by_r[lhs])
    {
      lhs_flops += term_right_first_flops(a_mats, b_mats, c_mats, *term);
    }
    for (auto const* term : terms_by_r[rhs])
    {
      rhs_flops += term_right_first_flops(a_mats, b_mats, c_mats, *term);
    }
    return lhs_flops > rhs_flops;
  });

  std::vector<int> devices(output_count, 0);
  std::vector<double> load_seconds(static_cast<std::size_t>(device_count), 0.0);
  std::vector<std::set<BcUseKey>> staged_bc(static_cast<std::size_t>(device_count));
  std::vector<std::set<MatrixUseKey>> staged_mats(static_cast<std::size_t>(device_count));

  auto matrix_transfer_seconds = [&](int device, MatrixUseKey key, MatrixFamily::Block block,
                                     std::optional<int> source_device, double gbps) {
    if (source_device.has_value() && *source_device == device)
    {
      return 0.0;
    }
    auto& staged = staged_mats[static_cast<std::size_t>(device)];
    if (staged.contains(key))
    {
      return 0.0;
    }
    return static_cast<double>(block.rows) * static_cast<double>(block.cols) * sizeof(double) / (gbps * 1.0e9);
  };

  for (std::size_t r : order)
  {
    int best_device = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (int device = 0; device < device_count; ++device)
    {
      double incremental = 0.0;
      std::set<BcUseKey> new_bc;
      std::set<MatrixUseKey> new_mats;
      for (auto const* term : terms_by_r[r])
      {
        BcUseKey const bc{.b = term->b, .c = term->c};
        if (!staged_bc[static_cast<std::size_t>(device)].contains(bc) && !new_bc.contains(bc))
        {
          incremental += gemm_flops(b_mats.block(term->b), c_mats.block(term->c)) / (gflops * 1.0e9);
          new_bc.insert(bc);
        }

        auto const a = a_mats.block(term->a);
        auto const b = b_mats.block(term->b);
        auto const c = c_mats.block(term->c);
        auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
        incremental += gemm_flops(a, intermediate) / (gflops * 1.0e9);

        MatrixUseKey const b_key{.family = 'B', .index = term->b};
        if (!new_mats.contains(b_key))
        {
          incremental += matrix_transfer_seconds(device, b_key, b, b_source_device[term->b], central_gbps);
          new_mats.insert(b_key);
        }

        if (count_environment_bytes)
        {
          MatrixUseKey const a_key{.family = 'A', .index = term->a};
          MatrixUseKey const c_key{.family = 'C', .index = term->c};
          if (!new_mats.contains(a_key))
          {
            incremental += matrix_transfer_seconds(device, a_key, a, std::nullopt, env_gbps);
            new_mats.insert(a_key);
          }
          if (!new_mats.contains(c_key))
          {
            incremental += matrix_transfer_seconds(device, c_key, c, std::nullopt, env_gbps);
            new_mats.insert(c_key);
          }
        }
      }

      auto projected = load_seconds;
      projected[static_cast<std::size_t>(device)] += incremental;
      double const score = *std::max_element(projected.begin(), projected.end());
      if (score < best_score)
      {
        best_score = score;
        best_device = device;
      }
    }

    devices[r] = best_device;
    for (auto const* term : terms_by_r[r])
    {
      BcUseKey const bc{.b = term->b, .c = term->c};
      if (staged_bc[static_cast<std::size_t>(best_device)].insert(bc).second)
      {
        load_seconds[static_cast<std::size_t>(best_device)] +=
            gemm_flops(b_mats.block(term->b), c_mats.block(term->c)) / (gflops * 1.0e9);
      }

      auto const a = a_mats.block(term->a);
      auto const b = b_mats.block(term->b);
      auto const c = c_mats.block(term->c);
      auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
      load_seconds[static_cast<std::size_t>(best_device)] += gemm_flops(a, intermediate) / (gflops * 1.0e9);

      MatrixUseKey const b_key{.family = 'B', .index = term->b};
      auto& best_staged_mats = staged_mats[static_cast<std::size_t>(best_device)];
      if (!best_staged_mats.contains(b_key))
      {
        load_seconds[static_cast<std::size_t>(best_device)] +=
            matrix_transfer_seconds(best_device, b_key, b, b_source_device[term->b], central_gbps);
        best_staged_mats.insert(b_key);
      }

      if (count_environment_bytes)
      {
        MatrixUseKey const a_key{.family = 'A', .index = term->a};
        MatrixUseKey const c_key{.family = 'C', .index = term->c};
        if (!best_staged_mats.contains(a_key))
        {
          load_seconds[static_cast<std::size_t>(best_device)] +=
              matrix_transfer_seconds(best_device, a_key, a, std::nullopt, env_gbps);
          best_staged_mats.insert(a_key);
        }
        if (!best_staged_mats.contains(c_key))
        {
          load_seconds[static_cast<std::size_t>(best_device)] +=
              matrix_transfer_seconds(best_device, c_key, c, std::nullopt, env_gbps);
          best_staged_mats.insert(c_key);
        }
      }
    }
  }

  if (log_cost_based_rabc_placement())
  {
    fmt::print(stderr, "[TENSORCONTRACTION][RABC_PLACEMENT] policy=cost devices={} outputs={}", device_count,
               output_count);
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
                 load_seconds[static_cast<std::size_t>(device)]);
    }
    fmt::print(stderr, "\n");
  }

  return devices;
}

std::vector<RabcPlacementRange> cost_based_rabc_output_ranges(MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                                                              MatrixFamily const& c_mats,
                                                              std::span<tensor::Matrix const> b_raw_mats,
                                                              std::span<EffectiveHamiltonianOperator::Term const> terms,
                                                              std::size_t output_count, tensor::Swapper& swapper)
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

  std::vector<RabcPlacementRange> ranges(static_cast<std::size_t>(range_count));
  std::size_t end = output_count;
  for (int used = range_count; used >= 1; --used)
  {
    std::size_t const begin = split[static_cast<std::size_t>(used)][end];
    ranges[static_cast<std::size_t>(used - 1)] =
        RabcPlacementRange{.device = used - 1,
                           .begin = begin,
                           .end = end,
                           .model_seconds = range_cost[static_cast<std::size_t>(used - 1)][begin][end]};
    end = begin;
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
    fmt::print(stderr, "\n");
  }

  return ranges;
}

std::vector<tensor::Matrix> matrix_subrange(std::span<tensor::Matrix const> mats, std::size_t begin, std::size_t end)
{
  return {mats.begin() + static_cast<std::ptrdiff_t>(begin), mats.begin() + static_cast<std::ptrdiff_t>(end)};
}

bool maybe_place_rabc_outputs(std::span<tensor::Matrix const> r_raw_mats, MatrixFamily const& a_mats,
                              MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                              std::span<tensor::Matrix const> b_raw_mats,
                              std::span<EffectiveHamiltonianOperator::Term const> terms, tensor::Swapper& swapper)
{
  bool const use_contiguous_cost = use_cost_based_rabc_placement();
  bool const use_block_cost = use_block_cost_based_rabc_placement();
  if ((!use_contiguous_cost && !use_block_cost) || swapper.getDeviceCount() <= 1)
  {
    return false;
  }

  if (swapper.anyPreStoreBuffer(std::vector<tensor::Matrix>{r_raw_mats.begin(), r_raw_mats.end()}))
  {
    return false;
  }

  if (use_contiguous_cost)
  {
    auto const ranges =
        cost_based_rabc_output_ranges(a_mats, b_mats, c_mats, b_raw_mats, terms, r_raw_mats.size(), swapper);
    for (auto const range : ranges)
    {
      swapper.registerGpuAllocationsCoalesced(matrix_subrange(r_raw_mats, range.begin, range.end), range.device);
    }
    return !ranges.empty();
  }

  auto const devices =
      greedy_rabc_output_devices(a_mats, b_mats, c_mats, b_raw_mats, terms, r_raw_mats.size(), swapper);
  for (std::size_t r = 0; r < r_raw_mats.size(); ++r)
  {
    swapper.registerGpuAllocation(r_raw_mats[r], devices[r]);
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
  bool const output_placement_selected =
      !use_legacy_planner &&
      maybe_place_rabc_outputs(r, impl_->a_mats, impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
                               impl_->variable_family == VariableFamily::Middle ? impl_->c_mats : x, b, impl_->terms,
                               swapper);
  if (!output_placement_selected)
  {
    arranger.localizeCoalescedForLinearAlgebra(raw_matrices(y), y.coalesced_values(), /*uploadFromHost=*/false);
  }

  if (!use_legacy_planner)
  {
    deterministic_right_first_apply(r, a, b, c, impl_->terms, swapper);
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
