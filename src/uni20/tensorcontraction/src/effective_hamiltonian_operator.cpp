#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/rabc_lanczos_fixture.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"
#include "Utils.h"

#include <fmt/core.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    std::vector<int> input_devices;
    std::vector<int> output_devices;
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

std::string rabc_placement_policy()
{
  auto const* planner = std::getenv("UNI20_TENSORCONTRACTION_RABC_PLACEMENT");
  if (planner == nullptr)
  {
    return {};
  }
  return std::string(planner);
}

bool use_manual_rabc_placement(std::string const& policy) { return policy == "manual" || policy == "layout"; }

bool log_rabc_placement()
{
  auto const* raw = std::getenv("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LOG");
  return raw != nullptr && (std::string(raw) == "1" || std::string(raw) == "true" || std::string(raw) == "on");
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

void fnv1a_update(std::uint64_t& hash, std::uint64_t value)
{
  constexpr std::uint64_t prime = 1099511628211ULL;
  for (int byte = 0; byte < 8; ++byte)
  {
    hash ^= (value >> (8 * byte)) & 0xffULL;
    hash *= prime;
  }
}

std::string output_shape_signature(std::span<tensor::Matrix const> r_raw_mats)
{
  std::uint64_t hash = 14695981039346656037ULL;
  fnv1a_update(hash, static_cast<std::uint64_t>(r_raw_mats.size()));
  for (auto const& mat : r_raw_mats)
  {
    fnv1a_update(hash, static_cast<std::uint64_t>(mat.getFirstDim()));
    fnv1a_update(hash, static_cast<std::uint64_t>(mat.getSecondDim()));
  }
  return fmt::format("fnv1a64:{:016x}", hash);
}

std::vector<int> parse_rabc_layout_text(std::string const& layout_text, std::size_t block_count, int device_count,
                                        char const* name)
{
  std::vector<int> devices;
  devices.reserve(block_count);
  std::size_t begin = 0;
  while (begin <= layout_text.size())
  {
    std::size_t const end = layout_text.find(',', begin);
    auto const token = trim_ascii(
        std::string_view(layout_text).substr(begin, end == std::string::npos ? std::string::npos : end - begin));
    if (token.empty())
    {
      throw std::invalid_argument(std::string(name) + " contains an empty device token");
    }

    std::size_t consumed = 0;
    int const device = std::stoi(token, &consumed);
    if (consumed != token.size() || device < 0 || device >= device_count)
    {
      throw std::invalid_argument(std::string(name) + " contains an invalid CUDA device id: " + token);
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
    throw std::invalid_argument(std::string(name) + " block count does not match the matrix family");
  }
  return devices;
}

std::vector<int> parse_manual_rabc_layout(std::size_t block_count, int device_count)
{
  auto const layout_text = optional_env_string("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT");
  if (!layout_text.has_value())
  {
    throw std::invalid_argument(
        "UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual requires UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT");
  }
  return parse_rabc_layout_text(*layout_text, block_count, device_count,
                                "UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT");
}

std::optional<std::vector<int>> parse_optional_rabc_layout(char const* name, std::size_t block_count, int device_count)
{
  auto const layout_text = optional_env_string(name);
  if (!layout_text.has_value())
  {
    return std::nullopt;
  }
  return parse_rabc_layout_text(*layout_text, block_count, device_count, name);
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
    ranges.push_back(ResidentOutputPlacementRange{.device = range_index, .begin = begin, .end = end});
    begin = end;
  }
  return ranges;
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

bool devices_match_default(std::span<int const> devices, std::span<tensor::Matrix const> mats, int device_count)
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
    std::size_t temporary_peer_bytes = 0;
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
    int const first_device = features.input_devices[term.b];
    int const output_device = features.output_devices[term.r];
    auto const first_device_index = static_cast<std::size_t>(first_device);
    auto const output_device_index = static_cast<std::size_t>(output_device);
    auto& first_features = features.devices[first_device_index];
    auto& output_features = features.devices[output_device_index];
    ++first_features.terms;

    BcUseKey const bc{.b = term.b, .c = term.c};
    if (staged_bc[first_device_index].insert(bc).second)
    {
      auto const b = b_mats.block(term.b);
      auto const c = c_mats.block(term.c);
      first_features.bc_flops += gemm_flops(b, c);
      first_features.intermediate_bytes += b.rows * c.cols * sizeof(double);
    }

    auto const a = a_mats.block(term.a);
    auto const b = b_mats.block(term.b);
    auto const c = c_mats.block(term.c);
    auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
    output_features.accumulate_flops += gemm_flops(a, intermediate);
    if (first_device != output_device)
    {
      output_features.temporary_peer_bytes += intermediate.rows * intermediate.cols * sizeof(double);
    }

    MatrixUseKey const b_key{.family = 'B', .index = term.b};
    if (staged_mats[first_device_index].insert(b_key).second)
    {
      ++first_features.unique_b;
      first_features.b_local_bytes += b_raw_mats[term.b].sizeInByte();
    }

    MatrixUseKey const a_key{.family = 'A', .index = term.a};
    if (staged_mats[output_device_index].insert(a_key).second)
    {
      ++output_features.unique_a;
      output_features.a_bytes += a.rows * a.cols * sizeof(double);
    }

    MatrixUseKey const c_key{.family = 'C', .index = term.c};
    if (staged_mats[first_device_index].insert(c_key).second)
    {
      ++first_features.unique_c;
      first_features.c_bytes += c.rows * c.cols * sizeof(double);
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
                          "\"b_local_bytes\":{},\"b_peer_bytes\":{},\"temporary_peer_bytes\":{},"
                          "\"a_bytes\":{},\"c_bytes\":{},"
                          "\"output_bytes\":{},\"intermediate_bytes\":{}}}",
                          item.device, item.input_blocks, item.output_blocks, item.terms, item.unique_bc, item.unique_a,
                          item.unique_b, item.unique_c, item.bc_flops, item.accumulate_flops, item.b_local_bytes,
                          item.b_peer_bytes, item.temporary_peer_bytes, item.a_bytes, item.c_bytes, item.output_bytes,
                          item.intermediate_bytes);
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
             "\"block_count\":{},\"output_shape_signature\":\"{}\",\"term_count\":{},\"enqueue_s\":{:.17g},"
             "\"sync_s\":{:.17g},"
             "\"wall_s\":{:.17g},\"gpu_s\":{:.17g},\"input_layout\":{},\"output_layout\":{},"
             "\"device_timings\":{},\"devices\":{}",
             index, policy, features.devices.size(), features.output_devices.size(), output_shape_signature(r_raw_mats),
             terms.size(), enqueue_seconds, sync_seconds, wall_seconds, max_gpu_seconds(timings),
             json_int_array(features.input_devices), json_int_array(features.output_devices),
             json_device_timings(timings), json_device_features(features.devices));
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
                 "\"first_device\":{},\"output_device\":{},"
                 "\"r_rows\":{},\"r_cols\":{},\"a_rows\":{},\"a_cols\":{},"
                 "\"b_rows\":{},\"b_cols\":{},\"c_rows\":{},\"c_cols\":{},"
                 "\"bc_flops\":{:.17g},\"accumulate_flops\":{:.17g},\"intermediate_bytes\":{}}}",
                 term.r, term.a, term.b, term.c, term.coefficient, features.output_devices[term.r],
                 features.input_devices[term.b], features.output_devices[term.r], r.getFirstDim(), r.getSecondDim(),
                 a.rows, a.cols, b.rows, b.cols, c.rows, c.cols, gemm_flops(b, c), gemm_flops(a, intermediate),
                 intermediate.rows * intermediate.cols * sizeof(double));
    }
    fmt::print(file, "]");
  }
  fmt::print(file, "}}\n");
  std::fclose(file);
}

bool ensure_rabc_placement_cache(ResidentOutputPlacementCache& cache, MatrixFamily const& b_mats,
                                 std::span<tensor::Matrix const> r_raw_mats, std::span<tensor::Matrix const> b_raw_mats,
                                 std::span<EffectiveHamiltonianOperator::Term const> terms,
                                 VariableFamily variable_family, tensor::Swapper& swapper)
{
  auto const policy = rabc_placement_policy();
  bool const use_manual = use_manual_rabc_placement(policy);
  int const device_count = swapper.getDeviceCount();
  if (!policy.empty() && !use_manual)
  {
    throw std::invalid_argument("unsupported TensorContraction R/A/B/C placement policy: " + policy);
  }
  if (!use_manual || device_count <= 1)
  {
    cache = ResidentOutputPlacementCache{};
    return false;
  }

  auto const shared_layout = optional_env_string("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT");
  auto const b_layout = optional_env_string("UNI20_TENSORCONTRACTION_RABC_B_LAYOUT");
  auto const r_layout = optional_env_string("UNI20_TENSORCONTRACTION_RABC_R_LAYOUT");
  auto const layout_key =
      fmt::format("shared={};b={};r={}", shared_layout.value_or(""), b_layout.value_or(""), r_layout.value_or(""));
  if (cache.policy == policy && cache.device_count == device_count && cache.output_count == r_raw_mats.size() &&
      cache.input_count == b_raw_mats.size() && cache.term_count == terms.size() && cache.layout_key == layout_key)
  {
    return !cache.input_devices.empty() || !cache.output_devices.empty();
  }

  ResidentOutputPlacementCache next;
  next.policy = policy;
  next.device_count = device_count;
  next.output_count = r_raw_mats.size();
  next.input_count = b_raw_mats.size();
  next.term_count = terms.size();
  next.layout_key = layout_key;

  if (variable_family == VariableFamily::Middle)
  {
    if (!center_block_layout_supported(b_mats, r_raw_mats, terms))
    {
      throw std::invalid_argument("manual R/A/B/C Hamiltonian placement requires matching B/R center block spaces");
    }
    auto shared = parse_manual_rabc_layout(r_raw_mats.size(), device_count);
    next.input_devices = shared;
    next.output_devices = std::move(shared);
  }
  else
  {
    auto parsed_b =
        parse_optional_rabc_layout("UNI20_TENSORCONTRACTION_RABC_B_LAYOUT", b_raw_mats.size(), device_count);
    auto parsed_r =
        parse_optional_rabc_layout("UNI20_TENSORCONTRACTION_RABC_R_LAYOUT", r_raw_mats.size(), device_count);
    if (!parsed_r.has_value() && shared_layout.has_value())
    {
      parsed_r = parse_rabc_layout_text(*shared_layout, r_raw_mats.size(), device_count,
                                        "UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT");
    }
    if (!parsed_b.has_value() && !parsed_r.has_value())
    {
      throw std::invalid_argument("manual R/A/B/C environment placement requires UNI20_TENSORCONTRACTION_RABC_B_LAYOUT "
                                  "and/or UNI20_TENSORCONTRACTION_RABC_R_LAYOUT");
    }
    next.input_devices = parsed_b.value_or(default_byte_balanced_devices(b_raw_mats, device_count));
    next.output_devices = parsed_r.value_or(default_byte_balanced_devices(r_raw_mats, device_count));
  }

  next.use_default_localization = devices_match_default(next.input_devices, b_raw_mats, device_count) &&
                                  devices_match_default(next.output_devices, r_raw_mats, device_count);
  if (log_rabc_placement())
  {
    fmt::print(stderr,
               "[TENSORCONTRACTION][RABC_PLACEMENT] policy=manual devices={} inputs={} outputs={} "
               "b_layout={} r_layout={}\n",
               device_count, b_raw_mats.size(), r_raw_mats.size(), json_int_array(next.input_devices),
               json_int_array(next.output_devices));
  }
  cache = std::move(next);
  return !cache.input_devices.empty() || !cache.output_devices.empty();
}

void ensure_pre_store_groups_by_device(std::span<tensor::Matrix const> mats, std::span<int const> devices,
                                       bool preserve_existing_content, tensor::Swapper& swapper)
{
  int const device_count = swapper.getDeviceCount();
  for (int device = 0; device < device_count; ++device)
  {
    std::vector<tensor::Matrix> group;
    for (std::size_t block = 0; block < mats.size(); ++block)
    {
      if (devices[block] == device)
      {
        group.push_back(mats[block]);
      }
    }
    if (!group.empty())
    {
      swapper.ensurePreStoreCoalescedOnDevice(group, device, preserve_existing_content);
    }
  }
}

bool maybe_place_rabc_families(std::span<tensor::Matrix const> input_raw_mats,
                               std::span<tensor::Matrix const> output_raw_mats, MatrixFamily const& b_mats,
                               std::span<EffectiveHamiltonianOperator::Term const> terms,
                               VariableFamily variable_family, tensor::Swapper& swapper,
                               ResidentOutputPlacementCache& cache)
{
  if (!ensure_rabc_placement_cache(cache, b_mats, output_raw_mats, input_raw_mats, terms, variable_family, swapper))
  {
    return false;
  }
  if (cache.use_default_localization)
  {
    return false;
  }

  ensure_pre_store_groups_by_device(input_raw_mats, cache.input_devices, /*preserve_existing_content=*/true, swapper);
  ensure_pre_store_groups_by_device(output_raw_mats, cache.output_devices, /*preserve_existing_content=*/false,
                                    swapper);
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

auto input_device_for(tensor::Swapper& swapper, tensor::Matrix b_mat) -> int
{
  auto [device_id, buffer] = swapper.getPreStoreBufferOrNone(b_mat);
  if (buffer != nullptr)
  {
    return device_id;
  }
  constexpr int fallback_device = 0;
  static_cast<void>(swapper.ensureLocalCopy(b_mat, fallback_device));
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

void deterministic_input_anchored_right_first_apply(std::span<tensor::Matrix const> r_mats,
                                                    std::span<tensor::Matrix const> a_mats,
                                                    std::span<tensor::Matrix const> b_mats,
                                                    std::span<tensor::Matrix const> c_mats,
                                                    std::span<EffectiveHamiltonianOperator::Term const> terms,
                                                    tensor::Swapper& swapper)
{
  std::vector<int> r_devices(r_mats.size(), 0);
  std::vector<int> b_devices(b_mats.size(), 0);
  std::vector<bool> r_written(r_mats.size(), false);
  for (std::size_t r = 0; r < r_mats.size(); ++r)
  {
    r_devices[r] = output_device_for(swapper, r_mats[r]);
  }
  for (std::size_t b = 0; b < b_mats.size(); ++b)
  {
    b_devices[b] = input_device_for(swapper, b_mats[b]);
  }

  std::map<RightFirstIntermediateKey, RightFirstIntermediate> intermediates;
  for (auto const& term : terms)
  {
    auto const target_device = r_devices.at(term.r);
    auto const first_device = b_devices.at(term.b);
    auto const& b_mat = b_mats[term.b];
    auto const& c_mat = c_mats[term.c];
    RightFirstIntermediateKey const key{
        .device_id = first_device, .b = checked_index(term.b), .c = checked_index(term.c)};
    auto [intermediate_it, inserted] = intermediates.try_emplace(key);
    if (inserted)
    {
      auto intermediate =
          tensor::Matrix(nullptr, checked_index(b_mat.getFirstDim()), checked_index(c_mat.getSecondDim()));
      swapper.registerGpuAllocation(intermediate, first_device);
      gemm_device_matrix(swapper, intermediate, b_mat, c_mat, 1.0, 0.0, first_device);
      intermediate_it->second.matrix = intermediate;
    }

    // This is the initial input-anchored planner: right-first only.  The first
    // GEMM runs where B_b lives, then the temporary is staged to R_r's owner for
    // final accumulation.  A later cost model can choose left-first per term.
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
  bool const output_placement_selected =
      maybe_place_rabc_families(b, r, impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
                                impl_->terms, impl_->variable_family, swapper, impl_->output_placement_cache);
  if (!output_placement_selected)
  {
    arranger.localizeCoalescedForLinearAlgebra(raw_matrices(y), y.coalesced_values(), /*uploadFromHost=*/false);
  }

  bool const trace_enabled = rabc_trace_enabled();
  auto trace_timings = trace_enabled ? start_rabc_trace_timing(swapper) : std::vector<RabcTraceDeviceTiming>{};
  auto const trace_index = trace_enabled ? next_rabc_trace_index() : 0;
  auto const enqueue_start = std::chrono::steady_clock::now();
  deterministic_input_anchored_right_first_apply(r, a, b, c, impl_->terms, swapper);
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
