#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/rabc_lanczos_fixture.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include "Swapper.hpp"
#include "Utils.h"

#include <fmt/core.h>

#include <algorithm>
#include <array>
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

struct ResidentRabcContext
{
    struct FirstStageProductPlan
    {
        int device_id = 0;
        std::size_t b = 0;
        std::size_t c = 0;
        tensor::Matrix matrix;
    };

    struct LocalPartialInput
    {
        std::size_t first_stage_product = 0;
        double coefficient = 1.0;
    };

    struct LocalPartialPlan
    {
        int device_id = 0;
        std::vector<LocalPartialInput> inputs;
        std::optional<tensor::Matrix> accumulation;
    };

    struct AssembledTemporaryPlan
    {
        std::size_t r = 0;
        std::size_t a = 0;
        std::vector<LocalPartialPlan> local_partials;
        std::optional<tensor::Matrix> output_accumulation;
    };

    VariableFamily variable_family = VariableFamily::Right;
    int device_count = 0;
    std::size_t input_count = 0;
    std::size_t output_count = 0;
    std::size_t term_count = 0;
    std::vector<int> input_devices;
    std::vector<int> output_devices;
    std::vector<FirstStageProductPlan> first_stage_products;
    std::vector<AssembledTemporaryPlan> assembled_temporaries;
    std::vector<tensor::Matrix> owned_mats;

    [[nodiscard]] bool matches(VariableFamily variable, int current_device_count,
                               std::span<int const> current_input_devices, std::span<int const> current_output_devices,
                               std::size_t current_term_count) const;
    void release(tensor::Swapper& swapper);
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
    ResidentOutputPlacementCache output_placement_cache;
    std::unique_ptr<ResidentRabcContext> resident_context;
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

    ~Impl();
    void initialize_runtime();
    ResidentRabcContext& ensure_resident_context(std::span<tensor::Matrix const> r_mats,
                                                 std::span<tensor::Matrix const> a_raw_mats,
                                                 std::span<tensor::Matrix const> b_raw_mats,
                                                 std::span<tensor::Matrix const> c_raw_mats, MatrixFamily const& c_mats,
                                                 tensor::Swapper& swapper);
    [[nodiscard]] bool host_backend() const { return swapper == nullptr; }
};

bool ResidentRabcContext::matches(VariableFamily variable, int current_device_count,
                                  std::span<int const> current_input_devices,
                                  std::span<int const> current_output_devices, std::size_t current_term_count) const
{
  return variable_family == variable && device_count == current_device_count && device_count > 0 &&
         current_input_devices.size() == input_count && current_output_devices.size() == output_count &&
         current_term_count == term_count &&
         std::equal(input_devices.begin(), input_devices.end(), current_input_devices.begin(),
                    current_input_devices.end()) &&
         std::equal(output_devices.begin(), output_devices.end(), current_output_devices.begin(),
                    current_output_devices.end());
}

void ResidentRabcContext::release(tensor::Swapper& swapper)
{
  for (auto const& matrix : owned_mats)
  {
    swapper.clear(matrix);
  }
  owned_mats.clear();
  first_stage_products.clear();
  assembled_temporaries.clear();
}

EffectiveHamiltonianOperator::Impl::~Impl()
{
  if (resident_context != nullptr && swapper != nullptr)
  {
    resident_context->release(*swapper);
  }
}

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

int checked_blas_vector_size(std::size_t value)
{
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    throw std::length_error("TensorContraction BLAS vector length must fit in int");
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

std::size_t matrix_element_count(std::span<tensor::Matrix const> mats, std::size_t begin, std::size_t end)
{
  std::size_t count = 0;
  for (std::size_t index = begin; index < end; ++index)
  {
    count += mats[index].size();
  }
  return count;
}

std::size_t matrix_element_offset(std::span<tensor::Matrix const> mats, std::size_t end)
{
  return matrix_element_count(mats, 0, end);
}

std::size_t matrix_byte_count(std::span<tensor::Matrix const> mats)
{
  std::size_t count = 0;
  for (auto const& mat : mats)
  {
    count += mat.sizeInByte();
  }
  return count;
}

std::vector<tensor::Matrix> matrix_range(std::span<tensor::Matrix const> mats, ResidentOutputPlacementRange range)
{
  return {mats.begin() + static_cast<std::ptrdiff_t>(range.begin),
          mats.begin() + static_cast<std::ptrdiff_t>(range.end)};
}

std::span<double const> const_value_range(std::span<double const> values, std::span<tensor::Matrix const> mats,
                                          ResidentOutputPlacementRange range)
{
  auto const offset = matrix_element_offset(mats, range.begin);
  auto const count = matrix_element_count(mats, range.begin, range.end);
  return values.subspan(offset, count);
}

std::span<double> value_range(std::span<double> values, std::span<tensor::Matrix const> mats,
                              ResidentOutputPlacementRange range)
{
  auto const offset = matrix_element_offset(mats, range.begin);
  auto const count = matrix_element_count(mats, range.begin, range.end);
  return values.subspan(offset, count);
}

bool localize_rabc_range(tensor::Swapper& swapper, std::span<tensor::Matrix const> mats, std::span<double const> values,
                         ResidentOutputPlacementRange range, bool upload_from_host, bool refresh_existing)
{
  auto const group = matrix_range(mats, range);
  auto const group_values = const_value_range(values, mats, range);
  bool const any_existing = swapper.anyPreStoreBuffer(group);
  auto const existing_device = swapper.commonPreStoreDevice(group);

  if (!any_existing)
  {
    if (upload_from_host)
    {
      swapper.uploadHostMatricesCoalesced(group, group_values, range.device);
    }
    else
    {
      swapper.registerGpuAllocationsCoalesced(group, range.device);
    }
    return true;
  }

  if (!existing_device.has_value() || *existing_device != range.device ||
      !swapper.preStoreBuffersAreCoalesced(group, range.device))
  {
    return false;
  }

  if (upload_from_host && refresh_existing)
  {
    return swapper.refreshHostMatricesToDeviceCoalesced(group, group_values, range.device);
  }
  return true;
}

void localize_rabc_individual(tensor::Swapper& swapper, std::span<tensor::Matrix const> mats, bool upload_from_host,
                              bool refresh_existing)
{
  auto const device_count = swapper.getDeviceCount();
  std::vector<std::size_t> bytes_per_device(static_cast<std::size_t>(device_count), 0);
  for (auto const& mat : mats)
  {
    auto [device_id, buffer] = swapper.getPreStoreBufferOrNone(mat);
    if (buffer != nullptr)
    {
      if (upload_from_host && refresh_existing)
      {
        swapper.refreshHostMatrixToDevice(mat.hostView());
      }
      continue;
    }

    auto const device =
        static_cast<int>(std::min_element(bytes_per_device.begin(), bytes_per_device.end()) - bytes_per_device.begin());
    if (upload_from_host)
    {
      swapper.uploadHostMatrix(mat.hostView(), device);
    }
    else
    {
      swapper.registerGpuAllocation(mat, device);
    }
    bytes_per_device[static_cast<std::size_t>(device)] += mat.sizeInByte();
  }
}

void localize_rabc_family(tensor::Swapper& swapper, std::span<tensor::Matrix const> mats,
                          std::span<double const> values, bool upload_from_host, bool refresh_existing)
{
  swapper.initMemPools();
  if (mats.empty())
  {
    return;
  }

  auto const total_bytes = matrix_byte_count(mats);
  if (total_bytes == values.size_bytes())
  {
    auto const ranges = default_byte_balanced_ranges(mats, swapper.getDeviceCount());
    bool localized = true;
    for (auto const range : ranges)
    {
      if (!localize_rabc_range(swapper, mats, values, range, upload_from_host, refresh_existing))
      {
        localized = false;
        break;
      }
    }
    if (localized)
    {
      CUDA_CALL(cudaSetDevice(0));
      return;
    }
  }

  localize_rabc_individual(swapper, mats, upload_from_host, refresh_existing);
  CUDA_CALL(cudaSetDevice(0));
}

bool synchronize_rabc_range_to_host(tensor::Swapper& swapper, std::span<tensor::Matrix const> mats,
                                    std::span<double> values, ResidentOutputPlacementRange range,
                                    std::vector<bool>& touched)
{
  auto const group = matrix_range(mats, range);
  if (!swapper.downloadDeviceMatricesToHostCoalesced(group, value_range(values, mats, range)))
  {
    return false;
  }
  auto const device = swapper.commonPreStoreDevice(group);
  if (device.has_value())
  {
    touched[static_cast<std::size_t>(*device)] = true;
  }
  return true;
}

void synchronize_rabc_family_to_host(tensor::Swapper& swapper, std::span<tensor::Matrix const> mats,
                                     std::span<double> values)
{
  swapper.initMemPools();
  auto const device_count = swapper.getDeviceCount();
  std::vector<bool> touched(static_cast<std::size_t>(device_count), false);
  auto const total_bytes = matrix_byte_count(mats);

  if (!mats.empty() && total_bytes == values.size_bytes())
  {
    auto const all_mats = matrix_range(mats, ResidentOutputPlacementRange{.device = 0, .begin = 0, .end = mats.size()});
    if (swapper.downloadDeviceMatricesToHostCoalesced(all_mats, values))
    {
      if (auto const device = swapper.commonPreStoreDevice(all_mats); device.has_value())
      {
        touched[static_cast<std::size_t>(*device)] = true;
      }
    }
    else
    {
      bool synchronized = true;
      for (auto const range : default_byte_balanced_ranges(mats, device_count))
      {
        if (!synchronize_rabc_range_to_host(swapper, mats, values, range, touched))
        {
          synchronized = false;
          break;
        }
      }
      if (!synchronized)
      {
        std::fill(touched.begin(), touched.end(), false);
        for (auto const& mat : mats)
        {
          auto const [device, buffer] = swapper.getPreStoreBufferOrNone(mat);
          if (buffer == nullptr)
          {
            continue;
          }
          swapper.downloadDeviceToHost(mat.hostView());
          touched[static_cast<std::size_t>(device)] = true;
        }
      }
    }
  }
  else
  {
    for (auto const& mat : mats)
    {
      auto const [device, buffer] = swapper.getPreStoreBufferOrNone(mat);
      if (buffer == nullptr)
      {
        continue;
      }
      swapper.downloadDeviceToHost(mat.hostView());
      touched[static_cast<std::size_t>(device)] = true;
    }
  }

  for (int device = 0; device < device_count; ++device)
  {
    if (touched[static_cast<std::size_t>(device)])
    {
      swapper.syncMemStream(device, "rabc_sync_to_host");
    }
  }
  CUDA_CALL(cudaSetDevice(0));
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
    std::size_t bc_gemms = 0;
    std::size_t final_gemms = 0;
    std::size_t direct_final_gemms = 0;
    std::size_t accumulation_groups = 0;
    std::size_t accumulation_terms = 0;
    std::size_t source_accumulation_groups = 0;
    std::size_t source_accumulation_terms = 0;
    std::size_t output_accumulation_groups = 0;
    std::size_t output_accumulation_terms = 0;
    std::size_t max_source_fan_in = 0;
    std::size_t max_output_fan_in = 0;
    std::size_t max_accumulation_fan_in = 0;
    std::size_t source_axpys = 0;
    std::size_t output_axpys = 0;
    std::size_t zero_fills = 0;
    std::size_t intermediate_matrices = 0;
    std::size_t temporary_matrices = 0;
    std::size_t temporary_peer_requests = 0;
    std::size_t temporary_peer_copies = 0;
    double bc_flops = 0.0;
    double accumulate_flops = 0.0;
    double temporary_accumulate_flops = 0.0;
    std::size_t b_local_bytes = 0;
    std::size_t b_peer_bytes = 0;
    std::size_t temporary_peer_request_bytes = 0;
    std::size_t temporary_peer_bytes = 0;
    std::size_t a_bytes = 0;
    std::size_t c_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t intermediate_bytes = 0;
    double intermediate_gemm_enqueue_s = 0.0;
    double final_gemm_enqueue_s = 0.0;
    double source_accumulation_enqueue_s = 0.0;
    double output_accumulation_enqueue_s = 0.0;
    double zero_fill_enqueue_s = 0.0;
};

struct RabcCostFeatures
{
    std::vector<int> input_devices;
    std::vector<int> output_devices;
    std::vector<RabcDeviceCostFeatures> devices;
};

struct RabcExecutionStats
{
    std::vector<RabcDeviceCostFeatures> devices;

    void reset(int device_count)
    {
      devices.assign(static_cast<std::size_t>(device_count), RabcDeviceCostFeatures{});
      for (int device = 0; device < device_count; ++device)
      {
        devices[static_cast<std::size_t>(device)].device = device;
      }
    }

    [[nodiscard]] RabcDeviceCostFeatures& device(int device_id) { return devices[static_cast<std::size_t>(device_id)]; }
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

struct RightFirstIntermediateKey
{
    int device_id = 0;
    int b = 0;
    int c = 0;

    auto operator<=>(RightFirstIntermediateKey const&) const = default;
};

struct RightFirstAccumulationKey
{
    int r = 0;
    int a = 0;

    auto operator<=>(RightFirstAccumulationKey const&) const = default;
};

struct TemporaryMigrationKey
{
    char kind = 'Y';
    int source_device = 0;
    int target_device = 0;
    int first = 0;
    int second = 0;

    auto operator<=>(TemporaryMigrationKey const&) const = default;
};

struct RightFirstTermWork
{
    EffectiveHamiltonianOperator::Term const* term = nullptr;
    RightFirstIntermediateKey intermediate;
};

auto right_first_work_by_accumulation(std::span<EffectiveHamiltonianOperator::Term const> terms,
                                      std::span<int const> input_devices)
    -> std::map<RightFirstAccumulationKey, std::vector<RightFirstTermWork>>
{
  std::map<RightFirstAccumulationKey, std::vector<RightFirstTermWork>> groups;
  for (auto const& term : terms)
  {
    RightFirstAccumulationKey const key{.r = checked_index(term.r), .a = checked_index(term.a)};
    groups[key].push_back(RightFirstTermWork{
        .term = &term,
        .intermediate = RightFirstIntermediateKey{
            .device_id = input_devices[term.b], .b = checked_index(term.b), .c = checked_index(term.c)}});
  }
  return groups;
}

auto combine_accumulation_inputs(std::span<RightFirstTermWork const> terms)
    -> std::vector<std::pair<RightFirstIntermediateKey, double>>
{
  std::map<RightFirstIntermediateKey, double> coefficients;
  for (auto const& work : terms)
  {
    coefficients[work.intermediate] += work.term->coefficient;
  }

  std::vector<std::pair<RightFirstIntermediateKey, double>> result;
  result.reserve(coefficients.size());
  for (auto const& [key, coefficient] : coefficients)
  {
    if (coefficient != 0.0)
    {
      result.emplace_back(key, coefficient);
    }
  }
  return result;
}

auto make_temporary_matrix(std::size_t rows, std::size_t cols) -> tensor::Matrix
{
  return tensor::Matrix(nullptr, checked_index(rows), checked_index(cols));
}

auto make_resident_rabc_context(VariableFamily variable_family, std::span<tensor::Matrix const> r_mats,
                                std::span<tensor::Matrix const> b_raw_mats, MatrixFamily const& c_mats,
                                std::span<EffectiveHamiltonianOperator::Term const> terms,
                                tensor::Swapper& swapper) -> std::unique_ptr<ResidentRabcContext>
{
  auto context = std::make_unique<ResidentRabcContext>();
  context->variable_family = variable_family;
  context->device_count = swapper.getDeviceCount();
  context->input_count = b_raw_mats.size();
  context->output_count = r_mats.size();
  context->term_count = terms.size();
  context->input_devices = resident_devices_for(swapper, b_raw_mats);
  context->output_devices = resident_devices_for(swapper, r_mats);

  auto const groups = right_first_work_by_accumulation(terms, context->input_devices);
  std::map<RightFirstIntermediateKey, std::size_t> intermediate_indices;
  for (auto const& [_, group_terms] : groups)
  {
    for (auto const& work : group_terms)
    {
      if (intermediate_indices.contains(work.intermediate))
      {
        continue;
      }

      auto const& b_mat = b_raw_mats[static_cast<std::size_t>(work.intermediate.b)];
      auto const c_block = c_mats.block(static_cast<std::size_t>(work.intermediate.c));
      auto intermediate = make_temporary_matrix(b_mat.getFirstDim(), c_block.cols);
      swapper.registerGpuAllocation(intermediate, work.intermediate.device_id);
      auto const index = context->first_stage_products.size();
      intermediate_indices.emplace(work.intermediate, index);
      context->first_stage_products.push_back(ResidentRabcContext::FirstStageProductPlan{
          .device_id = work.intermediate.device_id,
          .b = static_cast<std::size_t>(work.intermediate.b),
          .c = static_cast<std::size_t>(work.intermediate.c),
          .matrix = intermediate,
      });
      context->owned_mats.push_back(intermediate);
    }
  }

  for (auto const& [key, group_terms] : groups)
  {
    auto combined =
        combine_accumulation_inputs(std::span<RightFirstTermWork const>(group_terms.data(), group_terms.size()));
    if (combined.empty())
    {
      continue;
    }

    ResidentRabcContext::AssembledTemporaryPlan plan;
    plan.r = static_cast<std::size_t>(key.r);
    plan.a = static_cast<std::size_t>(key.a);

    std::map<int, std::vector<std::pair<RightFirstIntermediateKey, double>>> source_groups;
    for (auto const& input : combined)
    {
      source_groups[input.first.device_id].push_back(input);
    }

    for (auto const& [source_device, source_inputs] : source_groups)
    {
      ResidentRabcContext::LocalPartialPlan source_plan;
      source_plan.device_id = source_device;
      source_plan.inputs.reserve(source_inputs.size());
      for (auto const& [intermediate_key, coefficient] : source_inputs)
      {
        source_plan.inputs.push_back(ResidentRabcContext::LocalPartialInput{
            .first_stage_product = intermediate_indices.at(intermediate_key),
            .coefficient = coefficient,
        });
      }
      if (source_plan.inputs.size() > 1)
      {
        auto const& source_matrix =
            context->first_stage_products[source_plan.inputs.front().first_stage_product].matrix;
        auto q_matrix = make_temporary_matrix(source_matrix.getFirstDim(), source_matrix.getSecondDim());
        swapper.registerGpuAllocation(q_matrix, source_device);
        source_plan.accumulation = q_matrix;
        context->owned_mats.push_back(q_matrix);
      }
      plan.local_partials.push_back(std::move(source_plan));
    }

    if (plan.local_partials.size() > 1)
    {
      auto const first_intermediate = plan.local_partials.front().inputs.front().first_stage_product;
      auto const& source_matrix = context->first_stage_products[first_intermediate].matrix;
      auto const target_device = context->output_devices.at(plan.r);
      auto q_matrix = make_temporary_matrix(source_matrix.getFirstDim(), source_matrix.getSecondDim());
      swapper.registerGpuAllocation(q_matrix, target_device);
      plan.output_accumulation = q_matrix;
      context->owned_mats.push_back(q_matrix);
    }

    context->assembled_temporaries.push_back(std::move(plan));
  }

  return context;
}

void stage_right_first_dependencies(ResidentRabcContext const& context, std::span<tensor::Matrix const> a_raw_mats,
                                    std::span<tensor::Matrix const> c_raw_mats,
                                    std::span<EffectiveHamiltonianOperator::Term const> terms, tensor::Swapper& swapper)
{
  std::set<std::pair<std::size_t, int>> staged_c;
  std::set<std::pair<std::size_t, int>> staged_a;
  for (auto const& term : terms)
  {
    auto const first_device = context.input_devices.at(term.b);
    auto const output_device = context.output_devices.at(term.r);
    if (staged_c.emplace(term.c, first_device).second)
    {
      static_cast<void>(swapper.ensureLocalCopy(c_raw_mats[term.c], first_device));
    }
    if (staged_a.emplace(term.a, output_device).second)
    {
      static_cast<void>(swapper.ensureLocalCopy(a_raw_mats[term.a], output_device));
    }
  }
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
  std::set<TemporaryMigrationKey> migrated_temporaries;

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
    ++output_features.terms;

    BcUseKey const bc{.b = term.b, .c = term.c};
    if (staged_bc[first_device_index].insert(bc).second)
    {
      auto const b = b_mats.block(term.b);
      auto const c = c_mats.block(term.c);
      ++first_features.bc_gemms;
      ++first_features.intermediate_matrices;
      first_features.bc_flops += gemm_flops(b, c);
      first_features.intermediate_bytes += b.rows * c.cols * sizeof(double);
    }

    auto const a = a_mats.block(term.a);

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
      auto const c = c_mats.block(term.c);
      ++first_features.unique_c;
      first_features.c_bytes += c.rows * c.cols * sizeof(double);
    }
  }

  auto groups = right_first_work_by_accumulation(terms, features.input_devices);
  for (auto const& [key, group_terms] : groups)
  {
    if (group_terms.empty())
    {
      continue;
    }

    int const output_device = features.output_devices[static_cast<std::size_t>(key.r)];
    auto& output_features = features.devices[static_cast<std::size_t>(output_device)];
    auto combined =
        combine_accumulation_inputs(std::span<RightFirstTermWork const>(group_terms.data(), group_terms.size()));
    if (combined.empty())
    {
      continue;
    }

    auto const a = a_mats.block(static_cast<std::size_t>(key.a));
    auto const first_intermediate = combined.front().first;
    auto const b = b_mats.block(static_cast<std::size_t>(first_intermediate.b));
    auto const c = c_mats.block(static_cast<std::size_t>(first_intermediate.c));
    auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
    ++output_features.final_gemms;
    output_features.accumulate_flops += gemm_flops(a, intermediate);
    struct FeaturePartial
    {
        int device_id = 0;
        std::size_t bytes = 0;
        TemporaryMigrationKey migration_key;
    };

    std::map<int, std::vector<std::pair<RightFirstIntermediateKey, double>>> source_groups;
    for (auto const& input : combined)
    {
      source_groups[input.first.device_id].push_back(input);
    }

    std::vector<FeaturePartial> partials;
    partials.reserve(source_groups.size());
    for (auto const& [source_device, source_inputs] : source_groups)
    {
      if (source_inputs.size() == 1)
      {
        auto const& intermediate_key = source_inputs.front().first;
        auto const source_b = b_mats.block(static_cast<std::size_t>(intermediate_key.b));
        auto const source_c = c_mats.block(static_cast<std::size_t>(intermediate_key.c));
        auto const bytes = source_b.rows * source_c.cols * sizeof(double);
        partials.push_back(FeaturePartial{
            .device_id = source_device,
            .bytes = bytes,
            .migration_key =
                TemporaryMigrationKey{
                    .kind = 'Y',
                    .source_device = source_device,
                    .target_device = output_device,
                    .first = intermediate_key.b,
                    .second = intermediate_key.c,
                },
        });
        continue;
      }

      auto& source_features = features.devices[static_cast<std::size_t>(source_device)];
      ++source_features.accumulation_groups;
      ++source_features.source_accumulation_groups;
      source_features.accumulation_terms += source_inputs.size();
      source_features.source_accumulation_terms += source_inputs.size();
      source_features.max_source_fan_in = std::max(source_features.max_source_fan_in, source_inputs.size());
      source_features.max_accumulation_fan_in = std::max(source_features.max_accumulation_fan_in, source_inputs.size());
      source_features.source_axpys += source_inputs.size();
      ++source_features.zero_fills;
      ++source_features.temporary_matrices;
      std::size_t source_bytes = 0;
      for (auto const& [intermediate_key, coefficient] : source_inputs)
      {
        static_cast<void>(coefficient);
        auto const source_b = b_mats.block(static_cast<std::size_t>(intermediate_key.b));
        auto const source_c = c_mats.block(static_cast<std::size_t>(intermediate_key.c));
        source_bytes = source_b.rows * source_c.cols * sizeof(double);
        source_features.temporary_accumulate_flops += 2.0 * static_cast<double>(source_bytes / sizeof(double));
      }
      partials.push_back(FeaturePartial{
          .device_id = source_device,
          .bytes = source_bytes,
          .migration_key =
              TemporaryMigrationKey{
                  .kind = 'Q',
                  .source_device = source_device,
                  .target_device = output_device,
                  .first = key.r,
                  .second = key.a,
              },
      });
    }

    if (partials.size() == 1)
    {
      ++output_features.direct_final_gemms;
    }
    else
    {
      ++output_features.accumulation_groups;
      ++output_features.output_accumulation_groups;
      output_features.accumulation_terms += partials.size();
      output_features.output_accumulation_terms += partials.size();
      output_features.max_output_fan_in = std::max(output_features.max_output_fan_in, partials.size());
      output_features.max_accumulation_fan_in = std::max(output_features.max_accumulation_fan_in, partials.size());
      output_features.output_axpys += partials.size();
      ++output_features.zero_fills;
      ++output_features.temporary_matrices;
      for (auto const& partial : partials)
      {
        output_features.temporary_accumulate_flops += 2.0 * static_cast<double>(partial.bytes / sizeof(double));
      }
    }

    for (auto const& partial : partials)
    {
      if (partial.device_id == output_device)
      {
        continue;
      }
      ++output_features.temporary_peer_requests;
      output_features.temporary_peer_request_bytes += partial.bytes;
      if (migrated_temporaries.insert(partial.migration_key).second)
      {
        ++output_features.temporary_peer_copies;
        output_features.temporary_peer_bytes += partial.bytes;
      }
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
    result += fmt::format(
        "{{\"device\":{},\"input_blocks\":{},\"output_blocks\":{},\"terms\":{},"
        "\"unique_bc\":{},\"unique_a\":{},\"unique_b\":{},\"unique_c\":{},"
        "\"bc_gemms\":{},"
        "\"final_gemms\":{},\"direct_final_gemms\":{},"
        "\"accumulation_groups\":{},\"accumulation_terms\":{},"
        "\"source_accumulation_groups\":{},\"source_accumulation_terms\":{},"
        "\"output_accumulation_groups\":{},\"output_accumulation_terms\":{},"
        "\"max_source_fan_in\":{},\"max_output_fan_in\":{},\"max_accumulation_fan_in\":{},"
        "\"source_axpys\":{},\"output_axpys\":{},\"zero_fills\":{},"
        "\"intermediate_matrices\":{},\"temporary_matrices\":{},"
        "\"temporary_peer_requests\":{},\"temporary_peer_copies\":{},"
        "\"bc_flops\":{:.17g},\"accumulate_flops\":{:.17g},"
        "\"temporary_accumulate_flops\":{:.17g},"
        "\"b_local_bytes\":{},\"b_peer_bytes\":{},"
        "\"temporary_peer_request_bytes\":{},\"temporary_peer_bytes\":{},"
        "\"a_bytes\":{},\"c_bytes\":{},"
        "\"output_bytes\":{},\"intermediate_bytes\":{},"
        "\"intermediate_gemm_enqueue_s\":{:.17g},"
        "\"final_gemm_enqueue_s\":{:.17g},"
        "\"source_accumulation_enqueue_s\":{:.17g},"
        "\"output_accumulation_enqueue_s\":{:.17g},"
        "\"zero_fill_enqueue_s\":{:.17g}}}",
        item.device, item.input_blocks, item.output_blocks, item.terms, item.unique_bc, item.unique_a, item.unique_b,
        item.unique_c, item.bc_gemms, item.final_gemms, item.direct_final_gemms, item.accumulation_groups,
        item.accumulation_terms, item.source_accumulation_groups, item.source_accumulation_terms,
        item.output_accumulation_groups, item.output_accumulation_terms, item.max_source_fan_in, item.max_output_fan_in,
        item.max_accumulation_fan_in, item.source_axpys, item.output_axpys, item.zero_fills, item.intermediate_matrices,
        item.temporary_matrices, item.temporary_peer_requests, item.temporary_peer_copies, item.bc_flops,
        item.accumulate_flops, item.temporary_accumulate_flops, item.b_local_bytes, item.b_peer_bytes,
        item.temporary_peer_request_bytes, item.temporary_peer_bytes, item.a_bytes, item.c_bytes, item.output_bytes,
        item.intermediate_bytes, item.intermediate_gemm_enqueue_s, item.final_gemm_enqueue_s,
        item.source_accumulation_enqueue_s, item.output_accumulation_enqueue_s, item.zero_fill_enqueue_s);
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

std::uint64_t counter_delta(std::uint64_t stop, std::uint64_t start) { return stop >= start ? stop - start : 0; }

tensor::Swapper::RuntimeCounters subtract_runtime_counters(tensor::Swapper::RuntimeCounters const& stop,
                                                           tensor::Swapper::RuntimeCounters const& start)
{
  return tensor::Swapper::RuntimeCounters{
      .h2dCopies = counter_delta(stop.h2dCopies, start.h2dCopies),
      .h2dBytes = counter_delta(stop.h2dBytes, start.h2dBytes),
      .d2hCopies = counter_delta(stop.d2hCopies, start.d2hCopies),
      .d2hBytes = counter_delta(stop.d2hBytes, start.d2hBytes),
      .d2dCopies = counter_delta(stop.d2dCopies, start.d2dCopies),
      .d2dBytes = counter_delta(stop.d2dBytes, start.d2dBytes),
      .peerCopies = counter_delta(stop.peerCopies, start.peerCopies),
      .peerBytes = counter_delta(stop.peerBytes, start.peerBytes),
      .ensureLocalPeerCopies = counter_delta(stop.ensureLocalPeerCopies, start.ensureLocalPeerCopies),
      .ensureLocalPeerBytes = counter_delta(stop.ensureLocalPeerBytes, start.ensureLocalPeerBytes),
      .preStoreRelocateD2dCopies = counter_delta(stop.preStoreRelocateD2dCopies, start.preStoreRelocateD2dCopies),
      .preStoreRelocateD2dBytes = counter_delta(stop.preStoreRelocateD2dBytes, start.preStoreRelocateD2dBytes),
      .preStoreRelocatePeerCopies = counter_delta(stop.preStoreRelocatePeerCopies, start.preStoreRelocatePeerCopies),
      .preStoreRelocatePeerBytes = counter_delta(stop.preStoreRelocatePeerBytes, start.preStoreRelocatePeerBytes),
      .syncBufferPeerCopies = counter_delta(stop.syncBufferPeerCopies, start.syncBufferPeerCopies),
      .syncBufferPeerBytes = counter_delta(stop.syncBufferPeerBytes, start.syncBufferPeerBytes),
      .cudaEventCreate = counter_delta(stop.cudaEventCreate, start.cudaEventCreate),
      .cudaEventRecord = counter_delta(stop.cudaEventRecord, start.cudaEventRecord),
      .cudaEventWait = counter_delta(stop.cudaEventWait, start.cudaEventWait),
      .cudaEventQuery = counter_delta(stop.cudaEventQuery, start.cudaEventQuery),
      .cudaEventDestroy = counter_delta(stop.cudaEventDestroy, start.cudaEventDestroy),
      .cudaStreamSync = counter_delta(stop.cudaStreamSync, start.cudaStreamSync),
      .cudaAsyncFree = counter_delta(stop.cudaAsyncFree, start.cudaAsyncFree),
      .cudaAsyncFreeReclaim = counter_delta(stop.cudaAsyncFreeReclaim, start.cudaAsyncFreeReclaim),
      .cudaAsyncFreePoll = counter_delta(stop.cudaAsyncFreePoll, start.cudaAsyncFreePoll),
      .cudaPoolCacheHit = counter_delta(stop.cudaPoolCacheHit, start.cudaPoolCacheHit),
      .cudaPoolCacheMiss = counter_delta(stop.cudaPoolCacheMiss, start.cudaPoolCacheMiss),
      .cudaPoolCacheStore = counter_delta(stop.cudaPoolCacheStore, start.cudaPoolCacheStore),
      .cudaPoolCacheBypass = counter_delta(stop.cudaPoolCacheBypass, start.cudaPoolCacheBypass),
      .cudaPoolCacheRelease = counter_delta(stop.cudaPoolCacheRelease, start.cudaPoolCacheRelease),
  };
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

std::string json_runtime_counters(tensor::Swapper::RuntimeCounters const& counters)
{
  return fmt::format(
      "{{\"h2d_copies\":{},\"h2d_bytes\":{},\"d2h_copies\":{},\"d2h_bytes\":{},"
      "\"d2d_copies\":{},\"d2d_bytes\":{},\"peer_copies\":{},\"peer_bytes\":{},"
      "\"ensure_local_peer_copies\":{},\"ensure_local_peer_bytes\":{},"
      "\"pre_store_relocate_d2d_copies\":{},\"pre_store_relocate_d2d_bytes\":{},"
      "\"pre_store_relocate_peer_copies\":{},\"pre_store_relocate_peer_bytes\":{},"
      "\"sync_buffer_peer_copies\":{},\"sync_buffer_peer_bytes\":{},"
      "\"cuda_event_create\":{},\"cuda_event_record\":{},\"cuda_event_wait\":{},"
      "\"cuda_event_query\":{},\"cuda_event_destroy\":{},\"cuda_stream_sync\":{},"
      "\"cuda_async_free\":{},\"cuda_async_free_reclaim\":{},\"cuda_async_free_poll\":{},"
      "\"cuda_pool_cache_hit\":{},\"cuda_pool_cache_miss\":{},\"cuda_pool_cache_store\":{},"
      "\"cuda_pool_cache_bypass\":{},\"cuda_pool_cache_release\":{}}}",
      counters.h2dCopies, counters.h2dBytes, counters.d2hCopies, counters.d2hBytes, counters.d2dCopies,
      counters.d2dBytes, counters.peerCopies, counters.peerBytes, counters.ensureLocalPeerCopies,
      counters.ensureLocalPeerBytes, counters.preStoreRelocateD2dCopies, counters.preStoreRelocateD2dBytes,
      counters.preStoreRelocatePeerCopies, counters.preStoreRelocatePeerBytes, counters.syncBufferPeerCopies,
      counters.syncBufferPeerBytes, counters.cudaEventCreate, counters.cudaEventRecord, counters.cudaEventWait,
      counters.cudaEventQuery, counters.cudaEventDestroy, counters.cudaStreamSync, counters.cudaAsyncFree,
      counters.cudaAsyncFreeReclaim, counters.cudaAsyncFreePoll, counters.cudaPoolCacheHit, counters.cudaPoolCacheMiss,
      counters.cudaPoolCacheStore, counters.cudaPoolCacheBypass, counters.cudaPoolCacheRelease);
}

std::size_t layout_transitions(std::span<int const> devices)
{
  if (devices.empty())
  {
    return 0;
  }

  std::size_t transitions = 0;
  for (std::size_t index = 1; index < devices.size(); ++index)
  {
    if (devices[index] != devices[index - 1])
    {
      ++transitions;
    }
  }
  return transitions;
}

std::size_t active_device_count(std::span<int const> devices, std::size_t device_count)
{
  std::vector<bool> active(device_count, false);
  for (int device : devices)
  {
    if (device >= 0 && static_cast<std::size_t>(device) < active.size())
    {
      active[static_cast<std::size_t>(device)] = true;
    }
  }
  return static_cast<std::size_t>(std::ranges::count(active, true));
}

double max_block_fraction(std::span<int const> devices, std::size_t device_count)
{
  if (devices.empty())
  {
    return 0.0;
  }

  std::vector<std::size_t> counts(device_count, 0);
  for (int device : devices)
  {
    if (device >= 0 && static_cast<std::size_t>(device) < counts.size())
    {
      ++counts[static_cast<std::size_t>(device)];
    }
  }
  auto const max_count = *std::ranges::max_element(counts);
  return static_cast<double>(max_count) / static_cast<double>(devices.size());
}

double max_byte_fraction(std::span<int const> devices, std::span<tensor::Matrix const> mats, std::size_t device_count)
{
  std::size_t total_bytes = 0;
  std::vector<std::size_t> bytes(device_count, 0);
  for (std::size_t index = 0; index < devices.size() && index < mats.size(); ++index)
  {
    auto const matrix_bytes = mats[index].sizeInByte();
    total_bytes += matrix_bytes;
    auto const device = devices[index];
    if (device >= 0 && static_cast<std::size_t>(device) < bytes.size())
    {
      bytes[static_cast<std::size_t>(device)] += matrix_bytes;
    }
  }
  if (total_bytes == 0)
  {
    return 0.0;
  }
  auto const max_bytes = *std::ranges::max_element(bytes);
  return static_cast<double>(max_bytes) / static_cast<double>(total_bytes);
}

std::string json_one_layout_metrics(std::string_view name, std::span<int const> devices,
                                    std::span<tensor::Matrix const> mats, std::size_t device_count)
{
  auto const transitions = layout_transitions(devices);
  return fmt::format("\"{}\":{{\"blocks\":{},\"segments\":{},\"transitions\":{},\"active_devices\":{},"
                     "\"max_block_fraction\":{:.17g},\"max_byte_fraction\":{:.17g}}}",
                     name, devices.size(), devices.empty() ? 0 : transitions + 1, transitions,
                     active_device_count(devices, device_count), max_block_fraction(devices, device_count),
                     max_byte_fraction(devices, mats, device_count));
}

std::string json_layout_metrics(RabcCostFeatures const& features, std::span<tensor::Matrix const> b_raw_mats,
                                std::span<tensor::Matrix const> r_raw_mats)
{
  auto const device_count = features.devices.size();
  return fmt::format("{{{},{}}}", json_one_layout_metrics("input", features.input_devices, b_raw_mats, device_count),
                     json_one_layout_metrics("output", features.output_devices, r_raw_mats, device_count));
}

std::array<double, 5> quantile_summary(std::vector<double> values)
{
  if (values.empty())
  {
    return {0.0, 0.0, 0.0, 0.0, 0.0};
  }
  std::ranges::sort(values);
  auto const pick = [&](double fraction) {
    auto const index = static_cast<std::size_t>(
        std::clamp(fraction * static_cast<double>(values.size() - 1), 0.0, static_cast<double>(values.size() - 1)));
    return values[index];
  };
  return {values.front(), pick(0.5), pick(0.9), pick(0.99), values.back()};
}

std::string json_quantile_summary(std::vector<double> values)
{
  auto const summary = quantile_summary(std::move(values));
  return fmt::format("{{\"min\":{:.17g},\"q50\":{:.17g},\"q90\":{:.17g},\"q99\":{:.17g},\"max\":{:.17g}}}", summary[0],
                     summary[1], summary[2], summary[3], summary[4]);
}

std::string json_shape_summary(MatrixFamily const& a_mats, MatrixFamily const& b_mats, MatrixFamily const& c_mats,
                               std::span<tensor::Matrix const> r_raw_mats,
                               std::span<EffectiveHamiltonianOperator::Term const> terms)
{
  std::set<std::array<std::size_t, 8>> unique_shapes;
  std::vector<double> bc_flops;
  std::vector<double> accumulate_flops;
  std::vector<double> intermediate_bytes;
  std::vector<double> r_values;
  std::vector<double> b_values;
  bc_flops.reserve(terms.size());
  accumulate_flops.reserve(terms.size());
  intermediate_bytes.reserve(terms.size());
  r_values.reserve(terms.size());
  b_values.reserve(terms.size());

  for (auto const& term : terms)
  {
    auto const a = a_mats.block(term.a);
    auto const b = b_mats.block(term.b);
    auto const c = c_mats.block(term.c);
    auto const& r = r_raw_mats[term.r];
    auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
    unique_shapes.insert({static_cast<std::size_t>(r.getFirstDim()), static_cast<std::size_t>(r.getSecondDim()), a.rows,
                          a.cols, b.rows, b.cols, c.rows, c.cols});
    bc_flops.push_back(gemm_flops(b, c));
    accumulate_flops.push_back(gemm_flops(a, intermediate));
    intermediate_bytes.push_back(static_cast<double>(intermediate.rows * intermediate.cols * sizeof(double)));
    r_values.push_back(static_cast<double>(r.getFirstDim()) * static_cast<double>(r.getSecondDim()));
    b_values.push_back(static_cast<double>(b.rows) * static_cast<double>(b.cols));
  }

  return fmt::format("{{\"unique_term_shapes\":{},\"bc_flops\":{},\"accumulate_flops\":{},"
                     "\"intermediate_bytes\":{},\"r_values\":{},\"b_values\":{}}}",
                     unique_shapes.size(), json_quantile_summary(std::move(bc_flops)),
                     json_quantile_summary(std::move(accumulate_flops)),
                     json_quantile_summary(std::move(intermediate_bytes)), json_quantile_summary(std::move(r_values)),
                     json_quantile_summary(std::move(b_values)));
}

std::string json_feature_summary(std::span<RabcDeviceCostFeatures const> devices)
{
  RabcDeviceCostFeatures totals;
  RabcDeviceCostFeatures maximums;
  for (auto const& device : devices)
  {
    totals.terms += device.terms;
    totals.bc_gemms += device.bc_gemms;
    totals.final_gemms += device.final_gemms;
    totals.source_axpys += device.source_axpys;
    totals.output_axpys += device.output_axpys;
    totals.zero_fills += device.zero_fills;
    totals.temporary_peer_requests += device.temporary_peer_requests;
    totals.temporary_peer_copies += device.temporary_peer_copies;
    totals.temporary_peer_bytes += device.temporary_peer_bytes;
    totals.intermediate_bytes += device.intermediate_bytes;
    totals.output_bytes += device.output_bytes;
    totals.bc_flops += device.bc_flops;
    totals.accumulate_flops += device.accumulate_flops;
    totals.intermediate_gemm_enqueue_s += device.intermediate_gemm_enqueue_s;
    totals.final_gemm_enqueue_s += device.final_gemm_enqueue_s;
    totals.source_accumulation_enqueue_s += device.source_accumulation_enqueue_s;
    totals.output_accumulation_enqueue_s += device.output_accumulation_enqueue_s;
    totals.zero_fill_enqueue_s += device.zero_fill_enqueue_s;

    maximums.terms = std::max(maximums.terms, device.terms);
    maximums.unique_bc = std::max(maximums.unique_bc, device.unique_bc);
    maximums.bc_gemms = std::max(maximums.bc_gemms, device.bc_gemms);
    maximums.final_gemms = std::max(maximums.final_gemms, device.final_gemms);
    maximums.max_source_fan_in = std::max(maximums.max_source_fan_in, device.max_source_fan_in);
    maximums.max_output_fan_in = std::max(maximums.max_output_fan_in, device.max_output_fan_in);
    maximums.max_accumulation_fan_in = std::max(maximums.max_accumulation_fan_in, device.max_accumulation_fan_in);
    maximums.temporary_peer_bytes = std::max(maximums.temporary_peer_bytes, device.temporary_peer_bytes);
    maximums.intermediate_bytes = std::max(maximums.intermediate_bytes, device.intermediate_bytes);
    maximums.output_bytes = std::max(maximums.output_bytes, device.output_bytes);
    maximums.bc_flops = std::max(maximums.bc_flops, device.bc_flops);
    maximums.accumulate_flops = std::max(maximums.accumulate_flops, device.accumulate_flops);
    maximums.intermediate_gemm_enqueue_s =
        std::max(maximums.intermediate_gemm_enqueue_s, device.intermediate_gemm_enqueue_s);
    maximums.final_gemm_enqueue_s = std::max(maximums.final_gemm_enqueue_s, device.final_gemm_enqueue_s);
    maximums.source_accumulation_enqueue_s =
        std::max(maximums.source_accumulation_enqueue_s, device.source_accumulation_enqueue_s);
    maximums.output_accumulation_enqueue_s =
        std::max(maximums.output_accumulation_enqueue_s, device.output_accumulation_enqueue_s);
    maximums.zero_fill_enqueue_s = std::max(maximums.zero_fill_enqueue_s, device.zero_fill_enqueue_s);
  }

  return fmt::format(
      "{{\"total_terms\":{},\"total_bc_gemms\":{},\"total_final_gemms\":{},"
      "\"total_source_axpys\":{},\"total_output_axpys\":{},\"total_zero_fills\":{},"
      "\"total_temporary_peer_requests\":{},\"total_temporary_peer_copies\":{},"
      "\"total_temporary_peer_bytes\":{},\"total_intermediate_bytes\":{},\"total_output_bytes\":{},"
      "\"total_bc_flops\":{:.17g},\"total_accumulate_flops\":{:.17g},"
      "\"max_terms\":{},\"max_unique_bc\":{},\"max_bc_gemms\":{},\"max_final_gemms\":{},"
      "\"max_source_fan_in\":{},\"max_output_fan_in\":{},\"max_accumulation_fan_in\":{},"
      "\"max_temporary_peer_bytes\":{},\"max_intermediate_bytes\":{},\"max_output_bytes\":{},"
      "\"max_bc_flops\":{:.17g},\"max_accumulate_flops\":{:.17g},"
      "\"enqueue_sum_s\":{{\"intermediate_gemm\":{:.17g},\"final_gemm\":{:.17g},"
      "\"source_accumulation\":{:.17g},\"output_accumulation\":{:.17g},\"zero_fill\":{:.17g}}},"
      "\"enqueue_max_s\":{{\"intermediate_gemm\":{:.17g},\"final_gemm\":{:.17g},"
      "\"source_accumulation\":{:.17g},\"output_accumulation\":{:.17g},\"zero_fill\":{:.17g}}}}}",
      totals.terms, totals.bc_gemms, totals.final_gemms, totals.source_axpys, totals.output_axpys, totals.zero_fills,
      totals.temporary_peer_requests, totals.temporary_peer_copies, totals.temporary_peer_bytes,
      totals.intermediate_bytes, totals.output_bytes, totals.bc_flops, totals.accumulate_flops, maximums.terms,
      maximums.unique_bc, maximums.bc_gemms, maximums.final_gemms, maximums.max_source_fan_in,
      maximums.max_output_fan_in, maximums.max_accumulation_fan_in, maximums.temporary_peer_bytes,
      maximums.intermediate_bytes, maximums.output_bytes, maximums.bc_flops, maximums.accumulate_flops,
      totals.intermediate_gemm_enqueue_s, totals.final_gemm_enqueue_s, totals.source_accumulation_enqueue_s,
      totals.output_accumulation_enqueue_s, totals.zero_fill_enqueue_s, maximums.intermediate_gemm_enqueue_s,
      maximums.final_gemm_enqueue_s, maximums.source_accumulation_enqueue_s, maximums.output_accumulation_enqueue_s,
      maximums.zero_fill_enqueue_s);
}

void write_rabc_trace(std::uint64_t index, std::string const& policy, RabcCostFeatures const& features,
                      RabcExecutionStats const& execution_stats, MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                      MatrixFamily const& c_mats, std::span<tensor::Matrix const> r_raw_mats,
                      std::span<EffectiveHamiltonianOperator::Term const> terms, double enqueue_seconds,
                      double sync_seconds, double wall_seconds, std::span<RabcTraceDeviceTiming const> timings,
                      tensor::Swapper::RuntimeCounters const& runtime_counters)
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
             "\"device_timings\":{},\"devices\":{},\"execution_devices\":{},\"runtime_counters\":{},"
             "\"layout_metrics\":{},\"shape_summary\":{},\"feature_summary\":{},\"execution_summary\":{}",
             index, policy, features.devices.size(), features.output_devices.size(), output_shape_signature(r_raw_mats),
             terms.size(), enqueue_seconds, sync_seconds, wall_seconds, max_gpu_seconds(timings),
             json_int_array(features.input_devices), json_int_array(features.output_devices),
             json_device_timings(timings), json_device_features(features.devices),
             json_device_features(execution_stats.devices), json_runtime_counters(runtime_counters),
             json_layout_metrics(features, raw_matrices(b_mats), r_raw_mats),
             json_shape_summary(a_mats, b_mats, c_mats, r_raw_mats, terms), json_feature_summary(features.devices),
             json_feature_summary(execution_stats.devices));
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

enum class RabcGemmStage
{
  Intermediate,
  Final,
};

enum class RabcAxpyStage
{
  SourceAccumulation,
  OutputAccumulation,
};

enum class RabcTimedStage
{
  IntermediateGemm,
  FinalGemm,
  SourceAccumulation,
  OutputAccumulation,
  ZeroFill,
};

void add_enqueue_seconds(RabcExecutionStats* stats, int device_id, RabcTimedStage stage, double seconds)
{
  if (stats == nullptr)
  {
    return;
  }

  auto& device_stats = stats->device(device_id);
  switch (stage)
  {
    case RabcTimedStage::IntermediateGemm:
      device_stats.intermediate_gemm_enqueue_s += seconds;
      break;
    case RabcTimedStage::FinalGemm:
      device_stats.final_gemm_enqueue_s += seconds;
      break;
    case RabcTimedStage::SourceAccumulation:
      device_stats.source_accumulation_enqueue_s += seconds;
      break;
    case RabcTimedStage::OutputAccumulation:
      device_stats.output_accumulation_enqueue_s += seconds;
      break;
    case RabcTimedStage::ZeroFill:
      device_stats.zero_fill_enqueue_s += seconds;
      break;
  }
}

class RabcEnqueueTimer {
  public:
    RabcEnqueueTimer(RabcExecutionStats* stats, int device_id, RabcTimedStage stage)
        : stats_(stats), device_id_(device_id), stage_(stage)
    {
      if (stats_ != nullptr)
      {
        start_ = std::chrono::steady_clock::now();
      }
    }

    RabcEnqueueTimer(RabcEnqueueTimer const&) = delete;
    RabcEnqueueTimer& operator=(RabcEnqueueTimer const&) = delete;

    ~RabcEnqueueTimer()
    {
      if (stats_ == nullptr)
      {
        return;
      }
      auto const stop = std::chrono::steady_clock::now();
      add_enqueue_seconds(stats_, device_id_, stage_, std::chrono::duration<double>(stop - start_).count());
    }

  private:
    RabcExecutionStats* stats_ = nullptr;
    int device_id_ = 0;
    RabcTimedStage stage_ = RabcTimedStage::IntermediateGemm;
    std::chrono::steady_clock::time_point start_;
};

void note_temporary_peer_access(RabcExecutionStats* stats, tensor::Swapper& swapper, tensor::Matrix mat,
                                int source_device, int target_device)
{
  if (stats == nullptr || source_device == target_device)
  {
    return;
  }

  auto& device_stats = stats->device(target_device);
  ++device_stats.temporary_peer_requests;
  device_stats.temporary_peer_request_bytes += mat.sizeInByte();

  auto target_buffer = swapper.getGpuBufferOrNone(mat, target_device);
  if (target_buffer == nullptr || !target_buffer->contentValid())
  {
    ++device_stats.temporary_peer_copies;
    device_stats.temporary_peer_bytes += mat.sizeInByte();
  }
}

void zero_device_matrix(tensor::Swapper& swapper, tensor::Matrix mat, int device_id, RabcExecutionStats* stats)
{
  if (stats != nullptr)
  {
    ++stats->device(device_id).zero_fills;
  }
  RabcEnqueueTimer const timer(stats, device_id, RabcTimedStage::ZeroFill);
  auto buffer = require_buffer_on(swapper, mat, device_id);
  auto access = swapper.createAccessPlan({}, {buffer}, device_id);
  CUDA_CALL(cudaMemsetAsync(buffer->getPtr(), 0, mat.sizeInByte(), access.stream()));
}

void gemm_device_matrix(tensor::Swapper& swapper, tensor::Matrix result, tensor::Matrix lhs, tensor::Matrix rhs,
                        double alpha, double beta, int device_id, RabcGemmStage stage, RabcExecutionStats* stats)
{
  if (stats != nullptr)
  {
    auto& device_stats = stats->device(device_id);
    switch (stage)
    {
      case RabcGemmStage::Intermediate:
        ++device_stats.bc_gemms;
        break;
      case RabcGemmStage::Final:
        ++device_stats.final_gemms;
        break;
    }
  }
  RabcEnqueueTimer const timer(stats, device_id,
                               stage == RabcGemmStage::Intermediate ? RabcTimedStage::IntermediateGemm
                                                                    : RabcTimedStage::FinalGemm);
  auto lhs_buffer = swapper.ensureLocalCopy(lhs, device_id);
  auto rhs_buffer = swapper.ensureLocalCopy(rhs, device_id);
  auto result_buffer = require_buffer_on(swapper, result, device_id);
  auto access = swapper.createBlasAccessPlan({lhs_buffer, rhs_buffer}, {result_buffer}, device_id);

  CUBLAS_CALL(cublasDgemm(access.handle(), CUBLAS_OP_N, CUBLAS_OP_N, rhs.getSecondDim(), lhs.getFirstDim(),
                          lhs.getSecondDim(), &alpha, rhs_buffer->getPtr(), rhs.getSecondDim(), lhs_buffer->getPtr(),
                          lhs.getSecondDim(), &beta, result_buffer->getPtr(), result.getSecondDim()));
}

void axpy_device_matrix(tensor::Swapper& swapper, double alpha, tensor::Matrix source, tensor::Matrix target,
                        int device_id, RabcAxpyStage stage, RabcExecutionStats* stats)
{
  if (source.getFirstDim() != target.getFirstDim() || source.getSecondDim() != target.getSecondDim())
  {
    throw std::logic_error("TensorContraction deterministic RABC executor attempted to accumulate mismatched blocks");
  }

  if (stats != nullptr)
  {
    auto& device_stats = stats->device(device_id);
    switch (stage)
    {
      case RabcAxpyStage::SourceAccumulation:
        ++device_stats.source_axpys;
        break;
      case RabcAxpyStage::OutputAccumulation:
        ++device_stats.output_axpys;
        break;
    }
  }
  RabcEnqueueTimer const timer(stats, device_id,
                               stage == RabcAxpyStage::SourceAccumulation ? RabcTimedStage::SourceAccumulation
                                                                          : RabcTimedStage::OutputAccumulation);
  auto source_buffer = swapper.ensureLocalCopy(source, device_id);
  auto target_buffer = require_buffer_on(swapper, target, device_id);
  auto access = swapper.createBlasAccessPlan({source_buffer}, {target_buffer}, device_id);
  CUBLAS_CALL(cublasDaxpy(access.handle(), checked_blas_vector_size(source.size()), &alpha, source_buffer->getPtr(), 1,
                          target_buffer->getPtr(), 1));
}

struct RightFirstPartial
{
    int device_id = 0;
    tensor::Matrix matrix;
    double coefficient = 1.0;
};

void apply_resident_rabc_context(ResidentRabcContext& context, std::span<tensor::Matrix const> r_mats,
                                 std::span<tensor::Matrix const> a_mats, std::span<tensor::Matrix const> b_mats,
                                 std::span<tensor::Matrix const> c_mats, tensor::Swapper& swapper,
                                 RabcExecutionStats* stats)
{
  if (stats != nullptr)
  {
    stats->reset(swapper.getDeviceCount());
  }

  for (auto const& first_stage_product : context.first_stage_products)
  {
    if (stats != nullptr)
    {
      ++stats->device(first_stage_product.device_id).intermediate_matrices;
    }
    gemm_device_matrix(swapper, first_stage_product.matrix, b_mats[first_stage_product.b],
                       c_mats[first_stage_product.c], 1.0, 0.0, first_stage_product.device_id,
                       RabcGemmStage::Intermediate, stats);
  }

  std::vector<bool> r_written(r_mats.size(), false);
  std::vector<RightFirstPartial> partials;
  for (auto const& plan : context.assembled_temporaries)
  {
    auto const target_device = context.output_devices.at(plan.r);
    auto const beta = r_written[plan.r] ? 1.0 : 0.0;
    partials.clear();
    partials.reserve(plan.local_partials.size());

    for (auto const& local_partial : plan.local_partials)
    {
      if (!local_partial.accumulation.has_value())
      {
        auto const& input = local_partial.inputs.front();
        auto const& first_stage_product = context.first_stage_products.at(input.first_stage_product);
        partials.push_back(RightFirstPartial{
            .device_id = local_partial.device_id,
            .matrix = first_stage_product.matrix,
            .coefficient = input.coefficient,
        });
        continue;
      }

      auto const& q_matrix = *local_partial.accumulation;
      if (stats != nullptr)
      {
        auto& device_stats = stats->device(local_partial.device_id);
        ++device_stats.temporary_matrices;
        ++device_stats.accumulation_groups;
        ++device_stats.source_accumulation_groups;
        device_stats.accumulation_terms += local_partial.inputs.size();
        device_stats.source_accumulation_terms += local_partial.inputs.size();
        device_stats.max_source_fan_in = std::max(device_stats.max_source_fan_in, local_partial.inputs.size());
        device_stats.max_accumulation_fan_in =
            std::max(device_stats.max_accumulation_fan_in, local_partial.inputs.size());
      }
      zero_device_matrix(swapper, q_matrix, local_partial.device_id, stats);
      for (auto const& input : local_partial.inputs)
      {
        auto const& first_stage_product = context.first_stage_products.at(input.first_stage_product);
        axpy_device_matrix(swapper, input.coefficient, first_stage_product.matrix, q_matrix, local_partial.device_id,
                           RabcAxpyStage::SourceAccumulation, stats);
      }
      partials.push_back(
          RightFirstPartial{.device_id = local_partial.device_id, .matrix = q_matrix, .coefficient = 1.0});
    }

    if (!plan.output_accumulation.has_value())
    {
      auto const& partial = partials.front();
      if (stats != nullptr)
      {
        ++stats->device(target_device).direct_final_gemms;
      }
      note_temporary_peer_access(stats, swapper, partial.matrix, partial.device_id, target_device);
      gemm_device_matrix(swapper, r_mats[plan.r], a_mats[plan.a], partial.matrix, partial.coefficient, beta,
                         target_device, RabcGemmStage::Final, stats);
    }
    else
    {
      auto const& q_matrix = *plan.output_accumulation;
      if (stats != nullptr)
      {
        auto& device_stats = stats->device(target_device);
        ++device_stats.temporary_matrices;
        ++device_stats.accumulation_groups;
        ++device_stats.output_accumulation_groups;
        device_stats.accumulation_terms += partials.size();
        device_stats.output_accumulation_terms += partials.size();
        device_stats.max_output_fan_in = std::max(device_stats.max_output_fan_in, partials.size());
        device_stats.max_accumulation_fan_in = std::max(device_stats.max_accumulation_fan_in, partials.size());
      }
      zero_device_matrix(swapper, q_matrix, target_device, stats);
      for (auto const& partial : partials)
      {
        note_temporary_peer_access(stats, swapper, partial.matrix, partial.device_id, target_device);
        axpy_device_matrix(swapper, partial.coefficient, partial.matrix, q_matrix, target_device,
                           RabcAxpyStage::OutputAccumulation, stats);
      }
      gemm_device_matrix(swapper, r_mats[plan.r], a_mats[plan.a], q_matrix, 1.0, beta, target_device,
                         RabcGemmStage::Final, stats);
    }
    r_written[plan.r] = true;
  }

  for (std::size_t r = 0; r < r_mats.size(); ++r)
  {
    if (!r_written[r])
    {
      auto const device = context.output_devices.at(r);
      fmt::print(stderr,
                 "[TENSORCONTRACTION][RABC_WARNING] Output R block id={} shape={}x{} device={} received no "
                 "Hamiltonian terms; zeroing it explicitly.\n",
                 r_mats[r].getId(), r_mats[r].getFirstDim(), r_mats[r].getSecondDim(), device);
      zero_device_matrix(swapper, r_mats[r], device, stats);
    }
  }
}

} // namespace

ResidentRabcContext& EffectiveHamiltonianOperator::Impl::ensure_resident_context(
    std::span<tensor::Matrix const> r_mats, std::span<tensor::Matrix const> a_raw_mats,
    std::span<tensor::Matrix const> b_raw_mats, std::span<tensor::Matrix const> c_raw_mats, MatrixFamily const& c_mats,
    tensor::Swapper& swapper)
{
  auto const input_devices = resident_devices_for(swapper, b_raw_mats);
  auto const output_devices = resident_devices_for(swapper, r_mats);
  if (resident_context == nullptr || !resident_context->matches(variable_family, swapper.getDeviceCount(),
                                                                input_devices, output_devices, terms.size()))
  {
    if (resident_context != nullptr)
    {
      resident_context->release(swapper);
    }
    resident_context = make_resident_rabc_context(variable_family, r_mats, b_raw_mats, c_mats, terms, swapper);
  }
  stage_right_first_dependencies(*resident_context, a_raw_mats, c_raw_mats, terms, swapper);
  return *resident_context;
}

void EffectiveHamiltonianOperator::Impl::initialize_runtime()
{
  if (use_host_effective_hamiltonian_backend())
  {
    return;
  }
  swapper = std::make_unique<tensor::Swapper>();
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

void EffectiveHamiltonianOperator::write_term_structure(std::string const& path, std::uint64_t index) const
{
  auto const& a_mats = impl_->a_mats;
  auto const& b_mats = impl_->b_mats;
  auto const& c_mats = impl_->c_mats;
  auto const& r_mats = impl_->r_mats;
  auto const& terms = impl_->terms;
  std::size_t const block_count = r_mats.size();

  auto* file = std::fopen(path.c_str(), "a");
  if (file == nullptr)
  {
    throw std::runtime_error("failed to open R/A/B/C term-structure file: " + path);
  }

  // Single-device placeholder layout: R shares B's space, all on device 0.
  std::string layout = "[";
  for (std::size_t block = 0; block < block_count; ++block)
  {
    if (block != 0)
    {
      layout += ',';
    }
    layout += '0';
  }
  layout += ']';

  fmt::print(file,
             "{{\"kind\":\"rabc_matvec\",\"index\":{},\"policy\":\"structure\",\"device_count\":1,"
             "\"block_count\":{},\"term_count\":{},\"input_layout\":{},\"output_layout\":{},\"terms\":[",
             index, block_count, terms.size(), layout, layout);
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
    auto const r = r_mats.block(term.r);
    auto const intermediate = MatrixFamily::Block{.rows = b.rows, .cols = c.cols};
    fmt::print(file,
               "{{\"r\":{},\"a\":{},\"b\":{},\"c\":{},\"coefficient\":{:.17g},\"device\":0,"
               "\"r_rows\":{},\"r_cols\":{},\"a_rows\":{},\"a_cols\":{},"
               "\"b_rows\":{},\"b_cols\":{},\"c_rows\":{},\"c_cols\":{},"
               "\"bc_flops\":{},\"accumulate_flops\":{},\"intermediate_bytes\":{}}}",
               term.r, term.a, term.b, term.c, term.coefficient, r.rows, r.cols, a.rows, a.cols, b.rows, b.cols,
               c.rows, c.cols, static_cast<std::int64_t>(gemm_flops(b, c)),
               static_cast<std::int64_t>(gemm_flops(a, intermediate)),
               static_cast<std::int64_t>(b.rows * c.cols * sizeof(double)));
  }
  fmt::print(file, "]}}\n");
  std::fclose(file);
}

void EffectiveHamiltonianOperator::compile()
{
  if (impl_->is_compiled)
  {
    return;
  }

  validate_term_shapes(impl_->r_mats, impl_->a_mats, impl_->b_mats, impl_->c_mats, impl_->terms);
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
    auto& swapper = *impl_->swapper;
    auto const& r = raw_matrices(impl_->r_mats);
    auto const& a = raw_matrices(impl_->a_mats);
    auto const& b = raw_matrices(impl_->b_mats);
    auto const& c = raw_matrices(impl_->c_mats);

    localize_rabc_family(swapper, a, impl_->a_mats.coalesced_values(), /*upload_from_host=*/true,
                         /*refresh_existing=*/false);
    if (impl_->variable_family == VariableFamily::Middle)
    {
      localize_rabc_family(swapper, b, impl_->b_mats.coalesced_values(), /*upload_from_host=*/true,
                           /*refresh_existing=*/true);
      localize_rabc_family(swapper, c, impl_->c_mats.coalesced_values(), /*upload_from_host=*/true,
                           /*refresh_existing=*/false);
    }
    else
    {
      localize_rabc_family(swapper, b, impl_->b_mats.coalesced_values(), /*upload_from_host=*/true,
                           /*refresh_existing=*/false);
      localize_rabc_family(swapper, c, impl_->c_mats.coalesced_values(), /*upload_from_host=*/true,
                           /*refresh_existing=*/true);
    }
    bool const output_placement_selected = maybe_place_rabc_families(
        b, r, impl_->b_mats, impl_->terms, impl_->variable_family, swapper, impl_->output_placement_cache);
    if (!output_placement_selected)
    {
      localize_rabc_family(swapper, r, impl_->r_mats.coalesced_values(), /*upload_from_host=*/false,
                           /*refresh_existing=*/false);
    }
    auto& context = impl_->ensure_resident_context(r, a, b, c, impl_->c_mats, swapper);
    apply_resident_rabc_context(context, r, a, b, c, swapper, nullptr);
    synchronize_rabc_family_to_host(swapper, r, impl_->r_mats.coalesced_values());
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

  auto& swapper = algebra.resident_swapper();
  auto const& r = raw_matrices(y);
  auto const& a = raw_matrices(impl_->a_mats);
  auto const& b_family = impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats;
  auto const& c_family = impl_->variable_family == VariableFamily::Middle ? impl_->c_mats : x;
  auto const& b = raw_matrices(b_family);
  auto const& c = raw_matrices(c_family);

  validate_term_shapes(y, impl_->a_mats, impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
                       impl_->variable_family == VariableFamily::Middle ? impl_->c_mats : x, impl_->terms);

  // Static environments are host-authored but should not be refreshed during
  // every Krylov matvec.  The active Lanczos input/output vectors are already
  // resident in this same runtime.
  localize_rabc_family(swapper, a, impl_->a_mats.coalesced_values(), /*upload_from_host=*/true,
                       /*refresh_existing=*/false);
  if (impl_->variable_family == VariableFamily::Middle)
  {
    localize_rabc_family(swapper, c, impl_->c_mats.coalesced_values(), /*upload_from_host=*/true,
                         /*refresh_existing=*/false);
  }
  else
  {
    localize_rabc_family(swapper, b, impl_->b_mats.coalesced_values(), /*upload_from_host=*/true,
                         /*refresh_existing=*/false);
  }
  localize_rabc_family(swapper, raw_matrices(x), x.coalesced_values(), /*upload_from_host=*/false,
                       /*refresh_existing=*/false);
  bool const output_placement_selected =
      maybe_place_rabc_families(b, r, impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
                                impl_->terms, impl_->variable_family, swapper, impl_->output_placement_cache);
  if (!output_placement_selected)
  {
    localize_rabc_family(swapper, raw_matrices(y), y.coalesced_values(), /*upload_from_host=*/false,
                         /*refresh_existing=*/false);
  }
  auto& context = impl_->ensure_resident_context(r, a, b, c, c_family, swapper);

  bool const trace_enabled = rabc_trace_enabled();
  auto trace_timings = trace_enabled ? start_rabc_trace_timing(swapper) : std::vector<RabcTraceDeviceTiming>{};
  auto const runtime_counters_start = trace_enabled ? swapper.runtimeCounters() : tensor::Swapper::RuntimeCounters{};
  RabcExecutionStats execution_stats;
  auto const trace_index = trace_enabled ? next_rabc_trace_index() : 0;
  auto const enqueue_start = std::chrono::steady_clock::now();
  apply_resident_rabc_context(context, r, a, b, c, swapper, trace_enabled ? &execution_stats : nullptr);
  auto const enqueue_stop = std::chrono::steady_clock::now();
  if (trace_enabled)
  {
    auto const runtime_counters = subtract_runtime_counters(swapper.runtimeCounters(), runtime_counters_start);
    stop_rabc_trace_timing(trace_timings);
    finish_rabc_trace_timing(trace_timings);
    auto const sync_stop = std::chrono::steady_clock::now();
    auto const features = rabc_cost_features(
        impl_->a_mats, impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
        impl_->variable_family == VariableFamily::Middle ? impl_->c_mats : x, r, b, impl_->terms, swapper);
    write_rabc_trace(trace_index, rabc_placement_policy(), features, execution_stats, impl_->a_mats,
                     impl_->variable_family == VariableFamily::Middle ? x : impl_->b_mats,
                     impl_->variable_family == VariableFamily::Middle ? impl_->c_mats : x, r, impl_->terms,
                     std::chrono::duration<double>(enqueue_stop - enqueue_start).count(),
                     std::chrono::duration<double>(sync_stop - enqueue_stop).count(),
                     std::chrono::duration<double>(sync_stop - enqueue_start).count(), trace_timings, runtime_counters);
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
