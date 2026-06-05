#include <uni20/tensorcontraction/rabc_lanczos_fixture.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace uni20::tensorcontraction
{

namespace
{

constexpr std::array<char, 33> fixture_magic{'U', 'N', 'I', '2', '0', '_', 'R', 'A',  'B',  'C',  '_',
                                             'L', 'A', 'N', 'C', 'Z', 'O', 'S', '_',  'F',  'I',  'X',
                                             'T', 'U', 'R', 'E', '_', 'V', '1', '\n', '\0', '\0', '\0'};

auto checked_u64(std::size_t value) -> std::uint64_t
{
  if (value > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
  {
    throw std::length_error("R/A/B/C fixture size does not fit uint64_t");
  }
  return static_cast<std::uint64_t>(value);
}

auto checked_size(std::uint64_t value) -> std::size_t
{
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
  {
    throw std::length_error("R/A/B/C fixture size does not fit size_t");
  }
  return static_cast<std::size_t>(value);
}

auto scalar_count(MatrixFamily::Block block) -> std::size_t
{
  if (block.rows != 0 && block.cols > std::numeric_limits<std::size_t>::max() / block.rows)
  {
    throw std::length_error("R/A/B/C fixture block size overflows size_t");
  }
  return block.rows * block.cols;
}

template <typename T> void write_pod(std::ofstream& output, T const& value)
{
  output.write(reinterpret_cast<char const*>(&value), sizeof(T));
  if (!output)
  {
    throw std::runtime_error("failed to write R/A/B/C fixture");
  }
}

template <typename T> auto read_pod(std::ifstream& input) -> T
{
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input)
  {
    throw std::runtime_error("failed to read R/A/B/C fixture");
  }
  return value;
}

void write_blocks(std::ofstream& output, std::span<MatrixFamily::Block const> blocks)
{
  write_pod(output, checked_u64(blocks.size()));
  for (auto const block : blocks)
  {
    write_pod(output, checked_u64(block.rows));
    write_pod(output, checked_u64(block.cols));
  }
}

auto read_blocks(std::ifstream& input) -> std::vector<MatrixFamily::Block>
{
  auto const count = checked_size(read_pod<std::uint64_t>(input));
  std::vector<MatrixFamily::Block> blocks;
  blocks.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
  {
    auto const rows = checked_size(read_pod<std::uint64_t>(input));
    auto const cols = checked_size(read_pod<std::uint64_t>(input));
    blocks.push_back(MatrixFamily::Block{.rows = rows, .cols = cols});
  }
  return blocks;
}

void write_family(std::ofstream& output, MatrixFamily const& family)
{
  write_blocks(output, family.blocks());
  auto const values = family.coalesced_values();
  write_pod(output, checked_u64(values.size()));
  if (!values.empty())
  {
    output.write(reinterpret_cast<char const*>(values.data()), static_cast<std::streamsize>(values.size_bytes()));
    if (!output)
    {
      throw std::runtime_error("failed to write R/A/B/C fixture MatrixFamily values");
    }
  }
}

auto read_family(std::ifstream& input) -> MatrixFamily
{
  auto blocks = read_blocks(input);
  MatrixFamily family(blocks);
  auto values = family.coalesced_values();
  auto const stored_values = checked_size(read_pod<std::uint64_t>(input));
  if (stored_values != values.size())
  {
    throw std::runtime_error("R/A/B/C fixture MatrixFamily value count does not match block shapes");
  }
  if (!values.empty())
  {
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size_bytes()));
    if (!input)
    {
      throw std::runtime_error("failed to read R/A/B/C fixture MatrixFamily values");
    }
  }
  return family;
}

void write_terms(std::ofstream& output, std::span<EffectiveHamiltonianOperator::Term const> terms)
{
  write_pod(output, checked_u64(terms.size()));
  for (auto const& term : terms)
  {
    write_pod(output, checked_u64(term.r));
    write_pod(output, checked_u64(term.a));
    write_pod(output, checked_u64(term.b));
    write_pod(output, checked_u64(term.c));
    write_pod(output, term.coefficient);
  }
}

auto read_terms(std::ifstream& input) -> std::vector<EffectiveHamiltonianOperator::Term>
{
  auto const count = checked_size(read_pod<std::uint64_t>(input));
  std::vector<EffectiveHamiltonianOperator::Term> terms;
  terms.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
  {
    terms.push_back(EffectiveHamiltonianOperator::Term{
        .r = checked_size(read_pod<std::uint64_t>(input)),
        .a = checked_size(read_pod<std::uint64_t>(input)),
        .b = checked_size(read_pod<std::uint64_t>(input)),
        .c = checked_size(read_pod<std::uint64_t>(input)),
        .coefficient = read_pod<double>(input),
    });
  }
  return terms;
}

void validate_input_blocks(MatrixFamily const& input_vector)
{
  for (auto const block : input_vector.blocks())
  {
    static_cast<void>(scalar_count(block));
  }
}

} // namespace

auto RabcLanczosFixture::make_operator() -> EffectiveHamiltonianOperator
{
  return EffectiveHamiltonianOperator::variable_middle(std::move(a_mats), std::move(c_mats), input_vector.blocks(),
                                                       output_blocks, terms);
}

void write_variable_middle_rabc_fixture(std::string const& path, EffectiveHamiltonianOperator const& op,
                                        MatrixFamily const& input_vector)
{
  write_rabc_lanczos_fixture(path, capture_variable_middle_rabc_fixture(op, input_vector));
}

void write_rabc_lanczos_fixture(std::string const& path, RabcLanczosFixture const& fixture)
{
  std::ofstream output(path, std::ios::binary);
  if (!output)
  {
    throw std::runtime_error("failed to open R/A/B/C fixture for writing: " + path);
  }

  output.write(fixture_magic.data(), static_cast<std::streamsize>(fixture_magic.size()));
  if (!output)
  {
    throw std::runtime_error("failed to write R/A/B/C fixture header");
  }

  write_family(output, fixture.a_mats);
  write_family(output, fixture.c_mats);
  write_family(output, fixture.input_vector);
  write_blocks(output, fixture.output_blocks);
  write_terms(output, fixture.terms);
}

auto read_rabc_lanczos_fixture(std::string const& path) -> RabcLanczosFixture
{
  std::ifstream input(path, std::ios::binary);
  if (!input)
  {
    throw std::runtime_error("failed to open R/A/B/C fixture for reading: " + path);
  }

  std::array<char, fixture_magic.size()> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != fixture_magic)
  {
    throw std::runtime_error("R/A/B/C fixture has an unsupported format: " + path);
  }

  RabcLanczosFixture fixture{.a_mats = read_family(input),
                             .c_mats = read_family(input),
                             .input_vector = read_family(input),
                             .output_blocks = read_blocks(input),
                             .terms = read_terms(input)};
  validate_input_blocks(fixture.input_vector);
  return fixture;
}

} // namespace uni20::tensorcontraction
