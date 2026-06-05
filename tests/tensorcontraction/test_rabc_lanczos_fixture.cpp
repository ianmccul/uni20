#include <uni20/tensorcontraction/rabc_lanczos_fixture.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace utc = uni20::tensorcontraction;

namespace
{

auto make_family(std::vector<utc::MatrixFamily::Block> const& blocks,
                 std::vector<double> const& values) -> utc::MatrixFamily
{
  utc::MatrixFamily family(blocks);
  auto target = family.coalesced_values();
  if (target.size() != values.size())
  {
    throw std::logic_error("test fixture values do not match MatrixFamily shape");
  }
  std::copy(values.begin(), values.end(), target.begin());
  return family;
}

auto clone_family(utc::MatrixFamily const& source) -> utc::MatrixFamily
{
  utc::MatrixFamily clone(source.blocks());
  clone.assign(source);
  return clone;
}

auto temporary_fixture_path() -> std::filesystem::path
{
  auto const suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("uni20_rabc_lanczos_fixture_roundtrip_" + std::to_string(suffix) + ".bin");
}

void expect_family_equal(utc::MatrixFamily const& actual, utc::MatrixFamily const& expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t block = 0; block < actual.size(); ++block)
  {
    EXPECT_EQ(actual.block(block), expected.block(block));
  }
  auto const actual_values = actual.coalesced_values();
  auto const expected_values = expected.coalesced_values();
  ASSERT_EQ(actual_values.size(), expected_values.size());
  for (std::size_t i = 0; i < actual_values.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(actual_values[i], expected_values[i]);
  }
}

} // namespace

TEST(TensorContractionRabcLanczosFixtureTest, CapturesAndRoundTripsVariableMiddleOperator)
{
  std::vector<utc::MatrixFamily::Block> const a_blocks{utc::MatrixFamily::Block{.rows = 2, .cols = 2}};
  std::vector<utc::MatrixFamily::Block> const c_blocks{utc::MatrixFamily::Block{.rows = 1, .cols = 1}};
  std::vector<utc::MatrixFamily::Block> const input_blocks{utc::MatrixFamily::Block{.rows = 2, .cols = 1}};
  std::vector<utc::MatrixFamily::Block> const output_blocks{utc::MatrixFamily::Block{.rows = 2, .cols = 1}};
  std::vector<utc::EffectiveHamiltonianOperator::Term> const terms{
      utc::EffectiveHamiltonianOperator::Term{.r = 0, .a = 0, .b = 0, .c = 0, .coefficient = 1.5}};

  auto a = make_family(a_blocks, {2.0, 0.25, 0.25, 3.0});
  auto c = make_family(c_blocks, {4.0});
  auto input = make_family(input_blocks, {0.5, -0.75});
  auto op = utc::EffectiveHamiltonianOperator::variable_middle(
      clone_family(a), clone_family(c), input.blocks(), std::span{output_blocks.data(), output_blocks.size()}, terms);

  auto fixture = utc::capture_variable_middle_rabc_fixture(op, input);
  auto const path = temporary_fixture_path();
  utc::write_rabc_lanczos_fixture(path.string(), fixture);
  auto restored = utc::read_rabc_lanczos_fixture(path.string());
  std::filesystem::remove(path);

  expect_family_equal(restored.a_mats, a);
  expect_family_equal(restored.c_mats, c);
  expect_family_equal(restored.input_vector, input);
  ASSERT_EQ(restored.output_blocks.size(), output_blocks.size());
  EXPECT_EQ(restored.output_blocks.front(), output_blocks.front());
  ASSERT_EQ(restored.terms.size(), terms.size());
  EXPECT_EQ(restored.terms.front().r, terms.front().r);
  EXPECT_EQ(restored.terms.front().a, terms.front().a);
  EXPECT_EQ(restored.terms.front().b, terms.front().b);
  EXPECT_EQ(restored.terms.front().c, terms.front().c);
  EXPECT_DOUBLE_EQ(restored.terms.front().coefficient, terms.front().coefficient);

  auto restored_op = restored.make_operator();
  EXPECT_EQ(restored_op.term_count(), 1);
}
