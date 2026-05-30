#include <uni20/tensorcontraction/effective_hamiltonian_plan.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utc = uni20::tensorcontraction;

namespace
{

struct ParsedContractionData
{
    std::vector<utc::MatrixFamily::Block> r_blocks;
    std::vector<utc::MatrixFamily::Block> a_blocks;
    std::vector<utc::MatrixFamily::Block> b_blocks;
    std::vector<utc::MatrixFamily::Block> c_blocks;
    std::vector<utc::EffectiveHamiltonianPlan::Term> terms;
};

enum class Section
{
  None,
  R,
  A,
  B,
  C,
  F,
};

Section parse_section(std::string const& line)
{
  if (line == "RSize:") return Section::R;
  if (line == "ASize:") return Section::A;
  if (line == "BSize:") return Section::B;
  if (line == "CSize:") return Section::C;
  if (line == "F:") return Section::F;
  return Section::None;
}

void assign_block(std::vector<utc::MatrixFamily::Block>& blocks, std::size_t index, std::size_t rows, std::size_t cols)
{
  if (index >= blocks.size())
  {
    blocks.resize(index + 1);
  }
  blocks[index] = utc::MatrixFamily::Block{rows, cols};
}

ParsedContractionData read_original_format_data(std::string const& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::runtime_error("failed to open TensorContraction fixture: " + path);
  }

  ParsedContractionData data;
  Section section = Section::None;
  std::string line;
  while (std::getline(input, line))
  {
    if (line.empty())
    {
      continue;
    }
    if (auto next = parse_section(line); next != Section::None)
    {
      section = next;
      continue;
    }

    std::istringstream fields(line);
    if (section == Section::F)
    {
      utc::EffectiveHamiltonianPlan::Term term;
      fields >> term.r >> term.a >> term.b >> term.c >> term.coefficient;
      if (!fields)
      {
        throw std::runtime_error("invalid TensorContraction term line: " + line);
      }
      data.terms.push_back(term);
      continue;
    }

    std::size_t index = 0;
    std::size_t rows = 0;
    std::size_t cols = 0;
    fields >> index >> rows >> cols;
    if (!fields)
    {
      throw std::runtime_error("invalid TensorContraction block line: " + line);
    }

    switch (section)
    {
      case Section::R:
        assign_block(data.r_blocks, index, rows, cols);
        break;
      case Section::A:
        assign_block(data.a_blocks, index, rows, cols);
        break;
      case Section::B:
        assign_block(data.b_blocks, index, rows, cols);
        break;
      case Section::C:
        assign_block(data.c_blocks, index, rows, cols);
        break;
      case Section::None:
      case Section::F:
        throw std::runtime_error("TensorContraction data line appears before a section: " + line);
    }
  }

  return data;
}

utc::MatrixFamily make_family(std::vector<utc::MatrixFamily::Block> const& blocks)
{
  return utc::MatrixFamily(std::span{blocks.data(), blocks.size()});
}

} // namespace

TEST(TensorContractionDataFilePlanTest, CompilesOriginalFormatFixture)
{
  auto const path = std::string{UNI20_TENSORCONTRACTION_TEST_DATA_DIR} + "/original_format_small.dat";
  auto data = read_original_format_data(path);

  ASSERT_EQ(data.r_blocks.size(), 2);
  ASSERT_EQ(data.a_blocks.size(), 2);
  ASSERT_EQ(data.b_blocks.size(), 2);
  ASSERT_EQ(data.c_blocks.size(), 2);
  ASSERT_EQ(data.terms.size(), 2);

  utc::EffectiveHamiltonianPlan plan(make_family(data.r_blocks), make_family(data.a_blocks), make_family(data.b_blocks),
                                     make_family(data.c_blocks), data.terms);

  EXPECT_EQ(plan.term_count(), 2);
  EXPECT_FALSE(plan.compiled());
  plan.compile();
  EXPECT_TRUE(plan.compiled());
}
