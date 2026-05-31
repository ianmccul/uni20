#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace utc = uni20::tensorcontraction;

namespace
{

utc::MatrixFamily make_family(std::initializer_list<utc::MatrixFamily::Block> blocks)
{
  return utc::MatrixFamily(std::span{blocks.begin(), blocks.size()});
}

std::vector<double> expected_single_term(std::span<double const> a_values, std::span<double const> b_values,
                                         std::span<double const> c_values, double coefficient)
{
  std::vector<double> bc(3 * 4, 0.0);
  for (std::size_t row = 0; row < 3; ++row)
  {
    for (std::size_t col = 0; col < 4; ++col)
    {
      for (std::size_t inner = 0; inner < 5; ++inner)
      {
        bc[row * 4 + col] += b_values[row * 5 + inner] * c_values[inner * 4 + col];
      }
    }
  }

  std::vector<double> expected(2 * 4, 0.0);
  for (std::size_t row = 0; row < 2; ++row)
  {
    for (std::size_t col = 0; col < 4; ++col)
    {
      for (std::size_t inner = 0; inner < 3; ++inner)
      {
        expected[row * 4 + col] += coefficient * a_values[row * 3 + inner] * bc[inner * 4 + col];
      }
    }
  }
  return expected;
}

void expect_near(std::span<double const> actual, std::span<double const> expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(actual[i], expected[i], 1.0e-10);
  }
}

} // namespace

TEST(TensorContractionEffectiveHamiltonianOperatorTest, AppliesSingleTermMatvec)
{
  auto a = make_family({{2, 3}});
  auto b = make_family({{3, 5}});
  std::array input_blocks{utc::MatrixFamily::Block{5, 4}};
  std::array output_blocks{utc::MatrixFamily::Block{2, 4}};

  a.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  b.assign(0, std::array{1.0, 0.5, -1.0, 2.0, 1.5, 0.0, -0.5, 3.0, 1.0, 2.5, 2.0, 1.0, 0.0, -1.5, 0.25});
  auto const a_values = a.values(0);
  auto const b_values = b.values(0);
  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.25}};

  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  EXPECT_EQ(op.term_count(), 1);
  EXPECT_FALSE(op.compiled());

  auto x = op.make_input_vector();
  auto y = op.make_output_vector();
  x.assign(0, std::array{1.0,  2.0,  0.5, -1.0, 0.0,  1.5, 2.5, 3.0, -2.0, 1.0,
                         0.75, 0.25, 4.0, -0.5, 1.25, 2.0, 3.5, 0.0, -1.0, 1.0});

  auto const expected = expected_single_term(a_values, b_values, x.values(0), 1.25);
  op.apply(x, y);

  EXPECT_TRUE(op.compiled());
  expect_near(y.values(0), expected);
}

TEST(TensorContractionEffectiveHamiltonianOperatorTest, AppliesSingleTermWithVariableMiddle)
{
  auto a = make_family({{2, 3}});
  auto c = make_family({{5, 4}});
  std::array input_blocks{utc::MatrixFamily::Block{3, 5}};
  std::array output_blocks{utc::MatrixFamily::Block{2, 4}};

  a.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  c.assign(0, std::array{1.0,  2.0,  0.5, -1.0, 0.0,  1.5, 2.5, 3.0, -2.0, 1.0,
                         0.75, 0.25, 4.0, -0.5, 1.25, 2.0, 3.5, 0.0, -1.0, 1.0});
  auto const a_values = a.values(0);
  auto const c_values = c.values(0);
  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.25}};

  auto op = utc::EffectiveHamiltonianOperator::variable_middle(std::move(a), std::move(c), input_blocks, output_blocks,
                                                               terms);
  auto x = op.make_input_vector();
  auto y = op.make_output_vector();
  x.assign(0, std::array{1.0, 0.5, -1.0, 2.0, 1.5, 0.0, -0.5, 3.0, 1.0, 2.5, 2.0, 1.0, 0.0, -1.5, 0.25});

  auto const expected = expected_single_term(a_values, x.values(0), c_values, 1.25);
  op.apply(x, y);

  EXPECT_TRUE(op.compiled());
  expect_near(y.values(0), expected);
}

TEST(TensorContractionEffectiveHamiltonianOperatorTest, RepeatedApplyOverwritesOutput)
{
  auto a = make_family({{2, 3}});
  auto b = make_family({{3, 5}});
  std::array input_blocks{utc::MatrixFamily::Block{5, 4}};
  std::array output_blocks{utc::MatrixFamily::Block{2, 4}};

  a.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  b.assign(0, std::array{1.0, 0.5, -1.0, 2.0, 1.5, 0.0, -0.5, 3.0, 1.0, 2.5, 2.0, 1.0, 0.0, -1.5, 0.25});
  auto const a_values = a.values(0);
  auto const b_values = b.values(0);
  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.25}};

  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  auto x = op.make_input_vector();
  auto y = op.make_output_vector();

  x.assign(0, std::array{1.0,  2.0,  0.5, -1.0, 0.0,  1.5, 2.5, 3.0, -2.0, 1.0,
                         0.75, 0.25, 4.0, -0.5, 1.25, 2.0, 3.5, 0.0, -1.0, 1.0});
  op.apply(x, y);

  x.assign(0, std::array{0.5, -1.0, 2.0,  0.25, 1.25, -0.5, 0.0, 1.5, 2.5, -2.0,
                         3.0, 0.75, -1.5, 1.0,  0.25, 2.0,  1.0, 0.5, 4.0, -0.75});
  auto const expected = expected_single_term(a_values, b_values, x.values(0), 1.25);
  y.fill(42.0);
  op.apply(x, y);

  expect_near(y.values(0), expected);
}

TEST(TensorContractionEffectiveHamiltonianOperatorTest, OutputWorksWithVectorAlgebra)
{
  auto a = make_family({{2, 3}});
  auto b = make_family({{3, 5}});
  std::array input_blocks{utc::MatrixFamily::Block{5, 4}};
  std::array output_blocks{utc::MatrixFamily::Block{2, 4}};

  a.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  b.assign(0, std::array{1.0, 0.5, -1.0, 2.0, 1.5, 0.0, -0.5, 3.0, 1.0, 2.5, 2.0, 1.0, 0.0, -1.5, 0.25});
  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.25}};

  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  auto x = op.make_input_vector();
  auto y = op.make_output_vector();
  x.assign(0, std::array{1.0,  2.0,  0.5, -1.0, 0.0,  1.5, 2.5, 3.0, -2.0, 1.0,
                         0.75, 0.25, 4.0, -0.5, 1.25, 2.0, 3.5, 0.0, -1.0, 1.0});

  op.apply(x, y);
  auto z = utc::make_like(y);
  utc::copy(y, z);
  EXPECT_DOUBLE_EQ(utc::dot(y, z), utc::norm2(y));

  double const original_norm = utc::normalize(z);
  EXPECT_GT(original_norm, 0.0);
  EXPECT_NEAR(utc::norm(z), 1.0, 1.0e-14);
}

TEST(TensorContractionEffectiveHamiltonianOperatorTest, RejectsMismatchedInputOutputVectors)
{
  auto a = make_family({{2, 3}});
  auto b = make_family({{3, 5}});
  std::array input_blocks{utc::MatrixFamily::Block{5, 4}};
  std::array output_blocks{utc::MatrixFamily::Block{2, 4}};
  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0}};

  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  auto wrong_x = make_family({{4, 5}});
  auto y = op.make_output_vector();
  EXPECT_THROW(op.apply(wrong_x, y), std::invalid_argument);

  auto x = op.make_input_vector();
  auto wrong_y = make_family({{4, 2}});
  EXPECT_THROW(op.apply(x, wrong_y), std::invalid_argument);
}
