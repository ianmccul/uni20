#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
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

class EnvGuard {
    std::vector<std::pair<std::string, std::optional<std::string>>> saved_;

  public:
    explicit EnvGuard(std::initializer_list<char const*> names)
    {
      saved_.reserve(names.size());
      for (auto const* name : names)
      {
        if (auto const* value = std::getenv(name); value != nullptr)
        {
          saved_.push_back({name, std::string(value)});
        }
        else
        {
          saved_.push_back({name, std::nullopt});
        }
      }
    }

    EnvGuard(EnvGuard const&) = delete;
    EnvGuard& operator=(EnvGuard const&) = delete;

    ~EnvGuard()
    {
      for (auto const& [name, value] : saved_)
      {
        if (value.has_value())
        {
          setenv(name.c_str(), value->c_str(), 1);
        }
        else
        {
          unsetenv(name.c_str());
        }
      }
    }
};

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

TEST(TensorContractionEffectiveHamiltonianOperatorTest, AppliesEnvironmentStyleMultiTermContraction)
{
  auto a = make_family({{3, 2}, {3, 2}});
  auto b = make_family({{2, 2}});
  std::array input_blocks{utc::MatrixFamily::Block{2, 3}, utc::MatrixFamily::Block{2, 3}};
  std::array output_blocks{utc::MatrixFamily::Block{3, 3}};

  // A blocks are transposed MPS site tensors, B is a left identity
  // environment, and C blocks are the original site tensors.
  a.assign(0, std::array{1.0, 4.0, 2.0, 5.0, 3.0, 6.0});
  a.assign(1, std::array{7.0, 10.0, 8.0, 11.0, 9.0, 12.0});
  b.assign(0, std::array{1.0, 0.0, 0.0, 1.0});

  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0},
                   utc::EffectiveHamiltonianOperator::Term{0, 1, 0, 1, 1.0}};
  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  auto x = op.make_input_vector();
  auto y = op.make_output_vector();
  x.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  x.assign(1, std::array{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  std::array expected{166.0, 188.0, 210.0, 188.0, 214.0, 240.0, 210.0, 240.0, 270.0};
  op.apply(x, y);

  expect_near(y.values(0), expected);
}

TEST(TensorContractionEffectiveHamiltonianOperatorTest, AppliesVariableMiddleMultiTermContraction)
{
  auto a = make_family({{3, 2}, {3, 2}});
  auto c = make_family({{2, 3}, {2, 3}});
  std::array input_blocks{utc::MatrixFamily::Block{2, 2}};
  std::array output_blocks{utc::MatrixFamily::Block{3, 3}};

  a.assign(0, std::array{1.0, 4.0, 2.0, 5.0, 3.0, 6.0});
  a.assign(1, std::array{7.0, 10.0, 8.0, 11.0, 9.0, 12.0});
  c.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  c.assign(1, std::array{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0},
                   utc::EffectiveHamiltonianOperator::Term{0, 1, 0, 1, 1.0}};
  auto op = utc::EffectiveHamiltonianOperator::variable_middle(std::move(a), std::move(c), input_blocks, output_blocks,
                                                               terms);
  auto x = op.make_input_vector();
  auto y = op.make_output_vector();
  x.assign(0, std::array{1.0, 0.0, 0.0, 1.0});

  std::array expected{166.0, 188.0, 210.0, 188.0, 214.0, 240.0, 210.0, 240.0, 270.0};
  op.apply(x, y);

  expect_near(y.values(0), expected);
}

TEST(TensorContractionEffectiveHamiltonianOperatorTest, AppliesMultiOutputEnvironmentContraction)
{
  auto a = make_family({{1, 1}, {1, 1}});
  auto b = make_family({{1, 1}, {1, 1}});
  std::array input_blocks{utc::MatrixFamily::Block{1, 1}, utc::MatrixFamily::Block{1, 1}};
  std::array output_blocks{utc::MatrixFamily::Block{1, 1}, utc::MatrixFamily::Block{1, 1}};

  a.assign(0, std::array{2.0});
  a.assign(1, std::array{3.0});
  b.assign(0, std::array{5.0});
  b.assign(1, std::array{7.0});

  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0},
                   utc::EffectiveHamiltonianOperator::Term{1, 1, 1, 1, 1.0}};
  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  auto x = op.make_input_vector();
  auto y = op.make_output_vector();
  x.assign(0, std::array{11.0});
  x.assign(1, std::array{13.0});

  op.apply(x, y);

  EXPECT_DOUBLE_EQ(y.values(0)[0], 110.0);
  EXPECT_DOUBLE_EQ(y.values(1)[0], 273.0);
}

TEST(TensorContractionEffectiveHamiltonianOperatorTest, AppliesMultiOutputDenseEnvironmentContraction)
{
  auto a = make_family({{3, 2}, {3, 2}});
  auto b = make_family({{2, 2}});
  std::array input_blocks{utc::MatrixFamily::Block{2, 3}, utc::MatrixFamily::Block{2, 3}};
  std::array output_blocks{utc::MatrixFamily::Block{3, 3}, utc::MatrixFamily::Block{3, 3}};

  a.assign(0, std::array{1.0, 4.0, 2.0, 5.0, 3.0, 6.0});
  a.assign(1, std::array{7.0, 10.0, 8.0, 11.0, 9.0, 12.0});
  b.assign(0, std::array{1.0, 0.0, 0.0, 1.0});

  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0},
                   utc::EffectiveHamiltonianOperator::Term{1, 1, 0, 1, 1.0}};
  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  auto x = op.make_input_vector();
  auto y = op.make_output_vector();
  x.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  x.assign(1, std::array{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  op.apply(x, y);

  std::array expected0{17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0};
  std::array expected1{149.0, 166.0, 183.0, 166.0, 185.0, 204.0, 183.0, 204.0, 225.0};
  expect_near(y.values(0), expected0);
  expect_near(y.values(1), expected1);
}

TEST(TensorContractionEffectiveHamiltonianOperatorTest, AppliesSparseHeisenbergLikeEnvironmentContraction)
{
  auto a = make_family({{3, 2}, {3, 2}});
  auto b = make_family({{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}});
  std::array input_blocks{utc::MatrixFamily::Block{2, 3}, utc::MatrixFamily::Block{2, 3}};
  std::array output_blocks{utc::MatrixFamily::Block{3, 3}, utc::MatrixFamily::Block{3, 3},
                           utc::MatrixFamily::Block{3, 3}, utc::MatrixFamily::Block{3, 3},
                           utc::MatrixFamily::Block{3, 3}};

  a.assign(0, std::array{1.0, 4.0, 2.0, 5.0, 3.0, 6.0});
  a.assign(1, std::array{7.0, 10.0, 8.0, 11.0, 9.0, 12.0});
  b.assign(0, std::array{1.0, 0.0, 0.0, 1.0});
  for (std::size_t block = 1; block < b.size(); ++block)
  {
    b.assign(block, std::array{0.0, 0.0, 0.0, 0.0});
  }

  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0},
                   utc::EffectiveHamiltonianOperator::Term{0, 1, 0, 1, 1.0},
                   utc::EffectiveHamiltonianOperator::Term{1, 0, 0, 1, 1.0},
                   utc::EffectiveHamiltonianOperator::Term{2, 1, 0, 0, 1.0},
                   utc::EffectiveHamiltonianOperator::Term{3, 0, 0, 0, 0.5},
                   utc::EffectiveHamiltonianOperator::Term{3, 1, 0, 1, -0.5}};
  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  auto x = op.make_input_vector();
  auto y = op.make_output_vector();
  x.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  x.assign(1, std::array{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  op.apply(x, y);

  std::array expected0{166.0, 188.0, 210.0, 188.0, 214.0, 240.0, 210.0, 240.0, 270.0};
  expect_near(y.values(0), expected0);
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

TEST(TensorContractionEffectiveHamiltonianOperatorTest, ResidentCostPlacementMatchesReference)
{
  EnvGuard guard{
      "UNI20_TENSORCONTRACTION_BACKEND",
      "UNI20_TENSORCONTRACTION_DEVICES",
      "UNI20_TENSORCONTRACTION_RABC_PLACEMENT",
  };
  unsetenv("UNI20_TENSORCONTRACTION_BACKEND");
  setenv("UNI20_TENSORCONTRACTION_DEVICES", "all", 1);
  setenv("UNI20_TENSORCONTRACTION_RABC_PLACEMENT", "cost", 1);

  auto a = make_family({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
  auto c = make_family({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
  std::array input_blocks{utc::MatrixFamily::Block{1, 1}, utc::MatrixFamily::Block{1, 1},
                          utc::MatrixFamily::Block{1, 1}, utc::MatrixFamily::Block{1, 1}};
  std::array output_blocks{utc::MatrixFamily::Block{1, 1}, utc::MatrixFamily::Block{1, 1},
                           utc::MatrixFamily::Block{1, 1}, utc::MatrixFamily::Block{1, 1}};

  a.assign(0, std::array{2.0});
  a.assign(1, std::array{3.0});
  a.assign(2, std::array{5.0});
  a.assign(3, std::array{7.0});
  c.assign(0, std::array{11.0});
  c.assign(1, std::array{13.0});
  c.assign(2, std::array{17.0});
  c.assign(3, std::array{19.0});

  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0},
                   utc::EffectiveHamiltonianOperator::Term{1, 1, 1, 1, -0.5},
                   utc::EffectiveHamiltonianOperator::Term{2, 2, 2, 2, 0.25},
                   utc::EffectiveHamiltonianOperator::Term{3, 3, 3, 3, 2.0}};
  auto op = utc::EffectiveHamiltonianOperator::variable_middle(std::move(a), std::move(c), input_blocks, output_blocks,
                                                               terms);
  utc::VectorAlgebraEngine algebra;
  algebra.set_host_synchronization(false);

  auto x = op.make_input_vector();
  auto y = op.make_output_vector();
  x.assign(0, std::array{23.0});
  x.assign(1, std::array{29.0});
  x.assign(2, std::array{31.0});
  x.assign(3, std::array{37.0});
  algebra.upload(x);

  op.apply_resident(x, y, algebra);
  algebra.synchronize(y);

  EXPECT_DOUBLE_EQ(y.values(0)[0], 2.0 * 23.0 * 11.0);
  EXPECT_DOUBLE_EQ(y.values(1)[0], -0.5 * 3.0 * 29.0 * 13.0);
  EXPECT_DOUBLE_EQ(y.values(2)[0], 0.25 * 5.0 * 31.0 * 17.0);
  EXPECT_DOUBLE_EQ(y.values(3)[0], 2.0 * 7.0 * 37.0 * 19.0);
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
