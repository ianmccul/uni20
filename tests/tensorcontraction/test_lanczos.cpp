#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/lanczos.hpp>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace utc = uni20::tensorcontraction;

namespace
{

utc::MatrixFamily make_vector(std::initializer_list<double> values)
{
  std::array blocks{utc::MatrixFamily::Block{values.size(), 1}};
  utc::MatrixFamily x(blocks);
  x.assign(0, std::span{values.begin(), values.size()});
  return x;
}

void dense_matvec(std::span<double const> matrix, std::size_t n, utc::MatrixFamily const& x, utc::MatrixFamily& y)
{
  auto const x_values = x.values(0);
  auto y_values = y.values(0);
  std::fill(y_values.begin(), y_values.end(), 0.0);
  for (std::size_t row = 0; row < n; ++row)
  {
    for (std::size_t col = 0; col < n; ++col)
    {
      y_values[row] += matrix[row * n + col] * x_values[col];
    }
  }
}

} // namespace

TEST(TensorContractionLanczosTest, FindsLowestEigenpairForDiagonalMatrix)
{
  std::array matrix{4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0};
  auto guess = make_vector({1.0, 1.0, 1.0});
  utc::LanczosOptions options{.max_iterations = 6, .min_iterations = 2, .tolerance = 1.0e-12};

  auto result = utc::lanczos_lowest(guess, [&](auto const& x, auto& y) { dense_matvec(matrix, 3, x, y); }, options);

  EXPECT_NEAR(result.eigenvalue, -1.0, 1.0e-12);
  EXPECT_EQ(result.stop_reason, utc::LanczosStopReason::InvariantSubspace);
  EXPECT_LE(result.iterations, 3);
  EXPECT_NEAR(std::abs(guess.values(0)[2]), 1.0, 1.0e-12);
}

TEST(TensorContractionLanczosTest, FindsLowestEigenpairForSymmetricTwoByTwoMatrix)
{
  std::array matrix{2.0, 1.0, 1.0, 2.0};
  auto guess = make_vector({0.25, 1.0});
  utc::LanczosOptions options{.max_iterations = 4, .min_iterations = 2, .tolerance = 1.0e-12};

  auto result = utc::lanczos_lowest(guess, [&](auto const& x, auto& y) { dense_matvec(matrix, 2, x, y); }, options);

  EXPECT_NEAR(result.eigenvalue, 1.0, 1.0e-12);
  EXPECT_LE(result.iterations, 2);
  EXPECT_NEAR(std::abs(guess.values(0)[0]), 1.0 / std::sqrt(2.0), 1.0e-12);
  EXPECT_NEAR(std::abs(guess.values(0)[1]), 1.0 / std::sqrt(2.0), 1.0e-12);
  EXPECT_LT(guess.values(0)[0] * guess.values(0)[1], 0.0);
}

TEST(TensorContractionLanczosTest, ReportsMaxIterationsWhenToleranceIsNotMet)
{
  std::array matrix{3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0};
  auto guess = make_vector({1.0, 0.5, -0.25});
  utc::LanczosOptions options{.max_iterations = 2, .min_iterations = 2, .tolerance = 1.0e-16};

  auto result = utc::lanczos_lowest(guess, [&](auto const& x, auto& y) { dense_matvec(matrix, 3, x, y); }, options);

  EXPECT_EQ(result.stop_reason, utc::LanczosStopReason::MaxIterations);
  EXPECT_EQ(result.iterations, 2);
  EXPECT_LT(result.tolerance, 0.0);
  EXPECT_GT(result.residual_norm, 0.0);
}

TEST(TensorContractionLanczosTest, RejectsInvalidOptionsAndZeroInitialGuess)
{
  auto guess = make_vector({0.0, 0.0});
  std::array matrix{1.0, 0.0, 0.0, 2.0};
  EXPECT_THROW(utc::lanczos_lowest(guess, [&](auto const& x, auto& y) { dense_matvec(matrix, 2, x, y); }),
               std::invalid_argument);

  auto nonzero = make_vector({1.0, 0.0});
  utc::LanczosOptions options{.max_iterations = 1, .min_iterations = 2, .tolerance = 1.0e-12};
  EXPECT_THROW(utc::lanczos_lowest(
                   nonzero, [&](auto const& x, auto& y) { dense_matvec(matrix, 2, x, y); }, options),
               std::invalid_argument);
}

TEST(TensorContractionLanczosTest, RunsAgainstEffectiveHamiltonianOperator)
{
  auto a_blocks = std::array{utc::MatrixFamily::Block{2, 2}};
  auto b_blocks = std::array{utc::MatrixFamily::Block{2, 2}};
  auto input_blocks = std::array{utc::MatrixFamily::Block{2, 1}};
  auto output_blocks = std::array{utc::MatrixFamily::Block{2, 1}};
  utc::MatrixFamily a(a_blocks);
  utc::MatrixFamily b(b_blocks);
  a.assign(0, std::array{1.0, 0.0, 0.0, 1.0});
  b.assign(0, std::array{2.0, 1.0, 1.0, 2.0});
  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0}};
  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  auto guess = make_vector({0.25, 1.0});

  auto result = utc::lanczos_lowest(
      guess, [&](auto const& x, auto& y) { op.apply(x, y); },
      utc::LanczosOptions{.max_iterations = 4, .min_iterations = 2, .tolerance = 1.0e-12});

  EXPECT_NEAR(result.eigenvalue, 1.0, 1.0e-12);
  EXPECT_NEAR(utc::norm(guess), 1.0, 1.0e-12);
  EXPECT_LT(guess.values(0)[0] * guess.values(0)[1], 0.0);
}

TEST(TensorContractionLanczosTest, RunsResidentAgainstEffectiveHamiltonianOperator)
{
  auto a_blocks = std::array{utc::MatrixFamily::Block{2, 2}};
  auto b_blocks = std::array{utc::MatrixFamily::Block{2, 2}};
  auto input_blocks = std::array{utc::MatrixFamily::Block{2, 1}};
  auto output_blocks = std::array{utc::MatrixFamily::Block{2, 1}};
  utc::MatrixFamily a(a_blocks);
  utc::MatrixFamily b(b_blocks);
  a.assign(0, std::array{1.0, 0.0, 0.0, 1.0});
  b.assign(0, std::array{2.0, 1.0, 1.0, 2.0});
  std::array terms{utc::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0}};
  utc::EffectiveHamiltonianOperator op(std::move(a), std::move(b), input_blocks, output_blocks, terms);
  auto guess = make_vector({0.25, 1.0});
  utc::VectorAlgebraEngine algebra;

  auto apply = [&](utc::MatrixFamily const& x, utc::MatrixFamily& y) { op.apply_resident(x, y, algebra); };
  auto result = utc::lanczos_lowest_with_engine(
      guess, apply, algebra, utc::LanczosOptions{.max_iterations = 4, .min_iterations = 2, .tolerance = 1.0e-12});

  EXPECT_NEAR(result.eigenvalue, 1.0, 1.0e-12);
  algebra.synchronize(guess);
  EXPECT_NEAR(utc::norm(guess), 1.0, 1.0e-12);
  EXPECT_LT(guess.values(0)[0] * guess.values(0)[1], 0.0);
}
