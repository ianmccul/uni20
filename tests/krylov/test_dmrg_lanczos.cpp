#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/tensor_network/dmrg_lanczos.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace
{

TEST(DmrgLanczosTest, PerformsExactlyFourMatvecsWithoutFullReorthogonalization)
{
  using Vector = uni20::krylov::DenseHostVector<double>;
  std::vector<double> matrix(36);
  for (std::size_t index = 0; index < 6; ++index)
    matrix[index * 6 + index] = static_cast<double>(index + 1);
  uni20::krylov::DenseHostVectorOps<double> ops(6, std::move(matrix));
  Vector const initial{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  auto const result = uni20::tensor_network::dmrg_lanczos_ground_state<double>(ops, initial);

  EXPECT_EQ(result.iteration_count, 4);
  EXPECT_EQ(result.matvec_count, 4);
  EXPECT_EQ(ops.matvec_count(), 4);
  EXPECT_EQ(ops.inner_product_count(), 4);
  EXPECT_EQ(ops.norm_count(), 5);
  EXPECT_EQ(ops.set_zero_count(), 0);
  EXPECT_GT(result.energy, 1.0);
  EXPECT_LT(result.energy, 1.1);
  EXPECT_GT(result.residual_bound, 0.0);
  double norm_squared = 0.0;
  for (double const value : result.vector.values)
    norm_squared += value * value;
  EXPECT_NEAR(norm_squared, 1.0, 1.0e-13);
}

TEST(DmrgLanczosTest, StopsAtExactInvariantSubspaceBreakdown)
{
  using Vector = uni20::krylov::DenseHostVector<double>;
  uni20::krylov::DenseHostVectorOps<double> ops(3, {2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0});
  Vector const initial{{1.0, 0.0, 0.0}};

  auto const result = uni20::tensor_network::dmrg_lanczos_ground_state<double>(ops, initial, {.matvec_iterations = 4});

  EXPECT_EQ(result.iteration_count, 1);
  EXPECT_EQ(result.matvec_count, 1);
  EXPECT_DOUBLE_EQ(result.energy, 2.0);
  EXPECT_DOUBLE_EQ(result.residual_bound, 0.0);
  ASSERT_EQ(result.vector.values.size(), 3U);
  EXPECT_NEAR(std::abs(result.vector.values[0]), 1.0, 1.0e-14);
}

TEST(DmrgLanczosTest, KeepsTheFixedBudgetForAUniformlySmallHamiltonian)
{
  using Vector = uni20::krylov::DenseHostVector<double>;
  std::vector<double> matrix(36);
  for (std::size_t index = 0; index < 6; ++index)
    matrix[index * 6 + index] = 1.0e-20 * static_cast<double>(index + 1);
  uni20::krylov::DenseHostVectorOps<double> ops(6, std::move(matrix));
  Vector const initial{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  auto const result = uni20::tensor_network::dmrg_lanczos_ground_state<double>(ops, initial);

  EXPECT_EQ(result.matvec_count, 4);
  EXPECT_GT(result.energy, 1.0e-20);
  EXPECT_LT(result.energy, 1.1e-20);
}

TEST(DmrgLanczosTest, NormalizesUniformlyTinyInitialAndResidualVectors)
{
  using Vector = uni20::krylov::DenseHostVector<double>;
  uni20::krylov::DenseHostVectorOps<double> ops(2, {0.0, 1.0e-310, 1.0e-310, 0.0});
  Vector const initial{{1.0e-310, 0.0}};

  auto const result = uni20::tensor_network::dmrg_lanczos_ground_state<double>(ops, initial, {.matvec_iterations = 2});

  EXPECT_EQ(result.matvec_count, 2);
  EXPECT_TRUE(std::isfinite(result.energy));
  ASSERT_EQ(result.vector.values.size(), 2U);
  EXPECT_TRUE(std::isfinite(result.vector.values[0]));
  EXPECT_TRUE(std::isfinite(result.vector.values[1]));
  EXPECT_NEAR(std::hypot(result.vector.values[0], result.vector.values[1]), 1.0, 1.0e-14);
}

TEST(DmrgLanczosTest, SupportsComplexHermitianLocalWavefunctions)
{
  using Scalar = uni20::complex<double>;
  using Vector = uni20::krylov::DenseHostVector<Scalar>;
  uni20::krylov::DenseHostVectorOps<Scalar> ops(
      2, {Scalar{2.0, 0.0}, Scalar{0.0, 1.0}, Scalar{0.0, -1.0}, Scalar{3.0, 0.0}});
  Vector const initial{{Scalar{1.0, 0.0}, Scalar{0.0, 0.0}}};

  auto const result = uni20::tensor_network::dmrg_lanczos_ground_state<Scalar>(ops, initial, {.matvec_iterations = 4});

  EXPECT_EQ(result.matvec_count, 2);
  EXPECT_NEAR(result.energy, (5.0 - std::sqrt(5.0)) / 2.0, 1.0e-13);
  EXPECT_LT(result.residual_bound, 1.0e-13);
}

TEST(DmrgLanczosTest, RejectsAZeroIterationBudget)
{
  using Vector = uni20::krylov::DenseHostVector<double>;
  uni20::krylov::DenseHostVectorOps<double> ops(1, {1.0});
  Vector const initial{{1.0}};

  EXPECT_THROW(static_cast<void>(
                   uni20::tensor_network::dmrg_lanczos_ground_state<double>(ops, initial, {.matvec_iterations = 0})),
               std::invalid_argument);
  EXPECT_EQ(ops.matvec_count(), 0);
}

} // namespace
