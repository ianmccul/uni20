#include "known_matrices.hpp"
#include "krylov_test_types.hpp"

#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{

template <typename Scalar> class KrylovKnownMatricesRealTypedTest : public ::testing::Test {};

using RealTypes = uni20::krylov::test::KrylovRealTestTypes;
TYPED_TEST_SUITE(KrylovKnownMatricesRealTypedTest, RealTypes);

template <typename Scalar> [[nodiscard]] double eigenvalue_tolerance()
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return 8.0e-4;
  }
  else
  {
    return 2.0e-11;
  }
}

template <typename Scalar> [[nodiscard]] Scalar solver_tolerance()
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return Scalar{1.0e-5F};
  }
  else
  {
    return Scalar{1.0e-12};
  }
}

template <typename Scalar> [[nodiscard]] double stress_residual_tolerance()
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return 2.0e-4;
  }
  else
  {
    return 2.0e-10;
  }
}

template <typename Scalar> [[nodiscard]] uni20::krylov::DenseHostVector<Scalar> deterministic_initial(std::size_t n)
{
  uni20::krylov::DenseHostVector<Scalar> initial{{}};
  initial.values.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    initial.values.push_back(static_cast<Scalar>((i * i + 3 * i + 5) % 17 + 1));
  }
  return initial;
}

template <typename Scalar>
[[nodiscard]] uni20::krylov::DenseHostVector<Scalar> deterministic_initial_with_small_extremal_projection(std::size_t n)
{
  auto initial = deterministic_initial<Scalar>(n);
  if (n >= 6)
  {
    Scalar const small = std::is_same_v<Scalar, float> ? Scalar{1.0e-3F} : Scalar{1.0e-8};
    initial.values[0] = small;
    initial.values[1] = small;
    initial.values[2] = small;
    initial.values[n - 3] = small;
    initial.values[n - 2] = small;
    initial.values[n - 1] = small;
  }
  return initial;
}

template <typename Scalar>
void expect_values_near(std::vector<Scalar> const& actual, std::vector<Scalar> const& expected, double tolerance)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
  {
    EXPECT_NEAR(static_cast<double>(actual[i]), static_cast<double>(expected[i]), tolerance);
  }
}

template <typename Scalar>
[[nodiscard]] double relative_residual_norm(uni20::krylov::DenseHostVectorOps<Scalar>& ops,
                                            uni20::krylov::DenseHostVector<Scalar> const& eigenvector,
                                            Scalar eigenvalue)
{
  auto residual = ops.allocate_like(eigenvector);
  ops.matvec(residual, eigenvector);
  ops.axpy(residual, -eigenvalue, eigenvector);

  double norm_squared = 0.0;
  for (Scalar value : residual.values)
  {
    double const host_value = static_cast<double>(value);
    norm_squared += host_value * host_value;
  }
  return std::sqrt(norm_squared) / std::max(1.0, std::abs(static_cast<double>(eigenvalue)));
}

template <typename Scalar> void expect_unique_within(std::vector<Scalar> values, double minimum_separation)
{
  std::ranges::sort(values);
  for (std::size_t i = 1; i < values.size(); ++i)
  {
    double const separation = std::abs(static_cast<double>(values[i] - values[i - 1]));
    EXPECT_GT(separation, minimum_separation);
  }
}

template <typename Scalar>
void expect_largest_algebraic_eigenvalues(uni20::krylov::test_matrices::KnownSymmetricMatrix<Scalar> const& matrix,
                                          int eigenvalue_count)
{
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::KrylovDiagnosticsLevel;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<Scalar> ops(matrix.dimension, matrix.row_major_values);
  auto initial = deterministic_initial<Scalar>(matrix.dimension);

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = eigenvalue_count;
  params.krylov_dimension = std::min<int>(static_cast<int>(matrix.dimension), std::max(2 * eigenvalue_count + 2, 6));
  params.max_iterations = 400;
  params.tolerance = solver_tolerance<Scalar>();
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<Scalar>(ops, initial, params);

  ASSERT_EQ(result.status, 0) << matrix.metadata.id;
  ASSERT_EQ(result.converged_count, eigenvalue_count) << matrix.metadata.id;
  ASSERT_EQ(result.eigenvalues.size(), static_cast<std::size_t>(eigenvalue_count)) << matrix.metadata.id;

  std::vector<Scalar> expected;
  expected.reserve(static_cast<std::size_t>(eigenvalue_count));
  for (int i = 0; i < eigenvalue_count; ++i)
  {
    expected.push_back(
        matrix.eigenvalues_ascending[matrix.eigenvalues_ascending.size() - 1U - static_cast<std::size_t>(i)]);
  }
  expect_values_near(result.eigenvalues, expected, eigenvalue_tolerance<Scalar>());
}

template <typename Scalar>
void expect_smallest_algebraic_eigenvalues(uni20::krylov::test_matrices::KnownSymmetricMatrix<Scalar> const& matrix,
                                           int eigenvalue_count)
{
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::KrylovDiagnosticsLevel;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<Scalar> ops(matrix.dimension, matrix.row_major_values);
  auto initial = deterministic_initial<Scalar>(matrix.dimension);

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = eigenvalue_count;
  params.krylov_dimension = std::min<int>(static_cast<int>(matrix.dimension), std::max(2 * eigenvalue_count + 2, 6));
  params.max_iterations = 400;
  params.tolerance = solver_tolerance<Scalar>();
  params.spectrum = SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = false;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<Scalar>(ops, initial, params);

  ASSERT_EQ(result.status, 0) << matrix.metadata.id;
  ASSERT_EQ(result.converged_count, eigenvalue_count) << matrix.metadata.id;
  ASSERT_EQ(result.eigenvalues.size(), static_cast<std::size_t>(eigenvalue_count)) << matrix.metadata.id;

  std::vector<Scalar> expected(matrix.eigenvalues_ascending.begin(),
                               matrix.eigenvalues_ascending.begin() + eigenvalue_count);
  expect_values_near(result.eigenvalues, expected, eigenvalue_tolerance<Scalar>());
}

TEST(KrylovKnownMatrices, MetadataKeepsProvenanceAndOracleType)
{
  using uni20::krylov::test_matrices::MatrixOracleKind;

  auto const toeplitz = uni20::krylov::test_matrices::symmetric_tridiagonal_toeplitz<double>(5, 2.0, -1.0);
  EXPECT_EQ(toeplitz.metadata.id, std::string_view("symmetric_tridiagonal_toeplitz"));
  EXPECT_EQ(toeplitz.metadata.oracle, MatrixOracleKind::ExactEigenvaluesAndCondition);
  EXPECT_TRUE(toeplitz.metadata.symmetric);
  EXPECT_TRUE(toeplitz.metadata.sparse);

  auto const symmstoch = uni20::krylov::test_matrices::symmstoch_with_spectrum<double>({8.0, 5.0, 3.0, 1.0});
  EXPECT_EQ(symmstoch.metadata.license, std::string_view("BSD-2-Clause"));
  EXPECT_EQ(symmstoch.metadata.source, std::string_view("Anymatrix core/symmstoch and core/soules"));
  EXPECT_EQ(symmstoch.metadata.oracle, MatrixOracleKind::ExactEigenvaluesAndCondition);
  EXPECT_TRUE(symmstoch.metadata.symmetric);
  EXPECT_FALSE(symmstoch.metadata.sparse);
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, GeneratesExactTridiagonalToeplitzSpectrum)
{
  using Scalar = TypeParam;

  auto const matrix = uni20::krylov::test_matrices::symmetric_tridiagonal_toeplitz<Scalar>(4, Scalar{2}, Scalar{-1});
  ASSERT_EQ(matrix.dimension, 4);
  ASSERT_EQ(matrix.eigenvalues_ascending.size(), 4);
  EXPECT_NEAR(static_cast<double>(matrix.eigenvalues_ascending.front()), 0.3819660112501051,
              eigenvalue_tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(matrix.eigenvalues_ascending.back()), 3.618033988749895,
              eigenvalue_tolerance<Scalar>());
  ASSERT_TRUE(matrix.two_norm_condition_number.has_value());
  EXPECT_NEAR(static_cast<double>(*matrix.two_norm_condition_number), 9.47213595499958,
              20.0 * eigenvalue_tolerance<Scalar>());
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, GeneratesExactPathLaplacianSpectrum)
{
  using Scalar = TypeParam;

  auto const matrix = uni20::krylov::test_matrices::path_laplacian<Scalar>(5);
  ASSERT_EQ(matrix.dimension, 5);
  ASSERT_EQ(matrix.eigenvalues_ascending.size(), 5);
  EXPECT_NEAR(static_cast<double>(matrix.eigenvalues_ascending.front()), 0.0, eigenvalue_tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(matrix.eigenvalues_ascending.back()), 3.618033988749895,
              eigenvalue_tolerance<Scalar>());
  EXPECT_FALSE(matrix.two_norm_condition_number.has_value());
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, GeneratesAnymatrixSymmstochWithPrescribedSpectrum)
{
  using Scalar = TypeParam;

  auto const matrix =
      uni20::krylov::test_matrices::symmstoch_with_spectrum<Scalar>({Scalar{8}, Scalar{5}, Scalar{3}, Scalar{1}});
  ASSERT_EQ(matrix.dimension, 4);
  expect_values_near(matrix.eigenvalues_ascending,
                     std::vector<Scalar>{Scalar{0.125}, Scalar{0.375}, Scalar{0.625}, Scalar{1}},
                     eigenvalue_tolerance<Scalar>());
  ASSERT_TRUE(matrix.two_norm_condition_number.has_value());
  EXPECT_NEAR(static_cast<double>(*matrix.two_norm_condition_number), 8.0, eigenvalue_tolerance<Scalar>());
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, GeneratesAnymatrixBlockhouseCoordinateReflectorSpectrum)
{
  using Scalar = TypeParam;

  auto const matrix = uni20::krylov::test_matrices::blockhouse_coordinate_reflector<Scalar>(6, 2);
  expect_values_near(matrix.eigenvalues_ascending,
                     std::vector<Scalar>{Scalar{-1}, Scalar{-1}, Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}},
                     eigenvalue_tolerance<Scalar>());
  ASSERT_TRUE(matrix.two_norm_condition_number.has_value());
  EXPECT_EQ(*matrix.two_norm_condition_number, Scalar{1});
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, NativeLanczosSolvesToeplitzLargestAndSmallestAlgebraic)
{
  using Scalar = TypeParam;

  auto const matrix = uni20::krylov::test_matrices::symmetric_tridiagonal_toeplitz<Scalar>(12, Scalar{2}, Scalar{-1});
  expect_largest_algebraic_eigenvalues(matrix, 2);
  expect_smallest_algebraic_eigenvalues(matrix, 2);
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, NativeLanczosSolvesPathLaplacianLargestAlgebraic)
{
  using Scalar = TypeParam;

  auto const matrix = uni20::krylov::test_matrices::path_laplacian<Scalar>(12);
  expect_largest_algebraic_eigenvalues(matrix, 2);
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, StressLanczosGhostsWithClusteredExtremes)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::KrylovDiagnosticsLevel;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  auto const matrix = uni20::krylov::test_matrices::diagonal_clustered_extremes<Scalar>(96);
  DenseHostVectorOps<Scalar> ops(matrix.dimension, matrix.row_major_values);
  auto initial = deterministic_initial_with_small_extremal_projection<Scalar>(matrix.dimension);

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = 6;
  params.krylov_dimension = 20;
  params.max_iterations = 900;
  params.tolerance = solver_tolerance<Scalar>();
  params.spectrum = SpectrumPart::LargestMagnitude;
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<Scalar>(ops, initial, params);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.converged_count, params.eigenvalue_count);
  ASSERT_EQ(result.eigenvalues.size(), static_cast<std::size_t>(params.eigenvalue_count));
  ASSERT_EQ(result.eigenvectors.size(), result.eigenvalues.size());
  expect_unique_within(result.eigenvalues, std::is_same_v<Scalar, float> ? 0.2 : 1.0e-5);

  for (std::size_t i = 0; i < result.eigenvalues.size(); ++i)
  {
    EXPECT_LE(relative_residual_norm(ops, result.eigenvectors[i], result.eigenvalues[i]),
              stress_residual_tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, StressLargeShiftedLaplacianConvergesWithoutFalseBreakdown)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::KrylovDiagnosticsLevel;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  Scalar const shift = std::is_same_v<Scalar, float> ? Scalar{1.0e5F} : Scalar{1.0e12};
  auto const matrix = uni20::krylov::test_matrices::shifted_path_laplacian<Scalar>(24, shift);
  DenseHostVectorOps<Scalar> ops(matrix.dimension, matrix.row_major_values);
  auto initial = deterministic_initial<Scalar>(matrix.dimension);

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 10;
  params.max_iterations = 500;
  params.tolerance = solver_tolerance<Scalar>();
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<Scalar>(ops, initial, params);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.converged_count, params.eigenvalue_count);
  ASSERT_EQ(result.eigenvectors.size(), result.eigenvalues.size());
  for (std::size_t i = 0; i < result.eigenvalues.size(); ++i)
  {
    EXPECT_LE(relative_residual_norm(ops, result.eigenvectors[i], result.eigenvalues[i]),
              stress_residual_tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, StressInteriorSmallestMagnitudeFailsCleanlyInOrdinaryMode)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::KrylovDiagnosticsLevel;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  Scalar const gap = std::is_same_v<Scalar, float> ? Scalar{1.0e-3F} : Scalar{1.0e-9};
  auto const matrix = uni20::krylov::test_matrices::symmetric_interior_gap<Scalar>(96, gap);
  DenseHostVectorOps<Scalar> ops(matrix.dimension, matrix.row_major_values);
  auto initial = deterministic_initial<Scalar>(matrix.dimension);

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 10;
  params.max_iterations = 30;
  params.tolerance = solver_tolerance<Scalar>();
  params.spectrum = SpectrumPart::SmallestMagnitude;
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Full;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<Scalar>(ops, initial, params);

  EXPECT_NE(result.status, 0);
  EXPECT_LT(result.converged_count, params.eigenvalue_count);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->op_count, result.matvec_count);
  for (Scalar residual_bound : result.residual_bounds)
  {
    EXPECT_TRUE(std::isfinite(static_cast<double>(residual_bound)));
  }
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, StressClusteredWantedEndNeedsLargerSearchSpace)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::KrylovDiagnosticsLevel;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  Scalar const gap = std::is_same_v<Scalar, float> ? Scalar{2.0e-4F} : Scalar{1.0e-10};
  auto const matrix = uni20::krylov::test_matrices::diagonal_clustered_wanted_end<Scalar>(128, 8, gap);
  auto initial = deterministic_initial<Scalar>(matrix.dimension);

  SymmetricEigenParams<Scalar> cramped_params;
  cramped_params.eigenvalue_count = 6;
  cramped_params.krylov_dimension = 8;
  cramped_params.max_iterations = 80;
  cramped_params.tolerance = solver_tolerance<Scalar>();
  cramped_params.spectrum = SpectrumPart::LargestAlgebraic;
  cramped_params.compute_eigenvectors = true;
  cramped_params.diagnostics = KrylovDiagnosticsLevel::Full;

  DenseHostVectorOps<Scalar> cramped_ops(matrix.dimension, matrix.row_major_values);
  auto cramped = uni20::krylov::symmetric_lanczos_restarted_standard<Scalar>(cramped_ops, initial, cramped_params);

  EXPECT_NE(cramped.status, 0);
  EXPECT_LT(cramped.converged_count, cramped_params.eigenvalue_count);
  ASSERT_TRUE(cramped.diagnostics.has_value());
  EXPECT_EQ(cramped.diagnostics->op_count, cramped.matvec_count);

  SymmetricEigenParams<Scalar> roomy_params = cramped_params;
  roomy_params.krylov_dimension = 32;
  roomy_params.retained_ritz_count = 10;
  roomy_params.max_iterations = 1200;

  DenseHostVectorOps<Scalar> roomy_ops(matrix.dimension, matrix.row_major_values);
  auto roomy = uni20::krylov::symmetric_lanczos_restarted_standard<Scalar>(roomy_ops, initial, roomy_params);

  ASSERT_EQ(roomy.status, 0);
  ASSERT_EQ(roomy.converged_count, roomy_params.eigenvalue_count);
  ASSERT_EQ(roomy.eigenvalues.size(), static_cast<std::size_t>(roomy_params.eigenvalue_count));
  ASSERT_EQ(roomy.eigenvectors.size(), roomy.eigenvalues.size());

  std::vector<Scalar> expected;
  expected.reserve(static_cast<std::size_t>(roomy_params.eigenvalue_count));
  for (int i = 0; i < roomy_params.eigenvalue_count; ++i)
  {
    expected.push_back(
        matrix.eigenvalues_ascending[matrix.eigenvalues_ascending.size() - 1U - static_cast<std::size_t>(i)]);
  }
  expect_values_near(roomy.eigenvalues, expected, 10.0 * eigenvalue_tolerance<Scalar>());
  for (std::size_t i = 0; i < roomy.eigenvalues.size(); ++i)
  {
    EXPECT_LE(relative_residual_norm(roomy_ops, roomy.eigenvectors[i], roomy.eigenvalues[i]),
              stress_residual_tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovKnownMatricesRealTypedTest, NativeLanczosSolvesAnymatrixSymmstochLargestAlgebraic)
{
  using Scalar = TypeParam;

  auto const matrix = uni20::krylov::test_matrices::symmstoch_with_spectrum<Scalar>(
      {Scalar{13}, Scalar{8}, Scalar{5}, Scalar{3}, Scalar{2}, Scalar{1}});
  expect_largest_algebraic_eigenvalues(matrix, 2);
}

} // namespace
