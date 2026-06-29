#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>

#include "krylov_test_types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{

template <typename Scalar> class KrylovSymmetricLanczosRealTypedTest : public ::testing::Test {};
template <typename Scalar> class KrylovHermitianLanczosComplexTypedTest : public ::testing::Test {};

using RealTypes = uni20::krylov::test::KrylovRealTestTypes;
using ComplexTypes = uni20::krylov::test::KrylovComplexTestTypes;
TYPED_TEST_SUITE(KrylovSymmetricLanczosRealTypedTest, RealTypes);
TYPED_TEST_SUITE(KrylovHermitianLanczosComplexTypedTest, ComplexTypes);

template <typename Scalar> Scalar scalar_tolerance()
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return Scalar{1.0e-4F};
  }
  else
  {
    return Scalar{1.0e-11};
  }
}

std::vector<double> two_dimensional_laplacian(std::size_t nx)
{
  std::size_t const n = nx * nx;
  std::vector<double> matrix(n * n, 0.0);

  auto index = [nx](std::size_t row, std::size_t col) { return row * nx + col; };
  for (std::size_t row = 0; row < nx; ++row)
  {
    for (std::size_t col = 0; col < nx; ++col)
    {
      auto const center = index(row, col);
      matrix[center * n + center] = 4.0;
      if (row > 0)
      {
        matrix[center * n + index(row - 1, col)] = -1.0;
      }
      if (row + 1 < nx)
      {
        matrix[center * n + index(row + 1, col)] = -1.0;
      }
      if (col > 0)
      {
        matrix[center * n + index(row, col - 1)] = -1.0;
      }
      if (col + 1 < nx)
      {
        matrix[center * n + index(row, col + 1)] = -1.0;
      }
    }
  }

  return matrix;
}

std::vector<double> exact_two_dimensional_laplacian_eigenvalues(std::size_t nx)
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  std::vector<double> eigenvalues;
  eigenvalues.reserve(nx * nx);
  for (std::size_t row = 1; row <= nx; ++row)
  {
    for (std::size_t col = 1; col <= nx; ++col)
    {
      double const theta_row = static_cast<double>(row) * pi / static_cast<double>(nx + 1);
      double const theta_col = static_cast<double>(col) * pi / static_cast<double>(nx + 1);
      eigenvalues.push_back(4.0 - 2.0 * std::cos(theta_row) - 2.0 * std::cos(theta_col));
    }
  }
  std::sort(eigenvalues.begin(), eigenvalues.end());
  return eigenvalues;
}

uni20::krylov::Matrix<double> tridiagonal_matrix(std::vector<double> const& diagonal,
                                                 std::vector<double> const& subdiagonal)
{
  uni20::krylov::Matrix<double> matrix(diagonal.size(), diagonal.size());
  uni20::krylov::laset(matrix, 0.0, 0.0, uni20::krylov::MatrixFill::All);
  for (std::size_t i = 0; i < diagonal.size(); ++i)
  {
    matrix[i, i] = diagonal[i];
    if (i + 1 < diagonal.size())
    {
      matrix[i, i + 1] = subdiagonal[i];
      matrix[i + 1, i] = subdiagonal[i];
    }
  }
  return matrix;
}

uni20::krylov::Matrix<double> multiply(uni20::krylov::Matrix<double> const& lhs,
                                       uni20::krylov::Matrix<double> const& rhs)
{
  if (lhs.cols() != rhs.rows())
  {
    throw std::invalid_argument("test matrix multiplication size mismatch");
  }
  uni20::krylov::Matrix<double> result(lhs.rows(), rhs.cols());
  for (std::size_t col = 0; col < rhs.cols(); ++col)
  {
    for (std::size_t k = 0; k < lhs.cols(); ++k)
    {
      for (std::size_t row = 0; row < lhs.rows(); ++row)
      {
        result[row, col] += lhs[row, k] * rhs[k, col];
      }
    }
  }
  return result;
}

uni20::krylov::Matrix<double> transpose(uni20::krylov::Matrix<double> const& matrix)
{
  uni20::krylov::Matrix<double> result(matrix.cols(), matrix.rows());
  for (std::size_t col = 0; col < matrix.cols(); ++col)
  {
    for (std::size_t row = 0; row < matrix.rows(); ++row)
    {
      result[col, row] = matrix[row, col];
    }
  }
  return result;
}

void expect_matrix_near(uni20::krylov::Matrix<double> const& actual, uni20::krylov::Matrix<double> const& expected,
                        double tolerance)
{
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  for (std::size_t col = 0; col < actual.cols(); ++col)
  {
    for (std::size_t row = 0; row < actual.rows(); ++row)
    {
      EXPECT_NEAR((actual[row, col]), (expected[row, col]), tolerance);
    }
  }
}

void expect_vector_near(uni20::krylov::DenseHostVector<double> const& actual, std::vector<double> const& expected,
                        double tolerance)
{
  ASSERT_EQ(actual.values.size(), expected.size());
  for (std::size_t i = 0; i < actual.values.size(); ++i)
  {
    EXPECT_NEAR(actual.values[i], expected[i], tolerance);
  }
}

template <typename Scalar> std::vector<Scalar> diagonal_dense_matrix(std::vector<Scalar> const& diagonal)
{
  std::size_t const dimension = diagonal.size();
  std::vector<Scalar> matrix(dimension * dimension, Scalar{});
  for (std::size_t i = 0; i < dimension; ++i)
  {
    matrix[i * dimension + i] = diagonal[i];
  }
  return matrix;
}

template <typename Complex> std::vector<Complex> phase_laplacian_dense_matrix(std::size_t dimension)
{
  using Real = typename Complex::value_type;

  std::vector<Complex> matrix(dimension * dimension, Complex{});
  for (std::size_t i = 0; i < dimension; ++i)
  {
    matrix[i * dimension + i] = Complex{Real{2}};
    if (i + 1 < dimension)
    {
      Real const theta = Real{0.2} + Real{0.05} * static_cast<Real>(i);
      Complex const upper = -std::polar(Real{1}, theta);
      matrix[i * dimension + i + 1] = upper;
      matrix[(i + 1) * dimension + i] = std::conj(upper);
    }
  }
  return matrix;
}

std::vector<double> exact_path_laplacian_eigenvalues(std::size_t dimension)
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  std::vector<double> eigenvalues;
  eigenvalues.reserve(dimension);
  for (std::size_t k = 1; k <= dimension; ++k)
  {
    double const theta = static_cast<double>(k) * pi / static_cast<double>(dimension + 1);
    eigenvalues.push_back(2.0 - 2.0 * std::cos(theta));
  }
  return eigenvalues;
}

TEST(KrylovSymmetricLanczos, SolvesDiagonalLargestAlgebraicProblem)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(3, {
                                        2.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        3.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        5.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 3;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  EXPECT_NEAR(result.eigenvalues[0], 5.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[1], 3.0, 1.0e-12);
  EXPECT_LT(result.residual_bounds[0], 1.0e-12);
  EXPECT_LT(result.residual_bounds[1], 1.0e-12);
  EXPECT_TRUE(result.eigenvectors.empty());
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_EQ(result.iteration_count, 3);
  EXPECT_EQ(result.matvec_count, 3);
  EXPECT_EQ(result.status, 0);
  EXPECT_EQ(ops.matvec_count(), 3);
}

TYPED_TEST(KrylovSymmetricLanczosRealTypedTest, SolvesDiagonalLargestAlgebraicProblem)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<Scalar> ops(3, diagonal_dense_matrix<Scalar>({Scalar{2}, Scalar{3}, Scalar{5}}));
  DenseHostVector<Scalar> initial{{Scalar{1}, Scalar{1}, Scalar{1}}};

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 3;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_standard<Scalar>(ops, initial, params);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 5.0, static_cast<double>(scalar_tolerance<Scalar>()));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, static_cast<double>(scalar_tolerance<Scalar>()));
  EXPECT_LT(static_cast<double>(result.residual_bounds[0]), static_cast<double>(scalar_tolerance<Scalar>()));
  EXPECT_LT(static_cast<double>(result.residual_bounds[1]), static_cast<double>(scalar_tolerance<Scalar>()));
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_EQ(result.matvec_count, 3);
}

TYPED_TEST(KrylovHermitianLanczosComplexTypedTest, SolvesImaginaryOffDiagonalHermitianProblem)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  Complex const i{Real{}, Real{1}};
  DenseHostVectorOps<Complex> ops(2, {
                                         Complex{Real{2}},
                                         i,
                                         -i,
                                         Complex{Real{2}},
                                     });
  DenseHostVector<Complex> initial{{Complex{Real{1}}, Complex{Real{1}, Real{0.25}}}};

  SymmetricEigenParams<Real> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 2;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_standard<Complex>(ops, initial, params);

  Real const tolerance = std::is_same_v<Real, float> ? Real{1.0e-4F} : Real{1.0e-12};
  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 3.0, static_cast<double>(tolerance));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 1.0, static_cast<double>(tolerance));
  EXPECT_LT(static_cast<double>(result.residual_bounds[0]), static_cast<double>(tolerance));
  EXPECT_LT(static_cast<double>(result.residual_bounds[1]), static_cast<double>(tolerance));
}

TYPED_TEST(KrylovHermitianLanczosComplexTypedTest, RestartedSolveConvergesOnPhaseTwistedLaplacian)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  constexpr std::size_t dimension = 8;
  DenseHostVectorOps<Complex> ops(dimension, phase_laplacian_dense_matrix<Complex>(dimension));
  DenseHostVector<Complex> initial{{Complex{Real{1}, Real{0.1}}, Complex{Real{1.1}, Real{-0.2}},
                                    Complex{Real{0.9}, Real{0.3}}, Complex{Real{1.2}, Real{-0.4}},
                                    Complex{Real{0.8}, Real{0.5}}, Complex{Real{1.3}, Real{-0.6}},
                                    Complex{Real{0.7}, Real{0.7}}, Complex{Real{1.4}, Real{-0.8}}}};

  SymmetricEigenParams<Real> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 5;
  params.max_iterations = 80;
  params.tolerance = std::is_same_v<Real, float> ? Real{1.0e-5F} : Real{1.0e-10};
  params.spectrum = SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<Complex>(ops, initial, params);
  std::vector<double> exact = exact_path_laplacian_eigenvalues(dimension);

  Real const tolerance = std::is_same_v<Real, float> ? Real{5.0e-4F} : Real{1.0e-8};
  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), exact[0], static_cast<double>(tolerance));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), exact[1], static_cast<double>(tolerance));
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_EQ(ops.matvec_count(), result.matvec_count);
}

TYPED_TEST(KrylovHermitianLanczosComplexTypedTest, ShiftInvertSelectorMapsComplexHermitianVectorPath)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;
  using uni20::krylov::SymmetricSpectralTransform;
  using uni20::krylov::SymmetricSpectralTransformOptions;

  DenseHostVectorOps<Complex> ops(
      4, diagonal_dense_matrix<Complex>(
             {Complex{Real{1} / Real{8}}, Complex{Real{1} / Real{4}}, Complex{Real{1} / Real{2}}, Complex{Real{1}}}));
  DenseHostVector<Complex> initial{{Complex{Real{1}, Real{0.2}}, Complex{Real{1}, Real{-0.3}},
                                    Complex{Real{1}, Real{0.4}}, Complex{Real{1}, Real{-0.5}}}};

  SymmetricEigenParams<Real> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.spectrum = SpectrumPart::LargestMagnitude;
  params.compute_eigenvectors = false;

  SymmetricSpectralTransformOptions<Real> options;
  options.transform = SymmetricSpectralTransform::ShiftInvert;
  auto result = uni20::krylov::symmetric_lanczos_restarted_transformed<Complex>(ops, initial, params, options);

  Real const tolerance = std::is_same_v<Real, float> ? Real{1.0e-4F} : Real{1.0e-11};
  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 1.0, static_cast<double>(tolerance));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 2.0, static_cast<double>(tolerance));
}

TYPED_TEST(KrylovHermitianLanczosComplexTypedTest, GeneralizedRegularModeUsesComplexBMetricPath)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;
  using uni20::krylov::SymmetricSpectralTransform;
  using uni20::krylov::SymmetricSpectralTransformOptions;

  DenseHostVectorOps<Complex> op_ops(
      4, diagonal_dense_matrix<Complex>({Complex{Real{1}}, Complex{Real{3}}, Complex{Real{5}}, Complex{Real{7}}}));
  DenseHostVectorOps<Complex> b_ops(
      4, diagonal_dense_matrix<Complex>({Complex{Real{2}}, Complex{Real{3}}, Complex{Real{5}}, Complex{Real{11}}}));
  DenseHostVector<Complex> initial{{Complex{Real{1}, Real{0.1}}, Complex{Real{1}, Real{-0.2}},
                                    Complex{Real{1}, Real{0.3}}, Complex{Real{1}, Real{-0.4}}}};

  SymmetricEigenParams<Real> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  SymmetricSpectralTransformOptions<Real> options;
  options.transform = SymmetricSpectralTransform::Regular;
  auto result = uni20::krylov::symmetric_lanczos_restarted_generalized_transformed<Complex>(op_ops, b_ops, initial,
                                                                                            params, options);

  Real const tolerance = std::is_same_v<Real, float> ? Real{1.0e-4F} : Real{1.0e-11};
  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 7.0, static_cast<double>(tolerance));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 5.0, static_cast<double>(tolerance));
}

TEST(KrylovSymmetricLanczos, BuildsRitzVectorThroughMatrixFreeOps)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(3, {
                                        2.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        3.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        5.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 3;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = true;

  auto result = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 1);
  ASSERT_EQ(result.residual_bounds.size(), 1);
  ASSERT_EQ(result.eigenvectors.size(), 1);
  EXPECT_NEAR(result.eigenvalues[0], 5.0, 1.0e-12);
  EXPECT_LT(result.residual_bounds[0], 1.0e-12);

  DenseHostVector<double> applied{{0.0, 0.0, 0.0}};
  ops.matvec(applied, result.eigenvectors[0]);
  double residual_norm = 0.0;
  double vector_norm = 0.0;
  for (std::size_t i = 0; i < applied.values.size(); ++i)
  {
    double const residual = applied.values[i] - result.eigenvalues[0] * result.eigenvectors[0].values[i];
    residual_norm += residual * residual;
    vector_norm += result.eigenvectors[0].values[i] * result.eigenvectors[0].values[i];
  }

  EXPECT_LT(std::sqrt(residual_norm), 1.0e-12);
  EXPECT_NEAR(std::sqrt(vector_norm), 1.0, 1.0e-12);
}

TEST(KrylovSymmetricLanczos, ReportsUnconvergedTruncatedProjection)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(4, {
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        2.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        4.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        8.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.0, 1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 2;
  params.tolerance = 1.0e-14;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 1);
  ASSERT_EQ(result.residual_bounds.size(), 1);
  EXPECT_GT(result.residual_bounds[0], 1.0e-6);
  EXPECT_EQ(result.converged_count, 0);
  EXPECT_EQ(result.status, 1);
  EXPECT_EQ(result.iteration_count, 2);
  EXPECT_EQ(result.matvec_count, 2);
}

TEST(KrylovSymmetricLanczos, ReportsHappyBreakdownBeforeRequestedEigenvalueCount)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(4, {
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        2.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        4.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        8.0,
                                    });
  DenseHostVector<double> initial{{0.0, 0.0, 0.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.tolerance = 1.0e-14;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 1);
  ASSERT_EQ(result.residual_bounds.size(), 1);
  EXPECT_NEAR(result.eigenvalues[0], 8.0, 1.0e-12);
  EXPECT_EQ(result.converged_count, 1);
  EXPECT_EQ(result.status, 1);
}

TEST(KrylovSymmetricLanczos, RestartedSolveConvergesOnDiagonalLargestAlgebraicProblem)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(8, {
                                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 5;
  params.max_iterations = 60;
  params.tolerance = 1.0e-10;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_EQ(result.status, 0);
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_NEAR(result.eigenvalues[0], 8.0, 1.0e-8);
  EXPECT_NEAR(result.eigenvalues[1], 7.0, 1.0e-8);
  EXPECT_LT(result.residual_bounds[0], 1.0e-8);
  EXPECT_LT(result.residual_bounds[1], 1.0e-8);
  EXPECT_GT(result.matvec_count, params.krylov_dimension);
  EXPECT_EQ(ops.matvec_count(), result.matvec_count);
  EXPECT_FALSE(result.diagnostics.has_value());
}

TYPED_TEST(KrylovSymmetricLanczosRealTypedTest, RestartedSolveConvergesOnDiagonalLargestAlgebraicProblem)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<Scalar> ops(8, diagonal_dense_matrix<Scalar>({Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4},
                                                                   Scalar{5}, Scalar{6}, Scalar{7}, Scalar{8}}));
  DenseHostVector<Scalar> initial{
      {Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}}};

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 5;
  params.max_iterations = 80;
  params.tolerance = Scalar{1.0e-6};
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<Scalar>(ops, initial, params);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 8.0,
              static_cast<double>(Scalar{10} * scalar_tolerance<Scalar>()));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 7.0,
              static_cast<double>(Scalar{10} * scalar_tolerance<Scalar>()));
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_EQ(ops.matvec_count(), result.matvec_count);
}

TEST(KrylovSymmetricLanczos, RestartedSolveReportsFullDiagnosticsWhenRequested)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::KrylovDiagnosticsLevel;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(8, {
                                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 5;
  params.max_iterations = 60;
  params.tolerance = 1.0e-10;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;
  params.diagnostics = KrylovDiagnosticsLevel::Full;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params);

  ASSERT_EQ(result.status, 0);
  ASSERT_TRUE(result.diagnostics.has_value());
  auto const& diagnostics = *result.diagnostics;
  EXPECT_EQ(diagnostics.op_count, result.matvec_count);
  EXPECT_EQ(diagnostics.final_projected_dimension, params.krylov_dimension);
  EXPECT_EQ(diagnostics.final_ritz_values.size(), static_cast<std::size_t>(params.krylov_dimension));
  EXPECT_EQ(diagnostics.final_ritz_bounds.size(), diagnostics.final_ritz_values.size());
  EXPECT_EQ(diagnostics.final_selected_indices.size(), static_cast<std::size_t>(params.eigenvalue_count));
  ASSERT_FALSE(diagnostics.restart_cycles.empty());
  EXPECT_EQ(diagnostics.restart_count, static_cast<int>(diagnostics.restart_cycles.size()));

  auto const& first_cycle = diagnostics.restart_cycles.front();
  EXPECT_EQ(first_cycle.cycle, 0);
  EXPECT_EQ(first_cycle.projected_dimension, static_cast<std::size_t>(params.krylov_dimension));
  EXPECT_EQ(first_cycle.retained_count, static_cast<std::size_t>(params.eigenvalue_count));
  EXPECT_EQ(first_cycle.shift_count, static_cast<std::size_t>(params.krylov_dimension - params.eigenvalue_count));
  EXPECT_EQ(first_cycle.shifts.size(), first_cycle.shift_count);
  EXPECT_EQ(first_cycle.wanted_indices.size(), static_cast<std::size_t>(params.eigenvalue_count));
  EXPECT_EQ(first_cycle.shift_indices.size(), first_cycle.shift_count);
  EXPECT_EQ(first_cycle.ritz_values.size(), static_cast<std::size_t>(params.krylov_dimension));
  EXPECT_EQ(first_cycle.ritz_bounds.size(), first_cycle.ritz_values.size());
}

TEST(KrylovSymmetricLanczos, RestartedSolveReportsSummaryRestartCount)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::KrylovDiagnosticsLevel;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(8, {
                                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 5;
  params.max_iterations = 60;
  params.tolerance = 1.0e-10;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params);

  ASSERT_EQ(result.status, 0);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->op_count, result.matvec_count);
  EXPECT_GT(result.diagnostics->restart_count, 0);
  EXPECT_TRUE(result.diagnostics->restart_cycles.empty());
}

TEST(KrylovSymmetricLanczos, SolvesDiagonalBothEndsProblem)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(6, {
                                        -10.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0,   0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 2.0, 0.0, 0.0,
                                        0.0,   0.0, 0.0,  0.0, 4.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 9.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.5, 2.0, 2.5, 3.0, 3.5}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 3;
  params.krylov_dimension = 6;
  params.spectrum = SpectrumPart::BothEnds;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);

  std::sort(result.eigenvalues.begin(), result.eigenvalues.end());
  ASSERT_EQ(result.eigenvalues.size(), 3);
  EXPECT_EQ(result.status, 0);
  EXPECT_NEAR(result.eigenvalues[0], -10.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[1], 4.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[2], 9.0, 1.0e-12);
}

TEST(KrylovSymmetricLanczos, RestartedSolveHandlesExactBreakdown)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(4, {
                                        3.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        3.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        3.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        3.0,
                                    });
  DenseHostVector<double> initial{{1.0, 2.0, 3.0, 4.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 3;
  params.max_iterations = 10;
  params.tolerance = 1.0e-12;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 1);
  ASSERT_EQ(result.residual_bounds.size(), 1);
  EXPECT_EQ(result.status, 0);
  EXPECT_EQ(result.converged_count, 1);
  EXPECT_EQ(result.matvec_count, 1);
  EXPECT_NEAR(result.eigenvalues[0], 3.0, 1.0e-12);
  EXPECT_LT(result.residual_bounds[0], 1.0e-12);
}

TEST(KrylovSymmetricLanczos, RestartedSolveUsesBreakdownToleranceForTinyResidual)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  double constexpr coupling = 1.0e-14;
  DenseHostVectorOps<double> ops(2, {
                                        1.0,
                                        coupling,
                                        coupling,
                                        2.0,
                                    });
  DenseHostVector<double> initial{{1.0, 0.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 2;
  params.max_iterations = 10;
  params.tolerance = 1.0e-10;
  params.breakdown_tolerance = 1.0e-12;
  params.spectrum = SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 1);
  EXPECT_EQ(result.status, 0);
  EXPECT_EQ(result.converged_count, 1);
  EXPECT_EQ(result.iteration_count, 1);
  EXPECT_EQ(result.matvec_count, 1);
  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1.0e-12);
  EXPECT_LE(result.residual_bounds[0], coupling);
}

TEST(KrylovSymmetricLanczos, RestartedSolveReportsUnconvergedWhenIterationLimitIsReached)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(8, {
                                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 5;
  params.max_iterations = 1;
  params.tolerance = 1.0e-14;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params);

  EXPECT_EQ(result.status, 1);
  EXPECT_EQ(result.eigenvalues.size(), static_cast<std::size_t>(result.converged_count));
  EXPECT_LT(result.converged_count, 2);
  EXPECT_EQ(result.matvec_count, 5);
}

TEST(KrylovSymmetricLanczos, RestartedSolveRejectsInvalidRestartDimensions)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(3, {
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        2.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        3.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 2;
  EXPECT_THROW((void)uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params),
               std::invalid_argument);

  params.krylov_dimension = 3;
  params.retained_ritz_count = 3;
  EXPECT_THROW((void)uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params),
               std::invalid_argument);

  params.retained_ritz_count = 0;
  params.max_iterations = 0;
  EXPECT_THROW((void)uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params),
               std::invalid_argument);
}

TEST(KrylovSymmetricLanczos, RitzConvergenceUsesMachinePrecisionFloor)
{
  double const tolerance = 1.0e-8;
  double const tiny_ritz_value = 3.0e-12;
  double const floor = uni20::krylov::detail::ritz_convergence_floor<double>();

  ASSERT_GT(floor, tiny_ritz_value);
  EXPECT_FALSE(uni20::krylov::detail::symmetric_ritz_converged(tiny_ritz_value, 1.0e-10, tolerance));
  EXPECT_TRUE(uni20::krylov::detail::symmetric_ritz_converged(tiny_ritz_value, 0.5 * tolerance * floor, tolerance));
}

TEST(KrylovSymmetricLanczos, HermitianProjectionImaginaryToleranceUsesLocalActionScale)
{
  using Complex = uni20::complex<double>;

  Complex const projected_diagonal{0.0, 1.0e-10};

  EXPECT_THROW((void)uni20::krylov::detail::hermitian_projection_scalar(projected_diagonal), std::runtime_error);
  EXPECT_NO_THROW({
    double const real_part = uni20::krylov::detail::hermitian_projection_scalar(projected_diagonal, 1.0e8);
    EXPECT_EQ(real_part, 0.0);
  });
}

TEST(KrylovSymmetricLanczos, HermitianProjectionScaleUsesRealActionNorm)
{
  std::vector<double> const matrix{
      1.0,
      0.0, //
      0.0,
      1.0,
  };
  uni20::krylov::DenseHostVectorOps<double> ops(2, matrix);
  uni20::krylov::DenseHostVector<double> action{{3.0, 4.0}};

  EXPECT_EQ((uni20::krylov::detail::hermitian_projection_scale<double>(ops, action)), 5.0);
}

TEST(KrylovSymmetricLanczos, SelectsLargestAlgebraicRestartShiftsByResidualBound)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{-10.0, -3.0, -1.0, 2.0, 4.0, 9.0};
  std::vector<double> bounds{1.0, 10.0, 2.0, 4.0, 3.0, 5.0};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.tolerance = 0.5;
  params.spectrum = SpectrumPart::LargestAlgebraic;

  auto selection = uni20::krylov::select_symmetric_restart(values, bounds, params, 4);

  EXPECT_EQ(selection.wanted_indices, (std::vector<std::size_t>{4, 5}));
  EXPECT_EQ(selection.shift_indices, (std::vector<std::size_t>{1, 3, 2, 0}));
  EXPECT_EQ(selection.shifts, (std::vector<double>{-3.0, 2.0, -1.0, -10.0}));
  EXPECT_EQ(selection.wanted_converged, (std::vector<bool>{false, false}));
  EXPECT_EQ(selection.converged_count, 0);
}

TEST(KrylovSymmetricLanczos, SelectsLargestMagnitudeRestartShiftsByResidualBound)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{-10.0, -3.0, -1.0, 2.0, 4.0, 9.0};
  std::vector<double> bounds{1.0, 10.0, 2.0, 4.0, 3.0, 5.0};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.tolerance = 1.0;
  params.spectrum = SpectrumPart::LargestMagnitude;

  auto selection = uni20::krylov::select_symmetric_restart(values, bounds, params, 4);

  EXPECT_EQ(selection.wanted_indices, (std::vector<std::size_t>{5, 0}));
  EXPECT_EQ(selection.shift_indices, (std::vector<std::size_t>{1, 3, 4, 2}));
  EXPECT_EQ(selection.shifts, (std::vector<double>{-3.0, 2.0, 4.0, -1.0}));
  EXPECT_EQ(selection.wanted_converged, (std::vector<bool>{true, true}));
  EXPECT_EQ(selection.converged_count, 2);
}

TEST(KrylovSymmetricLanczos, SelectsThickRestartRetainedRitzValues)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{-10.0, -3.0, -1.0, 2.0, 4.0, 9.0};
  std::vector<double> bounds{1.0, 10.0, 2.0, 4.0, 3.0, 5.0};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.retained_ritz_count = 4;
  params.tolerance = 0.5;
  params.spectrum = SpectrumPart::LargestAlgebraic;

  auto selection = uni20::krylov::select_symmetric_restart(values, bounds, params, 4);

  EXPECT_EQ(selection.wanted_indices, (std::vector<std::size_t>{2, 3, 4, 5}));
  EXPECT_EQ(selection.shift_indices, (std::vector<std::size_t>{1, 0}));
  EXPECT_EQ(selection.shifts, (std::vector<double>{-3.0, -10.0}));
  EXPECT_EQ(selection.wanted_converged, (std::vector<bool>{false, false, false, false}));
  EXPECT_EQ(selection.converged_count, 0);
}

TEST(KrylovSymmetricLanczos, SelectsBothEndsRestartShiftsFromMiddleByResidualBound)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{-10.0, -3.0, -1.0, 2.0, 4.0, 9.0};
  std::vector<double> bounds{1.0, 10.0, 2.0, 4.0, 3.0, 5.0};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 3;
  params.tolerance = 1.0;
  params.spectrum = SpectrumPart::BothEnds;

  auto selection = uni20::krylov::select_symmetric_restart(values, bounds, params, 3);

  EXPECT_EQ(selection.wanted_indices, (std::vector<std::size_t>{0, 4, 5}));
  EXPECT_EQ(selection.shift_indices, (std::vector<std::size_t>{1, 3, 2}));
  EXPECT_EQ(selection.shifts, (std::vector<double>{-3.0, 2.0, -1.0}));
  EXPECT_EQ(selection.wanted_converged, (std::vector<bool>{true, true, true}));
  EXPECT_EQ(selection.converged_count, 3);
}

TEST(KrylovSymmetricLanczos, RejectsBothEndsSingleRequestedEigenvalue)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{1.0, 2.0, 3.0};
  std::vector<double> bounds{0.1, 0.1, 0.1};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.spectrum = SpectrumPart::BothEnds;

  EXPECT_THROW((void)uni20::krylov::select_symmetric_restart(values, bounds, params, 1), std::invalid_argument);
}

TEST(KrylovSymmetricLanczos, RestartPlanProtectsZeroBoundUnwantedRitzValues)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{-10.0, -3.0, -1.0, 2.0, 4.0, 9.0};
  std::vector<double> bounds{1.0, 0.0, 2.0, 0.0, 3.0, 5.0};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.spectrum = SpectrumPart::LargestAlgebraic;

  auto plan = uni20::krylov::plan_symmetric_restart(params, values, bounds, 0);

  EXPECT_EQ(plan.protected_zero_bound_count, 2);
  EXPECT_EQ(plan.protected_zero_bound_indices, (std::vector<std::size_t>{1, 3}));
  EXPECT_EQ(plan.retained_count, 4);
  EXPECT_EQ(plan.shift_count, 2);
  EXPECT_FALSE(plan.enlarged_for_partial_convergence);

  SymmetricEigenParams<double> restart_params = params;
  restart_params.retained_ritz_count = static_cast<int>(plan.retained_count);
  auto selection = uni20::krylov::select_symmetric_restart(values, bounds, restart_params, plan.shift_count,
                                                           plan.protected_zero_bound_indices);
  EXPECT_EQ(selection.shift_indices, (std::vector<std::size_t>{0}));
  EXPECT_EQ(selection.shifts, (std::vector<double>{-10.0}));
}

TEST(KrylovSymmetricLanczos, RestartPlanIncreasesRetainedBlockAfterPartialConvergence)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  std::vector<double> bounds(values.size(), 1.0);

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 4;
  params.spectrum = SpectrumPart::LargestAlgebraic;

  auto plan = uni20::krylov::plan_symmetric_restart(params, values, bounds, 2);

  EXPECT_EQ(plan.protected_zero_bound_count, 0);
  EXPECT_EQ(plan.retained_count, 6);
  EXPECT_EQ(plan.shift_count, 4);
  EXPECT_TRUE(plan.enlarged_for_partial_convergence);
}

TEST(KrylovSymmetricLanczos, RestartPlanUsesSingleVectorAntiStagnationBump)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> bounds(values.size(), 1.0);

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.spectrum = SpectrumPart::LargestAlgebraic;

  auto plan = uni20::krylov::plan_symmetric_restart(params, values, bounds, 0);

  EXPECT_EQ(plan.retained_count, 3);
  EXPECT_EQ(plan.shift_count, 3);
  EXPECT_TRUE(plan.enlarged_for_partial_convergence);
}

TEST(KrylovSymmetricLanczos, RejectsRetainedRitzCountBelowRequestedCount)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{1.0, 2.0, 3.0};
  std::vector<double> bounds{0.1, 0.1, 0.1};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.retained_ritz_count = 1;
  params.spectrum = SpectrumPart::LargestAlgebraic;

  EXPECT_THROW((void)uni20::krylov::select_symmetric_restart(values, bounds, params, 1), std::invalid_argument);
}

TEST(KrylovSymmetricLanczos, RejectsRetainedRitzCountAboveProjectedSize)
{
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::vector<double> values{1.0, 2.0, 3.0};
  std::vector<double> bounds{0.1, 0.1, 0.1};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.retained_ritz_count = 4;
  params.spectrum = SpectrumPart::LargestAlgebraic;

  EXPECT_THROW((void)uni20::krylov::select_symmetric_restart(values, bounds, params, 1), std::invalid_argument);
}

TEST(KrylovSymmetricLanczos, AppliesSymmetricQrShiftsAsOrthogonalSimilarity)
{
  std::vector<double> diagonal{1.0, 2.0, 4.0, 8.0};
  std::vector<double> subdiagonal{0.4, 0.3, 0.2};
  std::vector<double> shifts{1.5, 7.0};

  auto result = uni20::krylov::apply_symmetric_qr_shifts(diagonal, subdiagonal, shifts);

  auto original = tridiagonal_matrix(diagonal, subdiagonal);
  auto transformed = tridiagonal_matrix(result.diagonal, result.subdiagonal);
  auto q_t_a_q = multiply(transpose(result.transform), multiply(original, result.transform));
  expect_matrix_near(transformed, q_t_a_q, 1.0e-11);

  auto identity = multiply(transpose(result.transform), result.transform);
  uni20::krylov::Matrix<double> expected_identity(identity.rows(), identity.cols());
  uni20::krylov::laset(expected_identity, 1.0, 0.0, uni20::krylov::MatrixFill::All);
  expect_matrix_near(identity, expected_identity, 1.0e-12);
}

TEST(KrylovSymmetricLanczos, SymmetricQrShiftsPreserveTridiagonalEigenvalues)
{
  std::vector<double> diagonal{1.0, 2.0, 4.0, 8.0};
  std::vector<double> subdiagonal{0.4, 0.3, 0.2};
  std::vector<double> shifts{1.5, 7.0};

  auto before = uni20::krylov::symmetric_tridiagonal_eigensystem(diagonal, subdiagonal, false);
  auto shifted = uni20::krylov::apply_symmetric_qr_shifts(std::move(diagonal), std::move(subdiagonal), shifts);
  auto after = uni20::krylov::symmetric_tridiagonal_eigensystem(std::move(shifted.diagonal),
                                                                std::move(shifted.subdiagonal), false);

  ASSERT_EQ(before.eigenvalues.size(), after.eigenvalues.size());
  for (std::size_t i = 0; i < before.eigenvalues.size(); ++i)
  {
    EXPECT_NEAR(before.eigenvalues[i], after.eigenvalues[i], 1.0e-11);
  }
}

TEST(KrylovSymmetricLanczos, CompressesRestartedBasisAndResidualThroughMatrixFreeOps)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;

  DenseHostVectorOps<double> ops(4, {
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0,
                                    });
  std::vector<DenseHostVector<double>> basis{
      DenseHostVector<double>{{1.0, 0.0, 0.0, 0.0}},
      DenseHostVector<double>{{0.0, 1.0, 0.0, 0.0}},
      DenseHostVector<double>{{0.0, 0.0, 1.0, 0.0}},
      DenseHostVector<double>{{0.0, 0.0, 0.0, 1.0}},
  };
  DenseHostVector<double> residual{{10.0, 20.0, 30.0, 40.0}};

  std::vector<double> diagonal{1.0, 2.0, 4.0, 8.0};
  std::vector<double> subdiagonal{0.4, 0.3, 0.2};
  auto shifted = uni20::krylov::apply_symmetric_qr_shifts(diagonal, subdiagonal, std::vector<double>{1.5, 7.0});

  auto compressed = uni20::krylov::compress_symmetric_lanczos_restart<double>(ops, basis, residual, shifted, 2);

  ASSERT_EQ(compressed.basis.size(), 2);
  for (std::size_t column = 0; column < compressed.basis.size(); ++column)
  {
    std::vector<double> expected_column;
    expected_column.reserve(basis.size());
    for (std::size_t row = 0; row < basis.size(); ++row)
    {
      expected_column.push_back(shifted.transform[row, column]);
    }
    expect_vector_near(compressed.basis[column], expected_column, 1.0e-12);
  }

  double const sigma = shifted.transform[shifted.transform.rows() - 1, 1];
  double const beta = shifted.subdiagonal[1];
  std::vector<double> expected_residual;
  expected_residual.reserve(residual.values.size());
  for (std::size_t row = 0; row < residual.values.size(); ++row)
  {
    expected_residual.push_back(sigma * residual.values[row] + beta * shifted.transform[row, 2]);
  }
  expect_vector_near(compressed.residual, expected_residual, 1.0e-12);

  double expected_residual_norm = 0.0;
  for (double value : expected_residual)
  {
    expected_residual_norm += value * value;
  }
  expected_residual_norm = std::sqrt(expected_residual_norm);

  EXPECT_EQ(compressed.diagonal, (std::vector<double>{shifted.diagonal[0], shifted.diagonal[1]}));
  EXPECT_EQ(compressed.subdiagonal, (std::vector<double>{shifted.subdiagonal[0]}));
  EXPECT_NEAR(compressed.residual_scale, sigma, 1.0e-14);
  EXPECT_NEAR(compressed.trailing_coupling, beta, 1.0e-14);
  EXPECT_NEAR(compressed.residual_norm, expected_residual_norm, 1.0e-12);
  EXPECT_EQ(ops.matvec_count(), 0);
}

TEST(KrylovSymmetricLanczos, SolvesFullSubspaceTwoDimensionalLaplacian)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  std::size_t constexpr nx = 3;
  std::size_t constexpr n = nx * nx;
  DenseHostVectorOps<double> ops(n, two_dimensional_laplacian(nx));
  DenseHostVector<double> initial{{1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = static_cast<int>(n);
  params.spectrum = SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);
  auto exact = exact_two_dimensional_laplacian_eigenvalues(nx);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  for (std::size_t i = 0; i < result.eigenvalues.size(); ++i)
  {
    EXPECT_NEAR(result.eigenvalues[i], exact[i], 1.0e-11);
    EXPECT_LT(result.residual_bounds[i], 1.0e-10);
  }
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_EQ(result.status, 0);
}

TYPED_TEST(KrylovSymmetricLanczosRealTypedTest, ShiftInvertSmallestAlgebraicSelectorActsInTransformedSpace)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;
  using uni20::krylov::SymmetricSpectralTransform;
  using uni20::krylov::SymmetricSpectralTransformOptions;

  DenseHostVectorOps<Scalar> op_ops(4, {
                                           Scalar{1.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.5},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.25},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.125},
                                       });
  DenseHostVector<Scalar> initial{{Scalar{1.0}, Scalar{1.0}, Scalar{1.0}, Scalar{1.0}}};

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.spectrum = SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = false;

  SymmetricSpectralTransformOptions<Scalar> options;
  options.transform = SymmetricSpectralTransform::ShiftInvert;
  options.sigma = Scalar{0.0};
  auto result = uni20::krylov::symmetric_lanczos_restarted_transformed<Scalar>(op_ops, initial, params, options);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 8.0, static_cast<double>(scalar_tolerance<Scalar>()));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 4.0, static_cast<double>(scalar_tolerance<Scalar>()));
}

TYPED_TEST(KrylovSymmetricLanczosRealTypedTest, GeneralizedRegularModeUsesBMetric)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;
  using uni20::krylov::SymmetricSpectralTransform;
  using uni20::krylov::SymmetricSpectralTransformOptions;

  DenseHostVectorOps<Scalar> op_ops(4, {
                                           Scalar{1.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{3.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{5.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{7.0},
                                       });
  DenseHostVectorOps<Scalar> b_ops(4, {
                                          Scalar{2.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{3.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{5.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{11.0},
                                      });
  DenseHostVector<Scalar> initial{{Scalar{1.0}, Scalar{1.0}, Scalar{1.0}, Scalar{1.0}}};

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  SymmetricSpectralTransformOptions<Scalar> options;
  options.transform = SymmetricSpectralTransform::Regular;
  auto result = uni20::krylov::symmetric_lanczos_restarted_generalized_transformed<Scalar>(op_ops, b_ops, initial,
                                                                                           params, options);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 7.0, static_cast<double>(scalar_tolerance<Scalar>()));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 5.0, static_cast<double>(scalar_tolerance<Scalar>()));
}

TEST(KrylovSymmetricLanczos, RejectsZeroInitialVector)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(2, {
                                        1.0,
                                        0.0,
                                        0.0,
                                        2.0,
                                    });
  DenseHostVector<double> initial{{0.0, 0.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 2;

  EXPECT_THROW(uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params), std::invalid_argument);
}

TEST(KrylovSymmetricLanczos, RejectsInitialVectorWithWrongDimensionBeforeMatvec)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SymmetricEigenParams;

  DenseHostVectorOps<double> ops(3, {
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        2.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        3.0,
                                    });
  DenseHostVector<double> initial{{1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 2;

  EXPECT_THROW(uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params), std::invalid_argument);
  EXPECT_THROW((void)uni20::krylov::symmetric_lanczos_restarted_standard<double>(ops, initial, params),
               std::invalid_argument);
  EXPECT_EQ(ops.matvec_count(), 0);
}

} // namespace
