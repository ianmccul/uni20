#include <mplapack_config.h>
#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/dense_subspace.hpp>
#include <uni20/krylov/krylov_exponential.hpp>
#include <uni20/krylov/nonsymmetric_arnoldi.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>
#include <uni20/linalg/ops/matrix_set.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_LDBL)

TEST(MplapackBinary128KrylovSolversTest, SkipsLongDoubleAliasMode)
{
  GTEST_SKIP() << "configured MPLAPACK binary128 mode aliases long double";
}

#else

namespace
{

using Binary128 = mplapack_binary128_t;
using ComplexBinary128 = uni20::complex<Binary128>;
using Vector = uni20::krylov::DenseHostVector<Binary128>;
using ComplexVector = uni20::krylov::DenseHostVector<ComplexBinary128>;
using Ops = uni20::krylov::DenseHostVectorOps<Binary128>;
using ComplexOps = uni20::krylov::DenseHostVectorOps<ComplexBinary128>;

Binary128 abs_error(Binary128 actual, Binary128 expected) { return std::abs(actual - expected); }

Binary128 tolerance() { return static_cast<Binary128>(1.0e-25L); }

Binary128 binary_power_of_two(int exponent)
{
  Binary128 result{1};
  if (exponent >= 0)
  {
    for (int i = 0; i < exponent; ++i)
    {
      result *= Binary128{2};
    }
  }
  else
  {
    for (int i = 0; i < -exponent; ++i)
    {
      result /= Binary128{2};
    }
  }
  return result;
}

Binary128 below_double_resolution_gap() { return binary_power_of_two(-80); }

void expect_gap_is_binary128_only(Binary128 gap)
{
  EXPECT_TRUE(Binary128{1} + gap > Binary128{1});
  EXPECT_EQ(static_cast<double>(Binary128{1} + gap), 1.0);
}

void expect_value_increment_is_binary128_only(Binary128 value)
{
  EXPECT_TRUE(value > Binary128{1});
  EXPECT_EQ(static_cast<double>(value), 1.0);
}

std::vector<Binary128> diagonal_dense_matrix(std::vector<Binary128> const& diagonal)
{
  std::size_t const dimension = diagonal.size();
  std::vector<Binary128> matrix(dimension * dimension, Binary128{});
  for (std::size_t i = 0; i < dimension; ++i)
  {
    matrix[i * dimension + i] = diagonal[i];
  }
  return matrix;
}

Binary128 vector_norm(Vector const& vector)
{
  Binary128 sum{};
  for (Binary128 const value : vector.values)
  {
    sum += value * value;
  }
  return sum > Binary128{} ? std::sqrt(sum) : Binary128{};
}

Binary128 relative_residual(Ops& ops, Vector const& eigenvector, Binary128 eigenvalue)
{
  Vector residual = ops.allocate_like(eigenvector);
  ops.matvec(residual, eigenvector);
  ops.axpy(residual, -eigenvalue, eigenvector);
  Binary128 const residual_norm = vector_norm(residual);
  Binary128 const scale = std::max(Binary128{1}, std::abs(eigenvalue) * vector_norm(eigenvector));
  return residual_norm / scale;
}

} // namespace

TEST(MplapackBinary128KrylovSolversTest, TridiagonalProjectionResolvesGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  expect_gap_is_binary128_only(delta);

  std::vector<Binary128> diagonal{Binary128{1}, Binary128{1} + delta, Binary128{2}};
  std::vector<Binary128> subdiagonal{offdiagonal, Binary128{}};
  auto result = uni20::krylov::symmetric_tridiagonal_eigensystem(diagonal, subdiagonal, false);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  ASSERT_EQ(result.eigenvalues.size(), 3);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[1]), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], center - radius) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], center + radius) <= tolerance());
}

TEST(MplapackBinary128KrylovSolversTest, SymmetricLanczosResolvesDiagonalGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  Ops ops(4, diagonal_dense_matrix({Binary128{1}, Binary128{1} + delta, Binary128{2}, Binary128{3}}));
  Vector initial{{Binary128{1}, Binary128{1}, Binary128{1}, Binary128{1}}};

  uni20::krylov::SymmetricEigenParams<Binary128> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.tolerance = tolerance();
  params.spectrum = uni20::krylov::SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = true;

  auto result = uni20::krylov::symmetric_lanczos_standard<Binary128>(ops, initial, params);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.converged_count, 2);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.size(), 2);

  std::vector<Binary128> eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues);
  EXPECT_EQ(static_cast<double>(eigenvalues[0]), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1]), 1.0);
  EXPECT_TRUE(eigenvalues[1] > eigenvalues[0]);
  EXPECT_TRUE(abs_error(eigenvalues[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1], Binary128{1} + delta) <= tolerance());

  for (std::size_t i = 0; i < result.eigenvectors.size(); ++i)
  {
    EXPECT_TRUE(relative_residual(ops, result.eigenvectors[i], result.eigenvalues[i]) <= tolerance());
  }
}

TEST(MplapackBinary128KrylovSolversTest, RealSchurAndReorderUseBinary128ProjectedLAPACK)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::krylov::Matrix<Binary128> matrix(3, 3);
  uni20::linalg::set_matrix(matrix, Binary128{}, Binary128{});
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  matrix[2, 2] = Binary128{3};
  matrix[0, 2] = Binary128{0.125};
  matrix[1, 2] = Binary128{0.25};

  auto schur = uni20::krylov::real_schur(matrix, true);
  ASSERT_EQ(schur.eigenvalues.size(), 3);
  auto reordered = uni20::krylov::reorder_real_schur(std::move(schur), std::vector<std::size_t>{2});

  ASSERT_EQ(reordered.eigenvalues.size(), 3);
  EXPECT_TRUE(abs_error(reordered.eigenvalues[0].real(), Binary128{3}) <= tolerance());
  EXPECT_TRUE(abs_error(reordered.eigenvalues[0].imag(), Binary128{}) <= tolerance());
}

TEST(MplapackBinary128KrylovSolversTest, RealHessenbergSchurUsesBinary128ProjectedLAPACK)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::krylov::Matrix<Binary128> hessenberg(3, 3);
  uni20::linalg::set_matrix(hessenberg, Binary128{}, Binary128{});
  hessenberg[0, 0] = Binary128{1};
  hessenberg[1, 1] = Binary128{1} + delta;
  hessenberg[2, 2] = Binary128{2};
  hessenberg[0, 2] = Binary128{0.125};
  hessenberg[1, 2] = Binary128{0.25};

  auto schur = uni20::krylov::real_hessenberg_schur(hessenberg, true);
  ASSERT_EQ(schur.eigenvalues.size(), 3);
  std::ranges::sort(schur.eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  EXPECT_TRUE(abs_error(schur.eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(schur.eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
}

TEST(MplapackBinary128KrylovSolversTest, RealArnoldiResolvesTriangularGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  std::vector<Binary128> matrix{
      Binary128{1}, Binary128{0},         Binary128{0.125}, Binary128{0},    //
      Binary128{0}, Binary128{1} + delta, Binary128{0},     Binary128{0},    //
      Binary128{0}, Binary128{0},         Binary128{2},     Binary128{0.25}, //
      Binary128{0}, Binary128{0},         Binary128{0},     Binary128{3},
  };
  Ops ops(4, matrix);
  Vector initial{{Binary128{1}, Binary128{1}, Binary128{1}, Binary128{1}}};

  uni20::krylov::NonsymmetricEigenParams<Binary128> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.tolerance = tolerance();
  params.complex_pair_tolerance = tolerance();
  params.spectrum = uni20::krylov::SpectrumPart::SmallestReal;
  params.compute_eigenvectors = true;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_standard(ops, initial, params);

  ASSERT_EQ(result.status, uni20::krylov::NonsymmetricStatus::Converged);
  ASSERT_EQ(result.converged_count, 2);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);

  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  EXPECT_EQ(static_cast<double>(eigenvalues[0].real()), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1].real()), 1.0);
  EXPECT_TRUE(eigenvalues[1].real() > eigenvalues[0].real());
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
}

TEST(MplapackBinary128KrylovSolversTest, ComplexSchurAndReorderUseBinary128ProjectedLAPACK)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::krylov::Matrix<ComplexBinary128> matrix(3, 3);
  uni20::linalg::set_matrix(matrix, ComplexBinary128{}, ComplexBinary128{});
  matrix[0, 0] = ComplexBinary128{Binary128{1}, delta};
  matrix[1, 1] = ComplexBinary128{Binary128{1} + delta, Binary128{2} * delta};
  matrix[2, 2] = ComplexBinary128{Binary128{3}, Binary128{}};
  matrix[0, 2] = ComplexBinary128{Binary128{0.125}, Binary128{}};
  matrix[1, 2] = ComplexBinary128{Binary128{0.25}, Binary128{}};

  auto schur = uni20::krylov::complex_schur<Binary128>(matrix, true);
  ASSERT_EQ(schur.eigenvalues.size(), 3);
  auto reordered = uni20::krylov::reorder_complex_schur(std::move(schur), std::vector<std::size_t>{2});

  ASSERT_EQ(reordered.eigenvalues.size(), 3);
  EXPECT_TRUE(abs_error(reordered.eigenvalues[0].real(), Binary128{3}) <= tolerance());
  EXPECT_TRUE(abs_error(reordered.eigenvalues[0].imag(), Binary128{}) <= tolerance());
}

TEST(MplapackBinary128KrylovSolversTest, ComplexArnoldiResolvesComplexGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  std::vector<ComplexBinary128> matrix{
      ComplexBinary128{Binary128{1}, delta},
      ComplexBinary128{}, //
      ComplexBinary128{},
      ComplexBinary128{Binary128{1} + delta, Binary128{2} * delta},
  };
  ComplexOps ops(2, matrix);
  ComplexVector initial{{ComplexBinary128{Binary128{1}, Binary128{}}, ComplexBinary128{Binary128{1}, Binary128{}}}};

  uni20::krylov::NonsymmetricEigenParams<Binary128> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 2;
  params.tolerance = tolerance();
  params.complex_pair_tolerance = tolerance();
  params.spectrum = uni20::krylov::SpectrumPart::SmallestReal;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::complex_nonsymmetric_arnoldi_standard<Binary128>(ops, initial, params);

  ASSERT_EQ(result.status, uni20::krylov::NonsymmetricStatus::Converged);
  ASSERT_EQ(result.converged_count, 2);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  std::ranges::sort(result.eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  EXPECT_TRUE(abs_error(result.eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[0].imag(), delta) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1].imag(), Binary128{2} * delta) <= tolerance());
}

TEST(MplapackBinary128KrylovSolversTest, HermitianExponentialActionPreservesBinary128OnlyIncrement)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const exp_delta = std::exp(delta);
  Binary128 const exp_two_delta = std::exp(Binary128{2} * delta);
  expect_value_increment_is_binary128_only(exp_delta);
  expect_value_increment_is_binary128_only(exp_two_delta);

  std::vector<Binary128> matrix{
      delta,
      Binary128{}, //
      Binary128{},
      Binary128{2} * delta,
  };
  Ops ops(2, matrix);
  Vector initial{{Binary128{1}, Binary128{-1}}};

  uni20::krylov::KrylovExponentialParams<Binary128> params;
  params.krylov_dimension = 2;

  auto result = uni20::krylov::hermitian_krylov_exponential_action<Binary128>(ops, initial, Binary128{1}, params);

  ASSERT_EQ(result.projected_dimension, 2);
  ASSERT_EQ(result.matvec_count, 2);
  EXPECT_TRUE(abs_error(result.action.values[0], exp_delta) <= tolerance());
  EXPECT_TRUE(abs_error(result.action.values[1], -exp_two_delta) <= tolerance());
}

#endif
