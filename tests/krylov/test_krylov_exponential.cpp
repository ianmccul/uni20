#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/krylov_exponential.hpp>
#include <uni20/krylov/taylor_exponential.hpp>

#include "krylov_test_types.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{

using uni20::krylov::DenseHostVector;
using uni20::krylov::DenseHostVectorOps;
using uni20::krylov::KrylovDiagnosticsLevel;
using uni20::krylov::KrylovExponentialParams;
using uni20::krylov::TaylorExponentialParams;

template <typename Scalar> class KrylovExponentialRealTypedTest : public ::testing::Test {};
template <typename Scalar> class KrylovExponentialComplexTypedTest : public ::testing::Test {};

using RealTypes = uni20::krylov::test::KrylovRealTestTypes;
using ComplexTypes = uni20::krylov::test::KrylovComplexTestTypes;
TYPED_TEST_SUITE(KrylovExponentialRealTypedTest, RealTypes);
TYPED_TEST_SUITE(KrylovExponentialComplexTypedTest, ComplexTypes);

template <typename Scalar> double exponential_tolerance()
{
  using Real = uni20::make_real_t<Scalar>;
  if constexpr (std::is_same_v<Real, float>)
  {
    return 1.0e-4;
  }
  else
  {
    return 1.0e-11;
  }
}

template <typename Scalar> double orthogonality_diagnostic_tolerance()
{
  using Real = uni20::make_real_t<Scalar>;
  return 1000.0 * static_cast<double>(uni20::numeric_limits<Real>::epsilon());
}

template <typename Scalar>
void expect_vector_near(DenseHostVector<Scalar> const& actual, std::vector<Scalar> const& expected, double tolerance)
{
  ASSERT_EQ(actual.values.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    double const difference = static_cast<double>(std::abs(actual.values[i] - expected[i]));
    double const scale = std::max(1.0, static_cast<double>(std::abs(expected[i])));
    EXPECT_LE(difference, tolerance * scale) << "entry " << i;
  }
}

} // namespace

TYPED_TEST(KrylovExponentialRealTypedTest, HermitianFullSubspaceMatchesDiagonalExponential)
{
  using Scalar = TypeParam;

  std::vector<Scalar> matrix{Scalar{1}, Scalar{0}, Scalar{0}, Scalar{0}, Scalar{2},
                             Scalar{0}, Scalar{0}, Scalar{0}, Scalar{3}};
  DenseHostVectorOps<Scalar> ops(3, matrix);
  DenseHostVector<Scalar> initial{{Scalar{1}, Scalar{-2}, Scalar{0.5}}};

  KrylovExponentialParams<Scalar> params;
  params.krylov_dimension = 3;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;
  Scalar const time = Scalar{0.25};

  auto result = uni20::krylov::hermitian_krylov_exponential_action<Scalar>(ops, initial, time, params);

  std::vector<Scalar> expected{std::exp(time * Scalar{1}) * initial.values[0],
                               std::exp(time * Scalar{2}) * initial.values[1],
                               std::exp(time * Scalar{3}) * initial.values[2]};
  expect_vector_near(result.action, expected, exponential_tolerance<Scalar>());
  EXPECT_EQ(result.projected_dimension, 3);
  EXPECT_EQ(result.matvec_count, 3);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->projected_dimension, 3);
  EXPECT_LE(static_cast<double>(result.diagnostics->basis_max_diag_error),
            orthogonality_diagnostic_tolerance<Scalar>());
  EXPECT_LE(static_cast<double>(result.diagnostics->basis_max_offdiag), orthogonality_diagnostic_tolerance<Scalar>());
  EXPECT_LE(static_cast<double>(result.diagnostics->basis_frobenius_error),
            3.0 * orthogonality_diagnostic_tolerance<Scalar>());
  EXPECT_GE(result.diagnostics->max_reorthogonalization_passes, 1);
}

TYPED_TEST(KrylovExponentialRealTypedTest, HermitianActionPreservesTinyNonzeroInputScale)
{
  using Scalar = TypeParam;

  std::vector<Scalar> matrix{Scalar{1}, Scalar{0}, Scalar{0}, Scalar{1}};
  DenseHostVectorOps<Scalar> ops(2, matrix);
  DenseHostVector<Scalar> initial{{Scalar{1.0e-3}, Scalar{-2.0e-3}}};

  KrylovExponentialParams<Scalar> params;
  params.krylov_dimension = 1;
  params.breakdown_tolerance = Scalar{1.0e-2};
  Scalar const time = Scalar{0.25};

  auto result = uni20::krylov::hermitian_krylov_exponential_action<Scalar>(ops, initial, time, params);

  Scalar const factor = std::exp(time);
  std::vector<Scalar> expected{factor * initial.values[0], factor * initial.values[1]};
  expect_vector_near(result.action, expected, exponential_tolerance<Scalar>());
  EXPECT_EQ(result.projected_dimension, 1);
  EXPECT_EQ(result.matvec_count, 1);
}

TYPED_TEST(KrylovExponentialRealTypedTest, NonsymmetricFullSubspaceMatchesJordanExponential)
{
  using Scalar = TypeParam;

  Scalar const diagonal = Scalar{0.3};
  Scalar const off_diagonal = Scalar{2};
  std::vector<Scalar> matrix{diagonal, off_diagonal, Scalar{0}, diagonal};
  DenseHostVectorOps<Scalar> ops(2, matrix);
  DenseHostVector<Scalar> initial{{Scalar{2}, Scalar{-1}}};

  KrylovExponentialParams<Scalar> params;
  params.krylov_dimension = 2;
  Scalar const time = Scalar{0.4};

  auto result = uni20::krylov::nonsymmetric_krylov_exponential_action<Scalar>(ops, initial, time, params);

  Scalar const factor = std::exp(diagonal * time);
  std::vector<Scalar> expected{factor * (initial.values[0] + off_diagonal * time * initial.values[1]),
                               factor * initial.values[1]};
  expect_vector_near(result.action, expected, exponential_tolerance<Scalar>());
  EXPECT_EQ(result.projected_dimension, 2);
  EXPECT_EQ(result.matvec_count, 2);
}

TYPED_TEST(KrylovExponentialRealTypedTest, NonsymmetricActionPreservesTinyNonzeroInputScale)
{
  using Scalar = TypeParam;

  std::vector<Scalar> matrix{Scalar{1}, Scalar{0}, Scalar{0}, Scalar{1}};
  DenseHostVectorOps<Scalar> ops(2, matrix);
  DenseHostVector<Scalar> initial{{Scalar{1.0e-3}, Scalar{-2.0e-3}}};

  KrylovExponentialParams<Scalar> params;
  params.krylov_dimension = 1;
  params.breakdown_tolerance = Scalar{1.0e-2};
  Scalar const time = Scalar{-0.5};

  auto result = uni20::krylov::nonsymmetric_krylov_exponential_action<Scalar>(ops, initial, time, params);

  Scalar const factor = std::exp(time);
  std::vector<Scalar> expected{factor * initial.values[0], factor * initial.values[1]};
  expect_vector_near(result.action, expected, exponential_tolerance<Scalar>());
  EXPECT_EQ(result.projected_dimension, 1);
  EXPECT_EQ(result.matvec_count, 1);
}

TYPED_TEST(KrylovExponentialComplexTypedTest, NonsymmetricFullSubspaceMatchesComplexDiagonalExponential)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;

  std::vector<Scalar> matrix{Scalar{Real{0.1}, Real{0.3}},  Scalar{}, Scalar{}, Scalar{},
                             Scalar{Real{-0.2}, Real{0.5}}, Scalar{}, Scalar{}, Scalar{},
                             Scalar{Real{0.4}, Real{-0.1}}};
  DenseHostVectorOps<Scalar> ops(3, matrix);
  DenseHostVector<Scalar> initial{
      {Scalar{Real{1}, Real{-0.5}}, Scalar{Real{-2}, Real{1}}, Scalar{Real{0.25}, Real{0.75}}}};

  KrylovExponentialParams<Real> params;
  params.krylov_dimension = 3;
  Real const time = Real{0.35};

  auto result = uni20::krylov::nonsymmetric_krylov_exponential_action<Scalar>(ops, initial, time, params);

  std::vector<Scalar> expected{std::exp(time * matrix[0]) * initial.values[0],
                               std::exp(time * matrix[4]) * initial.values[1],
                               std::exp(time * matrix[8]) * initial.values[2]};
  expect_vector_near(result.action, expected, exponential_tolerance<Scalar>());
  EXPECT_EQ(result.projected_dimension, 3);
  EXPECT_EQ(result.matvec_count, 3);
}

TYPED_TEST(KrylovExponentialComplexTypedTest, HermitianActionAcceptsComplexTime)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;

  std::vector<Scalar> matrix{Scalar{Real{1}}, Scalar{}, Scalar{}, Scalar{Real{3}}};
  DenseHostVectorOps<Scalar> ops(2, matrix);
  DenseHostVector<Scalar> initial{{Scalar{Real{1}, Real{0.25}}, Scalar{Real{-0.5}, Real{0.75}}}};

  KrylovExponentialParams<Real> params;
  params.krylov_dimension = 2;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;
  Real const time = Real{0.4};
  Scalar const imaginary_time{Real{}, -time};

  auto result = uni20::krylov::hermitian_krylov_exponential_action<Scalar>(ops, initial, imaginary_time, params);

  std::vector<Scalar> expected{std::exp(imaginary_time * Scalar{Real{1}}) * initial.values[0],
                               std::exp(imaginary_time * Scalar{Real{3}}) * initial.values[1]};
  expect_vector_near(result.action, expected, exponential_tolerance<Scalar>());
  EXPECT_EQ(result.projected_dimension, 2);
  EXPECT_EQ(result.matvec_count, 2);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_LE(static_cast<double>(result.diagnostics->basis_max_diag_error),
            orthogonality_diagnostic_tolerance<Scalar>());
  EXPECT_LE(static_cast<double>(result.diagnostics->basis_max_offdiag), orthogonality_diagnostic_tolerance<Scalar>());
  EXPECT_LE(static_cast<double>(result.diagnostics->basis_frobenius_error),
            2.0 * orthogonality_diagnostic_tolerance<Scalar>());
  EXPECT_GE(result.diagnostics->max_reorthogonalization_passes, 1);
}

TYPED_TEST(KrylovExponentialComplexTypedTest, NonsymmetricActionAcceptsComplexTime)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;

  std::vector<Scalar> matrix{Scalar{Real{0.1}, Real{0.3}}, Scalar{}, Scalar{}, Scalar{Real{-0.2}, Real{0.5}}};
  DenseHostVectorOps<Scalar> ops(2, matrix);
  DenseHostVector<Scalar> initial{{Scalar{Real{1}, Real{-0.5}}, Scalar{Real{-2}, Real{1}}}};

  KrylovExponentialParams<Real> params;
  params.krylov_dimension = 2;
  Scalar const time{Real{0.2}, Real{-0.35}};

  auto result = uni20::krylov::nonsymmetric_krylov_exponential_action<Scalar>(ops, initial, time, params);

  std::vector<Scalar> expected{std::exp(time * matrix[0]) * initial.values[0],
                               std::exp(time * matrix[3]) * initial.values[1]};
  expect_vector_near(result.action, expected, exponential_tolerance<Scalar>());
  EXPECT_EQ(result.projected_dimension, 2);
  EXPECT_EQ(result.matvec_count, 2);
}

TYPED_TEST(KrylovExponentialComplexTypedTest, HermitianZeroVectorReturnsZeroWithoutMatvec)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;

  std::vector<Scalar> matrix{Scalar{Real{1}}, Scalar{}, Scalar{}, Scalar{Real{2}}};
  DenseHostVectorOps<Scalar> ops(2, matrix);
  DenseHostVector<Scalar> initial{{Scalar{}, Scalar{}}};

  KrylovExponentialParams<Real> params;
  params.krylov_dimension = 2;

  auto result = uni20::krylov::hermitian_krylov_exponential_action<Scalar>(ops, initial, Real{1}, params);

  expect_vector_near(result.action, std::vector<Scalar>{Scalar{}, Scalar{}}, exponential_tolerance<Scalar>());
  EXPECT_EQ(result.projected_dimension, 0);
  EXPECT_EQ(result.matvec_count, 0);
  EXPECT_TRUE(result.happy_breakdown);
}

TYPED_TEST(KrylovExponentialRealTypedTest, TaylorActionMatchesDiagonalExponential)
{
  using Scalar = TypeParam;

  std::vector<Scalar> matrix{Scalar{1}, Scalar{0}, Scalar{0}, Scalar{0}, Scalar{2},
                             Scalar{0}, Scalar{0}, Scalar{0}, Scalar{3}};
  DenseHostVectorOps<Scalar> ops(3, matrix);
  DenseHostVector<Scalar> initial{{Scalar{1}, Scalar{-2}, Scalar{0.5}}};

  TaylorExponentialParams<Scalar> params;
  params.tolerance = static_cast<Scalar>(exponential_tolerance<Scalar>());
  params.step_norm_limit = Scalar{0.25};
  params.diagnostics = KrylovDiagnosticsLevel::Summary;
  Scalar const time = Scalar{0.25};

  auto result = uni20::krylov::taylor_exponential_action<Scalar>(ops, initial, time, Scalar{3}, params);

  std::vector<Scalar> expected{std::exp(time * Scalar{1}) * initial.values[0],
                               std::exp(time * Scalar{2}) * initial.values[1],
                               std::exp(time * Scalar{3}) * initial.values[2]};
  expect_vector_near(result.action, expected, 5.0 * exponential_tolerance<Scalar>());
  EXPECT_TRUE(result.converged);
  EXPECT_GT(result.scaling_steps, 1);
  EXPECT_GT(result.matvec_count, 0);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->scaling_steps, result.scaling_steps);
  EXPECT_EQ(result.diagnostics->matvec_count, result.matvec_count);
}

TYPED_TEST(KrylovExponentialRealTypedTest, TaylorActionMatchesJordanExponential)
{
  using Scalar = TypeParam;

  Scalar const diagonal = Scalar{0.3};
  Scalar const off_diagonal = Scalar{2};
  std::vector<Scalar> matrix{diagonal, off_diagonal, Scalar{0}, diagonal};
  DenseHostVectorOps<Scalar> ops(2, matrix);
  DenseHostVector<Scalar> initial{{Scalar{2}, Scalar{-1}}};

  TaylorExponentialParams<Scalar> params;
  params.tolerance = static_cast<Scalar>(exponential_tolerance<Scalar>());
  Scalar const time = Scalar{0.4};

  auto result = uni20::krylov::taylor_exponential_action<Scalar>(ops, initial, time, Scalar{3}, params);

  Scalar const factor = std::exp(diagonal * time);
  std::vector<Scalar> expected{factor * (initial.values[0] + off_diagonal * time * initial.values[1]),
                               factor * initial.values[1]};
  expect_vector_near(result.action, expected, 5.0 * exponential_tolerance<Scalar>());
  EXPECT_TRUE(result.converged);
}

TYPED_TEST(KrylovExponentialComplexTypedTest, TaylorActionAcceptsComplexTime)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;

  std::vector<Scalar> matrix{Scalar{Real{1}}, Scalar{}, Scalar{}, Scalar{Real{3}}};
  DenseHostVectorOps<Scalar> ops(2, matrix);
  DenseHostVector<Scalar> initial{{Scalar{Real{1}, Real{0.25}}, Scalar{Real{-0.5}, Real{0.75}}}};

  TaylorExponentialParams<Real> params;
  params.tolerance = static_cast<Real>(exponential_tolerance<Scalar>());
  Real const time = Real{0.4};
  Scalar const imaginary_time{Real{}, -time};

  auto result = uni20::krylov::taylor_exponential_action<Scalar>(ops, initial, imaginary_time, Real{3}, params);

  std::vector<Scalar> expected{std::exp(imaginary_time * Scalar{Real{1}}) * initial.values[0],
                               std::exp(imaginary_time * Scalar{Real{3}}) * initial.values[1]};
  expect_vector_near(result.action, expected, 5.0 * exponential_tolerance<Scalar>());
  EXPECT_TRUE(result.converged);
}

TYPED_TEST(KrylovExponentialComplexTypedTest, TaylorZeroVectorReturnsZeroWithoutMatvec)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;

  std::vector<Scalar> matrix{Scalar{Real{1}}, Scalar{}, Scalar{}, Scalar{Real{2}}};
  DenseHostVectorOps<Scalar> ops(2, matrix);
  DenseHostVector<Scalar> initial{{Scalar{}, Scalar{}}};

  TaylorExponentialParams<Real> params;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::taylor_exponential_action<Scalar>(ops, initial, Real{1}, Real{2}, params);

  expect_vector_near(result.action, std::vector<Scalar>{Scalar{}, Scalar{}}, exponential_tolerance<Scalar>());
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.matvec_count, 0);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->matvec_count, 0);
}

TEST(KrylovExponentialTaylorAction, ThrowsWhenTaylorDegreeCannotMeetTolerance)
{
  using Scalar = double;

  std::vector<Scalar> matrix{Scalar{2}};
  DenseHostVectorOps<Scalar> ops(1, matrix);
  DenseHostVector<Scalar> initial{{Scalar{1}}};

  TaylorExponentialParams<Scalar> params;
  params.tolerance = Scalar{1.0e-14};
  params.step_norm_limit = Scalar{10};
  params.max_taylor_degree = 1;

  EXPECT_THROW((uni20::krylov::taylor_exponential_action<Scalar>(ops, initial, Scalar{1}, Scalar{2}, params)),
               std::runtime_error);
}
