#include <gtest/gtest.h>

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/backends/cpu/matrix_exponential.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <type_traits>

namespace
{

namespace cpu_linalg = uni20::linalg::backends::cpu;

template <typename Scalar> using DenseMatrix = cpu_linalg::DenseMatrix<Scalar>;

template <typename Scalar> double DefaultTolerance()
{
  if constexpr (std::is_same_v<uni20::make_real_t<Scalar>, float>)
  {
    return 1.0e-5;
  }
  else
  {
    return 1.0e-12;
  }
}

template <typename Scalar> double RelaxedTolerance()
{
  if constexpr (std::is_same_v<uni20::make_real_t<Scalar>, float>)
  {
    return 5.0e-4;
  }
  else
  {
    return 1.0e-10;
  }
}

template <typename Scalar>
void ExpectMatrixNear(DenseMatrix<Scalar> const& actual, DenseMatrix<Scalar> const& expected, double tol)
{
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());

  for (std::size_t i = 0; i < actual.rows(); ++i)
  {
    for (std::size_t j = 0; j < actual.cols(); ++j)
    {
      double const difference = static_cast<double>(std::abs(actual[i, j] - expected[i, j]));
      double const magnitude = std::max(1.0, static_cast<double>(std::abs(expected[i, j])));
      EXPECT_LE(difference, tol * magnitude)
          << "entry (" << i << ", " << j << ") differs: actual=" << actual[i, j] << " expected=" << expected[i, j];
    }
  }
}

template <typename Scalar> void ExpectFiniteBounded(Scalar const& value, double bound)
{
  if constexpr (uni20::Complex<Scalar>)
  {
    EXPECT_TRUE(std::isfinite(static_cast<double>(value.real()))) << "value=" << value;
    EXPECT_TRUE(std::isfinite(static_cast<double>(value.imag()))) << "value=" << value;
  }
  else
  {
    EXPECT_TRUE(std::isfinite(static_cast<double>(value))) << "value=" << value;
  }

  EXPECT_LT(static_cast<double>(std::abs(value)), bound) << "value=" << value;
}

template <typename Scalar> DenseMatrix<Scalar> MakeIdentity(std::size_t order)
{
  using Real = uni20::make_real_t<Scalar>;
  DenseMatrix<Scalar> identity(order, order);
  for (std::size_t i = 0; i < order; ++i)
  {
    for (std::size_t j = 0; j < order; ++j)
    {
      identity[i, j] = (i == j) ? Scalar(Real{1}) : Scalar();
    }
  }
  return identity;
}

template <typename Scalar> DenseMatrix<Scalar> MakeZeroMatrix(std::size_t order)
{
  return DenseMatrix<Scalar>(order, order);
}

template <typename Scalar> Scalar MakeScalarValue()
{
  using Real = uni20::make_real_t<Scalar>;
  if constexpr (uni20::Complex<Scalar>)
  {
    return Scalar(Real{2.0}, Real{-0.5});
  }
  else
  {
    return Scalar(Real{2.0});
  }
}

template <typename Scalar> Scalar MakeLargeOffDiagonal()
{
  using Real = uni20::make_real_t<Scalar>;
  return Scalar(Real{1000.0});
}

template <typename Scalar> Scalar MakeLargeDiagonal()
{
  using Real = uni20::make_real_t<Scalar>;
  return Scalar(Real{10.0});
}

template <typename Scalar> Scalar MakeZero() { return Scalar(); }

template <typename Scalar> Scalar MakeOne()
{
  using Real = uni20::make_real_t<Scalar>;
  return Scalar(Real{1});
}

template <typename Scalar> Scalar MakeNegativeOne()
{
  using Real = uni20::make_real_t<Scalar>;
  return Scalar(Real{-1});
}

template <typename Scalar> Scalar MakeLargeNilpotent()
{
  using Real = uni20::make_real_t<Scalar>;
  return Scalar(Real{1.0e3});
}

template <typename Scalar> uni20::make_real_t<Scalar> MakeHugeSkewTheta()
{
  using Real = uni20::make_real_t<Scalar>;
  if constexpr (std::is_same_v<Real, float>)
  {
    return Real{1.0e20F};
  }
  else
  {
    return Real{1.0e80};
  }
}

} // namespace

template <typename Scalar> class MatrixExponentialTypedTest : public ::testing::Test {};

using MatrixExponentialTypes = ::testing::Types<float, double, long double, uni20::complex<float>,
                                                uni20::complex<double>, uni20::complex<long double>>;
TYPED_TEST_SUITE(MatrixExponentialTypedTest, MatrixExponentialTypes);

TYPED_TEST(MatrixExponentialTypedTest, ZeroMatrixReturnsIdentity)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;
  DenseMatrix<Scalar> const matrix = MakeZeroMatrix<Scalar>(3);

  DenseMatrix<Scalar> const result = cpu_linalg::matrix_exponential(matrix, Real{1});
  DenseMatrix<Scalar> const expected = MakeIdentity<Scalar>(3);

  ExpectMatrixNear(result, expected, DefaultTolerance<Scalar>());
}

TYPED_TEST(MatrixExponentialTypedTest, RejectsNaNEntryBeforeZeroNormShortcut)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;
  DenseMatrix<Scalar> matrix(2, 2);
  matrix[0, 0] = Scalar(uni20::numeric_limits<Real>::quiet_NaN());

  EXPECT_THROW(cpu_linalg::matrix_exponential(matrix, Real{1}), std::overflow_error);
}

TYPED_TEST(MatrixExponentialTypedTest, ScalarMatrixMatchesScalarExponential)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;
  DenseMatrix<Scalar> matrix(1, 1);
  Scalar const entry = MakeScalarValue<Scalar>();
  matrix[0, 0] = entry;

  DenseMatrix<Scalar> const result = cpu_linalg::matrix_exponential(matrix, Real{1});

  Scalar const expected = std::exp(entry);
  double const tolerance = DefaultTolerance<Scalar>();
  double const diff = static_cast<double>(std::abs(result[0, 0] - expected));
  double const magnitude = std::max(1.0, static_cast<double>(std::abs(expected)));
  EXPECT_LE(diff, tolerance * magnitude);
}

TEST(MatrixExponentialTest, RealMatrixWithComplexMultiplierPromotesToComplex)
{
  DenseMatrix<double> matrix(1, 1);
  matrix[0, 0] = 2.0;

  uni20::complex<double> const time{0.0, std::numbers::pi / 2.0};
  auto const result = cpu_linalg::matrix_exponential(matrix, time);

  static_assert(std::is_same_v<decltype(result), DenseMatrix<uni20::complex<double>> const>);
  uni20::complex<double> const expected = std::exp(time * matrix[0, 0]);
  EXPECT_LE(std::abs(result[0, 0] - expected), 1.0e-12);
}

TEST(MatrixExponentialTest, ComplexMatrixWithComplexMultiplierUsesComplexTime)
{
  DenseMatrix<uni20::complex<double>> matrix(1, 1);
  matrix[0, 0] = uni20::complex<double>{1.0, -0.25};

  uni20::complex<double> const time{0.2, -0.3};
  auto const result = cpu_linalg::matrix_exponential(matrix, time);

  uni20::complex<double> const expected = std::exp(time * matrix[0, 0]);
  EXPECT_LE(std::abs(result[0, 0] - expected), 1.0e-12);
}

TYPED_TEST(MatrixExponentialTypedTest, SkewSymmetricGeneratesRotation)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;
  DenseMatrix<Scalar> matrix(2, 2);
  matrix[0, 0] = MakeZero<Scalar>();
  matrix[0, 1] = MakeNegativeOne<Scalar>();
  matrix[1, 0] = MakeOne<Scalar>();
  matrix[1, 1] = MakeZero<Scalar>();

  Real const angle = std::numbers::pi_v<Real> / Real{2};
  DenseMatrix<Scalar> const result = cpu_linalg::matrix_exponential(matrix, angle);

  DenseMatrix<Scalar> expected(2, 2);
  Real const cosine = std::cos(angle);
  Real const sine = std::sin(angle);
  expected[0, 0] = Scalar(cosine);
  expected[0, 1] = Scalar(-sine);
  expected[1, 0] = Scalar(sine);
  expected[1, 1] = Scalar(cosine);

  ExpectMatrixNear(result, expected, RelaxedTolerance<Scalar>());
}

TEST(MatrixExponentialTest, LongDoubleNilpotentUsesExtendedPrecisionScaling)
{
  DenseMatrix<long double> matrix(2, 2);
  long double const large = std::ldexp(1.0L, uni20::numeric_limits<long double>::max_exponent - 2);
  ASSERT_TRUE(std::isfinite(large));
  matrix[0, 1] = large;

  DenseMatrix<long double> const result = cpu_linalg::matrix_exponential(matrix, 1.0L);

  long double constexpr tolerance = 1.0e-12L;
  EXPECT_LE(std::abs(result[0, 0] - 1.0L), tolerance);
  EXPECT_LE(std::abs(result[1, 0]), tolerance);
  EXPECT_LE(std::abs(result[1, 1] - 1.0L), tolerance);
  EXPECT_LE(std::abs((result[0, 1] - large) / large), tolerance);
}

TYPED_TEST(MatrixExponentialTypedTest, HugeSkewSymmetricGeneratorStaysFinite)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;
  DenseMatrix<Scalar> matrix(2, 2);
  Real const theta = MakeHugeSkewTheta<Scalar>();
  matrix[0, 0] = MakeZero<Scalar>();
  matrix[0, 1] = Scalar(-theta);
  matrix[1, 0] = Scalar(theta);
  matrix[1, 1] = MakeZero<Scalar>();

  DenseMatrix<Scalar> const result = cpu_linalg::matrix_exponential(matrix, Real{1});

  ExpectFiniteBounded(result[0, 0], 2.0);
  ExpectFiniteBounded(result[0, 1], 2.0);
  ExpectFiniteBounded(result[1, 0], 2.0);
  ExpectFiniteBounded(result[1, 1], 2.0);
}

template <typename Scalar> void RunOverflowedOneNormSkewSymmetricGeneratorStaysFinite()
{
  using Real = uni20::make_real_t<Scalar>;
  Real const theta = Real{1.0e308};
  DenseMatrix<Scalar> matrix(3, 3);
  matrix[0, 0] = MakeZero<Scalar>();
  matrix[0, 1] = Scalar(-theta);
  matrix[0, 2] = Scalar(theta);
  matrix[1, 0] = Scalar(theta);
  matrix[1, 1] = MakeZero<Scalar>();
  matrix[1, 2] = Scalar(-theta);
  matrix[2, 0] = Scalar(-theta);
  matrix[2, 1] = Scalar(theta);
  matrix[2, 2] = MakeZero<Scalar>();

  DenseMatrix<Scalar> const result = cpu_linalg::matrix_exponential(matrix, Real{1});

  for (std::size_t i = 0; i < result.rows(); ++i)
  {
    for (std::size_t j = 0; j < result.cols(); ++j)
    {
      ExpectFiniteBounded(result[i, j], 2.0);
    }
  }
}

TEST(MatrixExponentialTest, OverflowedOneNormSkewSymmetricGeneratorStaysFinite)
{
  RunOverflowedOneNormSkewSymmetricGeneratorStaysFinite<double>();
  RunOverflowedOneNormSkewSymmetricGeneratorStaysFinite<uni20::complex<double>>();
}

TYPED_TEST(MatrixExponentialTypedTest, HighNormJordanBlockMatchesAnalyticSolution)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;
  DenseMatrix<Scalar> matrix(2, 2);
  Scalar const diag = MakeLargeDiagonal<Scalar>();
  Scalar const off = MakeLargeOffDiagonal<Scalar>();
  matrix[0, 0] = diag;
  matrix[0, 1] = off;
  matrix[1, 0] = MakeZero<Scalar>();
  matrix[1, 1] = diag;

  DenseMatrix<Scalar> const result = cpu_linalg::matrix_exponential(matrix, Real{1});

  Scalar const exp_diag = std::exp(diag);
  DenseMatrix<Scalar> expected(2, 2);
  expected[0, 0] = exp_diag;
  expected[0, 1] = exp_diag * off;
  expected[1, 0] = MakeZero<Scalar>();
  expected[1, 1] = exp_diag;

  ExpectMatrixNear(result, expected, RelaxedTolerance<Scalar>());
}

TYPED_TEST(MatrixExponentialTypedTest, NilpotentChainMatchesSeries)
{
  using Scalar = TypeParam;
  using Real = uni20::make_real_t<Scalar>;
  DenseMatrix<Scalar> matrix(3, 3);
  Scalar const large = MakeLargeNilpotent<Scalar>();
  matrix[0, 0] = MakeZero<Scalar>();
  matrix[0, 1] = large;
  matrix[0, 2] = MakeZero<Scalar>();
  matrix[1, 0] = MakeZero<Scalar>();
  matrix[1, 1] = MakeZero<Scalar>();
  matrix[1, 2] = large;
  matrix[2, 0] = MakeZero<Scalar>();
  matrix[2, 1] = MakeZero<Scalar>();
  matrix[2, 2] = MakeZero<Scalar>();

  DenseMatrix<Scalar> const result = cpu_linalg::matrix_exponential(matrix, Real{1});

  DenseMatrix<Scalar> expected = MakeIdentity<Scalar>(3);
  DenseMatrix<Scalar> const matrix_squared = cpu_linalg::matrix_power(matrix, 2);
  DenseMatrix<Scalar> const scaled_matrix = cpu_linalg::add(matrix, cpu_linalg::scale(matrix_squared, Real{0.5}));
  expected = cpu_linalg::add(expected, scaled_matrix);

  ExpectMatrixNear(result, expected, RelaxedTolerance<Scalar>());
}
