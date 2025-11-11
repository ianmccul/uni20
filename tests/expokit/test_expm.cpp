#include <gtest/gtest.h>

#include "core/scalar_concepts.hpp"
#include "expokit/expokitf.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <type_traits>

namespace
{

template <typename Scalar>
using Matrix = EXPOKIT::Matrix<Scalar>;

template <typename Scalar>
double DefaultTolerance()
{
    if constexpr (std::is_same_v<uni20::make_real_t<Scalar>, float>) {
        return 1.0e-5;
    } else {
        return 1.0e-12;
    }
}

template <typename Scalar>
double RelaxedTolerance()
{
    if constexpr (std::is_same_v<uni20::make_real_t<Scalar>, float>) {
        return 5.0e-4;
    } else {
        return 1.0e-10;
    }
}

template <typename Scalar>
void ExpectMatrixNear(Matrix<Scalar> const& actual, Matrix<Scalar> const& expected, double tol)
{
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());

    for (std::size_t i = 0; i < actual.rows(); ++i) {
        for (std::size_t j = 0; j < actual.cols(); ++j) {
            double const difference = static_cast<double>(std::abs(actual(i, j) - expected(i, j)));
            double const magnitude =
                std::max(1.0, static_cast<double>(std::abs(expected(i, j))));
            EXPECT_LE(difference, tol * magnitude)
                << "entry (" << i << ", " << j << ") differs: actual=" << actual(i, j)
                << " expected=" << expected(i, j);
        }
    }
}

template <typename Scalar>
Matrix<Scalar> MakeIdentity(std::size_t order)
{
    using Real = uni20::make_real_t<Scalar>;
    Matrix<Scalar> identity(order, order);
    for (std::size_t i = 0; i < order; ++i) {
        for (std::size_t j = 0; j < order; ++j) {
            identity(i, j) = (i == j) ? Scalar(Real{1}) : Scalar();
        }
    }
    return identity;
}

template <typename Scalar>
Matrix<Scalar> MakeZeroMatrix(std::size_t order)
{
    return Matrix<Scalar>(order, order);
}

template <typename Scalar>
Scalar MakeScalarValue()
{
    using Real = uni20::make_real_t<Scalar>;
    if constexpr (uni20::Complex<Scalar>) {
        return Scalar(Real{2.0}, Real{-0.5});
    } else {
        return Scalar(Real{2.0});
    }
}

template <typename Scalar>
Scalar MakeLargeOffDiagonal()
{
    using Real = uni20::make_real_t<Scalar>;
    return Scalar(Real{1000.0});
}

template <typename Scalar>
Scalar MakeLargeDiagonal()
{
    using Real = uni20::make_real_t<Scalar>;
    return Scalar(Real{10.0});
}

template <typename Scalar>
Scalar MakeZero()
{
    return Scalar();
}

template <typename Scalar>
Scalar MakeOne()
{
    using Real = uni20::make_real_t<Scalar>;
    return Scalar(Real{1});
}

template <typename Scalar>
Scalar MakeNegativeOne()
{
    using Real = uni20::make_real_t<Scalar>;
    return Scalar(Real{-1});
}

template <typename Scalar>
Scalar MakeLargeNilpotent()
{
    using Real = uni20::make_real_t<Scalar>;
    return Scalar(Real{1.0e3});
}

} // namespace

template <typename Scalar>
class ExpmTypedTest : public ::testing::Test
{};

using ExpmTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ExpmTypedTest, ExpmTypes);

TYPED_TEST(ExpmTypedTest, ZeroMatrixReturnsIdentity)
{
    using Scalar = TypeParam;
    using Real = uni20::make_real_t<Scalar>;
    Matrix<Scalar> const matrix = MakeZeroMatrix<Scalar>(3);

    Matrix<Scalar> const result = EXPOKIT::expm(matrix, Real{1});
    Matrix<Scalar> const expected = MakeIdentity<Scalar>(3);

    ExpectMatrixNear(result, expected, DefaultTolerance<Scalar>());
}

TYPED_TEST(ExpmTypedTest, ScalarMatrixMatchesScalarExponential)
{
    using Scalar = TypeParam;
    using Real = uni20::make_real_t<Scalar>;
    Matrix<Scalar> matrix(1, 1);
    Scalar const entry = MakeScalarValue<Scalar>();
    matrix(0, 0) = entry;

    Matrix<Scalar> const result = EXPOKIT::expm(matrix, Real{1});

    Scalar const expected = std::exp(entry);
    double const tolerance = DefaultTolerance<Scalar>();
    double const diff = static_cast<double>(std::abs(result(0, 0) - expected));
    double const magnitude = std::max(1.0, static_cast<double>(std::abs(expected)));
    EXPECT_LE(diff, tolerance * magnitude);
}

TYPED_TEST(ExpmTypedTest, SkewSymmetricGeneratesRotation)
{
    using Scalar = TypeParam;
    using Real = uni20::make_real_t<Scalar>;
    Matrix<Scalar> matrix(2, 2);
    matrix(0, 0) = MakeZero<Scalar>();
    matrix(0, 1) = MakeNegativeOne<Scalar>();
    matrix(1, 0) = MakeOne<Scalar>();
    matrix(1, 1) = MakeZero<Scalar>();

    Real const angle = static_cast<Real>(std::numbers::pi / 2.0);
    Matrix<Scalar> const result = EXPOKIT::expm(matrix, angle);

    Matrix<Scalar> expected(2, 2);
    Real const cosine = std::cos(angle);
    Real const sine = std::sin(angle);
    expected(0, 0) = Scalar(cosine);
    expected(0, 1) = Scalar(-sine);
    expected(1, 0) = Scalar(sine);
    expected(1, 1) = Scalar(cosine);

    ExpectMatrixNear(result, expected, RelaxedTolerance<Scalar>());
}

TYPED_TEST(ExpmTypedTest, HighNormJordanBlockMatchesAnalyticSolution)
{
    using Scalar = TypeParam;
    using Real = uni20::make_real_t<Scalar>;
    Matrix<Scalar> matrix(2, 2);
    Scalar const diag = MakeLargeDiagonal<Scalar>();
    Scalar const off = MakeLargeOffDiagonal<Scalar>();
    matrix(0, 0) = diag;
    matrix(0, 1) = off;
    matrix(1, 0) = MakeZero<Scalar>();
    matrix(1, 1) = diag;

    Matrix<Scalar> const result = EXPOKIT::expm(matrix, Real{1});

    Scalar const exp_diag = std::exp(diag);
    Matrix<Scalar> expected(2, 2);
    expected(0, 0) = exp_diag;
    expected(0, 1) = exp_diag * off;
    expected(1, 0) = MakeZero<Scalar>();
    expected(1, 1) = exp_diag;

    ExpectMatrixNear(result, expected, RelaxedTolerance<Scalar>());
}

TYPED_TEST(ExpmTypedTest, NilpotentChainMatchesSeries)
{
    using Scalar = TypeParam;
    using Real = uni20::make_real_t<Scalar>;
    Matrix<Scalar> matrix(3, 3);
    Scalar const large = MakeLargeNilpotent<Scalar>();
    matrix(0, 0) = MakeZero<Scalar>();
    matrix(0, 1) = large;
    matrix(0, 2) = MakeZero<Scalar>();
    matrix(1, 0) = MakeZero<Scalar>();
    matrix(1, 1) = MakeZero<Scalar>();
    matrix(1, 2) = large;
    matrix(2, 0) = MakeZero<Scalar>();
    matrix(2, 1) = MakeZero<Scalar>();
    matrix(2, 2) = MakeZero<Scalar>();

    Matrix<Scalar> const result = EXPOKIT::expm(matrix, Real{1});

    Matrix<Scalar> expected = MakeIdentity<Scalar>(3);
    Matrix<Scalar> const matrix_squared = EXPOKIT::matrix_power(matrix, 2);
    Matrix<Scalar> const scaled_matrix = EXPOKIT::add(matrix, EXPOKIT::scale(matrix_squared, Real{0.5}));
    expected = EXPOKIT::add(expected, scaled_matrix);

    ExpectMatrixNear(result, expected, RelaxedTolerance<Scalar>());
}

