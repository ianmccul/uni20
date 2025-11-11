#include <gtest/gtest.h>

#include "expokit/expokitf.h"

#include <cmath>
#include <complex>
#include <numbers>

namespace
{

using Complex = std::complex<double>;
using Matrix = EXPOKIT::Matrix<Complex>;

void ExpectMatrixNear(Matrix const& actual, Matrix const& expected, double tol)
{
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());

    for (std::size_t i = 0; i < actual.rows(); ++i) {
        for (std::size_t j = 0; j < actual.cols(); ++j) {
            Complex const value = actual(i, j);
            Complex const reference = expected(i, j);
            EXPECT_NEAR(std::real(value), std::real(reference), tol);
            EXPECT_NEAR(std::imag(value), std::imag(reference), tol);
        }
    }
}

Matrix MakeZeroMatrix(std::size_t order)
{
    return Matrix(order, order);
}

Matrix MakeIdentity(std::size_t order)
{
    Matrix identity(order, order);
    for (std::size_t i = 0; i < order; ++i) {
        for (std::size_t j = 0; j < order; ++j) {
            identity(i, j) = (i == j) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
        }
    }
    return identity;
}

} // namespace

TEST(ExpmTest, ZeroMatrixReturnsIdentity)
{
    Matrix matrix = MakeZeroMatrix(3);

    Matrix const result = EXPOKIT::expm(matrix, 1.0);
    Matrix const expected = MakeIdentity(3);

    ExpectMatrixNear(result, expected, 1.0e-12);
}

TEST(ExpmTest, ScalarMatrixMatchesScalarExponential)
{
    Matrix matrix(1, 1);
    matrix(0, 0) = Complex(2.0, -0.5);

    Matrix const result = EXPOKIT::expm(matrix, 1.0);

    Complex const expected = std::exp(matrix(0, 0));
    EXPECT_NEAR(std::real(result(0, 0)), std::real(expected), 1.0e-12);
    EXPECT_NEAR(std::imag(result(0, 0)), std::imag(expected), 1.0e-12);
}

TEST(ExpmTest, SkewHermitianGeneratesRotation)
{
    Matrix matrix(2, 2);
    matrix(0, 0) = Complex(0.0, 0.0);
    matrix(0, 1) = Complex(-1.0, 0.0);
    matrix(1, 0) = Complex(1.0, 0.0);
    matrix(1, 1) = Complex(0.0, 0.0);

    double const angle = std::numbers::pi / 2.0;
    Matrix const result = EXPOKIT::expm(matrix, angle);

    Matrix expected(2, 2);
    expected(0, 0) = Complex(std::cos(angle), 0.0);
    expected(0, 1) = Complex(-std::sin(angle), 0.0);
    expected(1, 0) = Complex(std::sin(angle), 0.0);
    expected(1, 1) = Complex(std::cos(angle), 0.0);

    ExpectMatrixNear(result, expected, 1.0e-9);
}
