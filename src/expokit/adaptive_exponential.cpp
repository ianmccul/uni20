#include "expokit/expokitf.h"
#include "expokit/matrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace EXPOKIT
{
namespace detail
{

constexpr std::array<double, 5> kThetaBounds{
    1.495585217958292e-2,  // theta_3
    2.539398330063230e-1,  // theta_5
    9.504178996162932e-1,  // theta_7
    2.097847961257068,     // theta_9
    5.371920351148152      // theta_13
};

template <typename Scalar>
Scalar to_scalar(double value)
{
    using Real = uni20::make_real_t<Scalar>;
    return Scalar(static_cast<Real>(value));
}

template <typename Scalar>
Matrix<Scalar> linear_combination(
    std::size_t rows, std::size_t cols,
    std::initializer_list<std::pair<Matrix<Scalar> const*, Scalar>> const& terms)
{
    Matrix<Scalar> result(rows, cols);
    for (auto const& term : terms) {
        Matrix<Scalar> const& mat = *term.first;
        Scalar const coefficient = term.second;
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                result(i, j) += coefficient * mat(i, j);
            }
        }
    }
    return result;
}

template <typename Scalar>
Matrix<Scalar> solve_pade(Matrix<Scalar> const& U, Matrix<Scalar> const& V)
{
    Matrix<Scalar> numerator = add(V, U);
    Matrix<Scalar> denominator = subtract(V, U);
    return solve_linear_system(denominator, numerator);
}

template <typename Scalar>
Matrix<Scalar> pade3(Matrix<Scalar> const& A)
{
    static constexpr std::array<double, 4> b{120.0, 60.0, 12.0, 1.0};
    std::size_t const n = A.rows();
    Matrix<Scalar> const identity = make_identity<Scalar>(n);
    Matrix<Scalar> const A2 = multiply(A, A);

    Matrix<Scalar> const tmp = linear_combination<Scalar>(
        n, n, {{&A2, to_scalar<Scalar>(b[3])}, {&identity, to_scalar<Scalar>(b[1])}});
    Matrix<Scalar> const U = multiply(A, tmp);

    Matrix<Scalar> const V = linear_combination<Scalar>(
        n, n, {{&A2, to_scalar<Scalar>(b[2])}, {&identity, to_scalar<Scalar>(b[0])}});

    return solve_pade(U, V);
}

template <typename Scalar>
Matrix<Scalar> pade5(Matrix<Scalar> const& A)
{
    static constexpr std::array<double, 6> b{30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0};
    std::size_t const n = A.rows();
    Matrix<Scalar> const identity = make_identity<Scalar>(n);
    Matrix<Scalar> const A2 = multiply(A, A);
    Matrix<Scalar> const A4 = multiply(A2, A2);

    Matrix<Scalar> const tmp = linear_combination<Scalar>(
        n, n,
        {{&A4, to_scalar<Scalar>(b[5])}, {&A2, to_scalar<Scalar>(b[3])}, {&identity, to_scalar<Scalar>(b[1])}});
    Matrix<Scalar> const U = multiply(A, tmp);

    Matrix<Scalar> const V = linear_combination<Scalar>(
        n, n,
        {{&A4, to_scalar<Scalar>(b[4])}, {&A2, to_scalar<Scalar>(b[2])}, {&identity, to_scalar<Scalar>(b[0])}});

    return solve_pade(U, V);
}

template <typename Scalar>
Matrix<Scalar> pade7(Matrix<Scalar> const& A)
{
    static constexpr std::array<double, 8> b{
        17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0};
    std::size_t const n = A.rows();
    Matrix<Scalar> const identity = make_identity<Scalar>(n);
    Matrix<Scalar> const A2 = multiply(A, A);
    Matrix<Scalar> const A4 = multiply(A2, A2);
    Matrix<Scalar> const A6 = multiply(A4, A2);

    Matrix<Scalar> const tmp = linear_combination<Scalar>(
        n, n,
        {{&A6, to_scalar<Scalar>(b[7])}, {&A4, to_scalar<Scalar>(b[5])}, {&A2, to_scalar<Scalar>(b[3])},
         {&identity, to_scalar<Scalar>(b[1])}});
    Matrix<Scalar> const U = multiply(A, tmp);

    Matrix<Scalar> const V = linear_combination<Scalar>(
        n, n,
        {{&A6, to_scalar<Scalar>(b[6])}, {&A4, to_scalar<Scalar>(b[4])}, {&A2, to_scalar<Scalar>(b[2])},
         {&identity, to_scalar<Scalar>(b[0])}});

    return solve_pade(U, V);
}

template <typename Scalar>
Matrix<Scalar> pade9(Matrix<Scalar> const& A)
{
    static constexpr std::array<double, 10> b{
        17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
        2162160.0, 110880.0, 3960.0, 90.0, 1.0};
    std::size_t const n = A.rows();
    Matrix<Scalar> const identity = make_identity<Scalar>(n);
    Matrix<Scalar> const A2 = multiply(A, A);
    Matrix<Scalar> const A4 = multiply(A2, A2);
    Matrix<Scalar> const A6 = multiply(A4, A2);
    Matrix<Scalar> const A8 = multiply(A6, A2);

    Matrix<Scalar> const tmp = linear_combination<Scalar>(
        n, n,
        {{&A8, to_scalar<Scalar>(b[9])}, {&A6, to_scalar<Scalar>(b[7])}, {&A4, to_scalar<Scalar>(b[5])},
         {&A2, to_scalar<Scalar>(b[3])}, {&identity, to_scalar<Scalar>(b[1])}});
    Matrix<Scalar> const U = multiply(A, tmp);

    Matrix<Scalar> const V = linear_combination<Scalar>(
        n, n,
        {{&A8, to_scalar<Scalar>(b[8])}, {&A6, to_scalar<Scalar>(b[6])}, {&A4, to_scalar<Scalar>(b[4])},
         {&A2, to_scalar<Scalar>(b[2])}, {&identity, to_scalar<Scalar>(b[0])}});

    return solve_pade(U, V);
}

template <typename Scalar>
Matrix<Scalar> pade13(
    Matrix<Scalar> const& A, Matrix<Scalar> const& A2, Matrix<Scalar> const& A4, Matrix<Scalar> const& A6)
{
    static constexpr std::array<double, 14> b{
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
        129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0,
        40840800.0, 960960.0, 16380.0, 182.0, 1.0};

    std::size_t const n = A.rows();
    Matrix<Scalar> const identity = make_identity<Scalar>(n);

    Matrix<Scalar> const first = linear_combination<Scalar>(
        n, n,
        {{&A6, to_scalar<Scalar>(b[13])}, {&A4, to_scalar<Scalar>(b[11])}, {&A2, to_scalar<Scalar>(b[9])}});
    Matrix<Scalar> tmp = multiply(A6, first);
    Matrix<Scalar> const second = linear_combination<Scalar>(
        n, n,
        {{&A6, to_scalar<Scalar>(b[7])}, {&A4, to_scalar<Scalar>(b[5])}, {&A2, to_scalar<Scalar>(b[3])},
         {&identity, to_scalar<Scalar>(b[1])}});
    tmp = add(tmp, second);
    Matrix<Scalar> const U = multiply(A, tmp);

    Matrix<Scalar> const third = linear_combination<Scalar>(
        n, n,
        {{&A6, to_scalar<Scalar>(b[12])}, {&A4, to_scalar<Scalar>(b[10])}, {&A2, to_scalar<Scalar>(b[8])}});
    Matrix<Scalar> V = multiply(A6, third);
    Matrix<Scalar> const fourth = linear_combination<Scalar>(
        n, n,
        {{&A6, to_scalar<Scalar>(b[6])}, {&A4, to_scalar<Scalar>(b[4])}, {&A2, to_scalar<Scalar>(b[2])},
         {&identity, to_scalar<Scalar>(b[0])}});
    V = add(V, fourth);

    return solve_pade(U, V);
}

template <typename Scalar>
int compute_scaling_exponent(Matrix<Scalar> const& A4, Matrix<Scalar> const& A6)
{
    double const norm4 = matrix_one_norm(A4);
    double const norm6 = matrix_one_norm(A6);
    double const d4 = std::pow(norm4, 0.25);
    double const d6 = std::pow(norm6, 1.0 / 6.0);
    double const eta = std::max(d4, d6);
    if (eta == 0.0) {
        return 0;
    }

    double const ratio = eta / kThetaBounds.back();
    if (ratio <= 1.0) {
        return 0;
    }

    double const exponent = std::log2(ratio);
    if (exponent <= 0.0) {
        return 0;
    }
    return static_cast<int>(std::ceil(exponent));
}

} // namespace detail

template <uni20::RealOrComplex Scalar>
Matrix<Scalar> expm(Matrix<Scalar> const& matrix, uni20::make_real_t<Scalar> t, int ideg)
{
    using Real = uni20::make_real_t<Scalar>;

    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("expm requires a square matrix");
    }

    if (matrix.rows() == 0) {
        return Matrix<Scalar>();
    }

    if (ideg < 1) {
        throw std::invalid_argument("expm requires a positive Pade degree");
    }

    std::size_t const n = matrix.rows();
    Matrix<Scalar> A = scale(matrix, Scalar(t));

    double const normA = matrix_one_norm(A);
    if (normA == 0.0) {
        return make_identity<Scalar>(n);
    }

    (void)ideg; // retained for compatibility

    if (normA <= detail::kThetaBounds[0]) {
        return detail::pade3(A);
    }
    if (normA <= detail::kThetaBounds[1]) {
        return detail::pade5(A);
    }
    if (normA <= detail::kThetaBounds[2]) {
        return detail::pade7(A);
    }
    if (normA <= detail::kThetaBounds[3]) {
        return detail::pade9(A);
    }

    Matrix<Scalar> const A2 = multiply(A, A);
    Matrix<Scalar> const A4 = multiply(A2, A2);
    Matrix<Scalar> const A6 = multiply(A4, A2);

    int const s = detail::compute_scaling_exponent(A4, A6);
    Real const scale_real = static_cast<Real>(std::ldexp(1.0, -s));
    Matrix<Scalar> const scaled_A = scale(A, scale_real);

    Matrix<Scalar> A2_scaled;
    Matrix<Scalar> A4_scaled;
    Matrix<Scalar> A6_scaled;
    if (s == 0) {
        A2_scaled = A2;
        A4_scaled = A4;
        A6_scaled = A6;
    } else {
        Real const scale_sq = scale_real * scale_real;
        Real const scale_pow4 = scale_sq * scale_sq;
        Real const scale_pow6 = scale_pow4 * scale_sq;
        A2_scaled = scale(A2, scale_sq);
        A4_scaled = scale(A4, scale_pow4);
        A6_scaled = scale(A6, scale_pow6);
    }

    Matrix<Scalar> result = detail::pade13(scaled_A, A2_scaled, A4_scaled, A6_scaled);
    for (int k = 0; k < s; ++k) {
        result = multiply(result, result);
    }

    return result;
}

template Matrix<float> expm(Matrix<float> const&, float, int);
template Matrix<double> expm(Matrix<double> const&, double, int);
template Matrix<std::complex<float>> expm(Matrix<std::complex<float>> const&, float, int);
template Matrix<std::complex<double>> expm(Matrix<std::complex<double>> const&, double, int);

} // namespace EXPOKIT

