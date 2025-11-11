#include "expokit/expokitf.h"
#include "expokit/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace EXPOKIT
{
namespace
{

using Complex = std::complex<double>;

struct PadeResult
{
    Matrix<Complex> matrix;
    bool odd = false;
};

PadeResult pade_step(Matrix<Complex> const& base, std::vector<Complex> const& coefficients,
                     Complex scale_factor)
{
    std::size_t const degree = coefficients.size() - 1;
    if (degree == 0) {
        return {make_identity<Complex>(base.rows()), true};
    }
    std::size_t const n = base.rows();

    Complex const scale_squared = scale_factor * scale_factor;
    Matrix<Complex> h2 = scale(multiply(base, base), scale_squared);

    Matrix<Complex> p = scale(make_identity<Complex>(n), coefficients[degree - 1]);
    Matrix<Complex> q = scale(make_identity<Complex>(n), coefficients[degree]);

    bool odd = true;
    for (std::size_t k = degree - 1; k > 0; --k) {
        if (odd) {
            Matrix<Complex> tmp = multiply(q, h2);
            for (std::size_t i = 0; i < n; ++i) {
                tmp(i, i) += coefficients[k - 1];
            }
            q.swap(tmp);
        } else {
            Matrix<Complex> tmp = multiply(p, h2);
            for (std::size_t i = 0; i < n; ++i) {
                tmp(i, i) += coefficients[k - 1];
            }
            p.swap(tmp);
        }
        odd = !odd;
    }

    if (odd) {
        Matrix<Complex> scaled = scale(q, scale_factor);
        q = multiply(scaled, base);
    } else {
        Matrix<Complex> scaled = scale(p, scale_factor);
        p = multiply(scaled, base);
    }

    Matrix<Complex> diff = subtract(q, p);
    Matrix<Complex> solved = solve_linear_system(diff, p);
    solved = scale(solved, 2.0);
    Matrix<Complex> identity = make_identity<Complex>(n);
    solved = add(solved, identity);

    return {std::move(solved), odd};
}

} // namespace

Matrix<Complex> expm(Matrix<Complex> const& matrix, double t, int ideg)
{
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("expm requires a square matrix");
    }

    if (matrix.rows() == 0) {
        return Matrix<Complex>();
    }

    if (ideg < 1) {
        throw std::invalid_argument("expm requires a positive Pade degree");
    }

    std::size_t const m = matrix.rows();

    double hnorm = matrix_one_norm(matrix);
    hnorm = std::abs(t) * hnorm;
    if (hnorm == 0.0) {
        return make_identity<Complex>(m);
    }

    double const log2 = std::log(2.0);
    int const ns = std::max(0, static_cast<int>(std::log(hnorm) / log2) + 2);
    Complex const scale_factor(std::ldexp(t, -ns), 0.0);

    std::vector<Complex> coefficients(static_cast<std::size_t>(ideg) + 1);
    coefficients[0] = Complex(1.0, 0.0);
    for (int k = 1; k <= ideg; ++k) {
        coefficients[static_cast<std::size_t>(k)] =
            coefficients[static_cast<std::size_t>(k - 1)]
            * static_cast<double>(ideg + 1 - k)
            / static_cast<double>(k * (2 * ideg + 1 - k));
    }

    PadeResult const pade = pade_step(matrix, coefficients, scale_factor);

    Matrix<Complex> exponential = pade.matrix;

    if (ns == 0 && pade.odd) {
        exponential = scale(exponential, -1.0);
    }

    for (int k = 0; k < ns; ++k) {
        exponential = multiply(exponential, exponential);
    }

    return exponential;
}

} // namespace EXPOKIT
