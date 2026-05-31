#pragma once

#include <uni20/tensorcontraction/matrix_family.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction
{

struct LanczosOptions
{
    int max_iterations = 20;
    int min_iterations = 2;
    double tolerance = 1.0e-10;
    double beta_tolerance = 1.0e-14;
    double orthogonality_tolerance = 1.0e-10;
};

enum class LanczosStopReason
{
  Converged,
  MaxIterations,
  InvariantSubspace,
  LossOfOrthogonality,
};

struct LanczosResult
{
    double eigenvalue = 0.0;
    int iterations = 0;
    double tolerance = 0.0;
    double residual_norm = 0.0;
    double spectral_diameter = 0.0;
    LanczosStopReason stop_reason = LanczosStopReason::MaxIterations;

    [[nodiscard]] bool converged() const noexcept { return stop_reason != LanczosStopReason::MaxIterations; }
};

namespace detail
{

struct DenseEigenpair
{
    double eigenvalue = 0.0;
    std::vector<double> eigenvector;
    double spectral_diameter = 0.0;
};

inline std::size_t dense_index(std::size_t n, std::size_t row, std::size_t col) { return row * n + col; }

inline DenseEigenpair lowest_eigenpair(std::vector<double> matrix, std::size_t n)
{
  if (n == 0)
  {
    throw std::invalid_argument("Lanczos dense eigensolver requires a non-empty matrix");
  }

  std::vector<double> eigenvectors(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
  {
    eigenvectors[dense_index(n, i, i)] = 1.0;
  }

  auto max_sweeps = std::max<std::size_t>(32, 16 * n * n);
  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep)
  {
    std::size_t p = 0;
    std::size_t q = 0;
    double max_offdiag = 0.0;
    for (std::size_t row = 0; row < n; ++row)
    {
      for (std::size_t col = row + 1; col < n; ++col)
      {
        double const value = std::abs(matrix[dense_index(n, row, col)]);
        if (value > max_offdiag)
        {
          max_offdiag = value;
          p = row;
          q = col;
        }
      }
    }

    if (max_offdiag <= 100.0 * std::numeric_limits<double>::epsilon())
    {
      break;
    }

    double const app = matrix[dense_index(n, p, p)];
    double const aqq = matrix[dense_index(n, q, q)];
    double const apq = matrix[dense_index(n, p, q)];
    double const tau = (aqq - app) / (2.0 * apq);
    double const sign = tau < 0.0 ? -1.0 : 1.0;
    double const t = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    double const c = 1.0 / std::sqrt(1.0 + t * t);
    double const s = t * c;

    for (std::size_t k = 0; k < n; ++k)
    {
      if (k == p || k == q)
      {
        continue;
      }
      double const akp = matrix[dense_index(n, k, p)];
      double const akq = matrix[dense_index(n, k, q)];
      matrix[dense_index(n, k, p)] = matrix[dense_index(n, p, k)] = c * akp - s * akq;
      matrix[dense_index(n, k, q)] = matrix[dense_index(n, q, k)] = s * akp + c * akq;
    }

    matrix[dense_index(n, p, p)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    matrix[dense_index(n, q, q)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    matrix[dense_index(n, p, q)] = 0.0;
    matrix[dense_index(n, q, p)] = 0.0;

    for (std::size_t k = 0; k < n; ++k)
    {
      double const vkp = eigenvectors[dense_index(n, k, p)];
      double const vkq = eigenvectors[dense_index(n, k, q)];
      eigenvectors[dense_index(n, k, p)] = c * vkp - s * vkq;
      eigenvectors[dense_index(n, k, q)] = s * vkp + c * vkq;
    }
  }

  std::size_t lowest = 0;
  std::size_t highest = 0;
  for (std::size_t i = 1; i < n; ++i)
  {
    if (matrix[dense_index(n, i, i)] < matrix[dense_index(n, lowest, lowest)])
    {
      lowest = i;
    }
    if (matrix[dense_index(n, i, i)] > matrix[dense_index(n, highest, highest)])
    {
      highest = i;
    }
  }

  DenseEigenpair result;
  result.eigenvalue = matrix[dense_index(n, lowest, lowest)];
  result.spectral_diameter = matrix[dense_index(n, highest, highest)] - result.eigenvalue;
  result.eigenvector.resize(n);
  for (std::size_t row = 0; row < n; ++row)
  {
    result.eigenvector[row] = eigenvectors[dense_index(n, row, lowest)];
  }
  return result;
}

inline std::vector<double> submatrix(std::vector<double> const& matrix, std::size_t stride, std::size_t n)
{
  std::vector<double> result(n * n, 0.0);
  for (std::size_t row = 0; row < n; ++row)
  {
    for (std::size_t col = 0; col < n; ++col)
    {
      result[dense_index(n, row, col)] = matrix[dense_index(stride, row, col)];
    }
  }
  return result;
}

inline MatrixFamily linear_combination(VectorAlgebraEngine& algebra, std::vector<MatrixFamily> const& vectors,
                                       std::vector<double> const& coefficients)
{
  if (vectors.empty() || vectors.size() != coefficients.size())
  {
    throw std::invalid_argument("Lanczos linear combination has inconsistent inputs");
  }

  auto result = make_like(vectors.front());
  algebra.zero(result);
  for (std::size_t i = 0; i < vectors.size(); ++i)
  {
    algebra.axpy(coefficients[i], vectors[i], result);
  }
  return result;
}

inline std::vector<double> coefficients_for(std::vector<double> const& source, std::size_t n)
{
  return std::vector<double>(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(n));
}

inline double scaled_tolerance(double residual_norm, double spectral_diameter)
{
  return residual_norm / (spectral_diameter == 0.0 ? 1.0 : spectral_diameter);
}

} // namespace detail

template <typename MatVec>
LanczosResult lanczos_lowest(MatrixFamily& guess, MatVec&& matvec, LanczosOptions options = {})
{
  if (options.max_iterations < 1)
  {
    throw std::invalid_argument("Lanczos requires at least one maximum iteration");
  }
  if (options.min_iterations < 1)
  {
    throw std::invalid_argument("Lanczos requires at least one minimum iteration");
  }
  if (options.min_iterations > options.max_iterations)
  {
    throw std::invalid_argument("Lanczos minimum iterations exceed maximum iterations");
  }

  std::vector<MatrixFamily> v;
  std::vector<MatrixFamily> hv;
  VectorAlgebraEngine algebra;
  std::vector<double> sub_h(
      static_cast<std::size_t>(options.max_iterations + 1) * static_cast<std::size_t>(options.max_iterations + 1), 0.0);
  auto const stride = static_cast<std::size_t>(options.max_iterations + 1);

  double beta = algebra.norm(guess);
  if (!(beta > 0.0) || std::isnan(beta))
  {
    throw std::invalid_argument("Lanczos initial guess must have finite non-zero norm");
  }

  // Keep pure Krylov vector algebra resident in the TensorContraction runtime.
  // The current matvec adapter still uses host MatrixFamily storage, so the
  // helper below is the only intended host/GPU crossing inside the iteration.
  algebra.upload(guess);
  algebra.set_host_synchronization(false);
  auto apply_matvec = [&](MatrixFamily& x, MatrixFamily& y) {
    algebra.synchronize(x);
    matvec(x, y);
    algebra.upload(y);
  };
  auto finish_with = [&](MatrixFamily& x) {
    algebra.copy(x, guess);
    algebra.synchronize(guess);
  };

  auto w = make_like(guess);
  algebra.copy(guess, w);
  algebra.scale(w, 1.0 / beta);
  v.push_back(std::move(w));

  w = make_like(guess);
  apply_matvec(v[0], w);
  hv.push_back(make_like(w));
  algebra.copy(w, hv.back());
  double alpha = algebra.dot(v[0], w);
  sub_h[detail::dense_index(stride, 0, 0)] = alpha;
  algebra.axpy(-alpha, v[0], w);

  alpha = algebra.dot(v[0], w);
  sub_h[detail::dense_index(stride, 0, 0)] += alpha;
  algebra.axpy(-alpha, v[0], w);

  beta = algebra.norm(w);
  if (beta < options.beta_tolerance)
  {
    finish_with(v[0]);
    return LanczosResult{.eigenvalue = sub_h[detail::dense_index(stride, 0, 0)],
                         .iterations = 1,
                         .tolerance = beta,
                         .residual_norm = beta,
                         .spectral_diameter = 0.0,
                         .stop_reason = LanczosStopReason::InvariantSubspace};
  }

  if (options.max_iterations == 1)
  {
    throw std::invalid_argument("Lanczos needs more than one iteration unless the initial vector already converges");
  }

  for (int i = 1; i < options.max_iterations; ++i)
  {
    auto const idx = static_cast<std::size_t>(i);
    sub_h[detail::dense_index(stride, idx, idx - 1)] = beta;
    sub_h[detail::dense_index(stride, idx - 1, idx)] = beta;

    algebra.scale(w, 1.0 / beta);
    v.push_back(std::move(w));

    w = make_like(guess);
    apply_matvec(v[idx], w);
    hv.push_back(make_like(w));
    algebra.copy(w, hv.back());
    algebra.axpy(-beta, v[idx - 1], w);
    algebra.axpy(-algebra.dot(v[idx - 1], w), v[idx - 1], w);

    alpha = algebra.dot(v[idx], w);
    sub_h[detail::dense_index(stride, idx, idx)] = alpha;
    algebra.axpy(-alpha, v[idx], w);
    alpha = algebra.dot(v[idx], w);
    sub_h[detail::dense_index(stride, idx, idx)] += alpha;
    algebra.axpy(-alpha, v[idx], w);

    beta = algebra.norm(w);
    if (beta < options.beta_tolerance)
    {
      auto eig = detail::lowest_eigenpair(detail::submatrix(sub_h, stride, idx + 1), idx + 1);
      auto y = detail::linear_combination(algebra, v, eig.eigenvector);
      finish_with(y);
      return LanczosResult{.eigenvalue = eig.eigenvalue,
                           .iterations = i + 1,
                           .tolerance = detail::scaled_tolerance(beta, eig.spectral_diameter),
                           .residual_norm = beta,
                           .spectral_diameter = eig.spectral_diameter,
                           .stop_reason = LanczosStopReason::InvariantSubspace};
    }

    double const overlap = std::abs(algebra.dot(v[0], v[idx]));
    if (overlap > options.orthogonality_tolerance)
    {
      auto eig = detail::lowest_eigenpair(detail::submatrix(sub_h, stride, idx), idx);
      auto const coefficients = detail::coefficients_for(eig.eigenvector, idx);
      auto active_v = std::vector<MatrixFamily>{};
      active_v.reserve(idx);
      for (std::size_t j = 0; j < idx; ++j)
      {
        active_v.push_back(make_like(v[j]));
        algebra.copy(v[j], active_v.back());
      }
      auto y = detail::linear_combination(algebra, active_v, coefficients);
      auto r = make_like(y);
      algebra.copy(y, r);
      algebra.scale(r, -eig.eigenvalue);
      for (std::size_t j = 0; j < idx; ++j)
      {
        algebra.axpy(coefficients[j], hv[j], r);
      }
      double const residual_norm = algebra.norm(r);
      finish_with(y);
      return LanczosResult{.eigenvalue = eig.eigenvalue,
                           .iterations = i + 1,
                           .tolerance = detail::scaled_tolerance(residual_norm, eig.spectral_diameter),
                           .residual_norm = residual_norm,
                           .spectral_diameter = eig.spectral_diameter,
                           .stop_reason = LanczosStopReason::LossOfOrthogonality};
    }

    auto eig = detail::lowest_eigenpair(detail::submatrix(sub_h, stride, idx + 1), idx + 1);
    auto y = detail::linear_combination(algebra, v, eig.eigenvector);
    auto r = make_like(y);
    algebra.copy(y, r);
    algebra.scale(r, -eig.eigenvalue);
    for (std::size_t j = 0; j <= idx; ++j)
    {
      algebra.axpy(eig.eigenvector[j], hv[j], r);
    }
    double const residual_norm = algebra.norm(r);
    double const denominator = eig.spectral_diameter == 0.0 ? 1.0 : eig.spectral_diameter;

    if (residual_norm < std::abs(options.tolerance * eig.spectral_diameter) && i + 1 >= options.min_iterations)
    {
      finish_with(y);
      return LanczosResult{.eigenvalue = eig.eigenvalue,
                           .iterations = i + 1,
                           .tolerance = residual_norm / denominator,
                           .residual_norm = residual_norm,
                           .spectral_diameter = eig.spectral_diameter,
                           .stop_reason = LanczosStopReason::Converged};
    }

    if (i == options.max_iterations - 1)
    {
      finish_with(y);
      return LanczosResult{.eigenvalue = eig.eigenvalue,
                           .iterations = i + 1,
                           .tolerance = -residual_norm / denominator,
                           .residual_norm = residual_norm,
                           .spectral_diameter = eig.spectral_diameter,
                           .stop_reason = LanczosStopReason::MaxIterations};
    }
  }

  throw std::logic_error("Lanczos reached unreachable control path");
}

} // namespace uni20::tensorcontraction
