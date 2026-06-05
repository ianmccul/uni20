#pragma once

#include <uni20/tensorcontraction/matrix_family.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
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

/// \brief Wall-clock and process-CPU timing for one profiled Lanczos substage.
struct LanczosStageTiming
{
    double wall_seconds = 0.0;
    double cpu_seconds = 0.0;
};

/// \brief Fine-grained timing data for one Lanczos solve.
struct LanczosTimings
{
    LanczosStageTiming workspace;
    LanczosStageTiming basis_setup;
    LanczosStageTiming matvec;
    LanczosStageTiming store_hamiltonian_vector;
    LanczosStageTiming orthogonalization;
    LanczosStageTiming reductions;
    LanczosStageTiming ritz_diagonalization;
    LanczosStageTiming ritz_vector;
    LanczosStageTiming residual_vector;
    LanczosStageTiming residual_norm;
    LanczosStageTiming finish;
    int matvec_count = 0;
    int ritz_diagonalization_count = 0;
    int ritz_vector_count = 0;
    int residual_vector_count = 0;
};

struct LanczosResult
{
    double eigenvalue = 0.0;
    int iterations = 0;
    double tolerance = 0.0;
    double residual_norm = 0.0;
    double spectral_diameter = 0.0;
    LanczosStopReason stop_reason = LanczosStopReason::MaxIterations;
    LanczosTimings timings;

    [[nodiscard]] bool converged() const noexcept { return stop_reason != LanczosStopReason::MaxIterations; }
};

namespace detail
{

/// \brief Wall-clock and process-CPU checkpoint for Lanczos profiling.
struct LanczosProfileCheckpoint
{
    std::chrono::steady_clock::time_point wall_time;
    double process_cpu_seconds = 0.0;
};

/// \brief Return process CPU seconds consumed by the current process.
/// \return CPU seconds accumulated by this process.
inline auto lanczos_profile_cpu_seconds() -> double
{
  return static_cast<double>(std::clock()) / static_cast<double>(CLOCKS_PER_SEC);
}

/// \brief Capture a Lanczos profile timing checkpoint.
/// \return Fine-grained Lanczos profiling checkpoint.
inline auto lanczos_profile_checkpoint() -> LanczosProfileCheckpoint
{
  return LanczosProfileCheckpoint{.wall_time = std::chrono::steady_clock::now(),
                                  .process_cpu_seconds = lanczos_profile_cpu_seconds()};
}

/// \brief Accumulate elapsed profile timing between checkpoints.
/// \param timing Timing accumulator to update.
/// \param start Initial checkpoint.
/// \param stop Final checkpoint.
inline void accumulate_lanczos_timing(LanczosStageTiming& timing, LanczosProfileCheckpoint const& start,
                                      LanczosProfileCheckpoint const& stop)
{
  timing.wall_seconds += std::chrono::duration<double>(stop.wall_time - start.wall_time).count();
  timing.cpu_seconds += stop.process_cpu_seconds - start.process_cpu_seconds;
}

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

struct MatrixFamilyAlgebra
{
    [[nodiscard]] double dot(MatrixFamily const& lhs, MatrixFamily const& rhs)
    {
      return tensorcontraction::dot(lhs, rhs);
    }

    [[nodiscard]] double norm2(MatrixFamily const& x) { return tensorcontraction::norm2(x); }

    [[nodiscard]] double norm(MatrixFamily const& x) { return tensorcontraction::norm(x); }

    void zero(MatrixFamily& x) { tensorcontraction::zero(x); }

    void copy(MatrixFamily const& source, MatrixFamily& target) { tensorcontraction::copy(source, target); }

    void scale(MatrixFamily& x, double alpha) { tensorcontraction::scale(x, alpha); }

    void axpy(double alpha, MatrixFamily const& x, MatrixFamily& y) { tensorcontraction::axpy(alpha, x, y); }
};

template <typename Algebra>
inline MatrixFamily linear_combination(Algebra& algebra, std::vector<MatrixFamily> const& vectors,
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

template <typename MatVec> void apply_matrix_family_matvec(MatVec& matvec, MatrixFamily const& x, MatrixFamily& y)
{
  if constexpr (requires { matvec.apply(x, y); })
  {
    matvec.apply(x, y);
  }
  else
  {
    matvec(x, y);
  }
}

} // namespace detail

template <typename MatVec, typename Algebra>
LanczosResult lanczos_lowest_with_engine(MatrixFamily& guess, MatVec&& matvec, Algebra& algebra,
                                         LanczosOptions options = {})
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
  std::vector<double> sub_h(
      static_cast<std::size_t>(options.max_iterations + 1) * static_cast<std::size_t>(options.max_iterations + 1), 0.0);
  auto const stride = static_cast<std::size_t>(options.max_iterations + 1);
  LanczosTimings timings;

  auto stage_start = detail::lanczos_profile_checkpoint();
  double beta = algebra.norm(guess);
  auto stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.reductions, stage_start, stage_stop);
  if (!(beta > 0.0) || std::isnan(beta))
  {
    throw std::invalid_argument("Lanczos initial guess must have finite non-zero norm");
  }

  auto make_result = [&](double eigenvalue, int iterations, double tolerance, double residual_norm,
                         double spectral_diameter, LanczosStopReason stop_reason) {
    return LanczosResult{.eigenvalue = eigenvalue,
                         .iterations = iterations,
                         .tolerance = tolerance,
                         .residual_norm = residual_norm,
                         .spectral_diameter = spectral_diameter,
                         .stop_reason = stop_reason,
                         .timings = timings};
  };

  auto finish_with = [&](MatrixFamily& x) {
    auto const start = detail::lanczos_profile_checkpoint();
    algebra.copy(x, guess);
    auto const stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.finish, start, stop);
  };

  auto solve_ritz_problem = [&](std::size_t n) {
    auto const start = detail::lanczos_profile_checkpoint();
    auto eig = detail::lowest_eigenpair(detail::submatrix(sub_h, stride, n), n);
    auto const stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.ritz_diagonalization, start, stop);
    ++timings.ritz_diagonalization_count;
    return eig;
  };

  auto make_ritz_vector = [&](std::vector<MatrixFamily> const& vectors, std::vector<double> const& coefficients) {
    auto const start = detail::lanczos_profile_checkpoint();
    auto y = detail::linear_combination(algebra, vectors, coefficients);
    auto const stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.ritz_vector, start, stop);
    ++timings.ritz_vector_count;
    return y;
  };

  auto make_residual_vector = [&](MatrixFamily const& y, double theta, std::vector<double> const& coefficients,
                                  std::size_t coefficient_count) {
    auto const start = detail::lanczos_profile_checkpoint();
    auto r = make_like(y);
    algebra.copy(y, r);
    algebra.scale(r, -theta);
    for (std::size_t j = 0; j < coefficient_count; ++j)
    {
      algebra.axpy(coefficients[j], hv[j], r);
    }
    auto const stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.residual_vector, start, stop);
    ++timings.residual_vector_count;
    return r;
  };

  auto compute_residual_norm = [&](MatrixFamily const& r) {
    auto const start = detail::lanczos_profile_checkpoint();
    double const residual_norm = algebra.norm(r);
    auto const stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.residual_norm, start, stop);
    return residual_norm;
  };

  stage_start = detail::lanczos_profile_checkpoint();
  auto w = make_like(guess);
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.workspace, stage_start, stage_stop);

  stage_start = stage_stop;
  algebra.copy(guess, w);
  algebra.scale(w, 1.0 / beta);
  v.push_back(std::move(w));
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.basis_setup, stage_start, stage_stop);

  stage_start = stage_stop;
  w = make_like(guess);
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.workspace, stage_start, stage_stop);

  stage_start = stage_stop;
  matvec(v[0], w);
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.matvec, stage_start, stage_stop);
  ++timings.matvec_count;

  stage_start = stage_stop;
  hv.push_back(make_like(w));
  algebra.copy(w, hv.back());
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.store_hamiltonian_vector, stage_start, stage_stop);

  stage_start = stage_stop;
  double alpha = algebra.dot(v[0], w);
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.reductions, stage_start, stage_stop);
  sub_h[detail::dense_index(stride, 0, 0)] = alpha;

  stage_start = stage_stop;
  algebra.axpy(-alpha, v[0], w);
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.orthogonalization, stage_start, stage_stop);

  stage_start = stage_stop;
  alpha = algebra.dot(v[0], w);
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.reductions, stage_start, stage_stop);
  sub_h[detail::dense_index(stride, 0, 0)] += alpha;

  stage_start = stage_stop;
  algebra.axpy(-alpha, v[0], w);
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.orthogonalization, stage_start, stage_stop);

  stage_start = stage_stop;
  beta = algebra.norm(w);
  stage_stop = detail::lanczos_profile_checkpoint();
  detail::accumulate_lanczos_timing(timings.reductions, stage_start, stage_stop);
  if (beta < options.beta_tolerance)
  {
    finish_with(v[0]);
    return make_result(sub_h[detail::dense_index(stride, 0, 0)], 1, beta, beta, 0.0,
                       LanczosStopReason::InvariantSubspace);
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

    stage_start = detail::lanczos_profile_checkpoint();
    algebra.scale(w, 1.0 / beta);
    v.push_back(std::move(w));
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.basis_setup, stage_start, stage_stop);

    stage_start = stage_stop;
    w = make_like(guess);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.workspace, stage_start, stage_stop);

    stage_start = stage_stop;
    matvec(v[idx], w);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.matvec, stage_start, stage_stop);
    ++timings.matvec_count;

    stage_start = stage_stop;
    hv.push_back(make_like(w));
    algebra.copy(w, hv.back());
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.store_hamiltonian_vector, stage_start, stage_stop);

    stage_start = stage_stop;
    algebra.axpy(-beta, v[idx - 1], w);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.orthogonalization, stage_start, stage_stop);

    stage_start = stage_stop;
    double previous_overlap = algebra.dot(v[idx - 1], w);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.reductions, stage_start, stage_stop);

    stage_start = stage_stop;
    algebra.axpy(-previous_overlap, v[idx - 1], w);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.orthogonalization, stage_start, stage_stop);

    stage_start = stage_stop;
    alpha = algebra.dot(v[idx], w);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.reductions, stage_start, stage_stop);
    sub_h[detail::dense_index(stride, idx, idx)] = alpha;

    stage_start = stage_stop;
    algebra.axpy(-alpha, v[idx], w);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.orthogonalization, stage_start, stage_stop);

    stage_start = stage_stop;
    alpha = algebra.dot(v[idx], w);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.reductions, stage_start, stage_stop);
    sub_h[detail::dense_index(stride, idx, idx)] += alpha;

    stage_start = stage_stop;
    algebra.axpy(-alpha, v[idx], w);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.orthogonalization, stage_start, stage_stop);

    stage_start = stage_stop;
    beta = algebra.norm(w);
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.reductions, stage_start, stage_stop);
    if (beta < options.beta_tolerance)
    {
      auto eig = solve_ritz_problem(idx + 1);
      auto y = make_ritz_vector(v, eig.eigenvector);
      finish_with(y);
      return make_result(eig.eigenvalue, i + 1, detail::scaled_tolerance(beta, eig.spectral_diameter), beta,
                         eig.spectral_diameter, LanczosStopReason::InvariantSubspace);
    }

    stage_start = stage_stop;
    double const overlap = std::abs(algebra.dot(v[0], v[idx]));
    stage_stop = detail::lanczos_profile_checkpoint();
    detail::accumulate_lanczos_timing(timings.reductions, stage_start, stage_stop);
    if (overlap > options.orthogonality_tolerance)
    {
      auto eig = solve_ritz_problem(idx);
      auto const coefficients = detail::coefficients_for(eig.eigenvector, idx);
      stage_start = detail::lanczos_profile_checkpoint();
      auto active_v = std::vector<MatrixFamily>{};
      active_v.reserve(idx);
      for (std::size_t j = 0; j < idx; ++j)
      {
        active_v.push_back(make_like(v[j]));
        algebra.copy(v[j], active_v.back());
      }
      stage_stop = detail::lanczos_profile_checkpoint();
      detail::accumulate_lanczos_timing(timings.workspace, stage_start, stage_stop);

      auto y = make_ritz_vector(active_v, coefficients);
      auto r = make_residual_vector(y, eig.eigenvalue, coefficients, idx);
      double const residual_norm = compute_residual_norm(r);
      finish_with(y);
      return make_result(eig.eigenvalue, i + 1, detail::scaled_tolerance(residual_norm, eig.spectral_diameter),
                         residual_norm, eig.spectral_diameter, LanczosStopReason::LossOfOrthogonality);
    }

    auto eig = solve_ritz_problem(idx + 1);
    auto y = make_ritz_vector(v, eig.eigenvector);
    auto r = make_residual_vector(y, eig.eigenvalue, eig.eigenvector, idx + 1);
    double const residual_norm = compute_residual_norm(r);
    double const denominator = eig.spectral_diameter == 0.0 ? 1.0 : eig.spectral_diameter;

    if (residual_norm < std::abs(options.tolerance * eig.spectral_diameter) && i + 1 >= options.min_iterations)
    {
      finish_with(y);
      return make_result(eig.eigenvalue, i + 1, residual_norm / denominator, residual_norm, eig.spectral_diameter,
                         LanczosStopReason::Converged);
    }

    if (i == options.max_iterations - 1)
    {
      finish_with(y);
      return make_result(eig.eigenvalue, i + 1, -residual_norm / denominator, residual_norm, eig.spectral_diameter,
                         LanczosStopReason::MaxIterations);
    }
  }

  throw std::logic_error("Lanczos reached unreachable control path");
}

template <typename MatVec>
LanczosResult lanczos_lowest(MatrixFamily& guess, MatVec&& matvec, LanczosOptions options = {})
{
  detail::MatrixFamilyAlgebra algebra;
  auto apply_matvec = [&](MatrixFamily const& x, MatrixFamily& y) { detail::apply_matrix_family_matvec(matvec, x, y); };
  return lanczos_lowest_with_engine(guess, apply_matvec, algebra, options);
}

} // namespace uni20::tensorcontraction
