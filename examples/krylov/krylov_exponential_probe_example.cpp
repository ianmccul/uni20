#include <uni20/common/presentation.hpp>
#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/krylov_exponential.hpp>
#include <uni20/krylov/taylor_exponential.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fmt/core.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{

namespace presentation = uni20::presentation;

using uni20::krylov::DenseHostVector;
using uni20::krylov::DenseHostVectorOps;
using uni20::krylov::KrylovDiagnosticsLevel;
using uni20::krylov::TaylorExponentialParams;

enum class ScalarPrecision
{
  Float32,
  Float64
};

struct Options
{
    ScalarPrecision precision = ScalarPrecision::Float64;
    double tolerance = 0.0;
    bool tolerance_was_set = false;
    bool check = false;
};

template <typename Scalar> using RealOf = uni20::make_real_t<Scalar>;

[[nodiscard]] bool starts_with(std::string_view value, std::string_view prefix)
{
  return value.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] std::string_view option_value(std::string_view argument, std::string_view prefix)
{
  if (!starts_with(argument, prefix))
  {
    throw std::invalid_argument(fmt::format("expected option prefix '{}'", prefix));
  }
  return argument.substr(prefix.size());
}

[[nodiscard]] double parse_double(std::string_view text, std::string_view option)
{
  try
  {
    return std::stod(std::string(text));
  }
  catch (std::exception const&)
  {
    throw std::invalid_argument(fmt::format("invalid numeric value for {}: {}", option, text));
  }
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view const argument = argv[i];
    if (argument == "--check")
    {
      options.check = true;
    }
    else if (argument == "--precision=float")
    {
      options.precision = ScalarPrecision::Float32;
    }
    else if (argument == "--precision=double")
    {
      options.precision = ScalarPrecision::Float64;
    }
    else if (starts_with(argument, "--tol="))
    {
      options.tolerance = parse_double(option_value(argument, "--tol="), "--tol");
      options.tolerance_was_set = true;
    }
    else if (argument == "--help")
    {
      fmt::print("Usage: krylov_exponential_probe_example [--precision=float|double] [--tol=value] [--check]\n");
      std::exit(0);
    }
    else
    {
      throw std::invalid_argument(fmt::format("unrecognized option: {}", argument));
    }
  }
  return options;
}

template <typename Real> [[nodiscard]] Real default_tolerance()
{
  if constexpr (std::is_same_v<Real, float>)
  {
    return Real{1.0e-4};
  }
  else
  {
    return Real{1.0e-8};
  }
}

template <typename Real> [[nodiscard]] std::string format_real(Real value)
{
  return fmt::format("{:.3e}", static_cast<double>(value));
}

template <typename TimeScalar> [[nodiscard]] std::string format_time(TimeScalar value)
{
  if constexpr (uni20::Complex<TimeScalar>)
  {
    return fmt::format("{:.3e}{:+.3e}i", static_cast<double>(value.real()), static_cast<double>(value.imag()));
  }
  else
  {
    return format_real(value);
  }
}

template <typename Scalar> [[nodiscard]] RealOf<Scalar> abs_squared(Scalar value)
{
  using Real = RealOf<Scalar>;
  if constexpr (uni20::Complex<Scalar>)
  {
    return static_cast<Real>(std::norm(value));
  }
  else
  {
    return value * value;
  }
}

template <typename Scalar> [[nodiscard]] RealOf<Scalar> vector_norm(DenseHostVector<Scalar> const& vector)
{
  using Real = RealOf<Scalar>;
  Real norm_squared{};
  for (Scalar const value : vector.values)
  {
    norm_squared += abs_squared(value);
  }
  return norm_squared > Real{} ? std::sqrt(norm_squared) : Real{};
}

template <typename Scalar>
[[nodiscard]] RealOf<Scalar> difference_norm(DenseHostVector<Scalar> const& lhs, DenseHostVector<Scalar> const& rhs)
{
  using Real = RealOf<Scalar>;
  if (lhs.values.size() != rhs.values.size())
  {
    throw std::invalid_argument("difference_norm requires equal vector sizes");
  }

  Real norm_squared{};
  for (std::size_t i = 0; i < lhs.values.size(); ++i)
  {
    norm_squared += abs_squared(lhs.values[i] - rhs.values[i]);
  }
  return norm_squared > Real{} ? std::sqrt(norm_squared) : Real{};
}

template <typename Scalar> [[nodiscard]] std::vector<Scalar> diagonal_matrix(std::vector<RealOf<Scalar>> const& values)
{
  std::vector<Scalar> matrix(values.size() * values.size(), Scalar{});
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    matrix[i * values.size() + i] = static_cast<Scalar>(values[i]);
  }
  return matrix;
}

template <typename Real> [[nodiscard]] std::vector<Real> arithmetic_spectrum(std::size_t n, Real first, Real last)
{
  std::vector<Real> values(n);
  if (n == 1)
  {
    values[0] = first;
    return values;
  }

  for (std::size_t i = 0; i < n; ++i)
  {
    Real const fraction = static_cast<Real>(i) / static_cast<Real>(n - 1);
    values[i] = first + fraction * (last - first);
  }
  return values;
}

template <typename Real> [[nodiscard]] std::vector<Real> clustered_spectrum(std::size_t n)
{
  std::vector<Real> values(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    if (i < n / 4)
    {
      values[i] = Real{0.02} * static_cast<Real>(i);
    }
    else
    {
      Real const fraction = static_cast<Real>(i - n / 4) / static_cast<Real>(n - n / 4 - 1);
      values[i] = Real{15} + fraction * Real{65};
    }
  }
  return values;
}

template <typename Scalar>
[[nodiscard]] DenseHostVector<Scalar> deterministic_initial(std::size_t n, RealOf<Scalar> scale)
{
  using Real = RealOf<Scalar>;
  DenseHostVector<Scalar> result{{}};
  result.values.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    Real const index = static_cast<Real>(i + 1);
    Real const real_part = std::sin(Real{0.37} * index) + Real{0.25} * std::cos(Real{0.11} * index * index);
    if constexpr (uni20::Complex<Scalar>)
    {
      Real const imag_part = Real{0.35} * std::cos(Real{0.23} * index + Real{0.01} * index * index);
      result.values.push_back(scale * Scalar{real_part, imag_part});
    }
    else
    {
      result.values.push_back(scale * real_part);
    }
  }
  return result;
}

template <typename Scalar, typename TimeScalar>
[[nodiscard]] DenseHostVector<Scalar> exact_diagonal_action(std::vector<RealOf<Scalar>> const& eigenvalues,
                                                            DenseHostVector<Scalar> const& initial, TimeScalar time)
{
  DenseHostVector<Scalar> result{{}};
  result.values.reserve(initial.values.size());
  Scalar const coefficient = static_cast<Scalar>(time);
  for (std::size_t i = 0; i < initial.values.size(); ++i)
  {
    result.values.push_back(std::exp(coefficient * static_cast<Scalar>(eigenvalues[i])) * initial.values[i]);
  }
  return result;
}

template <typename Real> [[nodiscard]] Real spectral_norm_bound(std::vector<Real> const& eigenvalues)
{
  Real bound{};
  for (Real const value : eigenvalues)
  {
    bound = std::max(bound, std::abs(value));
  }
  return bound;
}

template <typename Scalar>
[[nodiscard]] Scalar host_inner_product(DenseHostVector<Scalar> const& lhs, DenseHostVector<Scalar> const& rhs)
{
  if (lhs.values.size() != rhs.values.size())
  {
    throw std::invalid_argument("host_inner_product requires equal vector sizes");
  }

  Scalar result{};
  for (std::size_t i = 0; i < lhs.values.size(); ++i)
  {
    if constexpr (uni20::Complex<Scalar>)
    {
      result += std::conj(lhs.values[i]) * rhs.values[i];
    }
    else
    {
      result += lhs.values[i] * rhs.values[i];
    }
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] DenseHostVector<Scalar> diagonal_matvec(std::vector<RealOf<Scalar>> const& eigenvalues,
                                                      DenseHostVector<Scalar> const& vector)
{
  if (eigenvalues.size() != vector.values.size())
  {
    throw std::invalid_argument("diagonal_matvec requires matching dimensions");
  }

  DenseHostVector<Scalar> result{{}};
  result.values.resize(vector.values.size());
  for (std::size_t i = 0; i < vector.values.size(); ++i)
  {
    result.values[i] = static_cast<Scalar>(eigenvalues[i]) * vector.values[i];
  }
  return result;
}

template <typename Scalar> void host_axpy(DenseHostVector<Scalar>& y, Scalar alpha, DenseHostVector<Scalar> const& x)
{
  if (y.values.size() != x.values.size())
  {
    throw std::invalid_argument("host_axpy requires equal vector sizes");
  }

  for (std::size_t i = 0; i < y.values.size(); ++i)
  {
    y.values[i] += alpha * x.values[i];
  }
}

template <typename Scalar> void host_scale(DenseHostVector<Scalar>& vector, Scalar alpha)
{
  for (auto& value : vector.values)
  {
    value *= alpha;
  }
}

template <typename Scalar> void host_set_zero(DenseHostVector<Scalar>& vector)
{
  for (auto& value : vector.values)
  {
    value = Scalar{};
  }
}

template <typename Real> struct ProbeReorthogonalizationStats
{
    Real max_correction = Real{};
    Real max_correction_ratio = Real{};
    int max_passes = 0;
};

template <typename Real>
void record_probe_reorthogonalization_pass(ProbeReorthogonalizationStats<Real>* stats, Real max_correction,
                                           Real reference_norm, int pass_count)
{
  if (stats == nullptr)
  {
    return;
  }

  stats->max_correction = std::max(stats->max_correction, max_correction);
  if (reference_norm > Real{})
  {
    stats->max_correction_ratio = std::max(stats->max_correction_ratio, max_correction / reference_norm);
  }
  stats->max_passes = std::max(stats->max_passes, pass_count);
}

template <typename Real> struct ProbeBasisOrthogonality
{
    Real max_diag_error = Real{};
    Real max_offdiag = Real{};
    Real frobenius_error = Real{};
};

template <typename Scalar>
[[nodiscard]] ProbeBasisOrthogonality<RealOf<Scalar>>
probe_basis_orthogonality(std::vector<DenseHostVector<Scalar>> const& basis)
{
  using Real = RealOf<Scalar>;
  ProbeBasisOrthogonality<Real> result;
  long double frobenius_squared = 0.0L;
  for (std::size_t row = 0; row < basis.size(); ++row)
  {
    for (std::size_t col = row; col < basis.size(); ++col)
    {
      Scalar const entry = host_inner_product(basis[row], basis[col]);
      if (row == col)
      {
        Real const error = static_cast<Real>(std::abs(entry - Scalar{1}));
        result.max_diag_error = std::max(result.max_diag_error, error);
        frobenius_squared += static_cast<long double>(error) * static_cast<long double>(error);
      }
      else
      {
        Real const magnitude = static_cast<Real>(std::abs(entry));
        result.max_offdiag = std::max(result.max_offdiag, magnitude);
        frobenius_squared += 2.0L * static_cast<long double>(magnitude) * static_cast<long double>(magnitude);
      }
    }
  }
  result.frobenius_error = static_cast<Real>(std::sqrt(frobenius_squared));
  return result;
}

template <typename Scalar>
[[nodiscard]] RealOf<Scalar> orthogonalize_probe_residual(std::vector<DenseHostVector<Scalar>> const& basis,
                                                          DenseHostVector<Scalar>& residual,
                                                          ProbeReorthogonalizationStats<RealOf<Scalar>>* stats)
{
  using Real = RealOf<Scalar>;
  auto norm = [&]() { return vector_norm(residual); };
  auto orthogonalize_once = [&]() {
    Real max_correction{};
    for (auto const& basis_vector : basis)
    {
      Scalar const correction = host_inner_product(basis_vector, residual);
      max_correction = std::max(max_correction, static_cast<Real>(std::abs(correction)));
      host_axpy(residual, -correction, basis_vector);
    }
    return max_correction;
  };

  Real const original_norm = norm();
  Real const first_max_correction = orthogonalize_once();
  record_probe_reorthogonalization_pass(stats, first_max_correction, original_norm, 1);
  Real residual_norm = norm();
  if (original_norm == Real{} || residual_norm > Real{0.717} * original_norm)
  {
    return residual_norm;
  }

  Real const first_refined_norm = residual_norm;
  Real const second_max_correction = orthogonalize_once();
  record_probe_reorthogonalization_pass(stats, second_max_correction, first_refined_norm, 2);
  residual_norm = norm();
  if (first_refined_norm == Real{} || residual_norm <= Real{0.717} * first_refined_norm)
  {
    host_set_zero(residual);
    return Real{};
  }
  return residual_norm;
}

template <typename Scalar> [[nodiscard]] std::vector<Scalar> phi1_first_column(uni20::krylov::Matrix<Scalar> const& z)
{
  std::size_t const n = z.rows();
  if (z.cols() != n)
  {
    throw std::invalid_argument("phi1_first_column requires a square matrix");
  }

  uni20::krylov::Matrix<Scalar> augmented(n + 1, n + 1);
  for (std::size_t row = 0; row < n; ++row)
  {
    for (std::size_t col = 0; col < n; ++col)
    {
      augmented(row, col) = z(row, col);
    }
  }
  augmented(0, n) = Scalar{1};

  auto const exponential = uni20::krylov::matrix_exponential(augmented, uni20::make_real_t<Scalar>{1});
  std::vector<Scalar> column(n);
  for (std::size_t row = 0; row < n; ++row)
  {
    column[row] = exponential(row, n);
  }
  return column;
}

enum class LanczosProbeVariant
{
  FullReorthogonalized,
  LegacyThreeTerm
};

[[nodiscard]] std::string_view variant_name(LanczosProbeVariant variant)
{
  switch (variant)
  {
    case LanczosProbeVariant::FullReorthogonalized:
      return "full reorthogonalized Lanczos";
    case LanczosProbeVariant::LegacyThreeTerm:
      return "legacy three-term Lanczos";
  }
  return "unknown";
}

template <typename Scalar> struct LocalExponentialAction
{
    DenseHostVector<Scalar> action;
    RealOf<Scalar> residual_estimate = RealOf<Scalar>{};
    RealOf<Scalar> final_residual_norm = RealOf<Scalar>{};
    RealOf<Scalar> tail_coefficient = RealOf<Scalar>{};
    RealOf<Scalar> hermite_quadrature_estimate = RealOf<Scalar>{};
    RealOf<Scalar> saad_phi1_estimate = RealOf<Scalar>{};
    RealOf<Scalar> hochbruck_lubich_bound = RealOf<Scalar>{};
    RealOf<Scalar> basis_max_diag_error = RealOf<Scalar>{};
    RealOf<Scalar> basis_max_offdiag = RealOf<Scalar>{};
    RealOf<Scalar> basis_frobenius_error = RealOf<Scalar>{};
    RealOf<Scalar> max_reorthogonalization_correction_ratio = RealOf<Scalar>{};
    int max_reorthogonalization_passes = 0;
    int projected_dimension = 0;
    int matvec_count = 0;
};

template <typename Scalar, typename TimeScalar>
[[nodiscard]] LocalExponentialAction<Scalar>
local_lanczos_exponential_action(std::vector<RealOf<Scalar>> const& eigenvalues, DenseHostVector<Scalar> const& initial,
                                 TimeScalar time, int krylov_dimension, LanczosProbeVariant variant)
{
  using Real = RealOf<Scalar>;
  using ProjectedScalar = std::conditional_t<uni20::Complex<Scalar>, Scalar, Real>;

  if (krylov_dimension <= 0)
  {
    throw std::invalid_argument("legacy three-term probe requires a positive Krylov dimension");
  }
  if (static_cast<std::size_t>(krylov_dimension) > eigenvalues.size())
  {
    throw std::invalid_argument("legacy three-term probe requires krylov_dimension <= problem dimension");
  }

  Real const initial_norm = vector_norm(initial);
  DenseHostVector<Scalar> zero{{std::vector<Scalar>(initial.values.size(), Scalar{})}};
  if (initial_norm == Real{})
  {
    return LocalExponentialAction<Scalar>{.action = std::move(zero)};
  }

  DenseHostVector<Scalar> current = initial;
  host_scale(current, static_cast<Scalar>(Real{1} / initial_norm));
  DenseHostVector<Scalar> previous{{std::vector<Scalar>(initial.values.size(), Scalar{})}};

  std::vector<DenseHostVector<Scalar>> basis;
  std::vector<Real> diagonal;
  std::vector<Real> offdiagonal;
  basis.reserve(static_cast<std::size_t>(krylov_dimension));
  diagonal.reserve(static_cast<std::size_t>(krylov_dimension));
  offdiagonal.reserve(static_cast<std::size_t>(krylov_dimension));

  Real residual_norm{};
  ProbeReorthogonalizationStats<Real> reorthogonalization_stats;
  int matvec_count = 0;
  for (int step = 0; step < krylov_dimension; ++step)
  {
    basis.push_back(current);
    auto residual = diagonal_matvec(eigenvalues, current);
    ++matvec_count;

    Real const alpha = static_cast<Real>(std::real(host_inner_product(current, residual)));
    diagonal.push_back(alpha);
    host_axpy(residual, static_cast<Scalar>(-alpha), current);
    if (step > 0)
    {
      host_axpy(residual, static_cast<Scalar>(-offdiagonal.back()), previous);
    }

    residual_norm = variant == LanczosProbeVariant::FullReorthogonalized
                        ? orthogonalize_probe_residual(basis, residual, &reorthogonalization_stats)
                        : vector_norm(residual);
    offdiagonal.push_back(residual_norm);
    if (residual_norm == Real{})
    {
      break;
    }

    previous = current;
    current = std::move(residual);
    host_scale(current, static_cast<Scalar>(Real{1} / residual_norm));
  }

  std::size_t const projected_size = diagonal.size();
  uni20::krylov::Matrix<Real> projected(projected_size, projected_size);
  for (std::size_t i = 0; i < projected_size; ++i)
  {
    projected(i, i) = diagonal[i];
    if (i + 1 < projected_size)
    {
      projected(i, i + 1) = offdiagonal[i];
      projected(i + 1, i) = offdiagonal[i];
    }
  }

  uni20::krylov::Matrix<ProjectedScalar> scaled(projected_size, projected_size);
  ProjectedScalar const coefficient = static_cast<ProjectedScalar>(time);
  for (std::size_t row = 0; row < projected_size; ++row)
  {
    for (std::size_t col = 0; col < projected_size; ++col)
    {
      scaled(row, col) = coefficient * static_cast<ProjectedScalar>(projected(row, col));
    }
  }

  auto const exponential = uni20::krylov::matrix_exponential(scaled, uni20::make_real_t<ProjectedScalar>{1});
  auto const phi1_column = phi1_first_column(scaled);
  auto const orthogonality = probe_basis_orthogonality(basis);

  DenseHostVector<Scalar> action{{std::vector<Scalar>(initial.values.size(), Scalar{})}};
  std::vector<Scalar> coefficients(projected_size);
  for (std::size_t row = 0; row < projected_size; ++row)
  {
    coefficients[row] = static_cast<Scalar>(static_cast<ProjectedScalar>(initial_norm) * exponential(row, 0));
    host_axpy(action, coefficients[row], basis[row]);
  }

  // Estimator citation keys match docs/krylov_exponential_estimators.md.
  // defect/||v||: [BotchevGrimmHochbruck2013], [JiaLv2015].
  // Hquad and HL bound: [HochbruckLubich1997], [JaweckiAuzingerKoch2020].
  // Saad phi1: [Saad1992], [JiaLv2015].
  Real const tail_coefficient =
      projected_size == 0 ? Real{} : static_cast<Real>(std::abs(exponential(projected_size - 1, 0)));
  Real const time_magnitude = static_cast<Real>(std::abs(static_cast<ProjectedScalar>(time)));
  Real const residual_estimate =
      coefficients.empty() ? Real{} : residual_norm * static_cast<Real>(std::abs(coefficients.back()));
  Real const hermite_quadrature_estimate =
      projected_size == 0 ? Real{} : residual_estimate * time_magnitude / static_cast<Real>(projected_size);
  Real const phi1_tail = phi1_column.empty() ? Real{} : static_cast<Real>(std::abs(phi1_column.back()));
  Real const saad_phi1_estimate = initial_norm * residual_norm * time_magnitude * phi1_tail;

  Real hochbruck_lubich_bound = Real{};
  if (initial_norm > Real{} && residual_norm > Real{} && time_magnitude > Real{})
  {
    long double log_bound =
        std::log(static_cast<long double>(initial_norm)) + std::log(static_cast<long double>(residual_norm)) +
        static_cast<long double>(projected_size) * std::log(static_cast<long double>(time_magnitude)) -
        std::lgammal(static_cast<long double>(projected_size) + 1.0L);
    for (std::size_t i = 0; i + 1 < projected_size; ++i)
    {
      if (offdiagonal[i] == Real{})
      {
        log_bound = -uni20::numeric_limits<long double>::infinity();
        break;
      }
      log_bound += std::log(static_cast<long double>(offdiagonal[i]));
    }

    long double const log_min = std::log(static_cast<long double>(uni20::numeric_limits<Real>::min()));
    long double const log_max = std::log(static_cast<long double>(uni20::numeric_limits<Real>::max()));
    if (log_bound > log_max)
    {
      hochbruck_lubich_bound = uni20::numeric_limits<Real>::infinity();
    }
    else if (log_bound > log_min)
    {
      hochbruck_lubich_bound = static_cast<Real>(std::exp(log_bound));
    }
  }

  return LocalExponentialAction<Scalar>{.action = std::move(action),
                                        .residual_estimate = residual_estimate,
                                        .final_residual_norm = residual_norm,
                                        .tail_coefficient = tail_coefficient,
                                        .hermite_quadrature_estimate = hermite_quadrature_estimate,
                                        .saad_phi1_estimate = saad_phi1_estimate,
                                        .hochbruck_lubich_bound = hochbruck_lubich_bound,
                                        .basis_max_diag_error = orthogonality.max_diag_error,
                                        .basis_max_offdiag = orthogonality.max_offdiag,
                                        .basis_frobenius_error = orthogonality.frobenius_error,
                                        .max_reorthogonalization_correction_ratio =
                                            reorthogonalization_stats.max_correction_ratio,
                                        .max_reorthogonalization_passes = reorthogonalization_stats.max_passes,
                                        .projected_dimension = static_cast<int>(projected_size),
                                        .matvec_count = matvec_count};
}

template <typename Real> struct ProbeRow
{
    int krylov_dimension = 0;
    int lanczos_matvecs = 0;
    Real error_exact = Real{};
    Real error_taylor = Real{};
    Real residual_estimate = Real{};
    Real tail_coefficient = Real{};
    Real hermite_quadrature_estimate = Real{};
    Real saad_phi1_estimate = Real{};
    Real hochbruck_lubich_bound = Real{};
    Real basis_max_offdiag = Real{};
    Real basis_frobenius_error = Real{};
    Real max_reorthogonalization_correction_ratio = Real{};
    int max_reorthogonalization_passes = 0;
    Real estimate_ratio = Real{};
    bool error_pass = false;
    bool estimate_pass = false;
    bool coefficient_pass = false;
};

template <typename Real> struct ProbeTrajectory
{
    int first_error_pass_m = 0;
    int first_estimate_pass_m = 0;
    int first_coefficient_pass_m = 0;
    int best_error_m = 0;
    int best_estimate_m = 0;
    int best_coefficient_m = 0;
    int max_post_best_estimate_m = 0;
    int max_post_best_coefficient_m = 0;
    int final_m = 0;
    Real best_error_scaled = uni20::numeric_limits<Real>::infinity();
    Real estimate_at_best_error_scaled = uni20::numeric_limits<Real>::infinity();
    Real best_estimate_scaled = uni20::numeric_limits<Real>::infinity();
    Real error_at_best_estimate_scaled = uni20::numeric_limits<Real>::infinity();
    Real best_coefficient = uni20::numeric_limits<Real>::infinity();
    Real max_post_best_estimate_scaled = Real{};
    Real max_post_best_coefficient = Real{};
    Real post_best_estimate_growth = Real{1};
    Real post_best_coefficient_growth = Real{1};
    Real max_estimate_rebound = Real{1};
    Real max_coefficient_rebound = Real{1};
    Real final_error_scaled = Real{};
    Real final_estimate_scaled = Real{};
    Real final_coefficient = Real{};
    Real final_error_over_estimate = Real{};
};

template <typename Real> struct ProbeReport
{
    std::string name;
    std::string implementation;
    std::string scalar;
    std::string time;
    std::size_t dimension = 0;
    Real initial_norm = Real{};
    Real reference_norm = Real{};
    Real operator_norm_bound = Real{};
    Real target_relative_tolerance = Real{};
    Real target_absolute_tolerance = Real{};
    Real taylor_error_exact = Real{};
    Real taylor_error_scaled = Real{};
    Real taylor_estimate_scaled = Real{};
    int taylor_matvecs = 0;
    int taylor_scaling_steps = 0;
    int taylor_max_degree = 0;
    ProbeTrajectory<Real> trajectory;
    std::vector<ProbeRow<Real>> rows;
};

template <typename Real> [[nodiscard]] Real safe_ratio(Real numerator, Real denominator)
{
  return denominator > uni20::numeric_limits<Real>::min() ? numerator / denominator
                                                        : uni20::numeric_limits<Real>::infinity();
}

template <typename Real> [[nodiscard]] Real growth_ratio(Real numerator, Real denominator)
{
  if (denominator > uni20::numeric_limits<Real>::min())
  {
    return numerator / denominator;
  }
  return numerator > uni20::numeric_limits<Real>::min() ? uni20::numeric_limits<Real>::infinity() : Real{1};
}

template <typename Real>
[[nodiscard]] ProbeTrajectory<Real> analyze_probe_trajectory(std::vector<ProbeRow<Real>> const& rows, Real initial_norm)
{
  ProbeTrajectory<Real> trajectory;
  if (rows.empty() || initial_norm == Real{})
  {
    return trajectory;
  }

  for (auto const& row : rows)
  {
    Real const error_scaled = row.error_exact / initial_norm;
    Real const estimate_scaled = row.residual_estimate / initial_norm;
    if (trajectory.first_error_pass_m == 0 && row.error_pass)
    {
      trajectory.first_error_pass_m = row.krylov_dimension;
    }
    if (trajectory.first_estimate_pass_m == 0 && row.estimate_pass)
    {
      trajectory.first_estimate_pass_m = row.krylov_dimension;
    }
    if (trajectory.first_coefficient_pass_m == 0 && row.coefficient_pass)
    {
      trajectory.first_coefficient_pass_m = row.krylov_dimension;
    }
    if (error_scaled < trajectory.best_error_scaled)
    {
      trajectory.best_error_scaled = error_scaled;
      trajectory.estimate_at_best_error_scaled = estimate_scaled;
      trajectory.best_error_m = row.krylov_dimension;
    }
    if (estimate_scaled < trajectory.best_estimate_scaled)
    {
      trajectory.best_estimate_scaled = estimate_scaled;
      trajectory.error_at_best_estimate_scaled = error_scaled;
      trajectory.best_estimate_m = row.krylov_dimension;
    }
    if (row.tail_coefficient < trajectory.best_coefficient)
    {
      trajectory.best_coefficient = row.tail_coefficient;
      trajectory.best_coefficient_m = row.krylov_dimension;
    }
  }

  bool found_post_best = false;
  bool found_post_best_coefficient = false;
  Real running_min_estimate = uni20::numeric_limits<Real>::infinity();
  Real running_min_coefficient = uni20::numeric_limits<Real>::infinity();
  for (auto const& row : rows)
  {
    Real const estimate_scaled = row.residual_estimate / initial_norm;
    if (running_min_estimate < uni20::numeric_limits<Real>::infinity())
    {
      trajectory.max_estimate_rebound =
          std::max(trajectory.max_estimate_rebound, growth_ratio(estimate_scaled, running_min_estimate));
    }
    if (running_min_coefficient < uni20::numeric_limits<Real>::infinity())
    {
      trajectory.max_coefficient_rebound =
          std::max(trajectory.max_coefficient_rebound, growth_ratio(row.tail_coefficient, running_min_coefficient));
    }
    running_min_estimate = std::min(running_min_estimate, estimate_scaled);
    running_min_coefficient = std::min(running_min_coefficient, row.tail_coefficient);

    if (row.krylov_dimension <= trajectory.best_estimate_m)
    {
      if (row.krylov_dimension <= trajectory.best_coefficient_m)
      {
        continue;
      }
    }

    if (row.krylov_dimension > trajectory.best_estimate_m)
    {
      if (!found_post_best || estimate_scaled > trajectory.max_post_best_estimate_scaled)
      {
        trajectory.max_post_best_estimate_scaled = estimate_scaled;
        trajectory.max_post_best_estimate_m = row.krylov_dimension;
        found_post_best = true;
      }
    }

    if (row.krylov_dimension > trajectory.best_coefficient_m &&
        (!found_post_best_coefficient || row.tail_coefficient > trajectory.max_post_best_coefficient))
    {
      trajectory.max_post_best_coefficient = row.tail_coefficient;
      trajectory.max_post_best_coefficient_m = row.krylov_dimension;
      found_post_best_coefficient = true;
    }
  }

  auto const& final_row = rows.back();
  trajectory.final_m = final_row.krylov_dimension;
  trajectory.final_error_scaled = final_row.error_exact / initial_norm;
  trajectory.final_estimate_scaled = final_row.residual_estimate / initial_norm;
  trajectory.final_coefficient = final_row.tail_coefficient;
  trajectory.final_error_over_estimate = safe_ratio(trajectory.final_error_scaled, trajectory.final_estimate_scaled);
  if (found_post_best)
  {
    trajectory.post_best_estimate_growth =
        growth_ratio(trajectory.max_post_best_estimate_scaled, trajectory.best_estimate_scaled);
  }
  if (found_post_best_coefficient)
  {
    trajectory.post_best_coefficient_growth =
        growth_ratio(trajectory.max_post_best_coefficient, trajectory.best_coefficient);
  }

  return trajectory;
}

template <typename Scalar, typename TimeScalar>
[[nodiscard]] ProbeReport<RealOf<Scalar>>
run_probe_case(std::string name, LanczosProbeVariant variant, std::string scalar,
               std::vector<RealOf<Scalar>> eigenvalues, DenseHostVector<Scalar> initial, TimeScalar time,
               std::vector<int> krylov_dimensions, RealOf<Scalar> relative_tolerance)
{
  using Real = RealOf<Scalar>;

  Real const initial_norm = vector_norm(initial);
  Real const target_absolute_tolerance = relative_tolerance * initial_norm;
  Real const norm_bound = spectral_norm_bound(eigenvalues);
  auto const matrix = diagonal_matrix<Scalar>(eigenvalues);
  auto const exact = exact_diagonal_action(eigenvalues, initial, time);

  TaylorExponentialParams<Real> taylor_params;
  taylor_params.tolerance = std::max(uni20::numeric_limits<Real>::min(), target_absolute_tolerance * Real{1.0e-3});
  taylor_params.step_norm_limit = Real{0.5};
  taylor_params.diagnostics = KrylovDiagnosticsLevel::Summary;

  DenseHostVectorOps<Scalar> taylor_ops(eigenvalues.size(), matrix);
  auto taylor = uni20::krylov::taylor_exponential_action<Scalar>(taylor_ops, initial, time, norm_bound, taylor_params);

  ProbeReport<Real> report{.name = std::move(name),
                           .implementation = std::string(variant_name(variant)),
                           .scalar = std::move(scalar),
                           .time = format_time(time),
                           .dimension = eigenvalues.size(),
                           .initial_norm = initial_norm,
                           .reference_norm = vector_norm(exact),
                           .operator_norm_bound = norm_bound,
                           .target_relative_tolerance = relative_tolerance,
                           .target_absolute_tolerance = target_absolute_tolerance,
                           .taylor_error_exact = difference_norm(taylor.action, exact),
                           .taylor_error_scaled = difference_norm(taylor.action, exact) / initial_norm,
                           .taylor_estimate_scaled = taylor.estimated_error / initial_norm,
                           .taylor_matvecs = taylor.matvec_count,
                           .taylor_scaling_steps = taylor.scaling_steps,
                           .taylor_max_degree = taylor.max_degree_used,
                           .trajectory = {},
                           .rows = {}};

  for (int const dimension : krylov_dimensions)
  {
    LocalExponentialAction<Scalar> lanczos =
        local_lanczos_exponential_action(eigenvalues, initial, time, dimension, variant);

    Real const error_exact = difference_norm(lanczos.action, exact);
    Real const error_taylor = difference_norm(lanczos.action, taylor.action);
    Real const residual_estimate = lanczos.residual_estimate;
    Real const estimate_ratio =
        residual_estimate > Real{} ? error_exact / residual_estimate : uni20::numeric_limits<Real>::infinity();

    report.rows.push_back(
        ProbeRow<Real>{.krylov_dimension = dimension,
                       .lanczos_matvecs = lanczos.matvec_count,
                       .error_exact = error_exact,
                       .error_taylor = error_taylor,
                       .residual_estimate = residual_estimate,
                       .tail_coefficient = lanczos.tail_coefficient,
                       .hermite_quadrature_estimate = lanczos.hermite_quadrature_estimate,
                       .saad_phi1_estimate = lanczos.saad_phi1_estimate,
                       .hochbruck_lubich_bound = lanczos.hochbruck_lubich_bound,
                       .basis_max_offdiag = lanczos.basis_max_offdiag,
                       .basis_frobenius_error = lanczos.basis_frobenius_error,
                       .max_reorthogonalization_correction_ratio = lanczos.max_reorthogonalization_correction_ratio,
                       .max_reorthogonalization_passes = lanczos.max_reorthogonalization_passes,
                       .estimate_ratio = estimate_ratio,
                       .error_pass = error_exact <= target_absolute_tolerance,
                       .estimate_pass = residual_estimate <= target_absolute_tolerance,
                       .coefficient_pass = lanczos.tail_coefficient <= relative_tolerance});
  }

  report.trajectory = analyze_probe_trajectory(report.rows, initial_norm);
  return report;
}

[[nodiscard]] std::string format_dimension_or_dash(int value) { return value > 0 ? fmt::format("{}", value) : "-"; }

template <typename Real>
void append_probe_report(presentation::styled_text& text, presentation::output_policy const& policy,
                         ProbeReport<Real> const& probe)
{
  presentation::report_builder report(probe.name);
  bool const early_error_fail = !probe.rows.empty() && !probe.rows.front().error_pass;
  bool const any_error_pass = probe.trajectory.first_error_pass_m > 0;
  bool const taylor_pass = probe.taylor_error_scaled <= probe.target_relative_tolerance * Real{1.0e-1};
  report
      .status(any_error_pass && early_error_fail && taylor_pass ? presentation::semantic_glyph::success
                                                                : presentation::semantic_glyph::warning,
              any_error_pass && early_error_fail && taylor_pass ? "useful tolerance probe" : "inspect tolerance probe")
      .field("implementation", probe.implementation)
      .field("scalar", probe.scalar)
      .field("dimension", fmt::format("{}", probe.dimension))
      .field("time", probe.time)
      .field("||A|| bound", format_real(probe.operator_norm_bound))
      .field("||v||", format_real(probe.initial_norm))
      .field("||reference||", format_real(probe.reference_norm))
      .field("target rel tol", format_real(probe.target_relative_tolerance))
      .field("target abs tol", format_real(probe.target_absolute_tolerance));

  auto& taylor = report.table("Taylor Reference");
  taylor.column("matvecs").column("scales").column("max deg").column("tail/||v||").column("exact err/||v||");
  taylor.row(fmt::format("{}", probe.taylor_matvecs), fmt::format("{}", probe.taylor_scaling_steps),
             fmt::format("{}", probe.taylor_max_degree), format_real(probe.taylor_estimate_scaled),
             format_real(probe.taylor_error_scaled));

  auto& trajectory = report.table("Trajectory Diagnostics");
  trajectory.column("first err<=tol")
      .column("first est<=tol")
      .column("first coeff<=tol")
      .column("best err m")
      .column("best err/||v||")
      .column("best est m")
      .column("best est/||v||")
      .column("best coeff m")
      .column("best coeff")
      .column("max est rebound")
      .column("max coeff rebound")
      .column("final err/est");
  trajectory.row(
      format_dimension_or_dash(probe.trajectory.first_error_pass_m),
      format_dimension_or_dash(probe.trajectory.first_estimate_pass_m),
      format_dimension_or_dash(probe.trajectory.first_coefficient_pass_m),
      format_dimension_or_dash(probe.trajectory.best_error_m), format_real(probe.trajectory.best_error_scaled),
      format_dimension_or_dash(probe.trajectory.best_estimate_m), format_real(probe.trajectory.best_estimate_scaled),
      format_dimension_or_dash(probe.trajectory.best_coefficient_m), format_real(probe.trajectory.best_coefficient),
      format_real(probe.trajectory.max_estimate_rebound), format_real(probe.trajectory.max_coefficient_rebound),
      format_real(probe.trajectory.final_error_over_estimate));

  auto& rows = report.table("Lanczos Sweep");
  rows.column("m")
      .column("mv")
      .column("err/||v||")
      .column("tail coeff")
      .column("defect/||v||")
      .column("Hquad/||v||")
      .column("Saad phi1/||v||")
      .column("HL bound/||v||")
      .column("orth offdiag")
      .column("reorth ratio")
      .column("passes")
      .column("err/defect")
      .column("err/Saad")
      .column("err<=tol");

  for (auto const& row : probe.rows)
  {
    Real const initial_norm = probe.initial_norm;
    rows.row(fmt::format("{}", row.krylov_dimension), fmt::format("{}", row.lanczos_matvecs),
             format_real(row.error_exact / initial_norm), format_real(row.tail_coefficient),
             format_real(row.residual_estimate / initial_norm),
             format_real(row.hermite_quadrature_estimate / initial_norm),
             format_real(row.saad_phi1_estimate / initial_norm), format_real(row.hochbruck_lubich_bound / initial_norm),
             format_real(row.basis_max_offdiag), format_real(row.max_reorthogonalization_correction_ratio),
             fmt::format("{}", row.max_reorthogonalization_passes), format_real(row.estimate_ratio),
             format_real(safe_ratio(row.error_exact, row.saad_phi1_estimate)), row.error_pass ? "yes" : "no");
  }

  text.append(presentation::render_report(report, policy)).append("\n");
}

template <typename Real> [[nodiscard]] bool probe_passes(ProbeReport<Real> const& probe)
{
  if (probe.rows.empty())
  {
    return false;
  }

  bool const early_error_fail = !probe.rows.front().error_pass;
  bool const any_error_pass = probe.trajectory.first_error_pass_m > 0;
  bool const taylor_is_reference_quality = probe.taylor_error_scaled <= probe.target_relative_tolerance * Real{1.0e-1};
  return early_error_fail && any_error_pass && taylor_is_reference_quality;
}

template <typename Real> [[nodiscard]] std::vector<ProbeReport<Real>> run_all_probes(Options const& options)
{
  Real const tolerance = options.tolerance_was_set ? static_cast<Real>(options.tolerance) : default_tolerance<Real>();
  std::vector<ProbeReport<Real>> probes;

  {
    auto eigenvalues = arithmetic_spectrum<Real>(96, Real{0}, Real{80});
    auto dimensions = std::vector<int>{4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96};
    probes.push_back(run_probe_case<Real>(
        "wide positive spectrum, heat-time damping", LanczosProbeVariant::FullReorthogonalized, "real", eigenvalues,
        deterministic_initial<Real>(eigenvalues.size(), Real{1}), Real{-0.25}, dimensions, tolerance));
    probes.push_back(run_probe_case<Real>(
        "wide positive spectrum, heat-time damping", LanczosProbeVariant::LegacyThreeTerm, "real", eigenvalues,
        deterministic_initial<Real>(eigenvalues.size(), Real{1}), Real{-0.25}, dimensions, tolerance));
  }

  {
    auto eigenvalues = clustered_spectrum<Real>(96);
    auto dimensions = std::vector<int>{4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96};
    probes.push_back(run_probe_case<Real>(
        "same spectrum with tiny nonzero input", LanczosProbeVariant::FullReorthogonalized, "real", eigenvalues,
        deterministic_initial<Real>(eigenvalues.size(), Real{1.0e-8}), Real{-0.30}, dimensions, tolerance));
    probes.push_back(run_probe_case<Real>(
        "same spectrum with tiny nonzero input", LanczosProbeVariant::LegacyThreeTerm, "real", eigenvalues,
        deterministic_initial<Real>(eigenvalues.size(), Real{1.0e-8}), Real{-0.30}, dimensions, tolerance));
  }

  {
    using Complex = uni20::complex<Real>;
    auto eigenvalues = arithmetic_spectrum<Real>(96, Real{-30}, Real{30});
    auto dimensions = std::vector<int>{4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96};
    probes.push_back(run_probe_case<Complex>("wide signed spectrum, unitary complex time",
                                             LanczosProbeVariant::FullReorthogonalized, "complex", eigenvalues,
                                             deterministic_initial<Complex>(eigenvalues.size(), Real{1}),
                                             Complex{Real{}, Real{-0.75}}, dimensions, tolerance));
    probes.push_back(run_probe_case<Complex>("wide signed spectrum, unitary complex time",
                                             LanczosProbeVariant::LegacyThreeTerm, "complex", eigenvalues,
                                             deterministic_initial<Complex>(eigenvalues.size(), Real{1}),
                                             Complex{Real{}, Real{-0.75}}, dimensions, tolerance));
  }

  return probes;
}

template <typename Real> int run_with_precision(Options const& options)
{
  auto probes = run_all_probes<Real>(options);

  auto policy = presentation::terminal_policy(stdout);
  policy.glyphs = presentation::glyph_set::unicode;
  policy.charset = presentation::text_charset::utf8;
  policy.width = presentation::width_mode::display_cells;

  presentation::report_builder header("Lanczos vs Taylor exponential-action probes");
  header.field("precision", std::is_same_v<Real, float> ? "float" : "double")
      .field("relative tolerance",
             format_real(options.tolerance_was_set ? static_cast<Real>(options.tolerance) : default_tolerance<Real>()))
      .field("acceptance scale", "relative to ||v||")
      .field("reference", "exact diagonal action, with Taylor reported independently");

  presentation::styled_text report = presentation::render_report(header, policy);
  report.append("\n");
  bool all_passed = true;
  for (auto const& probe : probes)
  {
    append_probe_report(report, policy, probe);
    all_passed = probe_passes(probe) && all_passed;
  }

  fmt::print("{}", presentation::render_terminal(report, policy, stdout));
  if (options.check && !all_passed)
  {
    return 1;
  }
  return 0;
}

} // namespace

int main(int argc, char** argv)
try
{
  Options const options = parse_options(argc, argv);
  switch (options.precision)
  {
    case ScalarPrecision::Float32:
      return run_with_precision<float>(options);
    case ScalarPrecision::Float64:
      return run_with_precision<double>(options);
  }
  return 1;
}
catch (std::exception const& error)
{
  fmt::print(stderr, "error: {}\n", error.what());
  return 1;
}
