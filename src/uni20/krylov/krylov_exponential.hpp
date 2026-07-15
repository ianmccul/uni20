#pragma once

#include <uni20/krylov/dense_linalg.hpp>
#include <uni20/krylov/dense_subspace.hpp>
#include <uni20/krylov/matrix_free.hpp>
#include <uni20/krylov/nonsymmetric_arnoldi.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::krylov
{

/// \brief Parameters for Krylov exponential actions.
/// \tparam Scalar Real scalar type used for tolerances and the time parameter.
template <uni20::Real Scalar> struct KrylovExponentialParams
{
    /// \brief Fixed projection dimension, or adaptive maximum when `relative_tolerance > 0`.
    int krylov_dimension = 0;
    /// \brief Minimum projection dimension before adaptive acceptance; zero selects `1`.
    int minimum_krylov_dimension = 0;
    /// \brief Relative error target for nonexpansive Hermitian actions; zero keeps fixed-dimension mode.
    Scalar relative_tolerance = Scalar{};
    /// \brief Multiplier applied to the error estimate before comparing with the target.
    Scalar estimate_safety_factor = Scalar{1};
    /// \brief Absolute threshold for invariant-subspace breakdown; zero selects a machine default.
    Scalar breakdown_tolerance = Scalar{};
    /// \brief Permit adaptive acceptance for Hermitian actions not known automatically to be unitary.
    bool assume_nonexpansive = false;
    /// \brief Throw if adaptive Hermitian acceptance does not reach the requested tolerance.
    bool throw_on_nonconvergence = true;
    /// \brief Optional diagnostic retention level.
    KrylovDiagnosticsLevel diagnostics = KrylovDiagnosticsLevel::None;
};

/// \brief Optional diagnostic state for Krylov exponential actions.
/// \tparam Scalar Real scalar type.
template <uni20::Real Scalar> struct KrylovExponentialDiagnostics
{
    int op_count = 0;
    int projected_dimension = 0;
    Scalar initial_norm = Scalar{};
    Scalar final_residual_norm = Scalar{};
    Scalar target_error = Scalar{};
    /// \brief Best available action error estimate for the selected Krylov path.
    Scalar error_estimate = Scalar{};
    /// \brief Final-time defect estimate `||v|| h_{m+1,m} |e_m^* exp(t H_m) e_1|`.
    Scalar endpoint_defect_estimate = Scalar{};
    /// \brief Time-integrated Hermitian/Lanczos defect estimate; zero for nonsymmetric Arnoldi actions.
    Scalar defect_integral_estimate = Scalar{};
    Scalar basis_max_diag_error = Scalar{};
    Scalar basis_max_offdiag = Scalar{};
    Scalar basis_frobenius_error = Scalar{};
    Scalar max_reorthogonalization_correction = Scalar{};
    Scalar max_reorthogonalization_correction_ratio = Scalar{};
    int max_reorthogonalization_passes = 0;
    bool converged = true;
};

/// \brief Matrix-free approximation to `exp(t A) v`.
/// \tparam Scalar Vector-space scalar type.
/// \tparam Vector Opaque vector type.
template <uni20::RealOrComplex Scalar, typename Vector> struct KrylovExponentialResult
{
    Vector action;
    uni20::make_real_t<Scalar> error_estimate = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> endpoint_defect_estimate = uni20::make_real_t<Scalar>{};
    /// \brief Time-integrated Hermitian/Lanczos defect estimate; zero for nonsymmetric Arnoldi actions.
    uni20::make_real_t<Scalar> defect_integral_estimate = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> initial_norm = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> final_residual_norm = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> target_error = uni20::make_real_t<Scalar>{};
    int projected_dimension = 0;
    int matvec_count = 0;
    bool converged = true;
    bool happy_breakdown = false;
    std::optional<KrylovExponentialDiagnostics<uni20::make_real_t<Scalar>>> diagnostics;
};

/// \brief Return the default Krylov projection dimension for exponential actions.
[[nodiscard]] inline int default_exponential_krylov_dimension(std::size_t problem_dimension)
{
  if (problem_dimension > static_cast<std::size_t>(uni20::numeric_limits<int>::max()))
  {
    throw std::overflow_error("Krylov exponential problem dimension exceeds the supported int range");
  }
  return std::min(static_cast<int>(problem_dimension), 30);
}

/// \brief Return the default adaptive Krylov projection cap for exponential actions.
[[nodiscard]] inline int default_adaptive_exponential_krylov_dimension(std::size_t problem_dimension)
{
  if (problem_dimension > static_cast<std::size_t>(uni20::numeric_limits<int>::max()))
  {
    throw std::overflow_error("Krylov exponential problem dimension exceeds the supported int range");
  }
  return std::min(static_cast<int>(problem_dimension), 64);
}

template <uni20::Real Scalar>
[[nodiscard]] bool exponential_adaptive_acceptance_enabled(KrylovExponentialParams<Scalar> const& params)
{
  return params.relative_tolerance > Scalar{};
}

/// \brief Return the effective Krylov projection dimension for exponential actions.
template <uni20::Real Scalar>
[[nodiscard]] int effective_exponential_krylov_dimension(KrylovExponentialParams<Scalar> const& params,
                                                         std::size_t problem_dimension)
{
  if (params.krylov_dimension > 0)
  {
    return params.krylov_dimension;
  }
  return exponential_adaptive_acceptance_enabled(params)
             ? default_adaptive_exponential_krylov_dimension(problem_dimension)
             : default_exponential_krylov_dimension(problem_dimension);
}

namespace detail
{

template <uni20::Real Scalar> Scalar exponential_breakdown_tolerance(KrylovExponentialParams<Scalar> const& params)
{
  return params.breakdown_tolerance > Scalar{} ? params.breakdown_tolerance
                                               : Scalar{10} * uni20::numeric_limits<Scalar>::epsilon();
}

template <uni20::Real Scalar> void validate_krylov_exponential_params(KrylovExponentialParams<Scalar> const& params)
{
  if (params.krylov_dimension < 0)
  {
    throw std::invalid_argument("Krylov exponential requires a non-negative krylov_dimension");
  }
  if (params.minimum_krylov_dimension < 0)
  {
    throw std::invalid_argument("Krylov exponential requires a non-negative minimum_krylov_dimension");
  }
  if (params.relative_tolerance < Scalar{} || !detail::adl_isfinite(params.relative_tolerance))
  {
    throw std::invalid_argument("Krylov exponential requires a finite non-negative relative_tolerance");
  }
  if (params.estimate_safety_factor < Scalar{1} || !detail::adl_isfinite(params.estimate_safety_factor))
  {
    throw std::invalid_argument("Krylov exponential requires estimate_safety_factor >= 1");
  }
  if (params.breakdown_tolerance < Scalar{} || !detail::adl_isfinite(params.breakdown_tolerance))
  {
    throw std::invalid_argument("Krylov exponential requires a finite non-negative breakdown_tolerance");
  }
}

template <uni20::Real Scalar>
[[nodiscard]] int effective_exponential_minimum_krylov_dimension(KrylovExponentialParams<Scalar> const& params,
                                                                 int basis_limit)
{
  int const minimum = params.minimum_krylov_dimension > 0 ? params.minimum_krylov_dimension : 1;
  if (minimum > basis_limit)
  {
    throw std::invalid_argument("Krylov exponential requires minimum_krylov_dimension <= krylov_dimension");
  }
  return minimum;
}

template <uni20::Real Scalar>
[[nodiscard]] Scalar hermitian_exponential_target_error(KrylovExponentialParams<Scalar> const& params,
                                                        Scalar initial_norm)
{
  if (!exponential_adaptive_acceptance_enabled(params))
  {
    return Scalar{};
  }
  Scalar const target = params.relative_tolerance * initial_norm;
  if (!detail::adl_isfinite(target))
  {
    throw std::overflow_error("Krylov exponential relative tolerance target overflowed");
  }
  return target;
}

template <uni20::Real Scalar>
[[nodiscard]] bool exponential_estimate_satisfies_target(Scalar estimate, Scalar target_error,
                                                         KrylovExponentialParams<Scalar> const& params)
{
  if (!exponential_adaptive_acceptance_enabled(params))
  {
    return true;
  }
  return estimate <= target_error / params.estimate_safety_factor;
}

template <uni20::Real Real, typename TimeScalar> uni20::complex<Real> exponential_time_as_complex(TimeScalar time)
{
  if constexpr (uni20::Complex<TimeScalar>)
  {
    return static_cast<uni20::complex<Real>>(time);
  }
  else
  {
    return uni20::complex<Real>{static_cast<Real>(time), Real{}};
  }
}

template <uni20::Real Real, typename TimeScalar> bool hermitian_time_is_unitary(TimeScalar time)
{
  uni20::complex<Real> const complex_time = exponential_time_as_complex<Real>(time);
  Real const magnitude = static_cast<Real>(detail::adl_abs(complex_time));
  Real const real_part = static_cast<Real>(detail::adl_abs(detail::adl_real(complex_time)));
  Real const tolerance = Real{100} * uni20::numeric_limits<Real>::epsilon() * std::max(Real{1}, magnitude);
  return real_part <= tolerance;
}

template <uni20::Real Real, typename TimeScalar>
void validate_hermitian_adaptive_error_model(KrylovExponentialParams<Real> const& params, TimeScalar time)
{
  if (!exponential_adaptive_acceptance_enabled(params))
  {
    return;
  }
  if (params.assume_nonexpansive || hermitian_time_is_unitary<Real>(time))
  {
    return;
  }
  throw std::invalid_argument(
      "Hermitian Krylov adaptive tolerance requires a nonexpansive action; use pure imaginary time or set "
      "assume_nonexpansive when the caller has verified the model");
}

template <uni20::RealOrComplex Scalar, typename Vector> struct HermitianExponentialProjection
{
    std::vector<Vector> basis;
    Matrix<uni20::make_real_t<Scalar>> projected;
    uni20::make_real_t<Scalar> initial_norm = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> residual_norm = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> basis_max_diag_error = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> basis_max_offdiag = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> basis_frobenius_error = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> max_reorthogonalization_correction = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> max_reorthogonalization_correction_ratio = uni20::make_real_t<Scalar>{};
    int max_reorthogonalization_passes = 0;
    int op_count = 0;
    bool happy_breakdown = false;
};

template <uni20::Real Real>
Matrix<Real> make_hermitian_projected_tridiagonal(std::vector<Real> const& diagonal,
                                                  std::vector<Real> const& subdiagonal)
{
  if (diagonal.size() > 0 && subdiagonal.size() + 1 != diagonal.size())
  {
    throw std::invalid_argument("Hermitian Krylov tridiagonal data have inconsistent sizes");
  }
  Matrix<Real> projected(diagonal.size(), diagonal.size());
  for (std::size_t i = 0; i < diagonal.size(); ++i)
  {
    projected[i, i] = diagonal[i];
    if (i + 1 < diagonal.size())
    {
      projected[i, i + 1] = subdiagonal[i];
      projected[i + 1, i] = subdiagonal[i];
    }
  }
  return projected;
}

template <uni20::RealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
void record_exponential_basis_orthogonality(Ops& ops, HermitianExponentialProjection<Scalar, Vector>& projection)
{
  using Real = uni20::make_real_t<Scalar>;
  Real frobenius_squared{};
  for (std::size_t row = 0; row < projection.basis.size(); ++row)
  {
    for (std::size_t col = row; col < projection.basis.size(); ++col)
    {
      Scalar const entry = ops.inner_product(projection.basis[row], projection.basis[col]);
      if (row == col)
      {
        Real const error = static_cast<Real>(detail::adl_abs(entry - Scalar{1}));
        projection.basis_max_diag_error = std::max(projection.basis_max_diag_error, error);
        frobenius_squared += error * error;
      }
      else
      {
        Real const magnitude = static_cast<Real>(detail::adl_abs(entry));
        projection.basis_max_offdiag = std::max(projection.basis_max_offdiag, magnitude);
        frobenius_squared += Real{2} * magnitude * magnitude;
      }
    }
  }
  projection.basis_frobenius_error = static_cast<Real>(detail::adl_sqrt(frobenius_squared));
}

template <uni20::RealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops,
          typename StopPredicate>
HermitianExponentialProjection<Scalar, Vector>
hermitian_exponential_projection(Ops& ops, Vector const& initial,
                                 KrylovExponentialParams<uni20::make_real_t<Scalar>> const& params, int basis_limit,
                                 StopPredicate&& stop)
{
  using Real = uni20::make_real_t<Scalar>;
  std::size_t const problem_dimension = validate_matrix_free_dimensions(ops, initial, "Hermitian Krylov exponential");
  if (basis_limit <= 0)
  {
    throw std::invalid_argument("Hermitian Krylov exponential requires a positive Krylov dimension");
  }
  if (basis_limit > static_cast<int>(problem_dimension))
  {
    throw std::invalid_argument("Hermitian Krylov exponential requires krylov_dimension <= problem dimension");
  }

  Real const breakdown_tolerance = exponential_breakdown_tolerance(params);

  HermitianExponentialProjection<Scalar, Vector> result;
  result.basis.reserve(static_cast<std::size_t>(basis_limit));
  std::vector<Real> diagonal;
  std::vector<Real> subdiagonal;
  diagonal.reserve(static_cast<std::size_t>(basis_limit));
  subdiagonal.reserve(static_cast<std::size_t>(std::max(0, basis_limit - 1)));
  LanczosOrthogonalizationDiagnostics<Real> orthogonalization_diagnostics;
  auto* orthogonalization_diagnostics_ptr =
      params.diagnostics == KrylovDiagnosticsLevel::None ? nullptr : &orthogonalization_diagnostics;

  Vector start = ops.allocate_like(initial);
  ops.copy(start, initial);
  result.initial_norm = norm_or_inner_product<Scalar>(ops, start);
  if (result.initial_norm == Real{})
  {
    ops.set_zero(start);
    result.projected = Matrix<Real>();
    result.happy_breakdown = true;
    return result;
  }
  ops.scal(start, Scalar{1} / static_cast<Scalar>(result.initial_norm));
  result.basis.push_back(std::move(start));

  for (int step = 0; step < basis_limit; ++step)
  {
    Vector residual = ops.allocate_like(result.basis[static_cast<std::size_t>(step)]);
    ops.matvec(residual, result.basis[static_cast<std::size_t>(step)]);
    ++result.op_count;

    if (step > 0)
    {
      ops.axpy(residual, -static_cast<Scalar>(subdiagonal[static_cast<std::size_t>(step - 1)]),
               result.basis[static_cast<std::size_t>(step - 1)]);
    }

    Real const projection_scale = detail::hermitian_projection_scale<Scalar>(ops, residual);
    Real const alpha = detail::hermitian_projection_scalar(
        ops.inner_product(result.basis[static_cast<std::size_t>(step)], residual), projection_scale);
    diagonal.push_back(alpha);
    ops.axpy(residual, -static_cast<Scalar>(alpha), result.basis[static_cast<std::size_t>(step)]);

    Real const beta =
        detail::orthogonalize_lanczos_residual<Scalar>(ops, result.basis, residual, orthogonalization_diagnostics_ptr);
    result.residual_norm = beta;
    bool const happy_breakdown = beta <= breakdown_tolerance;
    bool const accepted = stop(result.initial_norm, diagonal, subdiagonal, beta, happy_breakdown);
    if (accepted || step + 1 == basis_limit || happy_breakdown)
    {
      result.happy_breakdown = happy_breakdown;
      break;
    }

    ops.scal(residual, Scalar{1} / static_cast<Scalar>(beta));
    subdiagonal.push_back(beta);
    result.basis.push_back(std::move(residual));
  }

  result.projected = make_hermitian_projected_tridiagonal(diagonal, subdiagonal);
  if (params.diagnostics != KrylovDiagnosticsLevel::None)
  {
    result.max_reorthogonalization_correction = orthogonalization_diagnostics.max_reorthogonalization_correction;
    result.max_reorthogonalization_correction_ratio =
        orthogonalization_diagnostics.max_reorthogonalization_correction_ratio;
    result.max_reorthogonalization_passes = orthogonalization_diagnostics.max_reorthogonalization_passes;
    record_exponential_basis_orthogonality<Scalar>(ops, result);
  }
  return result;
}

template <uni20::RealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
HermitianExponentialProjection<Scalar, Vector>
hermitian_exponential_projection(Ops& ops, Vector const& initial,
                                 KrylovExponentialParams<uni20::make_real_t<Scalar>> const& params, int basis_limit)
{
  auto never_stop = [](uni20::make_real_t<Scalar>, std::vector<uni20::make_real_t<Scalar>> const&,
                       std::vector<uni20::make_real_t<Scalar>> const&, uni20::make_real_t<Scalar>,
                       bool) static { return false; };
  return hermitian_exponential_projection<Scalar>(ops, initial, params, basis_limit, never_stop);
}

template <uni20::RealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
Vector combine_exponential_basis(Ops& ops, Vector const& prototype, std::vector<Vector> const& basis,
                                 std::vector<Scalar> const& coefficients, std::size_t basis_count)
{
  if (basis.size() < basis_count || basis_count != coefficients.size())
  {
    throw std::invalid_argument("Krylov exponential basis and coefficient sizes do not match");
  }

  Vector action = ops.allocate_like(prototype);
  ops.set_zero(action);
  for (std::size_t i = 0; i < basis_count; ++i)
  {
    if (coefficients[i] != Scalar{})
    {
      ops.axpy(action, coefficients[i], basis[i]);
    }
  }
  return action;
}

template <typename Scalar> std::vector<Scalar> first_column_scaled(Matrix<Scalar> const& exponential, Scalar scale)
{
  std::vector<Scalar> coefficients(static_cast<std::size_t>(exponential.rows()));
  for (uni20::index_type row = 0; row < exponential.rows(); ++row)
  {
    coefficients[row] = scale * exponential[row, 0];
  }
  return coefficients;
}

template <uni20::RealOrComplex Scalar>
std::vector<Scalar> first_real_column_scaled(Matrix<uni20::make_real_t<Scalar>> const& exponential,
                                             uni20::make_real_t<Scalar> scale)
{
  std::vector<Scalar> coefficients(static_cast<std::size_t>(exponential.rows()));
  for (uni20::index_type row = 0; row < exponential.rows(); ++row)
  {
    coefficients[row] = static_cast<Scalar>(scale * exponential[row, 0]);
  }
  return coefficients;
}

template <uni20::RealOrComplex Scalar>
uni20::make_real_t<Scalar> endpoint_defect_estimate(uni20::make_real_t<Scalar> residual_norm,
                                                    std::vector<Scalar> const& coefficients)
{
  // Final-time defect estimate h_{m+1,m} |e_m^* exp(t H_m) e_1| scaled by
  // ||v|| through the stored coefficients. See docs/krylov/exponential_estimators.md:
  // [BotchevGrimmHochbruck2013] for the residual/defect view and [JiaLv2015]
  // for related a posteriori estimates.
  if (coefficients.empty())
  {
    return uni20::make_real_t<Scalar>{};
  }
  return residual_norm * static_cast<uni20::make_real_t<Scalar>>(detail::adl_abs(coefficients.back()));
}

template <uni20::Real Real>
uni20::complex<Real> projected_defect_delta(TridiagonalEigensystem<Real> const& eigensystem, uni20::complex<Real> time,
                                            Real theta)
{
  using Complex = uni20::complex<Real>;
  std::size_t const dimension = eigensystem.eigenvalues.size();
  Complex result{};
  for (std::size_t col = 0; col < dimension; ++col)
  {
    Real const weight = eigensystem.eigenvectors[dimension - 1, col] * eigensystem.eigenvectors[0, col];
    result += Complex{weight, Real{}} * detail::adl_exp(time * (theta * eigensystem.eigenvalues[col]));
  }
  return result;
}

template <uni20::Real Real, typename TimeScalar>
Real hermitian_projected_defect_integral_estimate(Matrix<Real> const& projected, Real initial_norm, Real residual_norm,
                                                  TimeScalar time)
{
  if (projected.rows() != projected.cols())
  {
    throw std::invalid_argument("Hermitian Krylov defect integral requires a square projected matrix");
  }
  if (projected.rows() == 0 || initial_norm == Real{} || residual_norm == Real{})
  {
    return Real{};
  }

  uni20::complex<Real> const complex_time = exponential_time_as_complex<Real>(time);
  Real const time_magnitude = static_cast<Real>(detail::adl_abs(complex_time));
  if (time_magnitude == Real{})
  {
    return Real{};
  }

  std::size_t const dimension = static_cast<std::size_t>(projected.rows());
  std::vector<Real> diagonal(dimension);
  std::vector<Real> subdiagonal(dimension > 0 ? dimension - 1 : 0);
  for (std::size_t i = 0; i < dimension; ++i)
  {
    diagonal[i] = projected[i, i];
    if (i + 1 < dimension)
    {
      subdiagonal[i] = projected[i + 1, i];
    }
  }

  auto const eigensystem = symmetric_tridiagonal_eigensystem(std::move(diagonal), std::move(subdiagonal), true);

  // [JaweckiAuzingerKoch2020] writes the Krylov error through the defect
  // integral L_m(t)v = int_0^t exp((t-s)A) D_m(s)v ds.
  // For the unitary Hermitian case used by TDVP, ||exp((t-s)A)|| = 1 after
  // folding the -i into the time coefficient, so the global action error is
  // bounded by ||v|| h_{m+1,m} int_0^|t| |delta_m(s)| ds.  We integrate the
  // projected scalar delta over theta in [0, 1].  The projected matrix is tiny
  // compared with a tensor-network matvec, so a deterministic 1024-panel
  // Simpson rule is a deliberate simple policy rather than a performance risk.
  constexpr int panel_count = 1024;
  static_assert(panel_count % 2 == 0);

  Real weighted_sum = static_cast<Real>(detail::adl_abs(projected_defect_delta(eigensystem, complex_time, Real{}))) +
                      static_cast<Real>(detail::adl_abs(projected_defect_delta(eigensystem, complex_time, Real{1})));
  for (int panel = 1; panel < panel_count; ++panel)
  {
    Real const theta = static_cast<Real>(panel) / static_cast<Real>(panel_count);
    Real const weight = panel % 2 == 0 ? Real{2} : Real{4};
    weighted_sum +=
        weight * static_cast<Real>(detail::adl_abs(projected_defect_delta(eigensystem, complex_time, theta)));
  }

  Real const integral = weighted_sum / (Real{3} * static_cast<Real>(panel_count));
  return initial_norm * residual_norm * time_magnitude * integral;
}

template <typename TimeScalar, uni20::LapackScalar OutputScalar, typename InputScalar>
  requires std::constructible_from<OutputScalar, TimeScalar>
Matrix<OutputScalar> projected_exponential(Matrix<InputScalar> const& projected, TimeScalar time)
{
  Matrix<OutputScalar> scaled(projected.rows(), projected.cols());
  OutputScalar const coefficient = static_cast<OutputScalar>(time);
  for (uni20::index_type row = 0; row < projected.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < projected.cols(); ++col)
    {
      scaled[row, col] = coefficient * static_cast<OutputScalar>(projected[row, col]);
    }
  }
  Matrix<OutputScalar> result(projected.rows(), projected.cols());
  uni20::linalg::matrix_exponential(result, scaled, uni20::make_real_t<OutputScalar>{1});
  return result;
}

template <uni20::RealOrComplex Scalar, typename Vector>
KrylovExponentialDiagnostics<uni20::make_real_t<Scalar>>
make_exponential_diagnostics(KrylovExponentialResult<Scalar, Vector> const& result)
{
  return KrylovExponentialDiagnostics<uni20::make_real_t<Scalar>>{
      .op_count = result.matvec_count,
      .projected_dimension = result.projected_dimension,
      .initial_norm = result.initial_norm,
      .final_residual_norm = result.final_residual_norm,
      .target_error = result.target_error,
      .error_estimate = result.error_estimate,
      .endpoint_defect_estimate = result.endpoint_defect_estimate,
      .defect_integral_estimate = result.defect_integral_estimate,
      .converged = result.converged};
}

template <uni20::RealOrComplex Scalar, typename Vector>
void attach_exponential_diagnostics(KrylovExponentialResult<Scalar, Vector>& result, KrylovDiagnosticsLevel level,
                                    KrylovExponentialDiagnostics<uni20::make_real_t<Scalar>> diagnostics)
{
  if (level == KrylovDiagnosticsLevel::None)
  {
    return;
  }

  result.diagnostics = diagnostics;
}

template <uni20::RealOrComplex Scalar, typename Vector>
void attach_exponential_diagnostics(KrylovExponentialResult<Scalar, Vector>& result, KrylovDiagnosticsLevel level)
{
  attach_exponential_diagnostics(result, level, make_exponential_diagnostics(result));
}

template <uni20::LapackScalar Scalar, typename TimeScalar, typename Vector,
          KrylovMatrixFreeOperator<Vector, Scalar> Ops>
KrylovExponentialResult<Scalar, Vector>
make_hermitian_exponential_result(Ops& ops, Vector const& initial,
                                  HermitianExponentialProjection<Scalar, Vector> const& projection, TimeScalar time,
                                  KrylovExponentialParams<uni20::make_real_t<Scalar>> const& params)
{
  using Real = uni20::make_real_t<Scalar>;
  using ProjectedScalar = std::conditional_t<uni20::Complex<Scalar>, Scalar, Real>;

  Matrix<ProjectedScalar> const exponential =
      projected_exponential<TimeScalar, ProjectedScalar>(projection.projected, time);
  std::vector<Scalar> const coefficients =
      first_column_scaled(exponential, static_cast<ProjectedScalar>(projection.initial_norm));
  Vector action = combine_exponential_basis<Scalar>(ops, initial, projection.basis, coefficients, coefficients.size());
  Real const endpoint_defect = endpoint_defect_estimate<Scalar>(projection.residual_norm, coefficients);
  Real const defect_integral = hermitian_projected_defect_integral_estimate(
      projection.projected, projection.initial_norm, projection.residual_norm, time);
  Real const target_error = hermitian_exponential_target_error(params, projection.initial_norm);
  bool const converged =
      projection.happy_breakdown || exponential_estimate_satisfies_target(defect_integral, target_error, params);

  KrylovExponentialResult<Scalar, Vector> result{.action = std::move(action),
                                                 .error_estimate = defect_integral,
                                                 .endpoint_defect_estimate = endpoint_defect,
                                                 .defect_integral_estimate = defect_integral,
                                                 .initial_norm = projection.initial_norm,
                                                 .final_residual_norm = projection.residual_norm,
                                                 .target_error = target_error,
                                                 .projected_dimension = static_cast<int>(projection.projected.rows()),
                                                 .matvec_count = projection.op_count,
                                                 .converged = converged,
                                                 .happy_breakdown = projection.happy_breakdown,
                                                 .diagnostics = std::nullopt};
  auto diagnostics = make_exponential_diagnostics(result);
  diagnostics.basis_max_diag_error = projection.basis_max_diag_error;
  diagnostics.basis_max_offdiag = projection.basis_max_offdiag;
  diagnostics.basis_frobenius_error = projection.basis_frobenius_error;
  diagnostics.max_reorthogonalization_correction = projection.max_reorthogonalization_correction;
  diagnostics.max_reorthogonalization_correction_ratio = projection.max_reorthogonalization_correction_ratio;
  diagnostics.max_reorthogonalization_passes = projection.max_reorthogonalization_passes;
  attach_exponential_diagnostics(result, params.diagnostics, diagnostics);
  return result;
}

} // namespace detail

/// \brief Approximate `exp(t A) v` for a Hermitian matrix-free operator.
///
/// \details Builds a Lanczos projection `T_m`, evaluates `exp(t T_m)` with the
///          current CPU dense Padé backend, and returns
///          `||v|| V_m exp(t T_m) e_1`. When `relative_tolerance > 0`, pure
///          imaginary time is treated as the usual unitary `exp(-i t H)` case
///          and the routine stops at the first dimension whose direct
///          defect-integral estimate satisfies the relative target. Other
///          Hermitian nonexpansive actions may opt in with
///          `assume_nonexpansive`. `error_estimate` is the time-integrated
///          Hermitian defect estimate, while `endpoint_defect_estimate` keeps
///          the final-time sampled defect.
///          Real vector spaces require real time. Complex vector spaces also
///          accept complex time, so `exp(-i t H) v` can use the Hermitian
///          Lanczos projection without hiding the phase in the operator.
/// \tparam Scalar Real or complex vector-field scalar type with projected dense LAPACK coverage.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free operation provider.
template <uni20::LapackScalar Scalar, typename TimeScalar, typename Vector,
          KrylovMatrixFreeOperator<Vector, Scalar> Ops>
  requires(std::constructible_from<uni20::make_real_t<Scalar>, TimeScalar> ||
           (uni20::Complex<Scalar> && std::constructible_from<Scalar, TimeScalar>))
KrylovExponentialResult<Scalar, Vector>
hermitian_krylov_exponential_action(Ops& ops, Vector const& initial, TimeScalar time,
                                    KrylovExponentialParams<uni20::make_real_t<Scalar>> const& params = {})
{
  using Real = uni20::make_real_t<Scalar>;

  detail::validate_krylov_exponential_params(params);
  detail::validate_hermitian_adaptive_error_model(params, time);
  std::size_t const problem_dimension = validate_matrix_free_dimensions(ops, initial, "Hermitian Krylov exponential");
  int const basis_limit = effective_exponential_krylov_dimension(params, problem_dimension);
  int const minimum_dimension = detail::effective_exponential_minimum_krylov_dimension(params, basis_limit);
  bool const adaptive = exponential_adaptive_acceptance_enabled(params);

  auto adaptive_stop = [&](Real initial_norm, std::vector<Real> const& diagonal, std::vector<Real> const& subdiagonal,
                           Real residual_norm, bool happy_breakdown) {
    if (!adaptive || happy_breakdown || static_cast<int>(diagonal.size()) < minimum_dimension)
    {
      return false;
    }
    Matrix<Real> const projected = detail::make_hermitian_projected_tridiagonal(diagonal, subdiagonal);
    Real const estimate =
        detail::hermitian_projected_defect_integral_estimate(projected, initial_norm, residual_norm, time);
    Real const target_error = detail::hermitian_exponential_target_error(params, initial_norm);
    return detail::exponential_estimate_satisfies_target(estimate, target_error, params);
  };

  auto projection = detail::hermitian_exponential_projection<Scalar>(ops, initial, params, basis_limit, adaptive_stop);
  if (projection.initial_norm == Real{})
  {
    Vector action = ops.allocate_like(initial);
    ops.set_zero(action);
    KrylovExponentialResult<Scalar, Vector> result{.action = std::move(action),
                                                   .initial_norm = projection.initial_norm,
                                                   .final_residual_norm = projection.residual_norm,
                                                   .target_error = Real{},
                                                   .projected_dimension = 0,
                                                   .matvec_count = projection.op_count,
                                                   .converged = true,
                                                   .happy_breakdown = true,
                                                   .diagnostics = std::nullopt};
    detail::attach_exponential_diagnostics(result, params.diagnostics);
    return result;
  }

  auto result = detail::make_hermitian_exponential_result<Scalar>(ops, initial, projection, time, params);
  if (adaptive && !result.converged && params.throw_on_nonconvergence)
  {
    throw std::runtime_error("Hermitian Krylov exponential did not satisfy the requested relative_tolerance");
  }
  return result;
}

/// \brief Approximate `exp(t A) v` for a nonsymmetric matrix-free operator.
///
/// \details Builds a full-reorthogonalized Arnoldi projection `H_m`, evaluates
///          `exp(t H_m)` with the current CPU dense Padé backend, and
///          returns `||v|| V_m exp(t H_m) e_1`. This covers real and complex
///          nonsymmetric operators without inspecting vector storage. Complex
///          vector spaces also accept complex time.
/// \tparam Scalar Real or complex vector-field scalar type with projected dense LAPACK coverage.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free operation provider.
template <uni20::LapackScalar Scalar, typename TimeScalar, typename Vector,
          KrylovMatrixFreeOperator<Vector, Scalar> Ops>
  requires(std::constructible_from<uni20::make_real_t<Scalar>, TimeScalar> ||
           (uni20::Complex<Scalar> && std::constructible_from<Scalar, TimeScalar>))
KrylovExponentialResult<Scalar, Vector>
nonsymmetric_krylov_exponential_action(Ops& ops, Vector const& initial, TimeScalar time,
                                       KrylovExponentialParams<uni20::make_real_t<Scalar>> const& params = {})
{
  using Real = uni20::make_real_t<Scalar>;

  detail::validate_krylov_exponential_params(params);
  if (exponential_adaptive_acceptance_enabled(params))
  {
    throw std::invalid_argument("Nonsymmetric Krylov exponential adaptive relative_tolerance is not implemented yet");
  }

  std::size_t const problem_dimension =
      validate_matrix_free_dimensions(ops, initial, "nonsymmetric Krylov exponential");
  int const basis_limit = effective_exponential_krylov_dimension(params, problem_dimension);
  if (basis_limit <= 0)
  {
    throw std::invalid_argument("nonsymmetric Krylov exponential requires a positive Krylov dimension");
  }
  if (basis_limit > static_cast<int>(problem_dimension))
  {
    throw std::invalid_argument("nonsymmetric Krylov exponential requires krylov_dimension <= problem dimension");
  }

  Vector start = ops.allocate_like(initial);
  ops.copy(start, initial);
  Real const initial_norm = norm_or_inner_product<Scalar>(ops, start);
  if (initial_norm == Real{})
  {
    ops.set_zero(start);
    KrylovExponentialResult<Scalar, Vector> result{.action = std::move(start),
                                                   .initial_norm = initial_norm,
                                                   .final_residual_norm = Real{},
                                                   .target_error = Real{},
                                                   .projected_dimension = 0,
                                                   .converged = true,
                                                   .happy_breakdown = true,
                                                   .diagnostics = std::nullopt};
    detail::attach_exponential_diagnostics(result, params.diagnostics);
    return result;
  }

  ArnoldiFactorization<Scalar, Vector> factorization =
      arnoldi_factorize<Scalar>(ops, initial, basis_limit, detail::exponential_breakdown_tolerance(params));
  Matrix<Scalar> const projected = arnoldi_projected_hessenberg(factorization);
  Matrix<Scalar> const exponential = detail::projected_exponential<TimeScalar, Scalar>(projected, time);
  std::vector<Scalar> const coefficients = detail::first_column_scaled(exponential, static_cast<Scalar>(initial_norm));
  Vector action =
      detail::combine_exponential_basis<Scalar>(ops, initial, factorization.basis, coefficients, coefficients.size());
  Real const endpoint_defect = detail::endpoint_defect_estimate<Scalar>(factorization.residual_norm, coefficients);

  KrylovExponentialResult<Scalar, Vector> result{.action = std::move(action),
                                                 .error_estimate = endpoint_defect,
                                                 .endpoint_defect_estimate = endpoint_defect,
                                                 .initial_norm = initial_norm,
                                                 .final_residual_norm = factorization.residual_norm,
                                                 .target_error = Real{},
                                                 .projected_dimension = factorization.step_count,
                                                 .matvec_count = factorization.op_count,
                                                 .converged = true,
                                                 .happy_breakdown = factorization.happy_breakdown,
                                                 .diagnostics = std::nullopt};
  detail::attach_exponential_diagnostics(result, params.diagnostics);
  return result;
}

} // namespace uni20::krylov
