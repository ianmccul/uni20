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

/// \brief Parameters for a fixed-subspace Krylov exponential action.
/// \tparam Scalar Real scalar type used for tolerances and the time parameter.
template <uni20::Real Scalar> struct KrylovExponentialParams
{
    /// \brief Maximum projection dimension; zero selects the default policy.
    int krylov_dimension = 0;
    /// \brief Absolute threshold for invariant-subspace breakdown; zero selects a machine default.
    Scalar breakdown_tolerance = Scalar{};
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
    Scalar residual_estimate = Scalar{};
    Scalar basis_max_diag_error = Scalar{};
    Scalar basis_max_offdiag = Scalar{};
    Scalar basis_frobenius_error = Scalar{};
    Scalar max_reorthogonalization_correction = Scalar{};
    Scalar max_reorthogonalization_correction_ratio = Scalar{};
    int max_reorthogonalization_passes = 0;
};

/// \brief Matrix-free approximation to `exp(t A) v`.
/// \tparam Scalar Vector-space scalar type.
/// \tparam Vector Opaque vector type.
template <uni20::RealOrComplex Scalar, typename Vector> struct KrylovExponentialResult
{
    Vector action;
    uni20::make_real_t<Scalar> residual_estimate = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> initial_norm = uni20::make_real_t<Scalar>{};
    uni20::make_real_t<Scalar> final_residual_norm = uni20::make_real_t<Scalar>{};
    int projected_dimension = 0;
    int matvec_count = 0;
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

/// \brief Return the effective Krylov projection dimension for exponential actions.
template <uni20::Real Scalar>
[[nodiscard]] int effective_exponential_krylov_dimension(KrylovExponentialParams<Scalar> const& params,
                                                         std::size_t problem_dimension)
{
  return params.krylov_dimension > 0 ? params.krylov_dimension
                                     : default_exponential_krylov_dimension(problem_dimension);
}

namespace detail
{

template <uni20::Real Scalar> Scalar exponential_breakdown_tolerance(KrylovExponentialParams<Scalar> const& params)
{
  return params.breakdown_tolerance > Scalar{} ? params.breakdown_tolerance
                                               : Scalar{10} * uni20::numeric_limits<Scalar>::epsilon();
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
        Real const error = static_cast<Real>(std::abs(entry - Scalar{1}));
        projection.basis_max_diag_error = std::max(projection.basis_max_diag_error, error);
        frobenius_squared += error * error;
      }
      else
      {
        Real const magnitude = static_cast<Real>(std::abs(entry));
        projection.basis_max_offdiag = std::max(projection.basis_max_offdiag, magnitude);
        frobenius_squared += Real{2} * magnitude * magnitude;
      }
    }
  }
  projection.basis_frobenius_error = static_cast<Real>(std::sqrt(frobenius_squared));
}

template <uni20::RealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
HermitianExponentialProjection<Scalar, Vector>
hermitian_exponential_projection(Ops& ops, Vector const& initial,
                                 KrylovExponentialParams<uni20::make_real_t<Scalar>> const& params)
{
  using Real = uni20::make_real_t<Scalar>;
  std::size_t const problem_dimension = validate_matrix_free_dimensions(ops, initial, "Hermitian Krylov exponential");
  int const basis_limit = effective_exponential_krylov_dimension(params, problem_dimension);
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

    Real const alpha =
        detail::hermitian_projection_scalar(ops.inner_product(result.basis[static_cast<std::size_t>(step)], residual));
    diagonal.push_back(alpha);
    ops.axpy(residual, -static_cast<Scalar>(alpha), result.basis[static_cast<std::size_t>(step)]);

    Real const beta =
        detail::orthogonalize_lanczos_residual<Scalar>(ops, result.basis, residual, orthogonalization_diagnostics_ptr);
    result.residual_norm = beta;
    if (step + 1 == basis_limit || beta <= breakdown_tolerance)
    {
      result.happy_breakdown = beta <= breakdown_tolerance;
      break;
    }

    ops.scal(residual, Scalar{1} / static_cast<Scalar>(beta));
    subdiagonal.push_back(beta);
    result.basis.push_back(std::move(residual));
  }

  std::size_t const dimension = diagonal.size();
  result.projected = Matrix<Real>(dimension, dimension);
  for (std::size_t i = 0; i < dimension; ++i)
  {
    result.projected(i, i) = diagonal[i];
    if (i + 1 < dimension)
    {
      result.projected(i, i + 1) = subdiagonal[i];
      result.projected(i + 1, i) = subdiagonal[i];
    }
  }
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
  std::vector<Scalar> coefficients(exponential.rows());
  for (std::size_t row = 0; row < exponential.rows(); ++row)
  {
    coefficients[row] = scale * exponential(row, 0);
  }
  return coefficients;
}

template <uni20::RealOrComplex Scalar>
std::vector<Scalar> first_real_column_scaled(Matrix<uni20::make_real_t<Scalar>> const& exponential,
                                             uni20::make_real_t<Scalar> scale)
{
  std::vector<Scalar> coefficients(exponential.rows());
  for (std::size_t row = 0; row < exponential.rows(); ++row)
  {
    coefficients[row] = static_cast<Scalar>(scale * exponential(row, 0));
  }
  return coefficients;
}

template <uni20::RealOrComplex Scalar>
uni20::make_real_t<Scalar> last_coefficient_residual_estimate(uni20::make_real_t<Scalar> residual_norm,
                                                              std::vector<Scalar> const& coefficients)
{
  // Final-time defect estimate h_{m+1,m} |e_m^* exp(t H_m) e_1| scaled by
  // ||v|| through the stored coefficients. See docs/krylov_exponential_estimators.md:
  // [BotchevGrimmHochbruck2013] for the residual/defect view and [JiaLv2015]
  // for related a posteriori estimates.
  if (coefficients.empty())
  {
    return uni20::make_real_t<Scalar>{};
  }
  return residual_norm * static_cast<uni20::make_real_t<Scalar>>(std::abs(coefficients.back()));
}

template <typename TimeScalar, uni20::LapackScalar OutputScalar, typename InputScalar>
  requires std::constructible_from<OutputScalar, TimeScalar>
Matrix<OutputScalar> projected_exponential(Matrix<InputScalar> const& projected, TimeScalar time)
{
  Matrix<OutputScalar> scaled(projected.rows(), projected.cols());
  OutputScalar const coefficient = static_cast<OutputScalar>(time);
  for (std::size_t row = 0; row < projected.rows(); ++row)
  {
    for (std::size_t col = 0; col < projected.cols(); ++col)
    {
      scaled(row, col) = coefficient * static_cast<OutputScalar>(projected(row, col));
    }
  }
  return matrix_exponential(scaled, uni20::make_real_t<OutputScalar>{1});
}

template <uni20::RealOrComplex Scalar, typename Vector>
KrylovExponentialDiagnostics<uni20::make_real_t<Scalar>>
make_exponential_diagnostics(KrylovExponentialResult<Scalar, Vector> const& result)
{
  return KrylovExponentialDiagnostics<uni20::make_real_t<Scalar>>{.op_count = result.matvec_count,
                                                                  .projected_dimension = result.projected_dimension,
                                                                  .initial_norm = result.initial_norm,
                                                                  .final_residual_norm = result.final_residual_norm,
                                                                  .residual_estimate = result.residual_estimate};
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

} // namespace detail

/// \brief Approximate `exp(t A) v` for a Hermitian matrix-free operator.
///
/// \details Builds a Lanczos projection `T_m`, evaluates `exp(t T_m)` with the
///          current CPU dense Padé backend, and returns
///          `||v|| V_m exp(t T_m) e_1`. This first slice is fixed-subspace and
///          non-adaptive; the residual estimate is the last projected
///          exponential coefficient times the final Lanczos residual norm.
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
  using ProjectedScalar = std::conditional_t<uni20::Complex<Scalar>, Scalar, Real>;

  auto projection = detail::hermitian_exponential_projection<Scalar>(ops, initial, params);
  if (projection.initial_norm == Real{})
  {
    Vector action = ops.allocate_like(initial);
    ops.set_zero(action);
    KrylovExponentialResult<Scalar, Vector> result{.action = std::move(action),
                                                   .initial_norm = projection.initial_norm,
                                                   .final_residual_norm = projection.residual_norm,
                                                   .projected_dimension = 0,
                                                   .matvec_count = projection.op_count,
                                                   .happy_breakdown = true,
                                                   .diagnostics = std::nullopt};
    detail::attach_exponential_diagnostics(result, params.diagnostics);
    return result;
  }

  Matrix<ProjectedScalar> const exponential =
      detail::projected_exponential<TimeScalar, ProjectedScalar>(projection.projected, time);
  std::vector<Scalar> const coefficients =
      detail::first_column_scaled(exponential, static_cast<ProjectedScalar>(projection.initial_norm));
  Vector action =
      detail::combine_exponential_basis<Scalar>(ops, initial, projection.basis, coefficients, coefficients.size());

  KrylovExponentialResult<Scalar, Vector> result{
      .action = std::move(action),
      .residual_estimate = detail::last_coefficient_residual_estimate<Scalar>(projection.residual_norm, coefficients),
      .initial_norm = projection.initial_norm,
      .final_residual_norm = projection.residual_norm,
      .projected_dimension = static_cast<int>(projection.projected.rows()),
      .matvec_count = projection.op_count,
      .happy_breakdown = projection.happy_breakdown,
      .diagnostics = std::nullopt};
  auto diagnostics = detail::make_exponential_diagnostics(result);
  diagnostics.basis_max_diag_error = projection.basis_max_diag_error;
  diagnostics.basis_max_offdiag = projection.basis_max_offdiag;
  diagnostics.basis_frobenius_error = projection.basis_frobenius_error;
  diagnostics.max_reorthogonalization_correction = projection.max_reorthogonalization_correction;
  diagnostics.max_reorthogonalization_correction_ratio = projection.max_reorthogonalization_correction_ratio;
  diagnostics.max_reorthogonalization_passes = projection.max_reorthogonalization_passes;
  detail::attach_exponential_diagnostics(result, params.diagnostics, diagnostics);
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
                                                   .projected_dimension = 0,
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

  KrylovExponentialResult<Scalar, Vector> result{
      .action = std::move(action),
      .residual_estimate =
          detail::last_coefficient_residual_estimate<Scalar>(factorization.residual_norm, coefficients),
      .initial_norm = initial_norm,
      .final_residual_norm = factorization.residual_norm,
      .projected_dimension = factorization.step_count,
      .matvec_count = factorization.op_count,
      .happy_breakdown = factorization.happy_breakdown,
      .diagnostics = std::nullopt};
  detail::attach_exponential_diagnostics(result, params.diagnostics);
  return result;
}

} // namespace uni20::krylov
