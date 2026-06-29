/**
 * \file taylor_exponential.hpp
 * \brief Matrix-free Taylor action for validation-oriented exponential actions.
 */

#pragma once

#include <uni20/krylov/matrix_free.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20::krylov
{

/// \brief Parameters for conservative Taylor scaling exponential actions.
/// \tparam Scalar Real scalar type used for tolerances and norm bounds.
template <uni20::Real Scalar> struct TaylorExponentialParams
{
    /// \brief Absolute action tolerance; zero selects `100 * epsilon`.
    Scalar tolerance = Scalar{};
    /// \brief Target upper bound for `||h A||` in one scaled Taylor step.
    Scalar step_norm_limit = Scalar{0.5};
    /// \brief Maximum Taylor degree allowed within one scaled step.
    int max_taylor_degree = 200;
    /// \brief Maximum number of scaled Taylor steps allowed.
    int max_scaling_steps = 100000;
    /// \brief Throw if any scaled step fails to satisfy the tail bound.
    bool throw_on_nonconvergence = true;
    /// \brief Optional diagnostic retention level.
    KrylovDiagnosticsLevel diagnostics = KrylovDiagnosticsLevel::None;
};

/// \brief Diagnostic state for Taylor exponential actions.
/// \tparam Scalar Real scalar type used by diagnostics.
template <uni20::Real Scalar> struct TaylorExponentialDiagnostics
{
    int matvec_count = 0;
    int scaling_steps = 0;
    int max_degree_used = 0;
    int final_step_degree = 0;
    Scalar tolerance = Scalar{};
    Scalar operator_norm_bound = Scalar{};
    Scalar step_norm_bound = Scalar{};
    Scalar estimated_error = Scalar{};
    Scalar final_tail_bound = Scalar{};
    Scalar final_term_norm = Scalar{};
    bool converged = true;
};

/// \brief Result for validation-oriented Taylor exponential actions.
/// \tparam Scalar Vector-space scalar type.
/// \tparam Vector Opaque vector type.
template <uni20::RealOrComplex Scalar, typename Vector> struct TaylorExponentialResult
{
    Vector action;
    uni20::make_real_t<Scalar> estimated_error = uni20::make_real_t<Scalar>{};
    int matvec_count = 0;
    int scaling_steps = 0;
    int max_degree_used = 0;
    bool converged = true;
    std::optional<TaylorExponentialDiagnostics<uni20::make_real_t<Scalar>>> diagnostics;
};

namespace detail
{

template <uni20::Real Scalar> Scalar default_taylor_exponential_tolerance()
{
  return Scalar{100} * uni20::numeric_limits<Scalar>::epsilon();
}

template <uni20::Real Scalar>
Scalar effective_taylor_exponential_tolerance(TaylorExponentialParams<Scalar> const& params)
{
  return params.tolerance > Scalar{} ? params.tolerance : default_taylor_exponential_tolerance<Scalar>();
}

template <uni20::Real Scalar>
void validate_taylor_exponential_params(Scalar operator_norm_bound, TaylorExponentialParams<Scalar> const& params)
{
  if (!detail::adl_isfinite(operator_norm_bound) || operator_norm_bound < Scalar{})
  {
    throw std::invalid_argument("Taylor exponential requires a finite non-negative operator norm bound");
  }
  if (params.tolerance < Scalar{} || !detail::adl_isfinite(params.tolerance))
  {
    throw std::invalid_argument("Taylor exponential requires a finite non-negative tolerance");
  }
  if (params.step_norm_limit <= Scalar{} || !detail::adl_isfinite(params.step_norm_limit))
  {
    throw std::invalid_argument("Taylor exponential requires a positive finite step norm limit");
  }
  if (params.max_taylor_degree <= 0)
  {
    throw std::invalid_argument("Taylor exponential requires a positive maximum Taylor degree");
  }
  if (params.max_scaling_steps <= 0)
  {
    throw std::invalid_argument("Taylor exponential requires a positive maximum scaling step count");
  }
}

template <uni20::Real Scalar>
int taylor_scaling_steps(Scalar scaled_operator_norm, TaylorExponentialParams<Scalar> const& params)
{
  if (scaled_operator_norm == Scalar{})
  {
    return 1;
  }

  using std::ceil;

  Scalar const ratio = scaled_operator_norm / params.step_norm_limit;
  Scalar const steps = ceil(std::max(Scalar{1}, ratio));
  if (steps > static_cast<Scalar>(params.max_scaling_steps))
  {
    throw std::runtime_error("Taylor exponential requires more scaling steps than allowed");
  }
  if (steps > static_cast<Scalar>(uni20::numeric_limits<int>::max()))
  {
    throw std::overflow_error("Taylor exponential scaling step count exceeds int range");
  }
  return static_cast<int>(steps);
}

template <uni20::Real Scalar> Scalar taylor_tail_bound(Scalar term_norm, Scalar step_norm_bound, int degree)
{
  if (term_norm == Scalar{} || step_norm_bound == Scalar{})
  {
    return Scalar{};
  }

  Scalar const ratio = step_norm_bound / static_cast<Scalar>(degree + 1);
  if (ratio >= Scalar{1})
  {
    return uni20::numeric_limits<Scalar>::infinity();
  }
  return term_norm * ratio / (Scalar{1} - ratio);
}

template <uni20::Real Scalar>
Scalar amplified_tail_bound(Scalar local_tail_bound, Scalar step_norm_bound, int remaining_steps)
{
  if (local_tail_bound == Scalar{} || remaining_steps <= 0)
  {
    return local_tail_bound;
  }

  using std::exp;
  using std::log;

  Scalar const exponent = static_cast<Scalar>(remaining_steps) * step_norm_bound;
  Scalar const max_value = uni20::numeric_limits<Scalar>::max();
  if (exponent >= log(max_value) - log(local_tail_bound))
  {
    return uni20::numeric_limits<Scalar>::infinity();
  }
  return local_tail_bound * exp(exponent);
}

template <uni20::RealOrComplex Scalar, typename Vector>
void attach_taylor_exponential_diagnostics(TaylorExponentialResult<Scalar, Vector>& result,
                                           TaylorExponentialDiagnostics<uni20::make_real_t<Scalar>> const& diagnostics,
                                           KrylovDiagnosticsLevel level)
{
  if (level == KrylovDiagnosticsLevel::None)
  {
    return;
  }
  result.diagnostics = diagnostics;
}

} // namespace detail

/// \brief Compute `exp(t A) v` using conservative scaled Taylor series.
///
/// \details This validation-oriented action is algorithmically independent of
///          the Lanczos/Arnoldi projected exponential paths. The caller supplies
///          a finite upper bound for `||A||`; the routine chooses a number of
///          scaled steps so that each Taylor series sees
///          `||h A|| <= step_norm_limit`, then truncates using a geometric tail
///          bound. It is intentionally slower and more conservative than the
///          Krylov projected action, and it is not used as an automatic
///          fallback.
/// \tparam Scalar Real or complex vector-field scalar type.
/// \tparam TimeScalar Real time for real vector spaces; real or complex time
///         for complex vector spaces.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free operation provider.
/// \param ops Matrix-free operation object.
/// \param initial Initial vector.
/// \param time Exponential coefficient.
/// \param operator_norm_bound Finite upper bound for `||A||`.
/// \param params Taylor action parameters.
/// \return Approximation to `exp(t A) initial` and diagnostics.
template <uni20::RealOrComplex Scalar, typename TimeScalar, typename Vector,
          KrylovMatrixFreeOperator<Vector, Scalar> Ops>
  requires(std::constructible_from<uni20::make_real_t<Scalar>, TimeScalar> ||
           (uni20::Complex<Scalar> && std::constructible_from<Scalar, TimeScalar>))
TaylorExponentialResult<Scalar, Vector>
taylor_exponential_action(Ops& ops, Vector const& initial, TimeScalar time,
                          uni20::make_real_t<Scalar> operator_norm_bound,
                          TaylorExponentialParams<uni20::make_real_t<Scalar>> const& params = {})
{
  using Real = uni20::make_real_t<Scalar>;

  detail::validate_taylor_exponential_params(operator_norm_bound, params);
  validate_matrix_free_dimensions(ops, initial, "Taylor exponential");

  Scalar const time_scalar = static_cast<Scalar>(time);
  Real const time_abs = static_cast<Real>(detail::adl_abs(time_scalar));
  if (!detail::adl_isfinite(time_abs))
  {
    throw std::invalid_argument("Taylor exponential requires a finite time coefficient");
  }

  Real const tolerance = detail::effective_taylor_exponential_tolerance(params);
  Real const scaled_operator_norm = time_abs * operator_norm_bound;
  if (!detail::adl_isfinite(scaled_operator_norm))
  {
    throw std::overflow_error("Taylor exponential scaled operator norm overflowed");
  }
  int const scaling_steps = detail::taylor_scaling_steps(scaled_operator_norm, params);
  Real const step_norm_bound = scaled_operator_norm / static_cast<Real>(scaling_steps);
  Real const step_tolerance = tolerance / static_cast<Real>(scaling_steps);
  Scalar const step_time = time_scalar / Scalar{static_cast<Real>(scaling_steps)};

  Vector state = ops.allocate_like(initial);
  ops.copy(state, initial);

  TaylorExponentialDiagnostics<Real> diagnostics;
  diagnostics.scaling_steps = scaling_steps;
  diagnostics.tolerance = tolerance;
  diagnostics.operator_norm_bound = operator_norm_bound;
  diagnostics.step_norm_bound = step_norm_bound;

  if (time_abs == Real{} || operator_norm_bound == Real{} || norm_or_inner_product<Scalar>(ops, state) == Real{})
  {
    TaylorExponentialResult<Scalar, Vector> result{.action = std::move(state),
                                                   .estimated_error = Real{},
                                                   .matvec_count = 0,
                                                   .scaling_steps = scaling_steps,
                                                   .max_degree_used = 0,
                                                   .converged = true,
                                                   .diagnostics = std::nullopt};
    detail::attach_taylor_exponential_diagnostics(result, diagnostics, params.diagnostics);
    return result;
  }

  Vector step_result = ops.allocate_like(initial);
  Vector term = ops.allocate_like(initial);
  Vector scratch = ops.allocate_like(initial);

  bool all_steps_converged = true;
  Real estimated_error = Real{};
  int matvec_count = 0;
  int max_degree_used = 0;

  for (int step = 0; step < scaling_steps; ++step)
  {
    ops.copy(step_result, state);
    ops.copy(term, state);

    bool step_converged = false;
    Real final_tail_bound = uni20::numeric_limits<Real>::infinity();
    Real final_term_norm = uni20::numeric_limits<Real>::infinity();
    int degree_used = 0;

    for (int degree = 1; degree <= params.max_taylor_degree; ++degree)
    {
      ops.matvec(scratch, term);
      ++matvec_count;
      ops.scal(scratch, step_time / static_cast<Real>(degree));
      ops.copy(term, scratch);
      ops.axpy(step_result, Scalar{1}, term);

      final_term_norm = norm_or_inner_product<Scalar>(ops, term);
      final_tail_bound = detail::taylor_tail_bound(final_term_norm, step_norm_bound, degree);
      degree_used = degree;
      if (final_tail_bound <= step_tolerance)
      {
        step_converged = true;
        break;
      }
    }

    max_degree_used = std::max(max_degree_used, degree_used);
    diagnostics.final_step_degree = degree_used;
    diagnostics.final_tail_bound = final_tail_bound;
    diagnostics.final_term_norm = final_term_norm;
    estimated_error += detail::amplified_tail_bound(final_tail_bound, step_norm_bound, scaling_steps - step - 1);

    ops.copy(state, step_result);
    if (!step_converged)
    {
      all_steps_converged = false;
      if (params.throw_on_nonconvergence)
      {
        throw std::runtime_error("Taylor exponential did not satisfy the requested tolerance");
      }
      break;
    }
  }

  diagnostics.matvec_count = matvec_count;
  diagnostics.max_degree_used = max_degree_used;
  diagnostics.estimated_error = estimated_error;
  diagnostics.converged = all_steps_converged;

  TaylorExponentialResult<Scalar, Vector> result{.action = std::move(state),
                                                 .estimated_error = estimated_error,
                                                 .matvec_count = matvec_count,
                                                 .scaling_steps = scaling_steps,
                                                 .max_degree_used = max_degree_used,
                                                 .converged = all_steps_converged,
                                                 .diagnostics = std::nullopt};
  detail::attach_taylor_exponential_diagnostics(result, diagnostics, params.diagnostics);
  return result;
}

} // namespace uni20::krylov
