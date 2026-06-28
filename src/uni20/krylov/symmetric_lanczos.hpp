#pragma once

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/krylov/dense_subspace.hpp>
#include <uni20/krylov/matrix_free.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::krylov
{

/// \brief Ritz-value selection state used by symmetric implicit restart logic.
/// \tparam Scalar Real scalar type.
template <uni20::Real Scalar> struct SymmetricRestartSelection
{
    std::vector<std::size_t> wanted_indices;
    std::vector<std::size_t> shift_indices;
    std::vector<Scalar> shifts;
    std::vector<bool> wanted_converged;
    int converged_count = 0;
};

/// \brief Restart dimensions after implicit-restart edge-case adjustment.
/// \tparam Scalar Real scalar type.
template <uni20::Real Scalar> struct SymmetricRestartPlan
{
    std::size_t retained_count = 0;
    std::size_t shift_count = 0;
    std::size_t protected_zero_bound_count = 0;
    bool enlarged_for_partial_convergence = false;
};

/// \brief Result of applying implicit QR shifts to a symmetric tridiagonal projection.
/// \tparam Scalar Real scalar type.
template <uni20::Real Scalar> struct SymmetricQrShiftResult
{
    std::vector<Scalar> diagonal;
    std::vector<Scalar> subdiagonal;
    Matrix<Scalar> transform;
};

/// \brief Compressed symmetric Lanczos factorization after implicit restart.
/// \tparam Scalar Real scalar type.
/// \tparam Vector Opaque vector type.
template <uni20::Real Scalar, typename Vector> struct SymmetricCompressedLanczosFactorization
{
    std::vector<Vector> basis;
    Vector residual;
    std::vector<Scalar> diagonal;
    std::vector<Scalar> subdiagonal;
    Scalar residual_norm = Scalar{};
    Scalar residual_scale = Scalar{};
    Scalar trailing_coupling = Scalar{};
};

namespace detail
{
template <uni20::Real Scalar> struct LanczosOrthogonalizationDiagnostics
{
    Scalar max_reorthogonalization_correction = Scalar{};
    Scalar max_reorthogonalization_correction_ratio = Scalar{};
    int max_reorthogonalization_passes = 0;
};

template <uni20::Real Scalar>
void record_lanczos_orthogonalization_pass(LanczosOrthogonalizationDiagnostics<Scalar>* diagnostics,
                                           Scalar max_correction, Scalar reference_norm, int pass_count)
{
  if (diagnostics == nullptr)
  {
    return;
  }

  diagnostics->max_reorthogonalization_correction =
      std::max(diagnostics->max_reorthogonalization_correction, max_correction);
  if (reference_norm > Scalar{})
  {
    diagnostics->max_reorthogonalization_correction_ratio =
        std::max(diagnostics->max_reorthogonalization_correction_ratio, max_correction / reference_norm);
  }
  diagnostics->max_reorthogonalization_passes = std::max(diagnostics->max_reorthogonalization_passes, pass_count);
}

template <uni20::RealOrComplex Scalar, typename Vector, typename OpOps, typename BOps, typename MetricOps>
class SymmetricBMetricOps {
  public:
    SymmetricBMetricOps(OpOps& op_ops, BOps& b_ops, MetricOps& metric_ops)
        : op_ops_(op_ops), b_ops_(b_ops), metric_ops_(metric_ops)
    {}

    [[nodiscard]] std::size_t problem_dimension() const { return op_ops_.problem_dimension(); }

    [[nodiscard]] std::size_t vector_dimension(Vector const& x) const { return op_ops_.vector_dimension(x); }

    [[nodiscard]] Vector allocate_like(Vector const& x) { return op_ops_.allocate_like(x); }

    void copy(Vector& dst, Vector const& src) { op_ops_.copy(dst, src); }

    void axpy(Vector& y, Scalar alpha, Vector const& x) { op_ops_.axpy(y, alpha, x); }

    void scal(Vector& x, Scalar alpha) { op_ops_.scal(x, alpha); }

    void set_zero(Vector& x) { op_ops_.set_zero(x); }

    [[nodiscard]] Scalar inner_product(Vector const& x, Vector const& y)
    {
      if (!scratch_.has_value())
      {
        scratch_.emplace(b_ops_.allocate_like(y));
      }
      return metric_inner_product_or_apply<Scalar>(metric_ops_, b_ops_, op_ops_, x, y, *scratch_);
    }

    void matvec(Vector& y, Vector const& x) { op_ops_.matvec(y, x); }

  private:
    OpOps& op_ops_;
    BOps& b_ops_;
    MetricOps& metric_ops_;
    std::optional<Vector> scratch_;
};

template <uni20::Real Scalar> struct GivensRotation
{
    Scalar c = Scalar{1};
    Scalar s = Scalar{};
    Scalar r = Scalar{};
};

template <uni20::Real Scalar> GivensRotation<Scalar> lartg_like(Scalar f, Scalar g)
{
  if (g == Scalar{})
  {
    return GivensRotation<Scalar>{f < Scalar{} ? Scalar{-1} : Scalar{1}, Scalar{}, std::abs(f)};
  }
  if (f == Scalar{})
  {
    return GivensRotation<Scalar>{Scalar{}, g < Scalar{} ? Scalar{-1} : Scalar{1}, std::abs(g)};
  }

  Scalar const r = std::copysign(std::hypot(f, g), f);
  return GivensRotation<Scalar>{f / r, g / r, r};
}

template <uni20::Real Scalar>
void apply_givens_to_transform(Matrix<Scalar>& q, std::size_t lhs, std::size_t rhs, Scalar c, Scalar s)
{
  for (std::size_t row = 0; row < q.rows(); ++row)
  {
    Scalar const q_lhs = q(row, lhs);
    Scalar const q_rhs = q(row, rhs);
    q(row, lhs) = c * q_lhs + s * q_rhs;
    q(row, rhs) = -s * q_lhs + c * q_rhs;
  }
}

#if defined(__cpp_lib_constexpr_cmath)
template <uni20::Real Scalar> constexpr Scalar ritz_convergence_floor()
#else
template <uni20::Real Scalar> Scalar ritz_convergence_floor()
#endif
{
  return std::pow(uni20::numeric_limits<Scalar>::epsilon(), Scalar{2} / Scalar{3});
}

template <uni20::Real Scalar> bool symmetric_ritz_converged(Scalar value, Scalar residual_bound, Scalar tolerance)
{
  Scalar const eps23 = ritz_convergence_floor<Scalar>();
  return residual_bound <= tolerance * std::max(eps23, std::abs(value));
}

template <uni20::Real Scalar>
Scalar symmetric_breakdown_threshold(SymmetricEigenParams<Scalar> const& params, Scalar local_scale)
{
  Scalar const tolerance = params.breakdown_tolerance > Scalar{}
                               ? params.breakdown_tolerance
                               : Scalar{10} * uni20::numeric_limits<Scalar>::epsilon();
  return tolerance * std::max(Scalar{1}, local_scale);
}

template <uni20::Real Scalar>
bool symmetric_happy_breakdown(Scalar residual_norm, SymmetricEigenParams<Scalar> const& params, Scalar local_scale)
{
  return residual_norm <= symmetric_breakdown_threshold(params, local_scale);
}

template <uni20::RealOrComplex Scalar> uni20::make_real_t<Scalar> hermitian_projection_scalar(Scalar value)
{
  using Real = uni20::make_real_t<Scalar>;
  Real const real_part = static_cast<Real>(std::real(value));
  if constexpr (uni20::Complex<Scalar>)
  {
    Real const scale = std::max(Real{1}, std::abs(value));
    Real const tolerance = Real{100} * uni20::numeric_limits<Real>::epsilon() * scale;
    if (std::abs(std::imag(value)) > tolerance)
    {
      throw std::runtime_error("Hermitian Lanczos received a projected diagonal with non-negligible imaginary part");
    }
  }
  return real_part;
}

template <uni20::RealOrComplex Scalar, typename Vector, typename Ops>
uni20::make_real_t<Scalar>
orthogonalize_lanczos_residual(Ops& ops, std::vector<Vector> const& basis, Vector& residual,
                               LanczosOrthogonalizationDiagnostics<uni20::make_real_t<Scalar>>* diagnostics = nullptr)
{
  using Real = uni20::make_real_t<Scalar>;
  auto norm = [&]() { return norm_or_inner_product<Scalar>(ops, residual); };
  auto orthogonalize_once = [&]() {
    Real max_correction{};
    for (Vector const& basis_vector : basis)
    {
      Scalar const correction = ops.inner_product(basis_vector, residual);
      max_correction = std::max(max_correction, static_cast<Real>(std::abs(correction)));
      ops.axpy(residual, -correction, basis_vector);
    }
    return max_correction;
  };

  Real const original_norm = norm();
  Real const first_max_correction = orthogonalize_once();
  record_lanczos_orthogonalization_pass(diagnostics, first_max_correction, original_norm, 1);
  Real residual_norm = norm();
  if (original_norm == Real{} || residual_norm > Real{0.717} * original_norm)
  {
    return residual_norm;
  }

  Real const first_refined_norm = residual_norm;
  Real const second_max_correction = orthogonalize_once();
  record_lanczos_orthogonalization_pass(diagnostics, second_max_correction, first_refined_norm, 2);
  residual_norm = norm();
  if (first_refined_norm == Real{} || residual_norm <= Real{0.717} * first_refined_norm)
  {
    ops.set_zero(residual);
    return Real{};
  }
  return residual_norm;
}

enum class SymmetricResultPolicy
{
  Requested,
  ConvergedWanted
};

template <uni20::Real Scalar>
std::size_t effective_retained_ritz_count(SymmetricEigenParams<Scalar> const& params, std::size_t available_count)
{
  if (params.eigenvalue_count <= 0)
  {
    throw std::invalid_argument("symmetric restart selection requires a positive eigenvalue count");
  }
  if (params.spectrum == SpectrumPart::BothEnds && params.eigenvalue_count == 1)
  {
    throw std::invalid_argument("both-end symmetric selection requires eigenvalue_count > 1");
  }

  std::size_t const requested_count = static_cast<std::size_t>(params.eigenvalue_count);
  std::size_t const retained_count = params.retained_ritz_count > 0
                                         ? static_cast<std::size_t>(params.retained_ritz_count)
                                         : static_cast<std::size_t>(default_symmetric_retained_ritz_count(
                                               params.eigenvalue_count, static_cast<int>(available_count)));
  if (retained_count < requested_count)
  {
    throw std::invalid_argument("retained Ritz count must be at least the requested eigenvalue count");
  }
  if (retained_count > available_count)
  {
    throw std::invalid_argument("retained Ritz count cannot exceed the projected subspace size");
  }
  return retained_count;
}

template <uni20::Real Scalar>
std::vector<std::size_t> ordered_symmetric_ritz_indices(std::vector<Scalar> const& values, SpectrumPart spectrum)
{
  std::vector<std::size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});

  switch (spectrum)
  {
    case SpectrumPart::LargestMagnitude:
      std::ranges::sort(
          indices, [&](std::size_t lhs, std::size_t rhs) { return std::abs(values[lhs]) < std::abs(values[rhs]); });
      break;
    case SpectrumPart::SmallestMagnitude:
      std::ranges::sort(
          indices, [&](std::size_t lhs, std::size_t rhs) { return std::abs(values[lhs]) > std::abs(values[rhs]); });
      break;
    case SpectrumPart::LargestAlgebraic:
    case SpectrumPart::LargestReal:
      std::ranges::sort(indices, [&](std::size_t lhs, std::size_t rhs) { return values[lhs] < values[rhs]; });
      break;
    case SpectrumPart::SmallestAlgebraic:
    case SpectrumPart::SmallestReal:
      std::ranges::sort(indices, [&](std::size_t lhs, std::size_t rhs) { return values[lhs] > values[rhs]; });
      break;
    case SpectrumPart::BothEnds:
      std::ranges::sort(indices, [&](std::size_t lhs, std::size_t rhs) { return values[lhs] < values[rhs]; });
      break;
    case SpectrumPart::LargestImaginary:
    case SpectrumPart::SmallestImaginary:
      throw std::invalid_argument("imaginary spectrum selectors are not valid for symmetric/Hermitian Lanczos");
  }
  return indices;
}

template <uni20::Real Scalar>
std::vector<std::size_t> select_symmetric_ritz_indices(std::vector<Scalar> const& values,
                                                       SymmetricEigenParams<Scalar> const& params)
{
  if (params.spectrum == SpectrumPart::BothEnds && params.eigenvalue_count == 1)
  {
    throw std::invalid_argument("both-end symmetric selection requires eigenvalue_count > 1");
  }

  std::vector<std::size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});

  switch (params.spectrum)
  {
    case SpectrumPart::LargestMagnitude:
      std::ranges::sort(
          indices, [&](std::size_t lhs, std::size_t rhs) { return std::abs(values[lhs]) > std::abs(values[rhs]); });
      break;
    case SpectrumPart::SmallestMagnitude:
      std::ranges::sort(
          indices, [&](std::size_t lhs, std::size_t rhs) { return std::abs(values[lhs]) < std::abs(values[rhs]); });
      break;
    case SpectrumPart::LargestAlgebraic:
    case SpectrumPart::LargestReal:
      std::ranges::sort(indices, [&](std::size_t lhs, std::size_t rhs) { return values[lhs] > values[rhs]; });
      break;
    case SpectrumPart::SmallestAlgebraic:
    case SpectrumPart::SmallestReal:
      std::ranges::sort(indices, [&](std::size_t lhs, std::size_t rhs) { return values[lhs] < values[rhs]; });
      break;
    case SpectrumPart::BothEnds:
      std::ranges::sort(indices, [&](std::size_t lhs, std::size_t rhs) { return values[lhs] < values[rhs]; });
      {
        std::size_t const requested_count =
            std::min<std::size_t>(static_cast<std::size_t>(params.eigenvalue_count), indices.size());
        std::size_t const low_count = requested_count / 2;
        std::size_t const high_count = requested_count - low_count;
        std::vector<std::size_t> both_end_indices;
        both_end_indices.reserve(requested_count);
        both_end_indices.insert(both_end_indices.end(), indices.begin(),
                                indices.begin() + static_cast<std::ptrdiff_t>(low_count));
        both_end_indices.insert(both_end_indices.end(), indices.end() - static_cast<std::ptrdiff_t>(high_count),
                                indices.end());
        return both_end_indices;
      }
      break;
    case SpectrumPart::LargestImaginary:
    case SpectrumPart::SmallestImaginary:
      throw std::invalid_argument("imaginary spectrum selectors are not valid for symmetric/Hermitian Lanczos");
  }

  std::size_t const count = std::min<std::size_t>(static_cast<std::size_t>(params.eigenvalue_count), indices.size());
  indices.resize(count);
  return indices;
}

template <uni20::Real Scalar>
std::vector<Scalar> symmetric_ritz_residual_bounds(Scalar residual_norm,
                                                   TridiagonalEigensystem<Scalar> const& projected)
{
  std::vector<Scalar> residual_bounds(projected.eigenvalues.size(), Scalar{});
  for (std::size_t i = 0; i < residual_bounds.size(); ++i)
  {
    residual_bounds[i] = residual_norm * std::abs(projected.eigenvectors(projected.eigenvectors.rows() - 1, i));
  }
  return residual_bounds;
}

template <uni20::Real Scalar>
void record_symmetric_final_diagnostics(SymmetricLanczosDiagnostics<Scalar>& diagnostics,
                                        TridiagonalEigensystem<Scalar> const& projected,
                                        std::vector<Scalar> const& residual_bounds,
                                        SymmetricEigenParams<Scalar> const& params, Scalar residual_norm)
{
  diagnostics.final_projected_dimension = static_cast<int>(projected.eigenvalues.size());
  diagnostics.final_residual_norm = residual_norm;
  diagnostics.final_ritz_values = projected.eigenvalues;
  diagnostics.final_ritz_bounds = residual_bounds;
  diagnostics.final_selected_indices = select_symmetric_ritz_indices(projected.eigenvalues, params);
}

template <uni20::Real Scalar>
void record_symmetric_restart_cycle(SymmetricLanczosDiagnostics<Scalar>& diagnostics, int cycle,
                                    TridiagonalEigensystem<Scalar> const& projected,
                                    std::vector<Scalar> const& residual_bounds,
                                    SymmetricRestartPlan<Scalar> const& restart_plan,
                                    SymmetricRestartSelection<Scalar> const& selection, Scalar residual_norm)
{
  diagnostics.restart_count =
      std::max(diagnostics.restart_count, static_cast<int>(diagnostics.restart_cycles.size()) + 1);
  diagnostics.restart_cycles.push_back(SymmetricRestartCycleDiagnostics<Scalar>{
      .cycle = cycle,
      .converged_count = selection.converged_count,
      .projected_dimension = projected.eigenvalues.size(),
      .retained_count = restart_plan.retained_count,
      .shift_count = restart_plan.shift_count,
      .protected_zero_bound_count = restart_plan.protected_zero_bound_count,
      .enlarged_for_partial_convergence = restart_plan.enlarged_for_partial_convergence,
      .residual_norm = residual_norm,
      .ritz_values = projected.eigenvalues,
      .ritz_bounds = residual_bounds,
      .shifts = selection.shifts,
      .wanted_indices = selection.wanted_indices,
      .shift_indices = selection.shift_indices,
      .wanted_converged = selection.wanted_converged,
  });
}

template <uni20::LapackRealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
EigenResult<uni20::make_real_t<Scalar>, Vector>
make_symmetric_lanczos_result(Ops& ops, std::vector<Vector> const& basis, uni20::make_real_t<Scalar> residual_norm,
                              TridiagonalEigensystem<uni20::make_real_t<Scalar>>& projected,
                              SymmetricEigenParams<uni20::make_real_t<Scalar>> const& params, int iteration_count,
                              int matvec_count, SymmetricResultPolicy policy = SymmetricResultPolicy::Requested)
{
  using Real = uni20::make_real_t<Scalar>;
  Real const tolerance =
      params.tolerance > Real{} ? params.tolerance : Real{100} * uni20::numeric_limits<Real>::epsilon();

  EigenResult<Real, Vector> result;
  result.iteration_count = iteration_count;
  result.matvec_count = matvec_count;

  std::vector<std::size_t> selected = select_symmetric_ritz_indices(projected.eigenvalues, params);
  if (policy == SymmetricResultPolicy::ConvergedWanted)
  {
    std::erase_if(selected, [&](std::size_t ritz_index) {
      Real const residual_bound =
          residual_norm * std::abs(projected.eigenvectors(projected.eigenvectors.rows() - 1, ritz_index));
      return !symmetric_ritz_converged(projected.eigenvalues[ritz_index], residual_bound, tolerance);
    });
  }

  result.eigenvalues.reserve(selected.size());
  result.residual_bounds.reserve(selected.size());
  result.eigenvectors.reserve(params.compute_eigenvectors ? selected.size() : 0);
  for (std::size_t const ritz_index : selected)
  {
    result.eigenvalues.push_back(projected.eigenvalues[ritz_index]);
    Real const residual_bound =
        residual_norm * std::abs(projected.eigenvectors(projected.eigenvectors.rows() - 1, ritz_index));
    result.residual_bounds.push_back(residual_bound);
    if (symmetric_ritz_converged(projected.eigenvalues[ritz_index], residual_bound, tolerance))
    {
      ++result.converged_count;
    }
    if (params.compute_eigenvectors)
    {
      Vector vector = ops.allocate_like(basis.front());
      ops.set_zero(vector);
      for (std::size_t row = 0; row < basis.size(); ++row)
      {
        ops.axpy(vector, static_cast<Scalar>(projected.eigenvectors(row, ritz_index)), basis[row]);
      }
      result.eigenvectors.push_back(std::move(vector));
    }
  }

  result.status = result.converged_count == static_cast<int>(result.eigenvalues.size()) ? 0 : 1;
  if (policy == SymmetricResultPolicy::ConvergedWanted)
  {
    result.status = 0;
  }
  return result;
}
} // namespace detail

/// \brief Select wanted Ritz values and exact-shift candidates for symmetric restart.
///
/// \details Ritz values are ordered so the retained values occupy the final
///          `n_keep` positions, and the unwanted shift candidates are then
///          ordered by decreasing Ritz estimate. When `retained_ritz_count` is
///          not set, `n_keep` defaults to `nev` and the behavior is the minimal
///          implicit restart. The filtering step itself is separate.
/// \tparam Scalar Real vector-field scalar or complex vector-field scalar.
/// \param values Projected Ritz values.
/// \param residual_bounds Ritz residual estimates with the same length as \p values.
/// \param params Solver parameters defining `nev`, optional `n_keep`, selector,
///        and tolerance.
/// \param shift_count Number of exact shifts to select from the unwanted values.
/// \return Restart-selection state.
template <uni20::Real Scalar>
SymmetricRestartSelection<Scalar>
select_symmetric_restart(std::vector<Scalar> const& values, std::vector<Scalar> const& residual_bounds,
                         SymmetricEigenParams<Scalar> const& params, std::size_t shift_count)
{
  if (values.size() != residual_bounds.size())
  {
    throw std::invalid_argument("symmetric restart selection received inconsistent Ritz arrays");
  }
  std::size_t const wanted_count = detail::effective_retained_ritz_count(params, values.size());
  std::vector<std::size_t> ordered = detail::ordered_symmetric_ritz_indices(values, params.spectrum);
  std::vector<std::size_t> wanted_indices;
  std::vector<std::size_t> unwanted_indices;
  if (params.spectrum == SpectrumPart::BothEnds)
  {
    std::size_t const low_count = wanted_count / 2;
    std::size_t const high_count = wanted_count - low_count;
    wanted_indices.reserve(wanted_count);
    wanted_indices.insert(wanted_indices.end(), ordered.begin(),
                          ordered.begin() + static_cast<std::ptrdiff_t>(low_count));
    wanted_indices.insert(wanted_indices.end(), ordered.end() - static_cast<std::ptrdiff_t>(high_count), ordered.end());
    unwanted_indices.assign(ordered.begin() + static_cast<std::ptrdiff_t>(low_count),
                            ordered.end() - static_cast<std::ptrdiff_t>(high_count));
  }
  else
  {
    wanted_indices.assign(ordered.end() - static_cast<std::ptrdiff_t>(wanted_count), ordered.end());
    unwanted_indices.assign(ordered.begin(), ordered.end() - static_cast<std::ptrdiff_t>(wanted_count));
  }
  shift_count = std::min(shift_count, unwanted_indices.size());

  Scalar const tolerance =
      params.tolerance > Scalar{} ? params.tolerance : Scalar{100} * uni20::numeric_limits<Scalar>::epsilon();

  SymmetricRestartSelection<Scalar> selection;
  selection.wanted_indices = std::move(wanted_indices);
  selection.wanted_converged.reserve(selection.wanted_indices.size());
  for (std::size_t const index : selection.wanted_indices)
  {
    bool const converged = detail::symmetric_ritz_converged(values[index], residual_bounds[index], tolerance);
    selection.wanted_converged.push_back(converged);
    if (converged)
    {
      ++selection.converged_count;
    }
  }

  selection.shift_indices.assign(unwanted_indices.begin(),
                                 unwanted_indices.begin() + static_cast<std::ptrdiff_t>(shift_count));
  std::ranges::sort(selection.shift_indices, [&](std::size_t lhs, std::size_t rhs) {
    return std::abs(residual_bounds[lhs]) > std::abs(residual_bounds[rhs]);
  });
  selection.shifts.reserve(selection.shift_indices.size());
  for (std::size_t const index : selection.shift_indices)
  {
    selection.shifts.push_back(values[index]);
  }
  return selection;
}

/// \brief Adjust restart dimensions for implicit-restart edge cases.
///
/// \details The restart protects unwanted Ritz values whose residual estimates
///          are exactly zero, because they indicate a split-off leading block
///          that should not be removed by shifting. It also increases the
///          retained block when some, but not all, requested Ritz values have
///          converged, which prevents stagnation from repeatedly applying too
///          many shifts.
/// \tparam Scalar Real scalar type.
/// \param params Solver parameters defining `nev`, optional `n_keep`, and selector.
/// \param values Projected Ritz values.
/// \param residual_bounds Ritz residual estimates with the same length as \p values.
/// \param converged_count Number of currently converged requested Ritz values.
/// \return Adjusted restart dimensions.
template <uni20::Real Scalar>
SymmetricRestartPlan<Scalar> plan_symmetric_restart(SymmetricEigenParams<Scalar> const& params,
                                                    std::vector<Scalar> const& values,
                                                    std::vector<Scalar> const& residual_bounds, int converged_count)
{
  if (values.size() != residual_bounds.size())
  {
    throw std::invalid_argument("symmetric restart planning received inconsistent Ritz arrays");
  }

  std::size_t const order = values.size();
  std::size_t retained_count = detail::effective_retained_ritz_count(params, order);
  if (retained_count >= order)
  {
    throw std::invalid_argument("symmetric restart planning requires retained_count < projected size");
  }

  std::size_t shift_count = order - retained_count;
  SymmetricRestartPlan<Scalar> plan;
  SymmetricRestartSelection<Scalar> selection = select_symmetric_restart(values, residual_bounds, params, shift_count);

  for (std::size_t const index : selection.shift_indices)
  {
    if (residual_bounds[index] == Scalar{})
    {
      ++plan.protected_zero_bound_count;
    }
  }

  if (plan.protected_zero_bound_count > 0)
  {
    retained_count = std::min(order, retained_count + plan.protected_zero_bound_count);
    shift_count = order - retained_count;
  }

  if (converged_count < params.eigenvalue_count && shift_count > 0)
  {
    std::size_t const old_retained_count = retained_count;
    retained_count += std::min<std::size_t>(static_cast<std::size_t>(converged_count), shift_count / 2);
    if (retained_count == 1 && order >= 6)
    {
      retained_count = order / 2;
    }
    else if (retained_count == 1 && order > 2)
    {
      retained_count = 2;
    }
    retained_count = std::min(retained_count, order);
    shift_count = order - retained_count;
    plan.enlarged_for_partial_convergence = retained_count > old_retained_count;
  }

  plan.retained_count = retained_count;
  plan.shift_count = shift_count;
  return plan;
}

/// \brief Apply implicit shifted QR sweeps to a symmetric tridiagonal matrix.
///
/// \details Each shift is applied by Givens rotations that chase the QR bulge
///          through the tridiagonal, while the accumulated orthogonal transform
///          is returned for basis compression. Updating the opaque Lanczos
///          basis and residual is handled by a later restart slice.
/// \tparam Scalar Real scalar type.
/// \param diagonal Main diagonal of the tridiagonal matrix.
/// \param subdiagonal Subdiagonal entries, length `diagonal.size() - 1`.
/// \param shifts Exact shifts to apply.
/// \return Updated tridiagonal coefficients and accumulated orthogonal transform.
template <uni20::Real Scalar>
SymmetricQrShiftResult<Scalar> apply_symmetric_qr_shifts(std::vector<Scalar> diagonal, std::vector<Scalar> subdiagonal,
                                                         std::vector<Scalar> const& shifts)
{
  std::size_t const order = diagonal.size();
  if (subdiagonal.size() + 1 != order && !(order == 0 && subdiagonal.empty()))
  {
    throw std::invalid_argument("symmetric QR shifts received inconsistent tridiagonal sizes");
  }

  SymmetricQrShiftResult<Scalar> result;
  result.transform = Matrix<Scalar>(order, order);
  laset(result.transform, Scalar{1}, Scalar{}, MatrixFill::All);
  if (order == 0 || shifts.empty())
  {
    result.diagonal = std::move(diagonal);
    result.subdiagonal = std::move(subdiagonal);
    return result;
  }

  Scalar const epsilon = uni20::numeric_limits<Scalar>::epsilon();
  std::size_t itop = 0;
  for (Scalar const shift : shifts)
  {
    std::size_t istart = itop;
    while (istart + 1 < order)
    {
      std::size_t iend = order - 1;
      for (std::size_t i = istart; i + 1 < order; ++i)
      {
        Scalar const big = std::abs(diagonal[i]) + std::abs(diagonal[i + 1]);
        if (std::abs(subdiagonal[i]) <= epsilon * big)
        {
          subdiagonal[i] = Scalar{};
          iend = i;
          break;
        }
      }

      if (istart < iend)
      {
        Scalar f = diagonal[istart] - shift;
        Scalar g = subdiagonal[istart];
        auto rotation = detail::lartg_like(f, g);
        Scalar c = rotation.c;
        Scalar s = rotation.s;

        Scalar a1 = c * diagonal[istart] + s * subdiagonal[istart];
        Scalar a2 = c * subdiagonal[istart] + s * diagonal[istart + 1];
        Scalar a4 = c * diagonal[istart + 1] - s * subdiagonal[istart];
        Scalar a3 = c * subdiagonal[istart] - s * diagonal[istart];
        diagonal[istart] = c * a1 + s * a2;
        diagonal[istart + 1] = c * a4 - s * a3;
        subdiagonal[istart] = c * a3 + s * a4;
        detail::apply_givens_to_transform(result.transform, istart, istart + 1, c, s);

        for (std::size_t i = istart + 1; i < iend; ++i)
        {
          f = subdiagonal[i - 1];
          g = s * subdiagonal[i];
          subdiagonal[i] *= c;
          rotation = detail::lartg_like(f, g);
          c = rotation.c;
          s = rotation.s;
          Scalar r = rotation.r;
          if (r < Scalar{})
          {
            r = -r;
            c = -c;
            s = -s;
          }

          subdiagonal[i - 1] = r;
          a1 = c * diagonal[i] + s * subdiagonal[i];
          a2 = c * subdiagonal[i] + s * diagonal[i + 1];
          a3 = c * subdiagonal[i] - s * diagonal[i];
          a4 = c * diagonal[i + 1] - s * subdiagonal[i];
          diagonal[i] = c * a1 + s * a2;
          diagonal[i + 1] = c * a4 - s * a3;
          subdiagonal[i] = c * a3 + s * a4;
          detail::apply_givens_to_transform(result.transform, i, i + 1, c, s);
        }
      }

      if (iend > 0 && subdiagonal[iend - 1] < Scalar{})
      {
        subdiagonal[iend - 1] = -subdiagonal[iend - 1];
        for (std::size_t row = 0; row < result.transform.rows(); ++row)
        {
          result.transform(row, iend) = -result.transform(row, iend);
        }
      }

      istart = iend + 1;
    }

    for (std::size_t i = itop; i + 1 < order; ++i)
    {
      if (subdiagonal[i] > Scalar{})
      {
        break;
      }
      itop = i + 1;
    }
  }

  for (std::size_t i = itop; i + 1 < order; ++i)
  {
    Scalar const big = std::abs(diagonal[i]) + std::abs(diagonal[i + 1]);
    if (std::abs(subdiagonal[i]) <= epsilon * big)
    {
      subdiagonal[i] = Scalar{};
    }
  }

  result.diagonal = std::move(diagonal);
  result.subdiagonal = std::move(subdiagonal);
  return result;
}

/// \brief Compress a Lanczos factorization with the accumulated restart transform.
///
/// \details Given a factorization of order `kplusp`, it keeps the first
///          `retained_count` columns of `V Q`, truncates the transformed
///          tridiagonal projection, and updates the residual as
///          `r <- sigma_k r + beta_k (V Q)[:, k + 1]`, where
///          `sigma_k = Q(kplusp, k)` and `beta_k = T(k + 1, k)`.
///          All application-space vector work goes through the matrix-free
///          operation object.
/// \tparam Scalar Real scalar type.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free vector operation provider.
/// \param ops Matrix-free operation object.
/// \param basis Original Lanczos basis vectors.
/// \param residual Original residual vector.
/// \param shifted Shifted projected factorization and accumulated transform.
/// \param retained_count Number of transformed basis vectors to keep.
/// \return Compressed factorization ready for renewed basis expansion.
template <uni20::LapackRealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
SymmetricCompressedLanczosFactorization<uni20::make_real_t<Scalar>, Vector>
compress_symmetric_lanczos_restart(Ops& ops, std::vector<Vector> const& basis, Vector const& residual,
                                   SymmetricQrShiftResult<uni20::make_real_t<Scalar>> const& shifted,
                                   std::size_t retained_count)
{
  using Real = uni20::make_real_t<Scalar>;
  std::size_t const order = basis.size();
  if (order == 0)
  {
    throw std::invalid_argument("symmetric restart compression requires a nonempty basis");
  }
  if (shifted.diagonal.size() != order || shifted.subdiagonal.size() + 1 != order)
  {
    throw std::invalid_argument("symmetric restart compression received inconsistent tridiagonal sizes");
  }
  if (shifted.transform.rows() != order || shifted.transform.cols() != order)
  {
    throw std::invalid_argument("symmetric restart compression received an inconsistent transform size");
  }
  if (retained_count == 0 || retained_count > order)
  {
    throw std::invalid_argument("symmetric restart compression received an invalid retained count");
  }

  auto transformed_column = [&](std::size_t column) {
    Vector vector = ops.allocate_like(basis.front());
    ops.set_zero(vector);
    for (std::size_t row = 0; row < order; ++row)
    {
      Real const coefficient = shifted.transform(row, column);
      if (coefficient != Real{})
      {
        ops.axpy(vector, static_cast<Scalar>(coefficient), basis[row]);
      }
    }
    return vector;
  };

  SymmetricCompressedLanczosFactorization<Real, Vector> result{
      .basis = {},
      .residual = ops.allocate_like(residual),
      .diagonal = {},
      .subdiagonal = {},
      .residual_norm = Real{},
      .residual_scale = shifted.transform(order - 1, retained_count - 1),
      .trailing_coupling = retained_count < order ? shifted.subdiagonal[retained_count - 1] : Real{}};

  result.basis.reserve(retained_count);
  for (std::size_t column = 0; column < retained_count; ++column)
  {
    result.basis.push_back(transformed_column(column));
  }

  result.diagonal.assign(shifted.diagonal.begin(),
                         shifted.diagonal.begin() + static_cast<std::ptrdiff_t>(retained_count));
  if (retained_count > 1)
  {
    result.subdiagonal.assign(shifted.subdiagonal.begin(),
                              shifted.subdiagonal.begin() + static_cast<std::ptrdiff_t>(retained_count - 1));
  }

  ops.copy(result.residual, residual);
  ops.scal(result.residual, static_cast<Scalar>(result.residual_scale));
  if (result.trailing_coupling != Real{})
  {
    Vector next_vector = transformed_column(retained_count);
    ops.axpy(result.residual, static_cast<Scalar>(result.trailing_coupling), next_vector);
  }

  result.residual_norm = norm_or_inner_product<Scalar>(ops, result.residual);
  return result;
}

/// \brief Run the native symmetric/Hermitian Lanczos full-projection solver.
///
/// \details This implementation keeps the vector storage opaque and
///          communicates with the application only through the matrix-free
///          operation object. The current implementation uses full
///          reorthogonalization for deterministic, robust basis construction.
/// \tparam Scalar Real or complex vector-field scalar type with real LAPACK coverage.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free vector operation provider.
/// \param ops Matrix-free operation object.
/// \param initial Initial residual/start vector.
/// \param params Solver parameters.
/// \return Ritz values and optional Ritz vectors from the Lanczos projection.
template <uni20::LapackRealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
EigenResult<uni20::make_real_t<Scalar>, Vector>
symmetric_lanczos_standard(Ops& ops, Vector const& initial,
                           SymmetricEigenParams<uni20::make_real_t<Scalar>> const& params)
{
  using Real = uni20::make_real_t<Scalar>;
  if (params.eigenvalue_count <= 0)
  {
    throw std::invalid_argument("symmetric Lanczos requires a positive eigenvalue count");
  }

  std::size_t const problem_dimension = validate_matrix_free_dimensions(ops, initial, "symmetric Lanczos");
  int const basis_limit = effective_symmetric_krylov_dimension(params, problem_dimension);
  if (basis_limit <= 0)
  {
    throw std::invalid_argument("symmetric Lanczos requires a positive Krylov dimension");
  }
  if (basis_limit > static_cast<int>(problem_dimension))
  {
    throw std::invalid_argument("symmetric Lanczos requires krylov_dimension <= problem dimension");
  }
  if (params.eigenvalue_count > basis_limit)
  {
    throw std::invalid_argument("symmetric Lanczos requires eigenvalue_count <= krylov_dimension");
  }

  EigenResult<Real, Vector> result;
  std::vector<Vector> basis;
  std::vector<Real> diagonal;
  std::vector<Real> subdiagonal;
  Real final_residual_norm{};
  basis.reserve(static_cast<std::size_t>(basis_limit));
  diagonal.reserve(static_cast<std::size_t>(basis_limit));
  subdiagonal.reserve(static_cast<std::size_t>(basis_limit > 0 ? basis_limit - 1 : 0));

  Vector start = ops.allocate_like(initial);
  ops.copy(start, initial);
  Real const start_norm = norm_or_inner_product<Scalar>(ops, start);
  if (start_norm <= Real{})
  {
    throw std::invalid_argument("symmetric Lanczos initial vector has zero norm");
  }
  ops.scal(start, Scalar{1} / static_cast<Scalar>(start_norm));
  basis.push_back(std::move(start));

  for (int step = 0; step < basis_limit; ++step)
  {
    Vector w = ops.allocate_like(basis[static_cast<std::size_t>(step)]);
    ops.matvec(w, basis[static_cast<std::size_t>(step)]);
    ++result.matvec_count;

    if (step > 0)
    {
      ops.axpy(w, -static_cast<Scalar>(subdiagonal[static_cast<std::size_t>(step - 1)]),
               basis[static_cast<std::size_t>(step - 1)]);
    }

    Real const alpha = detail::hermitian_projection_scalar(ops.inner_product(basis[static_cast<std::size_t>(step)], w));
    diagonal.push_back(alpha);
    ops.axpy(w, -static_cast<Scalar>(alpha), basis[static_cast<std::size_t>(step)]);

    Real const beta = detail::orthogonalize_lanczos_residual<Scalar>(ops, basis, w);
    Real const previous_beta = step > 0 ? subdiagonal[static_cast<std::size_t>(step - 1)] : Real{};
    Real const breakdown_scale = std::max({Real{1}, std::abs(alpha), std::abs(previous_beta)});
    if (step + 1 == basis_limit || detail::symmetric_happy_breakdown(beta, params, breakdown_scale))
    {
      final_residual_norm = beta;
      break;
    }

    ops.scal(w, Scalar{1} / static_cast<Scalar>(beta));
    subdiagonal.push_back(beta);
    basis.push_back(std::move(w));
  }

  result.iteration_count = static_cast<int>(diagonal.size());
  auto projected = symmetric_tridiagonal_eigensystem(std::move(diagonal), std::move(subdiagonal), true);
  auto final_result = detail::make_symmetric_lanczos_result<Scalar>(ops, basis, final_residual_norm, projected, params,
                                                                    result.iteration_count, result.matvec_count);
  if (params.diagnostics != KrylovDiagnosticsLevel::None)
  {
    SymmetricLanczosDiagnostics<Real> diagnostics;
    diagnostics.op_count = result.matvec_count;
    std::vector<Real> const residual_bounds = detail::symmetric_ritz_residual_bounds(final_residual_norm, projected);
    detail::record_symmetric_final_diagnostics(diagnostics, projected, residual_bounds, params, final_residual_norm);
    final_result.diagnostics = std::move(diagnostics);
  }
  return final_result;
}

/// \brief Run a native symmetric/Hermitian Lanczos solve with implicit restarts.
///
/// \details This expands a Lanczos factorization to `ncv`, computes Ritz values
///          and residual bounds, applies exact shifts to the projected
///          tridiagonal, compresses the opaque basis through the accumulated
///          transform, and resumes expansion from the updated residual.
/// \tparam Scalar Real or complex vector-field scalar type with real LAPACK coverage.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free vector operation provider.
/// \param ops Matrix-free operation object.
/// \param initial Initial residual/start vector.
/// \param params Solver parameters.
/// \return Ritz values and optional Ritz vectors from the final projection.
template <uni20::LapackRealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
EigenResult<uni20::make_real_t<Scalar>, Vector>
symmetric_lanczos_restarted_standard(Ops& ops, Vector const& initial,
                                     SymmetricEigenParams<uni20::make_real_t<Scalar>> const& params)
{
  using Real = uni20::make_real_t<Scalar>;
  if (params.eigenvalue_count <= 0)
  {
    throw std::invalid_argument("restarted symmetric Lanczos requires a positive eigenvalue count");
  }
  if (params.max_iterations <= 0)
  {
    throw std::invalid_argument("restarted symmetric Lanczos requires a positive iteration limit");
  }

  std::size_t const problem_dimension = validate_matrix_free_dimensions(ops, initial, "restarted symmetric Lanczos");
  int const basis_limit = effective_symmetric_krylov_dimension(params, problem_dimension);
  if (basis_limit <= params.eigenvalue_count)
  {
    throw std::invalid_argument("restarted symmetric Lanczos requires eigenvalue_count < krylov_dimension");
  }
  if (basis_limit > static_cast<int>(problem_dimension))
  {
    throw std::invalid_argument("restarted symmetric Lanczos requires krylov_dimension <= problem dimension");
  }

  std::size_t const ncv = static_cast<std::size_t>(basis_limit);
  std::size_t const retained_count =
      static_cast<std::size_t>(effective_symmetric_retained_ritz_count(params, basis_limit));
  if (retained_count >= ncv)
  {
    throw std::invalid_argument("restarted symmetric Lanczos requires retained_ritz_count < krylov_dimension");
  }

  std::vector<Vector> basis;
  std::vector<Real> diagonal;
  std::vector<Real> subdiagonal;
  basis.reserve(ncv);
  diagonal.reserve(ncv);
  subdiagonal.reserve(ncv - 1);

  Vector start = ops.allocate_like(initial);
  ops.copy(start, initial);
  Real const start_norm = norm_or_inner_product<Scalar>(ops, start);
  if (start_norm <= Real{})
  {
    throw std::invalid_argument("restarted symmetric Lanczos initial vector has zero norm");
  }
  ops.scal(start, Scalar{1} / static_cast<Scalar>(start_norm));
  basis.push_back(std::move(start));

  int matvec_count = 0;
  int restart_cycle_count = 0;
  EigenResult<Real, Vector> last_result;
  EigenResult<Real, Vector> last_partial_exit_result;
  std::optional<SymmetricLanczosDiagnostics<Real>> diagnostics;
  if (params.diagnostics != KrylovDiagnosticsLevel::None)
  {
    diagnostics.emplace();
  }

  auto finish = [&](EigenResult<Real, Vector> result) {
    if (diagnostics.has_value())
    {
      diagnostics->op_count = matvec_count;
      result.diagnostics = *diagnostics;
    }
    return result;
  };

  for (int cycle = 0; cycle < params.max_iterations; ++cycle)
  {
    restart_cycle_count = cycle + 1;
    Real residual_norm{};
    bool cycle_happy_breakdown = false;
    Vector residual = ops.allocate_like(basis.front());
    ops.set_zero(residual);

    while (diagonal.size() < ncv)
    {
      std::size_t const step = diagonal.size();
      if (basis.size() != step + 1)
      {
        throw std::logic_error("restarted symmetric Lanczos basis/projection state is inconsistent");
      }

      Vector w = ops.allocate_like(basis[step]);
      ops.matvec(w, basis[step]);
      ++matvec_count;

      if (step > 0)
      {
        ops.axpy(w, -static_cast<Scalar>(subdiagonal[step - 1]), basis[step - 1]);
      }

      Real const alpha = detail::hermitian_projection_scalar(ops.inner_product(basis[step], w));
      diagonal.push_back(alpha);
      ops.axpy(w, -static_cast<Scalar>(alpha), basis[step]);

      Real const beta = detail::orthogonalize_lanczos_residual<Scalar>(ops, basis, w);
      Real const previous_beta = step > 0 ? subdiagonal[step - 1] : Real{};
      Real const breakdown_scale = std::max({Real{1}, std::abs(alpha), std::abs(previous_beta)});
      cycle_happy_breakdown = detail::symmetric_happy_breakdown(beta, params, breakdown_scale);
      residual = std::move(w);
      residual_norm = beta;
      if (diagonal.size() == ncv || cycle_happy_breakdown)
      {
        break;
      }

      ops.scal(residual, Scalar{1} / static_cast<Scalar>(beta));
      subdiagonal.push_back(beta);
      basis.push_back(std::move(residual));
    }

    auto projected_for_result = symmetric_tridiagonal_eigensystem(diagonal, subdiagonal, true);
    std::vector<Real> residual_bounds = detail::symmetric_ritz_residual_bounds(residual_norm, projected_for_result);
    if (diagnostics.has_value())
    {
      detail::record_symmetric_final_diagnostics(*diagnostics, projected_for_result, residual_bounds, params,
                                                 residual_norm);
    }
    last_result = detail::make_symmetric_lanczos_result<Scalar>(ops, basis, residual_norm, projected_for_result, params,
                                                                restart_cycle_count, matvec_count);
    last_partial_exit_result = detail::make_symmetric_lanczos_result<Scalar>(
        ops, basis, residual_norm, projected_for_result, params, restart_cycle_count, matvec_count,
        detail::SymmetricResultPolicy::ConvergedWanted);
    if (last_result.status == 0 || cycle_happy_breakdown || diagonal.size() < ncv)
    {
      return finish(std::move(last_result));
    }

    SymmetricRestartPlan<Real> const restart_plan =
        plan_symmetric_restart(params, projected_for_result.eigenvalues, residual_bounds, last_result.converged_count);
    SymmetricEigenParams<Real> restart_params = params;
    restart_params.retained_ritz_count = static_cast<int>(restart_plan.retained_count);
    SymmetricRestartSelection<Real> const selection = select_symmetric_restart(
        projected_for_result.eigenvalues, residual_bounds, restart_params, restart_plan.shift_count);
    if (diagnostics.has_value() && params.diagnostics == KrylovDiagnosticsLevel::Full)
    {
      detail::record_symmetric_restart_cycle(*diagnostics, cycle, projected_for_result, residual_bounds, restart_plan,
                                             selection, residual_norm);
    }
    if (selection.shifts.empty())
    {
      last_partial_exit_result.status = last_result.converged_count < params.eigenvalue_count ? 2 : 0;
      return finish(std::move(last_partial_exit_result));
    }

    auto shifted = apply_symmetric_qr_shifts(diagonal, subdiagonal, selection.shifts);
    auto compressed =
        compress_symmetric_lanczos_restart<Scalar>(ops, basis, residual, shifted, restart_plan.retained_count);

    basis = std::move(compressed.basis);
    diagonal = std::move(compressed.diagonal);
    subdiagonal = std::move(compressed.subdiagonal);
    residual = std::move(compressed.residual);
    residual_norm = compressed.residual_norm;
    Real const compressed_breakdown_scale =
        std::max({Real{1}, std::abs(compressed.residual_scale), std::abs(compressed.trailing_coupling)});

    if (detail::symmetric_happy_breakdown(residual_norm, params, compressed_breakdown_scale))
    {
      auto projected_after_restart = symmetric_tridiagonal_eigensystem(diagonal, subdiagonal, true);
      std::vector<Real> const after_restart_bounds =
          detail::symmetric_ritz_residual_bounds(residual_norm, projected_after_restart);
      if (diagnostics.has_value())
      {
        detail::record_symmetric_final_diagnostics(*diagnostics, projected_after_restart, after_restart_bounds, params,
                                                   residual_norm);
      }
      return finish(detail::make_symmetric_lanczos_result<Scalar>(ops, basis, residual_norm, projected_after_restart,
                                                                  params, restart_cycle_count, matvec_count));
    }

    ops.scal(residual, Scalar{1} / static_cast<Scalar>(residual_norm));
    subdiagonal.push_back(residual_norm);
    basis.push_back(std::move(residual));
  }

  if (last_result.status != 0)
  {
    last_partial_exit_result.status = 1;
    return finish(std::move(last_partial_exit_result));
  }
  return finish(std::move(last_result));
}

/// \brief Map a native symmetric transformed Ritz value back to the original eigenvalue scale.
/// \tparam Scalar Real or complex vector-field scalar type.
/// \param transformed_value Ritz value of the caller-supplied `OP`.
/// \param options Spectral transform represented by `OP`.
/// \return Eigenvalue in the original symmetric problem.
template <uni20::Real Scalar>
Scalar map_symmetric_transformed_eigenvalue(Scalar transformed_value,
                                            SymmetricSpectralTransformOptions<Scalar> const& options)
{
  switch (options.transform)
  {
    case SymmetricSpectralTransform::Regular:
    case SymmetricSpectralTransform::GeneralizedInverse:
      return transformed_value;
    case SymmetricSpectralTransform::ShiftInvert:
      if (transformed_value == Scalar{})
      {
        throw std::domain_error("shift-invert eigenvalue map received a zero transformed Ritz value");
      }
      return options.sigma + Scalar{1} / transformed_value;
    case SymmetricSpectralTransform::Buckling:
      if (transformed_value == Scalar{1})
      {
        throw std::domain_error("buckling eigenvalue map received a unit transformed Ritz value");
      }
      return options.sigma * transformed_value / (transformed_value - Scalar{1});
    case SymmetricSpectralTransform::Cayley:
      if (transformed_value == Scalar{1})
      {
        throw std::domain_error("Cayley eigenvalue map received a unit transformed Ritz value");
      }
      return options.sigma * (transformed_value + Scalar{1}) / (transformed_value - Scalar{1});
  }
  throw std::invalid_argument("unknown symmetric spectral transform");
}

/// \brief Run restarted symmetric Lanczos on a caller-supplied transformed operator.
///
/// \details The matrix-free operation object must implement the transformed
///          operator `OP` for the requested mode, for example `(A - sigma B)^-1 B`
///          for shift-invert. The native solver remains independent of the
///          workspace layout or linear-solver implementation used to apply
///          `OP`; this wrapper only maps converged Ritz values back to the
///          original eigenvalue scale. Reported residual bounds are those of
///          the transformed problem.
/// \tparam Scalar Real scalar type.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free transformed-operator provider.
/// \param op_ops Matrix-free operation object applying `OP`.
/// \param initial Initial residual/start vector.
/// \param params Solver parameters for the transformed problem.
/// \param options Transform metadata used for eigenvalue back-transformation.
/// \return Eigenvalues mapped to the original problem, with `OP` Ritz vectors.
template <uni20::LapackRealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
EigenResult<uni20::make_real_t<Scalar>, Vector>
symmetric_lanczos_restarted_transformed(Ops& op_ops, Vector const& initial,
                                        SymmetricEigenParams<uni20::make_real_t<Scalar>> const& params,
                                        SymmetricSpectralTransformOptions<uni20::make_real_t<Scalar>> const& options)
{
  auto result = symmetric_lanczos_restarted_standard<Scalar>(op_ops, initial, params);
  for (uni20::make_real_t<Scalar>& value : result.eigenvalues)
  {
    value = map_symmetric_transformed_eigenvalue(value, options);
  }
  return result;
}

/// \brief Run restarted symmetric Lanczos on a transformed generalized operator with a `B` metric.
///
/// \details This entry point is for generalized symmetric problems where the
///          caller supplies `OP`, such as `B^-1 A` or `(A - sigma B)^-1 B`,
///          and a separate matrix-free `B` operation. The Lanczos recurrence
///          uses the `B` inner product, so the transformed operator is treated
///          as self-adjoint in the generalized problem metric. The
///          returned eigenvalues are mapped back through \p options, and the
///          returned Ritz vectors are application-space vectors for the
///          original generalized problem.
/// \tparam Scalar Real or complex vector-field scalar type with real LAPACK coverage.
/// \tparam Vector Opaque vector type.
/// \tparam OpOps Matrix-free transformed-operator provider.
/// \tparam BOps Matrix-free `B` operation provider.
/// \tparam MetricOps Optional backend-owned metric inner-product provider.
/// \param op_ops Matrix-free operation object applying `OP`.
/// \param b_ops Matrix-free operation object applying `B`.
/// \param metric_ops Optional metric operation object. If it provides
///        `metric_inner_product(x, y)`, that is used directly; otherwise the
///        solver applies `B*y` into reusable scratch and calls
///        `op_ops.inner_product`.
/// \param initial Initial residual/start vector.
/// \param params Solver parameters for the transformed problem.
/// \param options Transform metadata used for eigenvalue back-transformation.
/// \return Eigenvalues mapped to the original problem, with generalized Ritz vectors.
template <uni20::LapackRealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> OpOps,
          KrylovMatrixFreeOperator<Vector, Scalar> BOps, typename MetricOps>
EigenResult<uni20::make_real_t<Scalar>, Vector> symmetric_lanczos_restarted_generalized_transformed(
    OpOps& op_ops, BOps& b_ops, MetricOps& metric_ops, Vector const& initial,
    SymmetricEigenParams<uni20::make_real_t<Scalar>> const& params,
    SymmetricSpectralTransformOptions<uni20::make_real_t<Scalar>> const& options)
{
  std::size_t const problem_dimension =
      validate_matrix_free_dimensions(op_ops, initial, "generalized restarted symmetric Lanczos");
  if (b_ops.problem_dimension() != problem_dimension || b_ops.vector_dimension(initial) != problem_dimension)
  {
    throw std::invalid_argument("generalized restarted symmetric Lanczos B operation dimension does not match");
  }

  detail::SymmetricBMetricOps<Scalar, Vector, OpOps, BOps, MetricOps> wrapped_ops(op_ops, b_ops, metric_ops);
  return symmetric_lanczos_restarted_transformed<Scalar>(wrapped_ops, initial, params, options);
}

/// \brief Run restarted symmetric Lanczos on a transformed generalized operator with default metric handling.
///
/// \details This convenience overload uses \p b_ops as the optional metric
///          customization provider. Backends can implement
///          `b_ops.metric_inner_product(x, y)` to own scratch, caching, or fused
///          backend collectives; otherwise the generic fallback applies `B*y`
///          into reusable solver scratch.
/// \tparam Scalar Real or complex vector-field scalar type with real LAPACK coverage.
/// \tparam Vector Opaque vector type.
/// \tparam OpOps Matrix-free transformed-operator provider.
/// \tparam BOps Matrix-free `B` operation provider.
/// \param op_ops Matrix-free operation object applying `OP`.
/// \param b_ops Matrix-free operation object applying `B`.
/// \param initial Initial residual/start vector.
/// \param params Solver parameters for the transformed problem.
/// \param options Transform metadata used for eigenvalue back-transformation.
/// \return Eigenvalues mapped to the original problem, with generalized Ritz vectors.
template <uni20::LapackRealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> OpOps,
          KrylovMatrixFreeOperator<Vector, Scalar> BOps>
EigenResult<uni20::make_real_t<Scalar>, Vector> symmetric_lanczos_restarted_generalized_transformed(
    OpOps& op_ops, BOps& b_ops, Vector const& initial, SymmetricEigenParams<uni20::make_real_t<Scalar>> const& params,
    SymmetricSpectralTransformOptions<uni20::make_real_t<Scalar>> const& options)
{
  return symmetric_lanczos_restarted_generalized_transformed<Scalar>(op_ops, b_ops, b_ops, initial, params, options);
}

} // namespace uni20::krylov
