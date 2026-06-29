#pragma once

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/krylov/detail_math.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace uni20::krylov
{

/// \brief Spectrum selector for matrix-free eigenvalue solvers.
enum class SpectrumPart
{
  LargestMagnitude,
  SmallestMagnitude,
  LargestAlgebraic,
  SmallestAlgebraic,
  BothEnds,
  LargestReal,
  SmallestReal,
  LargestImaginary,
  SmallestImaginary
};

/// \brief Amount of solver-internal diagnostic state to retain.
enum class KrylovDiagnosticsLevel
{
  None,
  Summary,
  Full
};

/// \brief Policy for real nonsymmetric solves when wanted Ritz values may be complex.
enum class RealNonsymmetricPolicy
{
  /// Accept only Ritz values that are numerically real; report genuinely complex wanted pairs.
  RequireRealEigenpairs,
  /// Allow real Schur two-plane output for complex conjugate pairs once that output is implemented.
  AllowRealSchurPairs,
  /// Stop with a promotion recommendation when wanted Ritz values are genuinely complex.
  PromoteToComplexSuggested
};

/// \brief Native nonsymmetric solver status.
enum class NonsymmetricStatus
{
  Converged,
  NotConverged,
  ComplexPairEncountered,
  ComplexPromotionRecommended,
  RealSchurPairRequired,
  AmbiguousReality,
  Breakdown,
  InvalidInput
};

/// \brief Numerical reality classification for a complex Ritz value from a real Arnoldi solve.
enum class RitzReality
{
  Real,
  Ambiguous,
  Complex
};

/// \brief Transform represented by the caller-supplied symmetric Krylov operator.
enum class SymmetricTransformKind
{
  Regular,
  GeneralizedInverse,
  ShiftInvert,
  Buckling,
  Cayley
};

/// \brief Backward-compatible name for symmetric transform kinds.
using SymmetricSpectralTransform = SymmetricTransformKind;

/// \brief Native symmetric transform options for mapping `OP` Ritz values.
template <uni20::Real Scalar> struct SymmetricTransformOptions
{
    SymmetricTransformKind transform = SymmetricTransformKind::Regular;
    Scalar sigma = Scalar{};
};

/// \brief Backward-compatible name for symmetric transform options.
template <uni20::Real Scalar> using SymmetricSpectralTransformOptions = SymmetricTransformOptions<Scalar>;

/// \brief Parameters for a native nonsymmetric eigenvalue solve.
///
/// \details Phase 4 starts with a real-double Arnoldi implementation. The API
///          already exposes the real-mode policy needed by transfer-matrix
///          workflows: a real matrix-free operator can preserve a real Arnoldi
///          basis while detecting when wanted Ritz values require complex
///          eigenvectors or a real Schur two-plane.
template <uni20::Real Scalar> struct NonsymmetricEigenParams
{
    int eigenvalue_count = 1;
    int retained_ritz_count = 0;
    int krylov_dimension = 0;
    int max_iterations = 300;
    Scalar tolerance = Scalar{};
    Scalar complex_pair_tolerance = Scalar{};
    SpectrumPart spectrum = SpectrumPart::LargestMagnitude;
    bool compute_eigenvectors = true;
    RealNonsymmetricPolicy real_policy = RealNonsymmetricPolicy::RequireRealEigenpairs;
    KrylovDiagnosticsLevel diagnostics = KrylovDiagnosticsLevel::None;
};

/// \brief Optional diagnostic state for nonsymmetric Arnoldi solves.
/// \tparam Scalar Real scalar type used by the Arnoldi basis.
template <uni20::Real Scalar> struct NonsymmetricArnoldiDiagnostics
{
    int restart_count = 0;
    int op_count = 0;
    int final_projected_dimension = 0;
    Scalar final_residual_norm = Scalar{};
    Scalar projected_departure_from_normality = Scalar{};
    std::vector<uni20::complex<Scalar>> final_ritz_values;
    std::vector<Scalar> final_ritz_bounds;
    std::vector<RitzReality> final_ritz_reality;
};

/// \brief Matrix-free nonsymmetric eigenvalue result.
///
/// \details For real nonsymmetric Arnoldi, eigenvalues may be complex even
///          though vectors and matrix-vector products are real. Real
///          eigenvectors are populated only when every selected Ritz value is
///          numerically real. If any selected Ritz value is complex or
///          ambiguous, `right_eigenvectors` is left empty and callers should
///          inspect `status` and `reality`. Complex eigenvector output and real
///          Schur two-plane output are reserved for later nonsymmetric phases.
template <uni20::Real Scalar, typename Vector> struct NonsymmetricEigenResult
{
    std::vector<uni20::complex<Scalar>> eigenvalues;
    std::vector<Scalar> residual_bounds;
    std::vector<Vector> right_eigenvectors;
    std::vector<RitzReality> reality;
    int converged_count = 0;
    int iteration_count = 0;
    int matvec_count = 0;
    NonsymmetricStatus status = NonsymmetricStatus::Converged;
    std::optional<NonsymmetricArnoldiDiagnostics<Scalar>> diagnostics;
};

/// \brief Common parameters for a symmetric or Hermitian eigenvalue solve.
template <uni20::Real Scalar> struct SymmetricEigenParams
{
    /// \brief Number of converged eigenpairs requested by the caller.
    int eigenvalue_count = 1;
    /// \brief Optional number of Ritz vectors retained internally after restart.
    ///
    /// \details A value of zero selects `retained_ritz_count == eigenvalue_count`.
    ///          Thick-restart paths may set this larger than
    ///          `eigenvalue_count`, subject to
    ///          `eigenvalue_count <= retained_ritz_count < krylov_dimension`.
    int retained_ritz_count = 0;
    /// \brief Maximum Arnoldi/Lanczos subspace dimension.
    int krylov_dimension = 0;
    int max_iterations = 300;
    Scalar tolerance = Scalar{};
    /// \brief Relative threshold for declaring Lanczos invariant-subspace breakdown.
    ///
    /// \details A value of zero selects a machine-precision default. This is
    ///          independent of the user Ritz convergence tolerance.
    Scalar breakdown_tolerance = Scalar{};
    SpectrumPart spectrum = SpectrumPart::LargestMagnitude;
    bool compute_eigenvectors = true;
    KrylovDiagnosticsLevel diagnostics = KrylovDiagnosticsLevel::None;
};

/// \brief Return the default symmetric/Hermitian Krylov subspace dimension.
///
/// \details The default follows the common implicitly restarted Krylov guideline
///          `ncv = min(n, max(20, 2*nev + 1))`. Callers may still set
///          `krylov_dimension` explicitly when memory is tight or a hard
///          clustered problem needs a larger search space.
[[nodiscard]] inline int default_symmetric_krylov_dimension(int eigenvalue_count, std::size_t problem_dimension)
{
  if (eigenvalue_count <= 0)
  {
    throw std::invalid_argument("default symmetric Krylov dimension requires a positive eigenvalue count");
  }
  if (problem_dimension > static_cast<std::size_t>(uni20::numeric_limits<int>::max()))
  {
    throw std::overflow_error("Krylov problem dimension exceeds the supported int range");
  }
  int const dimension = static_cast<int>(problem_dimension);
  int const target = std::max(20, 2 * eigenvalue_count + 1);
  return std::min(dimension, target);
}

/// \brief Return the default nonsymmetric Arnoldi subspace dimension.
///
/// \details Nonsymmetric restarts are more sensitive to nonnormality and
///          complex conjugate Schur blocks than Hermitian Lanczos restarts, so
///          the native default is deliberately roomier than the minimal
///          symmetric/Hermitian default: `ncv = min(n, max(20, 6*nev + 8))`.
[[nodiscard]] inline int default_nonsymmetric_krylov_dimension(int eigenvalue_count, std::size_t problem_dimension)
{
  if (eigenvalue_count <= 0)
  {
    throw std::invalid_argument("default nonsymmetric Krylov dimension requires a positive eigenvalue count");
  }
  if (problem_dimension > static_cast<std::size_t>(uni20::numeric_limits<int>::max()))
  {
    throw std::overflow_error("Krylov problem dimension exceeds the supported int range");
  }
  int const dimension = static_cast<int>(problem_dimension);
  int const target = std::max(20, 6 * eigenvalue_count + 8);
  return std::min(dimension, target);
}

/// \brief Return the effective symmetric/Hermitian Krylov subspace dimension.
template <uni20::Real Scalar>
[[nodiscard]] int effective_symmetric_krylov_dimension(SymmetricEigenParams<Scalar> const& params,
                                                       std::size_t problem_dimension)
{
  return params.krylov_dimension > 0 ? params.krylov_dimension
                                     : default_symmetric_krylov_dimension(params.eigenvalue_count, problem_dimension);
}

/// \brief Return the effective nonsymmetric Arnoldi subspace dimension.
template <uni20::Real Scalar>
[[nodiscard]] int effective_nonsymmetric_krylov_dimension(NonsymmetricEigenParams<Scalar> const& params,
                                                          std::size_t problem_dimension)
{
  return params.krylov_dimension > 0
             ? params.krylov_dimension
             : default_nonsymmetric_krylov_dimension(params.eigenvalue_count, problem_dimension);
}

/// \brief Return the default symmetric/Hermitian retained Ritz count.
///
/// \details Hermitian problems default to a minimal implicit restart: keep
///          exactly the requested invariant subspace and use the remaining
///          Krylov directions as shifts. Clustered problems may set
///          `retained_ritz_count` larger to get thick-restart guard vectors.
[[nodiscard]] inline int default_symmetric_retained_ritz_count(int eigenvalue_count, int krylov_dimension)
{
  if (eigenvalue_count <= 0)
  {
    throw std::invalid_argument("default symmetric retained Ritz count requires a positive eigenvalue count");
  }
  if (krylov_dimension <= eigenvalue_count)
  {
    return eigenvalue_count;
  }
  return eigenvalue_count;
}

/// \brief Return the default nonsymmetric retained Schur dimension.
///
/// \details The native nonsymmetric default keeps guard vectors by retaining
///          `max(2*nev, nev+4)` Schur dimensions, clamped below `ncv`. This
///          improves robustness on nonnormal and clustered cases while still
///          leaving room for restart filtering.
[[nodiscard]] inline int default_nonsymmetric_retained_ritz_count(int eigenvalue_count, int krylov_dimension)
{
  if (eigenvalue_count <= 0)
  {
    throw std::invalid_argument("default nonsymmetric retained Ritz count requires a positive eigenvalue count");
  }
  if (krylov_dimension <= eigenvalue_count)
  {
    return eigenvalue_count;
  }
  int const target = std::max(2 * eigenvalue_count, eigenvalue_count + 4);
  return std::min(krylov_dimension - 1, target);
}

/// \brief Return the effective symmetric/Hermitian retained Ritz count.
template <uni20::Real Scalar>
[[nodiscard]] int effective_symmetric_retained_ritz_count(SymmetricEigenParams<Scalar> const& params,
                                                          int krylov_dimension)
{
  int const retained_count = params.retained_ritz_count > 0
                                 ? params.retained_ritz_count
                                 : default_symmetric_retained_ritz_count(params.eigenvalue_count, krylov_dimension);
  if (retained_count < params.eigenvalue_count)
  {
    throw std::invalid_argument("retained Ritz count must be at least the requested eigenvalue count");
  }
  if (retained_count >= krylov_dimension)
  {
    throw std::invalid_argument("retained Ritz count must be smaller than the Krylov dimension");
  }
  return retained_count;
}

/// \brief Return the effective nonsymmetric retained Ritz or Schur count.
template <uni20::Real Scalar>
[[nodiscard]] int effective_nonsymmetric_retained_ritz_count(NonsymmetricEigenParams<Scalar> const& params,
                                                             int krylov_dimension)
{
  int const retained_count = params.retained_ritz_count > 0
                                 ? params.retained_ritz_count
                                 : default_nonsymmetric_retained_ritz_count(params.eigenvalue_count, krylov_dimension);
  if (retained_count < params.eigenvalue_count)
  {
    throw std::invalid_argument("nonsymmetric retained Ritz count is below eigenvalue count");
  }
  if (retained_count >= krylov_dimension)
  {
    throw std::invalid_argument("nonsymmetric retained Ritz count must be smaller than the Krylov dimension");
  }
  return retained_count;
}

/// \brief Diagnostic state for one symmetric Lanczos restart cycle.
/// \tparam Scalar Real scalar type.
template <uni20::Real Scalar> struct SymmetricRestartCycleDiagnostics
{
    int cycle = 0;
    int converged_count = 0;
    std::size_t projected_dimension = 0;
    std::size_t retained_count = 0;
    std::size_t shift_count = 0;
    std::size_t protected_zero_bound_count = 0;
    bool enlarged_for_partial_convergence = false;
    Scalar residual_norm = Scalar{};
    std::vector<Scalar> ritz_values;
    std::vector<Scalar> ritz_bounds;
    std::vector<Scalar> shifts;
    std::vector<std::size_t> wanted_indices;
    std::vector<std::size_t> shift_indices;
    std::vector<bool> wanted_converged;
};

/// \brief Optional diagnostic state for symmetric Lanczos solves.
/// \tparam Scalar Real scalar type.
template <uni20::Real Scalar> struct SymmetricLanczosDiagnostics
{
    int restart_count = 0;
    int op_count = 0;
    int final_projected_dimension = 0;
    Scalar final_residual_norm = Scalar{};
    std::vector<Scalar> final_ritz_values;
    std::vector<Scalar> final_ritz_bounds;
    std::vector<std::size_t> final_selected_indices;
    std::vector<SymmetricRestartCycleDiagnostics<Scalar>> restart_cycles;
};

/// \brief Matrix-free eigenvalue result with solver-owned vector objects.
template <typename Scalar, typename Vector> struct EigenResult
{
    std::vector<Scalar> eigenvalues;
    std::vector<Scalar> residual_bounds;
    std::vector<Vector> eigenvectors;
    int converged_count = 0;
    int iteration_count = 0;
    int matvec_count = 0;
    int status = 0;
    std::optional<SymmetricLanczosDiagnostics<Scalar>> diagnostics;
};

/// \brief Concept for opaque vector-space operations required by Krylov solvers.
///
/// \details Native Krylov solvers keep solver state local to each call and do
///          not use hidden global workspaces. They are therefore reentrant when
///          each solve receives independent operation/vector state, and nested
///          native solves from inside a matrix-free callback are valid. If an
///          implementation is shared across concurrent solves, that
///          implementation is responsible for synchronizing its own mutable
///          backend/cache state. Krylov callbacks use the local convention that
///          mutable/output vector arguments appear first, for example
///          `copy(y, x)`, `axpy(y, alpha, x)`, and `matvec(y, x)`.
///          `inner_product(x, y)` returns the host scalar for `<x,y>`; for
///          complex scalar types it is conjugate-linear in `x` and linear in
///          `y`. Norms are exposed separately through optional `norm(x)`, which
///          returns a real host scalar.
template <typename Ops, typename Vector, typename Scalar>
concept KrylovVectorOps = requires(Ops ops, Vector x, Vector y, Scalar alpha) {
  { ops.problem_dimension() } -> std::convertible_to<std::size_t>;
  { ops.vector_dimension(x) } -> std::convertible_to<std::size_t>;
  { ops.allocate_like(x) } -> std::same_as<Vector>;
  { ops.copy(y, x) } -> std::same_as<void>;
  { ops.axpy(y, alpha, x) } -> std::same_as<void>;
  { ops.scal(y, alpha) } -> std::same_as<void>;
  { ops.set_zero(y) } -> std::same_as<void>;
  { ops.inner_product(x, y) } -> std::convertible_to<Scalar>;
};

/// \brief Concept for an output-first Krylov operator application `y <- OP*x`.
template <typename Op, typename Vector>
concept KrylovOperator = requires(Op op, Vector x, Vector y) {
  { op.matvec(y, x) } -> std::same_as<void>;
};

/// \brief Composite concept for current adapters that own vector ops and `OP*x`.
///
/// \details This is the role consumed by the current symmetric Lanczos
///          implementation. Future public entry points can split vector
///          operations and operators into separate objects while preserving this
///          output-first callback convention.
template <typename Ops, typename Vector, typename Scalar>
concept KrylovMatrixFreeOperator = KrylovVectorOps<Ops, Vector, Scalar> && KrylovOperator<Ops, Vector>;

/// \brief Backward-compatible name for the combined vector/operator concept.
template <typename Ops, typename Vector, typename Scalar>
concept MatrixFreeVectorOps = KrylovMatrixFreeOperator<Ops, Vector, Scalar>;

/// \brief Optional customization point for backend-owned metric inner products.
///
/// \details Implementations may provide this operation to compute `x^T B y`
///          directly, using backend-owned scratch storage, cached `B*y`
///          values, fused GPU kernels, or MPI collectives. Generic solvers
///          should use `metric_inner_product_or_apply` so backends without this
///          customization still get a correct default through `metric.apply` or
///          `B.matvec`.
template <typename Ops, typename Vector, typename Scalar>
concept KrylovMetricInnerProduct = requires(Ops ops, Vector x, Vector y) {
  { ops.metric_inner_product(x, y) } -> std::convertible_to<Scalar>;
};

/// \brief Backward-compatible name for direct metric inner-product customization.
template <typename Ops, typename Vector, typename Scalar>
concept MetricInnerProductOps = KrylovMetricInnerProduct<Ops, Vector, Scalar>;

/// \brief Optional customization point for direct vector norms.
///
/// \details A backend may compute norms more accurately or efficiently than a
///          generic `sqrt(real(inner_product(x, x)))`, for example by using a
///          fused device reduction or an MPI norm collective. Solvers should
///          call `norm_or_inner_product` so backends without this customization
///          still get a correct default from the inner product.
template <typename Ops, typename Vector, typename Scalar>
concept KrylovNorm = requires(Ops ops, Vector x) {
  { ops.norm(x) } -> std::convertible_to<uni20::make_real_t<Scalar>>;
};

/// \brief Evaluate a vector norm, using a backend override when available.
///
/// \details The fallback computes `sqrt(max(0, real(inner_product(x, x))))`.
///          The result type is the real counterpart of \p Scalar, so complex
///          vector spaces return a real norm without duplicating scalar-type
///          metadata in the matrix-free interface.
template <typename Scalar, typename Vector, typename Ops>
uni20::make_real_t<Scalar> norm_or_inner_product(Ops& ops, Vector const& x)
{
  using RealScalar = uni20::make_real_t<Scalar>;
  if constexpr (KrylovNorm<Ops, Vector, Scalar>)
  {
    return static_cast<RealScalar>(ops.norm(x));
  }
  else
  {
    Scalar const norm_squared = ops.inner_product(x, x);
    RealScalar const real_norm_squared = static_cast<RealScalar>(detail::adl_real(norm_squared));
    return real_norm_squared > RealScalar{} ? detail::adl_sqrt(real_norm_squared) : RealScalar{};
  }
}

/// \brief Default imaginary-part tolerance for deciding whether a real Arnoldi Ritz value is real.
/// \tparam Scalar Real scalar type.
/// \return A conservative scale multiplier derived from machine precision.
template <uni20::Real Scalar> Scalar default_complex_pair_tolerance()
{
  return detail::adl_sqrt(uni20::numeric_limits<Scalar>::epsilon());
}

/// \brief Classify whether a complex Ritz value is numerically real.
///
/// \details This is intended for real nonsymmetric Arnoldi. A real operator can
///          have complex-conjugate Ritz pairs; transfer-matrix workflows often
///          want to remain in a real path when the wanted fixed point is real,
///          but detect genuine or borderline time-reversal-symmetry breaking.
///          Values with `|imag(theta)| <= tolerance * scale` are classified as
///          real. Values within \p ambiguity_factor of that boundary are
///          classified as ambiguous.
/// \tparam Scalar Real scalar type.
/// \param theta Ritz value to classify.
/// \param tolerance Relative imaginary-part tolerance. If zero or negative, a
///        machine-precision default is used.
/// \param scale Problem or projected-matrix scale. Values below one are raised
///        to one.
/// \param ambiguity_factor Width of the ambiguous band above the real threshold.
/// \return Reality classification for \p theta.
template <uni20::Real Scalar>
RitzReality classify_ritz_reality(uni20::complex<Scalar> theta, Scalar tolerance = Scalar{}, Scalar scale = Scalar{1},
                                  Scalar ambiguity_factor = Scalar{10})
{
  Scalar const effective_tolerance = tolerance > Scalar{} ? tolerance : default_complex_pair_tolerance<Scalar>();
  Scalar const effective_scale = std::max({Scalar{1}, detail::adl_abs(theta), detail::adl_abs(scale)});
  Scalar const threshold = effective_tolerance * effective_scale;
  Scalar const imaginary = detail::adl_abs(theta.imag());
  if (imaginary <= threshold)
  {
    return RitzReality::Real;
  }
  if (imaginary <= ambiguity_factor * threshold)
  {
    return RitzReality::Ambiguous;
  }
  return RitzReality::Complex;
}

/// \brief Evaluate a metric inner product, using a backend override when available.
///
/// \details If \p metric_ops has `metric_inner_product(x, y)`, that operation
///          is used directly. Otherwise the default computes `B*y` into
///          \p scratch with \p b_ops and returns `inner_ops.inner_product(x,
///          scratch)`.
template <typename Scalar, typename Vector, typename MetricOps, typename BOps, typename InnerOps>
Scalar metric_inner_product_or_apply(MetricOps& metric_ops, BOps& b_ops, InnerOps& inner_ops, Vector const& x,
                                     Vector const& y, Vector& scratch)
{
  if constexpr (MetricInnerProductOps<MetricOps, Vector, Scalar>)
  {
    return metric_ops.metric_inner_product(x, y);
  }
  else
  {
    b_ops.matvec(scratch, y);
    return inner_ops.inner_product(x, scratch);
  }
}

/// \brief Validate that an initial/prototype vector matches the matrix-free problem dimension.
template <typename Ops, typename Vector>
std::size_t validate_matrix_free_dimensions(Ops const& ops, Vector const& vector, char const* context)
{
  std::size_t const problem_dimension = ops.problem_dimension();
  std::size_t const vector_dimension = ops.vector_dimension(vector);
  if (problem_dimension == 0)
  {
    throw std::invalid_argument(std::string(context) + " requires a non-empty problem dimension");
  }
  if (vector_dimension != problem_dimension)
  {
    throw std::invalid_argument(std::string(context) + " vector dimension does not match operator dimension");
  }
  return problem_dimension;
}

} // namespace uni20::krylov
