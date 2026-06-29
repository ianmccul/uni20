#pragma once

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/krylov/dense_subspace.hpp>
#include <uni20/krylov/matrix_free.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20::krylov
{

/// \brief Arnoldi factorization for a nonsymmetric matrix-free operator.
/// \tparam Scalar Vector-space scalar type.
/// \tparam Vector Opaque vector type.
template <uni20::RealOrComplex Scalar, typename Vector> struct ArnoldiFactorization
{
    std::vector<Vector> basis;
    Matrix<Scalar> hessenberg;
    int step_count = 0;
    int op_count = 0;
    uni20::make_real_t<Scalar> residual_norm = uni20::make_real_t<Scalar>{};
    bool happy_breakdown = false;
};

/// \brief Ritz data extracted from a real Arnoldi Hessenberg projection.
/// \tparam Scalar Real scalar type.
template <uni20::Real Scalar> struct ArnoldiRitzExtraction
{
    std::vector<uni20::complex<Scalar>> ritz_values;
    std::vector<Scalar> residual_bounds;
    std::vector<RitzReality> reality;
    Matrix<uni20::complex<Scalar>> projected_right_eigenvectors;
};

/// \brief Block-aware selection result for real Schur restart.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSchurRestartSelection
{
    std::vector<std::size_t> block_indices;
    std::size_t scalar_count = 0;
    bool enlarged_to_preserve_block = false;
};

/// \brief Compressed real Arnoldi relation after a Schur restart.
///
/// \details Represents `A V_k = V_k T_k + f b^T`, where `V_k` is the
///          transformed retained basis, `T_k` is the leading reordered real
///          Schur form, `f` is the application-space residual vector, and
///          `b` is the dense residual-coupling row. This is the state that a
///          Krylov-Schur restart expands back to `ncv`.
/// \tparam Scalar Real scalar type.
/// \tparam Vector Opaque vector type.
template <uni20::LapackReal Scalar, typename Vector> struct RealSchurCompressedArnoldiFactorization
{
    std::vector<Vector> basis;
    Vector residual;
    Matrix<Scalar> schur_form;
    std::vector<Scalar> residual_coupling;
    std::vector<uni20::complex<Scalar>> eigenvalues;
    std::vector<RealSchurBlock<Scalar>> blocks;
    Scalar residual_norm = Scalar{};
    bool happy_breakdown = false;
    bool enlarged_to_preserve_block = false;
};

/// \brief Compressed complex Arnoldi relation after a Schur restart.
///
/// \details Represents `A V_k = V_k T_k + f b^T`, where `V_k` is the
///          transformed retained basis, `T_k` is the leading reordered complex
///          Schur form, `f` is the application-space residual vector, and `b`
///          is the dense residual-coupling row.
/// \tparam Real Underlying real precision.
/// \tparam Vector Opaque vector type.
template <uni20::LapackComplexReal Real, typename Vector> struct ComplexSchurCompressedArnoldiFactorization
{
    using Complex = uni20::complex<Real>;

    std::vector<Vector> basis;
    Vector residual;
    Matrix<Complex> schur_form;
    std::vector<Complex> residual_coupling;
    std::vector<Complex> eigenvalues;
    Real residual_norm = Real{};
    bool happy_breakdown = false;
};

namespace detail
{
#if defined(__cpp_lib_constexpr_cmath)
template <uni20::Real Scalar> constexpr Scalar nonsymmetric_convergence_floor()
#else
template <uni20::Real Scalar> Scalar nonsymmetric_convergence_floor()
#endif
{
  return adl_pow(uni20::numeric_limits<Scalar>::epsilon(), Scalar{2} / Scalar{3});
}

template <uni20::Real Scalar>
bool nonsymmetric_ritz_converged(uni20::complex<Scalar> value, Scalar residual_bound, Scalar tolerance)
{
  Scalar const eps23 = nonsymmetric_convergence_floor<Scalar>();
  return residual_bound <= tolerance * std::max(eps23, adl_abs(value));
}

template <typename Compare> void sort_indices(std::vector<std::size_t>& indices, Compare&& compare)
{
  std::ranges::sort(indices, [&](std::size_t lhs, std::size_t rhs) {
    if (compare(lhs, rhs))
    {
      return true;
    }
    if (compare(rhs, lhs))
    {
      return false;
    }
    return lhs < rhs;
  });
}

template <uni20::LapackReal Scalar> Scalar schur_block_score(RealSchurBlock<Scalar> const& block, SpectrumPart spectrum)
{
  auto const first = block.first_eigenvalue;
  auto const second = block.size == 2 ? block.second_eigenvalue : block.first_eigenvalue;
  switch (spectrum)
  {
    case SpectrumPart::LargestMagnitude:
      return std::max(adl_abs(first), adl_abs(second));
    case SpectrumPart::SmallestMagnitude:
      return std::min(adl_abs(first), adl_abs(second));
    case SpectrumPart::LargestReal:
      return std::max(first.real(), second.real());
    case SpectrumPart::SmallestReal:
      return std::min(first.real(), second.real());
    case SpectrumPart::LargestImaginary:
      return std::max(first.imag(), second.imag());
    case SpectrumPart::SmallestImaginary:
      return std::min(first.imag(), second.imag());
    case SpectrumPart::LargestAlgebraic:
    case SpectrumPart::SmallestAlgebraic:
    case SpectrumPart::BothEnds:
      throw std::invalid_argument("algebraic and both-end selectors are not valid for nonsymmetric Schur restart");
  }
  return Scalar{};
}

template <uni20::LapackReal Scalar> bool larger_is_better(SpectrumPart spectrum)
{
  switch (spectrum)
  {
    case SpectrumPart::LargestMagnitude:
    case SpectrumPart::LargestReal:
    case SpectrumPart::LargestImaginary:
      return true;
    case SpectrumPart::SmallestMagnitude:
    case SpectrumPart::SmallestReal:
    case SpectrumPart::SmallestImaginary:
      return false;
    case SpectrumPart::LargestAlgebraic:
    case SpectrumPart::SmallestAlgebraic:
    case SpectrumPart::BothEnds:
      throw std::invalid_argument("algebraic and both-end selectors are not valid for nonsymmetric Schur restart");
  }
  return true;
}

template <uni20::Real Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
NonsymmetricEigenResult<Scalar, Vector> make_real_nonsymmetric_arnoldi_result(
    Ops& ops, ArnoldiFactorization<Scalar, Vector> const& factorization, ArnoldiRitzExtraction<Scalar> const& ritz,
    std::vector<std::size_t> const& selected, NonsymmetricEigenParams<Scalar> const& params, int iteration_count,
    int matvec_count, int restart_count, Scalar projected_departure)
{
  Scalar const tolerance =
      params.tolerance > Scalar{} ? params.tolerance : Scalar{100} * uni20::numeric_limits<Scalar>::epsilon();

  NonsymmetricEigenResult<Scalar, Vector> result;
  result.iteration_count = iteration_count;
  result.matvec_count = matvec_count;
  result.eigenvalues.reserve(selected.size());
  result.residual_bounds.reserve(selected.size());
  result.reality.reserve(selected.size());
  result.right_eigenvectors.reserve(params.compute_eigenvectors ? selected.size() : 0);

  bool saw_complex = false;
  bool saw_ambiguous = false;
  for (std::size_t const index : selected)
  {
    uni20::complex<Scalar> const value = ritz.ritz_values[index];
    Scalar const residual_bound = ritz.residual_bounds[index];
    RitzReality const reality = ritz.reality[index];

    result.eigenvalues.push_back(value);
    result.residual_bounds.push_back(residual_bound);
    result.reality.push_back(reality);
    if (nonsymmetric_ritz_converged(value, residual_bound, tolerance))
    {
      ++result.converged_count;
    }

    saw_complex = saw_complex || reality == RitzReality::Complex;
    saw_ambiguous = saw_ambiguous || reality == RitzReality::Ambiguous;
    if (params.compute_eigenvectors && reality == RitzReality::Real)
    {
      Vector vector = ops.allocate_like(factorization.basis.front());
      ops.set_zero(vector);
      for (std::size_t row = 0; row < static_cast<std::size_t>(factorization.step_count); ++row)
      {
        ops.axpy(vector, ritz.projected_right_eigenvectors[row, index].real(), factorization.basis[row]);
      }
      result.right_eigenvectors.push_back(std::move(vector));
    }
  }

  if (params.diagnostics != KrylovDiagnosticsLevel::None)
  {
    NonsymmetricArnoldiDiagnostics<Scalar> diagnostics;
    diagnostics.restart_count = restart_count;
    diagnostics.op_count = matvec_count;
    diagnostics.final_projected_dimension = factorization.step_count;
    diagnostics.final_residual_norm = factorization.residual_norm;
    diagnostics.projected_departure_from_normality = projected_departure;
    diagnostics.final_ritz_values = ritz.ritz_values;
    diagnostics.final_ritz_bounds = ritz.residual_bounds;
    diagnostics.final_ritz_reality = ritz.reality;
    result.diagnostics = std::move(diagnostics);
  }

  if (saw_ambiguous)
  {
    result.status = NonsymmetricStatus::AmbiguousReality;
  }
  else if (saw_complex)
  {
    switch (params.real_policy)
    {
      case RealNonsymmetricPolicy::RequireRealEigenpairs:
        result.status = NonsymmetricStatus::ComplexPairEncountered;
        break;
      case RealNonsymmetricPolicy::PromoteToComplexSuggested:
        result.status = NonsymmetricStatus::ComplexPromotionRecommended;
        break;
      case RealNonsymmetricPolicy::AllowRealSchurPairs:
        result.status = NonsymmetricStatus::RealSchurPairRequired;
        break;
    }
  }
  else
  {
    result.status = result.converged_count == params.eigenvalue_count ? NonsymmetricStatus::Converged
                                                                      : NonsymmetricStatus::NotConverged;
  }
  if (result.eigenvalues.size() < static_cast<std::size_t>(params.eigenvalue_count))
  {
    result.status = factorization.happy_breakdown ? NonsymmetricStatus::Breakdown : NonsymmetricStatus::NotConverged;
  }

  return result;
}

template <uni20::LapackComplexReal Real, typename Vector, KrylovMatrixFreeOperator<Vector, uni20::complex<Real>> Ops>
NonsymmetricEigenResult<Real, Vector> make_complex_nonsymmetric_arnoldi_result(
    Ops& ops, ArnoldiFactorization<uni20::complex<Real>, Vector> const& factorization,
    ArnoldiRitzExtraction<Real> const& ritz, std::vector<std::size_t> const& selected,
    NonsymmetricEigenParams<Real> const& params, int iteration_count, int matvec_count, int restart_count,
    Real projected_departure)
{
  using Complex = uni20::complex<Real>;

  Real const tolerance =
      params.tolerance > Real{} ? params.tolerance : Real{100} * uni20::numeric_limits<Real>::epsilon();

  NonsymmetricEigenResult<Real, Vector> result;
  result.iteration_count = iteration_count;
  result.matvec_count = matvec_count;
  result.eigenvalues.reserve(selected.size());
  result.residual_bounds.reserve(selected.size());
  result.reality.reserve(selected.size());
  result.right_eigenvectors.reserve(params.compute_eigenvectors ? selected.size() : 0);

  for (std::size_t const index : selected)
  {
    Complex const value = ritz.ritz_values[index];
    Real const residual_bound = ritz.residual_bounds[index];

    result.eigenvalues.push_back(value);
    result.residual_bounds.push_back(residual_bound);
    result.reality.push_back(classify_ritz_reality(value, params.complex_pair_tolerance));
    if (nonsymmetric_ritz_converged(value, residual_bound, tolerance))
    {
      ++result.converged_count;
    }

    if (params.compute_eigenvectors)
    {
      Vector vector = ops.allocate_like(factorization.basis.front());
      ops.set_zero(vector);
      for (std::size_t row = 0; row < static_cast<std::size_t>(factorization.step_count); ++row)
      {
        ops.axpy(vector, ritz.projected_right_eigenvectors[row, index], factorization.basis[row]);
      }
      result.right_eigenvectors.push_back(std::move(vector));
    }
  }

  if (params.diagnostics != KrylovDiagnosticsLevel::None)
  {
    NonsymmetricArnoldiDiagnostics<Real> diagnostics;
    diagnostics.restart_count = restart_count;
    diagnostics.op_count = matvec_count;
    diagnostics.final_projected_dimension = factorization.step_count;
    diagnostics.final_residual_norm = factorization.residual_norm;
    diagnostics.projected_departure_from_normality = projected_departure;
    diagnostics.final_ritz_values = ritz.ritz_values;
    diagnostics.final_ritz_bounds = ritz.residual_bounds;
    diagnostics.final_ritz_reality = ritz.reality;
    result.diagnostics = std::move(diagnostics);
  }

  result.status = result.converged_count == params.eigenvalue_count ? NonsymmetricStatus::Converged
                                                                    : NonsymmetricStatus::NotConverged;
  if (result.eigenvalues.size() < static_cast<std::size_t>(params.eigenvalue_count))
  {
    result.status = factorization.happy_breakdown ? NonsymmetricStatus::Breakdown : NonsymmetricStatus::NotConverged;
  }
  return result;
}
} // namespace detail

/// \brief Measure departure from normality of a projected matrix.
///
/// \details Returns `||H^* H - H H^*||_F / max(1, ||H||_F^2)`. The value is
///          zero for normal projected matrices and grows as the projected
///          Hessenberg matrix becomes non-normal. This is a diagnostic of the
///          local projected problem only; it does not inspect the original
///          matrix-free operator.
/// \tparam Scalar Real or complex scalar type.
/// \param matrix Square projected matrix.
/// \return Scale-normalized Frobenius departure from normality.
template <uni20::RealOrComplex Scalar>
uni20::make_real_t<Scalar> projected_departure_from_normality(Matrix<Scalar> const& matrix)
{
  if (matrix.rows() != matrix.cols())
  {
    throw std::invalid_argument("projected_departure_from_normality requires a square matrix");
  }

  using Real = uni20::make_real_t<Scalar>;

  std::size_t const order = matrix.rows();
  Real norm_squared{};
  for (std::size_t col = 0; col < order; ++col)
  {
    for (std::size_t row = 0; row < order; ++row)
    {
      Scalar const value = matrix[row, col];
      norm_squared += detail::abs_squared(value);
    }
  }

  Real commutator_norm_squared{};
  for (std::size_t col = 0; col < order; ++col)
  {
    for (std::size_t row = 0; row < order; ++row)
    {
      Scalar hth{};
      Scalar hht{};
      for (std::size_t inner = 0; inner < order; ++inner)
      {
        hth += detail::conjugate_if_complex(matrix[inner, row]) * matrix[inner, col];
        hht += matrix[row, inner] * detail::conjugate_if_complex(matrix[col, inner]);
      }
      Scalar const difference = hth - hht;
      commutator_norm_squared += detail::abs_squared(difference);
    }
  }

  Real const scale = std::max(Real{1}, norm_squared);
  return detail::adl_sqrt(commutator_norm_squared) / scale;
}

/// \brief Build a full-reorthogonalized Arnoldi factorization.
///
/// \details This constructs `A V_m = V_{m+1} H_m` when no happy breakdown is
///          detected. On happy breakdown, the final subdiagonal is zero and the
///          basis spans an invariant subspace. Vector storage and operations are
///          entirely delegated to \p ops; only the small Hessenberg projection is
///          stored locally.
/// \tparam Scalar Vector-space scalar type.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free operator/vector operation type.
/// \param ops Matrix-free vector operations and operator application.
/// \param initial Initial vector used to seed the Arnoldi basis.
/// \param max_steps Maximum number of Arnoldi steps.
/// \param breakdown_tolerance Absolute residual norm threshold for happy breakdown.
/// \return Arnoldi factorization.
template <uni20::RealOrComplex Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
ArnoldiFactorization<Scalar, Vector>
arnoldi_factorize(Ops& ops, Vector const& initial, int max_steps,
                  uni20::make_real_t<Scalar> breakdown_tolerance = uni20::make_real_t<Scalar>{})
{
  validate_matrix_free_dimensions(ops, initial, "arnoldi_factorize");
  if (max_steps <= 0)
  {
    throw std::invalid_argument("arnoldi_factorize requires a positive step count");
  }

  using Real = uni20::make_real_t<Scalar>;

  Real const effective_breakdown_tolerance =
      breakdown_tolerance > Real{} ? breakdown_tolerance : Real{10} * uni20::numeric_limits<Real>::epsilon();

  Vector q0 = ops.allocate_like(initial);
  ops.copy(q0, initial);
  Real const initial_norm = norm_or_inner_product<Scalar>(ops, q0);
  if (initial_norm == Real{})
  {
    throw std::invalid_argument("arnoldi_factorize received a zero initial vector");
  }
  ops.scal(q0, Scalar{1} / static_cast<Scalar>(initial_norm));

  ArnoldiFactorization<Scalar, Vector> factorization;
  factorization.hessenberg =
      Matrix<Scalar>(static_cast<std::size_t>(max_steps) + 1, static_cast<std::size_t>(max_steps));
  factorization.basis.push_back(std::move(q0));

  for (int step = 0; step < max_steps; ++step)
  {
    Vector residual = ops.allocate_like(initial);
    ops.matvec(residual, factorization.basis[static_cast<std::size_t>(step)]);
    ++factorization.op_count;

    for (int pass = 0; pass < 2; ++pass)
    {
      for (int row = 0; row <= step; ++row)
      {
        Scalar const coefficient = ops.inner_product(factorization.basis[static_cast<std::size_t>(row)], residual);
        factorization.hessenberg[static_cast<std::size_t>(row), static_cast<std::size_t>(step)] += coefficient;
        ops.axpy(residual, -coefficient, factorization.basis[static_cast<std::size_t>(row)]);
      }
    }

    Real const beta = norm_or_inner_product<Scalar>(ops, residual);
    factorization.residual_norm = beta;
    factorization.step_count = step + 1;
    if (beta <= effective_breakdown_tolerance)
    {
      factorization.happy_breakdown = true;
      break;
    }

    factorization.hessenberg[static_cast<std::size_t>(step + 1), static_cast<std::size_t>(step)] =
        static_cast<Scalar>(beta);
    ops.scal(residual, Scalar{1} / static_cast<Scalar>(beta));
    factorization.basis.push_back(std::move(residual));
  }

  return factorization;
}

/// \brief Copy the square projected Hessenberg block from an Arnoldi factorization.
/// \tparam Scalar Vector-space scalar type.
/// \tparam Vector Opaque vector type used by the Arnoldi basis.
/// \param factorization Arnoldi factorization to extract from.
/// \return The leading `m`-by-`m` projected Hessenberg matrix.
template <uni20::RealOrComplex Scalar, typename Vector>
Matrix<Scalar> arnoldi_projected_hessenberg(ArnoldiFactorization<Scalar, Vector> const& factorization)
{
  if (factorization.step_count <= 0)
  {
    throw std::invalid_argument("arnoldi_projected_hessenberg requires a non-empty Arnoldi factorization");
  }

  std::size_t const projected_dimension = static_cast<std::size_t>(factorization.step_count);
  if (factorization.hessenberg.rows() < projected_dimension || factorization.hessenberg.cols() < projected_dimension)
  {
    throw std::invalid_argument("arnoldi_projected_hessenberg received an inconsistent Hessenberg projection");
  }

  Matrix<Scalar> projected(projected_dimension, projected_dimension);
  for (std::size_t col = 0; col < projected_dimension; ++col)
  {
    for (std::size_t row = 0; row < projected_dimension; ++row)
    {
      projected[row, col] = factorization.hessenberg[row, col];
    }
  }
  return projected;
}

/// \brief Extract projected Ritz values and residual estimates from a real Arnoldi factorization.
///
/// \details The Arnoldi relation gives
///          `A V_m y - theta V_m y = h_{m+1,m} (e_m^T y) v_{m+1}` before
///          happy breakdown. This routine solves the small projected
///          Hessenberg problem and reports that residual bound without touching
///          the caller-owned vector backend.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \tparam Vector Opaque vector type used by the Arnoldi basis.
/// \param factorization Arnoldi factorization to extract from.
/// \param complex_pair_tolerance Relative imaginary-part tolerance for reality classification.
/// \param scale Optional projected-problem scale for reality classification.
/// \return Complex Ritz values, projected right eigenvectors, residual bounds, and reality classifications.
template <uni20::LapackReal Scalar, typename Vector>
ArnoldiRitzExtraction<Scalar> extract_arnoldi_ritz(ArnoldiFactorization<Scalar, Vector> const& factorization,
                                                   Scalar complex_pair_tolerance = Scalar{}, Scalar scale = Scalar{1})
{
  if (factorization.step_count <= 0)
  {
    throw std::invalid_argument("extract_arnoldi_ritz requires a non-empty Arnoldi factorization");
  }

  std::size_t const projected_dimension = static_cast<std::size_t>(factorization.step_count);
  if (factorization.hessenberg.rows() < projected_dimension || factorization.hessenberg.cols() < projected_dimension)
  {
    throw std::invalid_argument("extract_arnoldi_ritz received an inconsistent Hessenberg projection");
  }

  Matrix<Scalar> projected = arnoldi_projected_hessenberg(factorization);
  Scalar hessenberg_scale = Scalar{1};
  for (std::size_t col = 0; col < projected_dimension; ++col)
  {
    for (std::size_t row = 0; row < projected_dimension; ++row)
    {
      Scalar const value = factorization.hessenberg[row, col];
      hessenberg_scale = std::max(hessenberg_scale, detail::adl_abs(value));
    }
  }

  auto eigensystem = real_nonsymmetric_eigensystem(std::move(projected), true);

  ArnoldiRitzExtraction<Scalar> result;
  result.ritz_values = std::move(eigensystem.eigenvalues);
  result.projected_right_eigenvectors = std::move(eigensystem.right_eigenvectors);
  result.residual_bounds.resize(projected_dimension, Scalar{});
  result.reality.resize(projected_dimension, RitzReality::Real);

  Scalar const beta = factorization.happy_breakdown
                          ? Scalar{}
                          : detail::adl_abs(factorization.hessenberg[projected_dimension, projected_dimension - 1]);
  Scalar const classification_scale = std::max(scale, hessenberg_scale);
  for (std::size_t col = 0; col < projected_dimension; ++col)
  {
    Scalar norm_squared = Scalar{};
    for (std::size_t row = 0; row < projected_dimension; ++row)
    {
      norm_squared += detail::abs_squared(result.projected_right_eigenvectors[row, col]);
    }
    Scalar const vector_norm = detail::adl_sqrt(norm_squared);
    if (vector_norm == Scalar{})
    {
      throw std::runtime_error("extract_arnoldi_ritz received a zero projected eigenvector from LAPACK");
    }
    Scalar const last_component = detail::adl_abs(result.projected_right_eigenvectors[projected_dimension - 1, col]);
    result.residual_bounds[col] = beta * last_component / vector_norm;
    result.reality[col] = classify_ritz_reality(result.ritz_values[col], complex_pair_tolerance, classification_scale);
  }

  return result;
}

/// \brief Extract projected Ritz values from a complex Arnoldi factorization.
///
/// \details The Arnoldi relation gives
///          `A V_m y - theta V_m y = h_{m+1,m} (e_m^T y) v_{m+1}` before
///          happy breakdown. This routine solves the small complex projected
///          Hessenberg problem and reports that residual bound.
/// \tparam Real Real precision whose `uni20::complex<Real>` has dense LAPACK coverage.
/// \tparam Vector Opaque vector type used by the Arnoldi basis.
/// \param factorization Complex Arnoldi factorization to extract from.
/// \param complex_pair_tolerance Relative imaginary-part tolerance for reality classification.
/// \param scale Optional projected-problem scale for reality classification.
/// \return Complex Ritz values, projected right eigenvectors, residual bounds, and reality classifications.
template <uni20::LapackComplexReal Real, typename Vector>
ArnoldiRitzExtraction<Real>
extract_complex_arnoldi_ritz(ArnoldiFactorization<uni20::complex<Real>, Vector> const& factorization,
                             Real complex_pair_tolerance = Real{}, Real scale = Real{1})
{
  if (factorization.step_count <= 0)
  {
    throw std::invalid_argument("extract_complex_arnoldi_ritz requires a non-empty Arnoldi factorization");
  }

  using Complex = uni20::complex<Real>;

  std::size_t const projected_dimension = static_cast<std::size_t>(factorization.step_count);
  if (factorization.hessenberg.rows() < projected_dimension || factorization.hessenberg.cols() < projected_dimension)
  {
    throw std::invalid_argument("extract_complex_arnoldi_ritz received an inconsistent Hessenberg projection");
  }

  Matrix<Complex> projected = arnoldi_projected_hessenberg(factorization);
  Real hessenberg_scale = Real{1};
  for (std::size_t col = 0; col < projected_dimension; ++col)
  {
    for (std::size_t row = 0; row < projected_dimension; ++row)
    {
      hessenberg_scale = std::max(hessenberg_scale, detail::adl_abs(factorization.hessenberg[row, col]));
    }
  }

  auto eigensystem = complex_nonsymmetric_eigensystem<Real>(std::move(projected), true);

  ArnoldiRitzExtraction<Real> result;
  result.ritz_values = std::move(eigensystem.eigenvalues);
  result.projected_right_eigenvectors = std::move(eigensystem.right_eigenvectors);
  result.residual_bounds.resize(projected_dimension, Real{});
  result.reality.resize(projected_dimension, RitzReality::Real);

  Real const beta = factorization.happy_breakdown
                        ? Real{}
                        : detail::adl_abs(factorization.hessenberg[projected_dimension, projected_dimension - 1]);
  Real const classification_scale = std::max(scale, hessenberg_scale);
  for (std::size_t col = 0; col < projected_dimension; ++col)
  {
    Real norm_squared = Real{};
    for (std::size_t row = 0; row < projected_dimension; ++row)
    {
      norm_squared += detail::abs_squared(result.projected_right_eigenvectors[row, col]);
    }
    Real const vector_norm = detail::adl_sqrt(norm_squared);
    if (vector_norm == Real{})
    {
      throw std::runtime_error("extract_complex_arnoldi_ritz received a zero projected eigenvector from LAPACK");
    }
    Real const last_component = detail::adl_abs(result.projected_right_eigenvectors[projected_dimension - 1, col]);
    result.residual_bounds[col] = beta * last_component / vector_norm;
    result.reality[col] = classify_ritz_reality(result.ritz_values[col], complex_pair_tolerance, classification_scale);
  }

  return result;
}

/// \brief Select wanted Ritz indices for a nonsymmetric projected spectrum.
///
/// \details Nonsymmetric projected eigenvalues may be complex. `LargestReal`
///          and `SmallestReal` use the real part, imaginary selectors use the
///          signed imaginary part, and magnitude selectors use complex
///          magnitude. Algebraic and both-end selectors are deliberately
///          rejected because they are symmetric/Hermitian concepts.
/// \tparam Scalar Real scalar type.
/// \param values Projected Ritz values.
/// \param params Solver parameters defining `nev` and selector.
/// \return Indices of selected wanted Ritz values.
template <uni20::Real Scalar>
std::vector<std::size_t> select_nonsymmetric_ritz_indices(std::vector<uni20::complex<Scalar>> const& values,
                                                          NonsymmetricEigenParams<Scalar> const& params)
{
  if (params.eigenvalue_count <= 0)
  {
    throw std::invalid_argument("nonsymmetric Ritz selection requires a positive eigenvalue count");
  }
  if (static_cast<std::size_t>(params.eigenvalue_count) > values.size())
  {
    throw std::invalid_argument("nonsymmetric Ritz selection requires eigenvalue_count <= projected size");
  }

  std::vector<std::size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});

  switch (params.spectrum)
  {
    case SpectrumPart::LargestMagnitude:
      detail::sort_indices(indices, [&](std::size_t lhs, std::size_t rhs) {
        return detail::adl_abs(values[lhs]) > detail::adl_abs(values[rhs]);
      });
      break;
    case SpectrumPart::SmallestMagnitude:
      detail::sort_indices(indices, [&](std::size_t lhs, std::size_t rhs) {
        return detail::adl_abs(values[lhs]) < detail::adl_abs(values[rhs]);
      });
      break;
    case SpectrumPart::LargestReal:
      detail::sort_indices(indices,
                           [&](std::size_t lhs, std::size_t rhs) { return values[lhs].real() > values[rhs].real(); });
      break;
    case SpectrumPart::SmallestReal:
      detail::sort_indices(indices,
                           [&](std::size_t lhs, std::size_t rhs) { return values[lhs].real() < values[rhs].real(); });
      break;
    case SpectrumPart::LargestImaginary:
      detail::sort_indices(indices,
                           [&](std::size_t lhs, std::size_t rhs) { return values[lhs].imag() > values[rhs].imag(); });
      break;
    case SpectrumPart::SmallestImaginary:
      detail::sort_indices(indices,
                           [&](std::size_t lhs, std::size_t rhs) { return values[lhs].imag() < values[rhs].imag(); });
      break;
    case SpectrumPart::LargestAlgebraic:
    case SpectrumPart::SmallestAlgebraic:
    case SpectrumPart::BothEnds:
      throw std::invalid_argument("algebraic and both-end selectors are not valid for nonsymmetric Arnoldi");
  }

  indices.resize(static_cast<std::size_t>(params.eigenvalue_count));
  return indices;
}

namespace detail
{
template <typename Scalar> std::vector<std::size_t> all_ritz_indices(std::vector<uni20::complex<Scalar>> const& values)
{
  std::vector<std::size_t> indices(values.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  return indices;
}
} // namespace detail

/// \brief Select retained Ritz or Schur indices for a nonsymmetric thick restart.
/// \tparam Scalar Real scalar type.
/// \param values Projected Ritz or Schur eigenvalues.
/// \param params Solver parameters defining selector and retained count.
/// \return Indices of retained values in wanted order.
template <uni20::Real Scalar>
std::vector<std::size_t> select_nonsymmetric_restart_indices(std::vector<uni20::complex<Scalar>> const& values,
                                                             NonsymmetricEigenParams<Scalar> const& params)
{
  NonsymmetricEigenParams<Scalar> restart_params = params;
  restart_params.eigenvalue_count = effective_nonsymmetric_retained_ritz_count(params, static_cast<int>(values.size()));
  return select_nonsymmetric_ritz_indices(values, restart_params);
}

/// \brief Select real Schur blocks for a nonsymmetric thick restart.
///
/// \details The requested retained dimension is interpreted as a number of
///          scalar Schur dimensions. Since real arithmetic represents complex
///          conjugate eigenpairs as 2x2 Schur blocks, the returned
///          `scalar_count` may exceed the requested count by one when needed
///          to avoid cutting a complex block.
/// \tparam Scalar Real scalar type.
/// \param blocks Real Schur blocks in current projected order.
/// \param params Nonsymmetric solve parameters defining selector and retained count.
/// \return Selected block indices in wanted order and the corresponding scalar dimension.
template <uni20::LapackReal Scalar>
RealSchurRestartSelection<Scalar> select_real_schur_restart_blocks(std::vector<RealSchurBlock<Scalar>> const& blocks,
                                                                   NonsymmetricEigenParams<Scalar> const& params)
{
  if (params.eigenvalue_count <= 0)
  {
    throw std::invalid_argument("nonsymmetric Schur restart requires a positive eigenvalue count");
  }
  std::size_t total_scalar_count = 0;
  for (auto const& block : blocks)
  {
    if (block.size != 1 && block.size != 2)
    {
      throw std::invalid_argument("nonsymmetric Schur restart received an invalid block size");
    }
    total_scalar_count += block.size;
  }
  int const requested_count = effective_nonsymmetric_retained_ritz_count(params, static_cast<int>(total_scalar_count));

  std::vector<std::size_t> indices(blocks.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  bool const descending = detail::larger_is_better<Scalar>(params.spectrum);
  detail::sort_indices(indices, [&](std::size_t lhs, std::size_t rhs) {
    Scalar const lhs_score = detail::schur_block_score(blocks[lhs], params.spectrum);
    Scalar const rhs_score = detail::schur_block_score(blocks[rhs], params.spectrum);
    return descending ? lhs_score > rhs_score : lhs_score < rhs_score;
  });

  RealSchurRestartSelection<Scalar> selection;
  for (std::size_t const index : indices)
  {
    if (selection.scalar_count >= static_cast<std::size_t>(requested_count))
    {
      break;
    }
    selection.block_indices.push_back(index);
    selection.scalar_count += blocks[index].size;
  }
  selection.enlarged_to_preserve_block = selection.scalar_count > static_cast<std::size_t>(requested_count);
  return selection;
}

/// \brief Compress an Arnoldi factorization through real Schur block reordering.
///
/// \details Computes the real Schur form of the projected Hessenberg matrix,
///          reorders selected Schur blocks to the leading part, transforms the
///          opaque Arnoldi basis by the Schur vectors, and returns the
///          compressed Krylov-Schur relation `A V_k = V_k T_k + f b^T`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free vector operation provider.
/// \param ops Matrix-free vector operations.
/// \param factorization Input Arnoldi factorization.
/// \param params Nonsymmetric solve parameters used for block selection.
/// \return Compressed restart relation ready for Krylov-Schur expansion.
template <uni20::LapackReal Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
RealSchurCompressedArnoldiFactorization<Scalar, Vector>
compress_real_schur_arnoldi_restart(Ops& ops, ArnoldiFactorization<Scalar, Vector> const& factorization,
                                    NonsymmetricEigenParams<Scalar> const& params)
{
  if (factorization.step_count <= 0)
  {
    throw std::invalid_argument("real Schur Arnoldi compression requires a non-empty factorization");
  }

  std::size_t const order = static_cast<std::size_t>(factorization.step_count);
  if (factorization.basis.size() < order || (!factorization.happy_breakdown && factorization.basis.size() <= order))
  {
    throw std::invalid_argument("real Schur Arnoldi compression received an inconsistent basis");
  }

  Matrix<Scalar> projected = arnoldi_projected_hessenberg(factorization);
  auto schur = real_schur(std::move(projected), true);
  RealSchurRestartSelection<Scalar> selection = select_real_schur_restart_blocks(schur.blocks, params);
  if (selection.scalar_count == 0 || selection.scalar_count > order)
  {
    throw std::invalid_argument("real Schur Arnoldi compression selected an invalid retained dimension");
  }
  schur = reorder_real_schur(std::move(schur), selection.block_indices);

  std::size_t const retained_count = selection.scalar_count;
  auto transformed_column = [&](std::size_t column) {
    Vector vector = ops.allocate_like(factorization.basis.front());
    ops.set_zero(vector);
    for (std::size_t row = 0; row < order; ++row)
    {
      Scalar const coefficient = schur.schur_vectors[row, column];
      if (coefficient != Scalar{})
      {
        ops.axpy(vector, coefficient, factorization.basis[row]);
      }
    }
    return vector;
  };

  RealSchurCompressedArnoldiFactorization<Scalar, Vector> result{
      .basis = {},
      .residual = ops.allocate_like(factorization.basis.front()),
      .schur_form = Matrix<Scalar>(retained_count, retained_count),
      .residual_coupling = std::vector<Scalar>(retained_count, Scalar{}),
      .eigenvalues = {},
      .blocks = {},
      .residual_norm = Scalar{},
      .happy_breakdown = factorization.happy_breakdown,
      .enlarged_to_preserve_block = selection.enlarged_to_preserve_block};

  result.basis.reserve(retained_count);
  for (std::size_t column = 0; column < retained_count; ++column)
  {
    result.basis.push_back(transformed_column(column));
  }

  for (std::size_t col = 0; col < retained_count; ++col)
  {
    for (std::size_t row = 0; row < retained_count; ++row)
    {
      result.schur_form[row, col] = schur.schur_form[row, col];
    }
  }

  result.eigenvalues.assign(schur.eigenvalues.begin(),
                            schur.eigenvalues.begin() + static_cast<std::ptrdiff_t>(retained_count));
  for (auto const& block : schur.blocks)
  {
    if (block.begin >= retained_count)
    {
      break;
    }
    if (block.begin + block.size > retained_count)
    {
      throw std::logic_error("real Schur Arnoldi compression split a Schur block");
    }
    result.blocks.push_back(block);
  }

  ops.set_zero(result.residual);
  if (!factorization.happy_breakdown)
  {
    Scalar const beta = factorization.hessenberg[order, order - 1];
    ops.copy(result.residual, factorization.basis[order]);
    ops.scal(result.residual, beta);
    for (std::size_t col = 0; col < retained_count; ++col)
    {
      result.residual_coupling[col] = schur.schur_vectors[order - 1, col];
    }
  }
  result.residual_norm = norm_or_inner_product<Scalar>(ops, result.residual);
  return result;
}

/// \brief Expand a compressed real Schur Arnoldi relation back to a requested dimension.
///
/// \details The compressed relation has the form
///          `A V_k = V_k T_k + f b^T`. If `f` is nonzero, this routine
///          normalizes it as the next Arnoldi vector, writes
///          `||f|| b^T` into the next projected row, and resumes full
///          reorthogonalized Arnoldi expansion until `target_step_count` is
///          reached or happy breakdown occurs. The returned factorization uses
///          the same storage convention as `arnoldi_factorize`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free vector operation provider.
/// \param ops Matrix-free vector operations and operator application.
/// \param compressed Compressed Schur restart relation.
/// \param target_step_count Desired projected dimension after expansion.
/// \param breakdown_tolerance Absolute residual norm threshold for happy breakdown.
/// \return Expanded Arnoldi relation.
template <uni20::LapackReal Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
ArnoldiFactorization<Scalar, Vector>
expand_real_schur_arnoldi_restart(Ops& ops, RealSchurCompressedArnoldiFactorization<Scalar, Vector> const& compressed,
                                  std::size_t target_step_count, Scalar breakdown_tolerance = Scalar{})
{
  std::size_t const retained_count = compressed.basis.size();
  if (retained_count == 0)
  {
    throw std::invalid_argument("real Schur Arnoldi expansion requires a nonempty retained basis");
  }
  if (target_step_count < retained_count)
  {
    throw std::invalid_argument("real Schur Arnoldi expansion target is smaller than retained basis");
  }
  if (compressed.schur_form.rows() != retained_count || compressed.schur_form.cols() != retained_count ||
      compressed.residual_coupling.size() != retained_count)
  {
    throw std::invalid_argument("real Schur Arnoldi expansion received inconsistent compressed dimensions");
  }
  std::size_t const problem_dimension =
      validate_matrix_free_dimensions(ops, compressed.basis.front(), "real Schur Arnoldi expansion");
  if (target_step_count > problem_dimension)
  {
    throw std::invalid_argument("real Schur Arnoldi expansion target exceeds problem dimension");
  }

  Scalar const effective_breakdown_tolerance =
      breakdown_tolerance > Scalar{} ? breakdown_tolerance : Scalar{10} * uni20::numeric_limits<Scalar>::epsilon();

  ArnoldiFactorization<Scalar, Vector> factorization;
  factorization.hessenberg = Matrix<Scalar>(target_step_count + 1, target_step_count);
  factorization.basis.reserve(target_step_count + 1);
  for (auto const& vector : compressed.basis)
  {
    Vector copy = ops.allocate_like(vector);
    ops.copy(copy, vector);
    factorization.basis.push_back(std::move(copy));
  }

  for (std::size_t col = 0; col < retained_count; ++col)
  {
    for (std::size_t row = 0; row < retained_count; ++row)
    {
      factorization.hessenberg[row, col] = compressed.schur_form[row, col];
    }
  }

  factorization.step_count = static_cast<int>(retained_count);
  factorization.residual_norm = compressed.residual_norm;
  if (compressed.residual_norm <= effective_breakdown_tolerance)
  {
    factorization.happy_breakdown =
        compressed.happy_breakdown || compressed.residual_norm <= effective_breakdown_tolerance;
    return factorization;
  }

  Vector next_vector = ops.allocate_like(compressed.residual);
  ops.copy(next_vector, compressed.residual);
  ops.scal(next_vector, Scalar{1} / compressed.residual_norm);
  factorization.basis.push_back(std::move(next_vector));
  for (std::size_t col = 0; col < retained_count; ++col)
  {
    factorization.hessenberg[retained_count, col] = compressed.residual_norm * compressed.residual_coupling[col];
  }
  if (target_step_count == retained_count)
  {
    return factorization;
  }

  for (std::size_t step = retained_count; step < target_step_count; ++step)
  {
    Vector residual = ops.allocate_like(factorization.basis[step]);
    ops.matvec(residual, factorization.basis[step]);
    ++factorization.op_count;

    for (int pass = 0; pass < 2; ++pass)
    {
      for (std::size_t row = 0; row <= step; ++row)
      {
        Scalar const coefficient = ops.inner_product(factorization.basis[row], residual);
        factorization.hessenberg[row, step] += coefficient;
        ops.axpy(residual, -coefficient, factorization.basis[row]);
      }
    }

    Scalar const beta = norm_or_inner_product<Scalar>(ops, residual);
    factorization.residual_norm = beta;
    factorization.step_count = static_cast<int>(step + 1);
    if (beta <= effective_breakdown_tolerance)
    {
      factorization.happy_breakdown = true;
      break;
    }

    factorization.hessenberg[step + 1, step] = beta;
    ops.scal(residual, Scalar{1} / beta);
    factorization.basis.push_back(std::move(residual));
  }

  return factorization;
}

/// \brief Compress a complex Arnoldi factorization through complex Schur reordering.
///
/// \details Computes the complex Schur form of the projected Hessenberg
///          matrix, reorders selected Schur entries to the leading part,
///          transforms the opaque Arnoldi basis by the Schur vectors, and
///          returns the compressed Krylov-Schur relation
///          `A V_k = V_k T_k + f b^T`.
/// \tparam Real Real precision whose `uni20::complex<Real>` has dense LAPACK coverage.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free vector operation provider.
/// \param ops Matrix-free vector operations.
/// \param factorization Input Arnoldi factorization.
/// \param params Nonsymmetric solve parameters used for Schur entry selection.
/// \return Compressed restart relation ready for Krylov-Schur expansion.
template <uni20::LapackComplexReal Real, typename Vector, KrylovMatrixFreeOperator<Vector, uni20::complex<Real>> Ops>
ComplexSchurCompressedArnoldiFactorization<Real, Vector>
compress_complex_schur_arnoldi_restart(Ops& ops,
                                       ArnoldiFactorization<uni20::complex<Real>, Vector> const& factorization,
                                       NonsymmetricEigenParams<Real> const& params)
{
  if (factorization.step_count <= 0)
  {
    throw std::invalid_argument("complex Schur Arnoldi compression requires a non-empty factorization");
  }

  using Complex = uni20::complex<Real>;

  std::size_t const order = static_cast<std::size_t>(factorization.step_count);
  if (factorization.basis.size() < order || (!factorization.happy_breakdown && factorization.basis.size() <= order))
  {
    throw std::invalid_argument("complex Schur Arnoldi compression received an inconsistent basis");
  }

  Matrix<Complex> projected = arnoldi_projected_hessenberg(factorization);
  auto schur = complex_schur<Real>(std::move(projected), true);
  std::vector<std::size_t> const retained_indices = select_nonsymmetric_restart_indices(schur.eigenvalues, params);
  if (retained_indices.empty() || retained_indices.size() > order)
  {
    throw std::invalid_argument("complex Schur Arnoldi compression selected an invalid retained dimension");
  }
  schur = reorder_complex_schur<Real>(std::move(schur), retained_indices);

  std::size_t const retained_count = retained_indices.size();
  auto transformed_column = [&](std::size_t column) {
    Vector vector = ops.allocate_like(factorization.basis.front());
    ops.set_zero(vector);
    for (std::size_t row = 0; row < order; ++row)
    {
      Complex const coefficient = schur.schur_vectors[row, column];
      if (coefficient != Complex{})
      {
        ops.axpy(vector, coefficient, factorization.basis[row]);
      }
    }
    return vector;
  };

  ComplexSchurCompressedArnoldiFactorization<Real, Vector> result{
      .basis = {},
      .residual = ops.allocate_like(factorization.basis.front()),
      .schur_form = Matrix<Complex>(retained_count, retained_count),
      .residual_coupling = std::vector<Complex>(retained_count, Complex{}),
      .eigenvalues = {},
      .residual_norm = Real{},
      .happy_breakdown = factorization.happy_breakdown};

  result.basis.reserve(retained_count);
  for (std::size_t column = 0; column < retained_count; ++column)
  {
    result.basis.push_back(transformed_column(column));
  }

  for (std::size_t col = 0; col < retained_count; ++col)
  {
    for (std::size_t row = 0; row < retained_count; ++row)
    {
      result.schur_form[row, col] = schur.schur_form[row, col];
    }
  }

  result.eigenvalues.assign(schur.eigenvalues.begin(),
                            schur.eigenvalues.begin() + static_cast<std::ptrdiff_t>(retained_count));

  ops.set_zero(result.residual);
  if (!factorization.happy_breakdown)
  {
    Complex const beta = factorization.hessenberg[order, order - 1];
    ops.copy(result.residual, factorization.basis[order]);
    ops.scal(result.residual, beta);
    for (std::size_t col = 0; col < retained_count; ++col)
    {
      result.residual_coupling[col] = schur.schur_vectors[order - 1, col];
    }
  }
  result.residual_norm = norm_or_inner_product<Complex>(ops, result.residual);
  return result;
}

/// \brief Expand a compressed complex Schur Arnoldi relation back to a requested dimension.
///
/// \details The compressed relation has the form
///          `A V_k = V_k T_k + f b^T`. If `f` is nonzero, this routine
///          normalizes it as the next Arnoldi vector, writes `||f|| b^T` into
///          the next projected row, and resumes full reorthogonalized Arnoldi
///          expansion until `target_step_count` is reached or happy breakdown
///          occurs.
/// \tparam Real Real precision whose `uni20::complex<Real>` has dense LAPACK coverage.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free vector operation provider.
/// \param ops Matrix-free vector operations and operator application.
/// \param compressed Compressed Schur restart relation.
/// \param target_step_count Desired projected dimension after expansion.
/// \param breakdown_tolerance Absolute residual norm threshold for happy breakdown.
/// \return Expanded Arnoldi relation.
template <uni20::LapackComplexReal Real, typename Vector, KrylovMatrixFreeOperator<Vector, uni20::complex<Real>> Ops>
ArnoldiFactorization<uni20::complex<Real>, Vector>
expand_complex_schur_arnoldi_restart(Ops& ops,
                                     ComplexSchurCompressedArnoldiFactorization<Real, Vector> const& compressed,
                                     std::size_t target_step_count, Real breakdown_tolerance = Real{})
{
  using Complex = uni20::complex<Real>;

  std::size_t const retained_count = compressed.basis.size();
  if (retained_count == 0)
  {
    throw std::invalid_argument("complex Schur Arnoldi expansion requires a nonempty retained basis");
  }
  if (target_step_count < retained_count)
  {
    throw std::invalid_argument("complex Schur Arnoldi expansion target is smaller than retained basis");
  }
  if (compressed.schur_form.rows() != retained_count || compressed.schur_form.cols() != retained_count ||
      compressed.residual_coupling.size() != retained_count)
  {
    throw std::invalid_argument("complex Schur Arnoldi expansion received inconsistent compressed dimensions");
  }
  std::size_t const problem_dimension =
      validate_matrix_free_dimensions(ops, compressed.basis.front(), "complex Schur Arnoldi expansion");
  if (target_step_count > problem_dimension)
  {
    throw std::invalid_argument("complex Schur Arnoldi expansion target exceeds problem dimension");
  }

  Real const effective_breakdown_tolerance =
      breakdown_tolerance > Real{} ? breakdown_tolerance : Real{10} * uni20::numeric_limits<Real>::epsilon();

  ArnoldiFactorization<Complex, Vector> factorization;
  factorization.hessenberg = Matrix<Complex>(target_step_count + 1, target_step_count);
  factorization.basis.reserve(target_step_count + 1);
  for (auto const& vector : compressed.basis)
  {
    Vector copy = ops.allocate_like(vector);
    ops.copy(copy, vector);
    factorization.basis.push_back(std::move(copy));
  }

  for (std::size_t col = 0; col < retained_count; ++col)
  {
    for (std::size_t row = 0; row < retained_count; ++row)
    {
      factorization.hessenberg[row, col] = compressed.schur_form[row, col];
    }
  }

  factorization.step_count = static_cast<int>(retained_count);
  factorization.residual_norm = compressed.residual_norm;
  if (compressed.residual_norm <= effective_breakdown_tolerance)
  {
    factorization.happy_breakdown =
        compressed.happy_breakdown || compressed.residual_norm <= effective_breakdown_tolerance;
    return factorization;
  }

  Vector next_vector = ops.allocate_like(compressed.residual);
  ops.copy(next_vector, compressed.residual);
  ops.scal(next_vector, Complex{1} / compressed.residual_norm);
  factorization.basis.push_back(std::move(next_vector));
  for (std::size_t col = 0; col < retained_count; ++col)
  {
    factorization.hessenberg[retained_count, col] = compressed.residual_norm * compressed.residual_coupling[col];
  }
  if (target_step_count == retained_count)
  {
    return factorization;
  }

  for (std::size_t step = retained_count; step < target_step_count; ++step)
  {
    Vector residual = ops.allocate_like(factorization.basis[step]);
    ops.matvec(residual, factorization.basis[step]);
    ++factorization.op_count;

    for (int pass = 0; pass < 2; ++pass)
    {
      for (std::size_t row = 0; row <= step; ++row)
      {
        Complex const coefficient = ops.inner_product(factorization.basis[row], residual);
        factorization.hessenberg[row, step] += coefficient;
        ops.axpy(residual, -coefficient, factorization.basis[row]);
      }
    }

    Real const beta = norm_or_inner_product<Complex>(ops, residual);
    factorization.residual_norm = beta;
    factorization.step_count = static_cast<int>(step + 1);
    if (beta <= effective_breakdown_tolerance)
    {
      factorization.happy_breakdown = true;
      break;
    }

    factorization.hessenberg[step + 1, step] = beta;
    ops.scal(residual, Complex{1} / beta);
    factorization.basis.push_back(std::move(residual));
  }

  return factorization;
}

/// \brief Run the first native non-restarted real nonsymmetric Arnoldi solver.
///
/// \details This Phase 4 entry point keeps the Arnoldi basis and matrix-free
///          operations real, extracts complex projected Ritz values, and
///          reports when wanted Ritz values require complex arithmetic or a
///          future real-Schur two-plane output path.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free operator/vector operation type.
/// \param ops Matrix-free vector operations and operator application.
/// \param initial Initial residual/start vector.
/// \param params Solver parameters.
/// \return Selected Ritz values, residual bounds, optional real Ritz vectors, and diagnostics.
template <uni20::LapackReal Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
NonsymmetricEigenResult<Scalar, Vector>
real_nonsymmetric_arnoldi_standard(Ops& ops, Vector const& initial, NonsymmetricEigenParams<Scalar> const& params)
{
  if (params.eigenvalue_count <= 0)
  {
    throw std::invalid_argument("real nonsymmetric Arnoldi requires a positive eigenvalue count");
  }

  std::size_t const problem_dimension = validate_matrix_free_dimensions(ops, initial, "real nonsymmetric Arnoldi");
  int const basis_limit = effective_nonsymmetric_krylov_dimension(params, problem_dimension);
  if (basis_limit <= 0)
  {
    throw std::invalid_argument("real nonsymmetric Arnoldi requires a positive Krylov dimension");
  }
  if (basis_limit > static_cast<int>(problem_dimension))
  {
    throw std::invalid_argument("real nonsymmetric Arnoldi requires krylov_dimension <= problem dimension");
  }
  if (params.eigenvalue_count > basis_limit)
  {
    throw std::invalid_argument("real nonsymmetric Arnoldi requires eigenvalue_count <= krylov_dimension");
  }

  ArnoldiFactorization<Scalar, Vector> factorization = arnoldi_factorize<Scalar>(ops, initial, basis_limit);
  ArnoldiRitzExtraction<Scalar> ritz = extract_arnoldi_ritz(factorization, params.complex_pair_tolerance);
  bool const undersized_projection = ritz.ritz_values.size() < static_cast<std::size_t>(params.eigenvalue_count);
  std::vector<std::size_t> const selected = undersized_projection
                                                ? detail::all_ritz_indices(ritz.ritz_values)
                                                : select_nonsymmetric_ritz_indices(ritz.ritz_values, params);
  Scalar const projected_departure = projected_departure_from_normality(arnoldi_projected_hessenberg(factorization));
  return detail::make_real_nonsymmetric_arnoldi_result(ops, factorization, ritz, selected, params,
                                                       factorization.op_count, factorization.op_count, 0,
                                                       projected_departure);
}

/// \brief Run a restarted complex nonsymmetric Arnoldi solver.
///
/// \details This Phase 5 entry point keeps the Arnoldi basis and matrix-free
///          operations complex, extracts complex projected Ritz values, and
///          uses complex Schur thick restarts when the initial projected
///          subspace does not satisfy the convergence criterion.
/// \tparam Real Underlying real precision.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free operator/vector operation type.
/// \param ops Matrix-free vector operations and operator application.
/// \param initial Initial residual/start vector.
/// \param params Solver parameters. Tolerances and residuals use \p Real.
/// \return Selected Ritz values, residual bounds, optional complex Ritz vectors, and diagnostics.
template <uni20::LapackComplexReal Real, typename Vector, KrylovMatrixFreeOperator<Vector, uni20::complex<Real>> Ops>
NonsymmetricEigenResult<Real, Vector> complex_nonsymmetric_arnoldi_standard(Ops& ops, Vector const& initial,
                                                                            NonsymmetricEigenParams<Real> const& params)
{
  if (params.eigenvalue_count <= 0)
  {
    throw std::invalid_argument("complex nonsymmetric Arnoldi requires a positive eigenvalue count");
  }

  using Complex = uni20::complex<Real>;

  std::size_t const problem_dimension = validate_matrix_free_dimensions(ops, initial, "complex nonsymmetric Arnoldi");
  int const basis_limit = effective_nonsymmetric_krylov_dimension(params, problem_dimension);
  if (basis_limit <= 0)
  {
    throw std::invalid_argument("complex nonsymmetric Arnoldi requires a positive Krylov dimension");
  }
  if (basis_limit > static_cast<int>(problem_dimension))
  {
    throw std::invalid_argument("complex nonsymmetric Arnoldi requires krylov_dimension <= problem dimension");
  }
  if (params.eigenvalue_count > basis_limit)
  {
    throw std::invalid_argument("complex nonsymmetric Arnoldi requires eigenvalue_count <= krylov_dimension");
  }
  int const requested_retained_count = effective_nonsymmetric_retained_ritz_count(params, basis_limit);
  if (params.max_iterations < params.eigenvalue_count)
  {
    throw std::invalid_argument("complex nonsymmetric Arnoldi iteration limit is too small for nev");
  }

  int const initial_steps = std::min(basis_limit, params.max_iterations);
  ArnoldiFactorization<Complex, Vector> factorization = arnoldi_factorize<Complex>(ops, initial, initial_steps);
  int matvec_count = factorization.op_count;
  int restart_count = 0;

  while (true)
  {
    ArnoldiRitzExtraction<Real> ritz = extract_complex_arnoldi_ritz<Real>(factorization, params.complex_pair_tolerance);
    bool const undersized_projection = ritz.ritz_values.size() < static_cast<std::size_t>(params.eigenvalue_count);
    std::vector<std::size_t> const selected = undersized_projection
                                                  ? detail::all_ritz_indices(ritz.ritz_values)
                                                  : select_nonsymmetric_ritz_indices(ritz.ritz_values, params);
    Real const projected_departure = projected_departure_from_normality(arnoldi_projected_hessenberg(factorization));
    NonsymmetricEigenResult<Real, Vector> result = detail::make_complex_nonsymmetric_arnoldi_result(
        ops, factorization, ritz, selected, params, matvec_count, matvec_count, restart_count, projected_departure);

    bool const all_selected_converged = result.converged_count == static_cast<int>(selected.size());
    if (all_selected_converged || factorization.happy_breakdown || matvec_count >= params.max_iterations)
    {
      return result;
    }

    if (static_cast<std::size_t>(factorization.step_count) <= static_cast<std::size_t>(requested_retained_count))
    {
      return result;
    }

    auto compressed = compress_complex_schur_arnoldi_restart<Real>(ops, factorization, params);
    ++restart_count;
    if (compressed.basis.size() >= static_cast<std::size_t>(basis_limit))
    {
      return result;
    }

    int const remaining_matvecs = params.max_iterations - matvec_count;
    if (remaining_matvecs <= 0)
    {
      return result;
    }
    std::size_t const expansion_count = std::min<std::size_t>(
        static_cast<std::size_t>(basis_limit) - compressed.basis.size(), static_cast<std::size_t>(remaining_matvecs));
    if (expansion_count == 0)
    {
      return result;
    }
    std::size_t const target_step_count = compressed.basis.size() + expansion_count;
    factorization = expand_complex_schur_arnoldi_restart<Real>(ops, compressed, target_step_count);
    matvec_count += factorization.op_count;
  }
}

/// \brief Run a restarted native real nonsymmetric Arnoldi solver.
///
/// \details This is a Krylov-Schur style real Arnoldi restart loop. Each cycle
///          expands a real Arnoldi factorization to `krylov_dimension`, checks
///          Ritz convergence, reorders the projected real Schur form to retain
///          wanted Schur blocks, compresses the opaque basis through matrix-free
///          `axpy` operations, and resumes expansion from the compressed
///          relation. Complex wanted Ritz values are tolerated while
///          unconverged, then reported through the configured real-mode policy.
/// \tparam Vector Opaque vector type.
/// \tparam Ops Matrix-free operator/vector operation type.
/// \param ops Matrix-free vector operations and operator application.
/// \param initial Initial residual/start vector.
/// \param params Solver parameters.
/// \return Selected Ritz values, residual bounds, optional real Ritz vectors, and diagnostics.
template <uni20::LapackReal Scalar, typename Vector, KrylovMatrixFreeOperator<Vector, Scalar> Ops>
NonsymmetricEigenResult<Scalar, Vector>
real_nonsymmetric_arnoldi_restarted_standard(Ops& ops, Vector const& initial,
                                             NonsymmetricEigenParams<Scalar> const& params)
{
  if (params.eigenvalue_count <= 0)
  {
    throw std::invalid_argument("restarted real nonsymmetric Arnoldi requires a positive eigenvalue count");
  }
  if (params.max_iterations <= 0)
  {
    throw std::invalid_argument("restarted real nonsymmetric Arnoldi requires a positive iteration limit");
  }

  std::size_t const problem_dimension =
      validate_matrix_free_dimensions(ops, initial, "restarted real nonsymmetric Arnoldi");
  int const basis_limit = effective_nonsymmetric_krylov_dimension(params, problem_dimension);
  if (basis_limit <= 0)
  {
    throw std::invalid_argument("restarted real nonsymmetric Arnoldi requires a positive Krylov dimension");
  }
  if (basis_limit > static_cast<int>(problem_dimension))
  {
    throw std::invalid_argument("restarted real nonsymmetric Arnoldi requires krylov_dimension <= problem dimension");
  }
  if (params.eigenvalue_count > basis_limit)
  {
    throw std::invalid_argument("restarted real nonsymmetric Arnoldi requires eigenvalue_count <= krylov_dimension");
  }

  int const requested_retained_count = effective_nonsymmetric_retained_ritz_count(params, basis_limit);
  if (params.max_iterations < params.eigenvalue_count)
  {
    throw std::invalid_argument("restarted real nonsymmetric Arnoldi iteration limit is too small for nev");
  }

  int const initial_steps = std::min(basis_limit, params.max_iterations);
  ArnoldiFactorization<Scalar, Vector> factorization = arnoldi_factorize<Scalar>(ops, initial, initial_steps);
  int matvec_count = factorization.op_count;
  int restart_count = 0;

  while (true)
  {
    ArnoldiRitzExtraction<Scalar> ritz = extract_arnoldi_ritz(factorization, params.complex_pair_tolerance);
    bool const undersized_projection = ritz.ritz_values.size() < static_cast<std::size_t>(params.eigenvalue_count);
    std::vector<std::size_t> const selected = undersized_projection
                                                  ? detail::all_ritz_indices(ritz.ritz_values)
                                                  : select_nonsymmetric_ritz_indices(ritz.ritz_values, params);
    Scalar const projected_departure = projected_departure_from_normality(arnoldi_projected_hessenberg(factorization));
    NonsymmetricEigenResult<Scalar, Vector> result = detail::make_real_nonsymmetric_arnoldi_result(
        ops, factorization, ritz, selected, params, matvec_count, matvec_count, restart_count, projected_departure);

    bool const all_selected_converged = result.converged_count == static_cast<int>(selected.size());
    if (all_selected_converged || factorization.happy_breakdown || matvec_count >= params.max_iterations)
    {
      return result;
    }

    if (static_cast<std::size_t>(factorization.step_count) <= static_cast<std::size_t>(requested_retained_count))
    {
      return result;
    }

    auto compressed = compress_real_schur_arnoldi_restart<Scalar>(ops, factorization, params);
    ++restart_count;
    if (compressed.basis.size() >= static_cast<std::size_t>(basis_limit))
    {
      return result;
    }

    int const remaining_matvecs = params.max_iterations - matvec_count;
    if (remaining_matvecs <= 0)
    {
      return result;
    }
    std::size_t const expansion_count = std::min<std::size_t>(
        static_cast<std::size_t>(basis_limit) - compressed.basis.size(), static_cast<std::size_t>(remaining_matvecs));
    if (expansion_count == 0)
    {
      return result;
    }
    std::size_t const target_step_count = compressed.basis.size() + expansion_count;
    factorization = expand_real_schur_arnoldi_restart<Scalar>(ops, compressed, target_step_count);
    matvec_count += factorization.op_count;
  }
}

} // namespace uni20::krylov
