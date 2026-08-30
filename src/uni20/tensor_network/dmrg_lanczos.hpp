/**
 * \file dmrg_lanczos.hpp
 * \ingroup tensor_network
 * \brief Defines the lightweight fixed-step Lanczos solve used by finite DMRG.
 */

#pragma once

#include <uni20/core/math.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/krylov/dense_subspace.hpp>
#include <uni20/krylov/matrix_free.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{

/// \brief Numerical policy for one lightweight DMRG local Lanczos solve.
/// \details The solve performs at most the requested number of effective-Hamiltonian
///          applications and returns the smallest Ritz vector without applying a
///          convergence criterion or restarting the Krylov basis.
/// \tparam Real Real component type of the local wavefunction scalar.
template <uni20::Real Real> struct DmrgLanczosOptions
{
    /// \brief Maximum number of effective-Hamiltonian applications per local update.
    std::size_t matvec_iterations = 4;
};

/// \brief Approximate local ground state produced by fixed-step DMRG Lanczos.
/// \tparam Real Real component type of the local wavefunction scalar.
/// \tparam Vector Opaque local wavefunction type.
template <uni20::Real Real, class Vector> struct DmrgLanczosResult
{
    /// \brief Smallest Ritz value of the final projected problem.
    Real energy;
    /// \brief Final three-term-recurrence Ritz residual estimate.
    Real residual_bound;
    /// \brief Smallest Ritz vector reconstructed in the original vector space.
    Vector vector;
    /// \brief Number of completed Lanczos projection steps.
    int iteration_count;
    /// \brief Number of effective-Hamiltonian applications.
    int matvec_count;
};

namespace detail
{

template <uni20::LapackRealOrComplex Scalar>
auto dmrg_lanczos_projection_scalar(Scalar value) -> uni20::make_real_t<Scalar>
{
  using Real = uni20::make_real_t<Scalar>;
  using std::abs;
  Real const real_part = static_cast<Real>(uni20::real(value));
  if constexpr (uni20::Complex<Scalar>)
  {
    Real const scale = std::max(Real{1}, abs(real_part));
    Real const tolerance = Real{100} * uni20::numeric_limits<Real>::epsilon() * scale;
    if (abs(static_cast<Real>(uni20::imag(value))) > tolerance)
      throw std::runtime_error("DMRG Lanczos received a non-real projected diagonal");
  }
  return real_part;
}

template <uni20::Real Real> auto dmrg_lanczos_breakdown(Real beta, Real alpha, Real previous_beta) -> bool
{
  using std::abs;
  Real const scale = std::max(abs(alpha), abs(previous_beta));
  return beta <= Real{10} * uni20::numeric_limits<Real>::epsilon() * scale;
}

template <uni20::LapackRealOrComplex Scalar, class Vector, class Ops>
void normalize_dmrg_lanczos_vector(Ops& ops, Vector& vector, uni20::make_real_t<Scalar> norm)
{
  using Real = uni20::make_real_t<Scalar>;
  Real const maximum = uni20::numeric_limits<Real>::max();
  if (norm >= Real{1} / maximum)
  {
    ops.scal(vector, Scalar{1} / static_cast<Scalar>(norm));
    return;
  }

  // A uniformly tiny vector still has a meaningful direction, but its direct
  // reciprocal is not representable. Lift it into the normal range first.
  ops.scal(vector, static_cast<Scalar>(maximum));
  Real const lifted_norm = krylov::norm_or_inner_product<Scalar>(ops, vector);
  if (!(lifted_norm > Real{}) || !uni20::isfinite(lifted_norm))
    throw std::runtime_error("DMRG Lanczos could not normalize a finite nonzero vector");
  ops.scal(vector, Scalar{1} / static_cast<Scalar>(lifted_norm));
}

} // namespace detail

/// \brief Perform a lightweight fixed-step Lanczos local ground-state update.
/// \details This is the DMRG local optimization policy, not a general
///          convergence-seeking eigensolver. It uses the Hermitian three-term
///          recurrence without full reorthogonalization, solves one small real
///          tridiagonal projected problem, and always returns its smallest Ritz
///          pair. Exact invariant-subspace breakdown may stop before the requested
///          number of matvecs. The basis is limited by the problem dimension.
/// \throws std::invalid_argument If the problem, initial vector, or iteration count is invalid.
/// \throws std::runtime_error If a complex projection has a non-negligible imaginary diagonal.
/// \tparam Scalar Real or complex local wavefunction scalar with LAPACK coverage.
/// \tparam Vector Opaque local wavefunction type.
/// \tparam Ops Matrix-free vector and effective-Hamiltonian operations.
/// \param ops Matrix-free operations for the fixed local problem.
/// \param initial Current two-site wavefunction used as the starting vector.
/// \param options Fixed local work policy.
/// \return Approximate smallest Ritz pair and local-work diagnostics.
template <uni20::LapackRealOrComplex Scalar, class Vector, krylov::KrylovMatrixFreeOperator<Vector, Scalar> Ops>
[[nodiscard]] auto dmrg_lanczos_ground_state(Ops& ops, Vector const& initial,
                                             DmrgLanczosOptions<uni20::make_real_t<Scalar>> const& options = {})
    -> DmrgLanczosResult<uni20::make_real_t<Scalar>, Vector>
{
  using Real = uni20::make_real_t<Scalar>;
  using std::abs;

  if (options.matvec_iterations == 0)
    throw std::invalid_argument("DMRG Lanczos requires at least one matvec iteration");
  std::size_t const problem_dimension = krylov::validate_matrix_free_dimensions(ops, initial, "DMRG Lanczos");
  std::size_t const basis_limit = std::min(problem_dimension, options.matvec_iterations);

  std::vector<Vector> basis;
  std::vector<Real> diagonal;
  std::vector<Real> subdiagonal;
  basis.reserve(basis_limit);
  diagonal.reserve(basis_limit);
  subdiagonal.reserve(basis_limit - 1);

  Vector start = ops.allocate_like(initial);
  ops.copy(start, initial);
  Real const start_norm = krylov::norm_or_inner_product<Scalar>(ops, start);
  if (!(start_norm > Real{}) || !uni20::isfinite(start_norm))
    throw std::invalid_argument("DMRG Lanczos initial vector must have a finite nonzero norm");
  detail::normalize_dmrg_lanczos_vector<Scalar>(ops, start, start_norm);
  basis.push_back(std::move(start));

  Real final_residual_norm{};
  int matvec_count = 0;
  for (std::size_t step = 0; step < basis_limit; ++step)
  {
    Vector residual = ops.allocate_like(basis[step]);
    ops.matvec(residual, basis[step]);
    ++matvec_count;

    Real const previous_beta = step == 0 ? Real{} : subdiagonal[step - 1];
    if (step > 0) ops.axpy(residual, -static_cast<Scalar>(previous_beta), basis[step - 1]);

    Real const alpha = detail::dmrg_lanczos_projection_scalar<Scalar>(ops.inner_product(basis[step], residual));
    diagonal.push_back(alpha);
    ops.axpy(residual, -static_cast<Scalar>(alpha), basis[step]);

    Real const beta = krylov::norm_or_inner_product<Scalar>(ops, residual);
    if (!uni20::isfinite(beta)) throw std::runtime_error("DMRG Lanczos produced a non-finite residual norm");
    final_residual_norm = beta;
    if (step + 1 == basis_limit || detail::dmrg_lanczos_breakdown(beta, alpha, previous_beta)) break;

    detail::normalize_dmrg_lanczos_vector<Scalar>(ops, residual, beta);
    subdiagonal.push_back(beta);
    basis.push_back(std::move(residual));
  }

  auto projected = krylov::symmetric_tridiagonal_eigensystem(std::move(diagonal), std::move(subdiagonal), true);
  Real const final_coefficient = projected.eigenvectors[basis.size() - 1, 0];

  Vector vector = ops.allocate_like(initial);
  ops.copy(vector, basis.front());
  ops.scal(vector, static_cast<Scalar>(projected.eigenvectors[0, 0]));
  for (std::size_t row = 1; row < basis.size(); ++row)
    ops.axpy(vector, static_cast<Scalar>(projected.eigenvectors[row, 0]), basis[row]);

  return {.energy = projected.eigenvalues.front(),
          .residual_bound = final_residual_norm * abs(final_coefficient),
          .vector = std::move(vector),
          .iteration_count = static_cast<int>(basis.size()),
          .matvec_count = matvec_count};
}

} // namespace uni20::tensor_network
