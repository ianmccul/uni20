#pragma once

/**
 * \file lapack.hpp
 * \ingroup backend_lapack
 * \brief Checked LAPACK wrappers over provider-specific unchecked overloads.
 */

#include <uni20/backend/lapack/common.hpp>
#include <uni20/backend/lapack/reference/band.hpp>
#include <uni20/backend/lapack/reference/general.hpp>
#include <uni20/backend/lapack/reference/norms.hpp>
#include <uni20/backend/lapack/reference/tridiagonal.hpp>
#include <uni20/config.hpp>
#include <uni20/core/scalar_concepts.hpp>

#include <stdexcept>
#include <string>

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
#include <uni20/backend/lapack/mplapack/band.hpp>
#include <uni20/backend/lapack/mplapack/general.hpp>
#include <uni20/backend/lapack/mplapack/norms.hpp>
#include <uni20/backend/lapack/mplapack/tridiagonal.hpp>
#endif

namespace uni20::lapack
{

namespace detail
{

[[noreturn]] inline void throw_invalid_argument_info(char const* routine, blas_int info)
{
  throw std::invalid_argument("LAPACK " + std::string(routine) + " received invalid argument " + std::to_string(-info));
}

[[noreturn]] inline void throw_runtime_info(char const* routine, char const* message, blas_int info)
{
  throw std::runtime_error("LAPACK " + std::string(routine) + " " + message + " (info=" + std::to_string(info) + ")");
}

inline void check_invalid_argument(char const* routine, blas_int info)
{
  if (info < 0)
  {
    throw_invalid_argument_info(routine, info);
  }
}

inline void check_singular(char const* routine, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 0)
  {
    throw_runtime_info(routine, "found a singular matrix", info);
  }
}

inline void check_positive_definite(char const* routine, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 0)
  {
    throw_runtime_info(routine, "found a matrix that is not positive definite", info);
  }
}

inline void check_singular_diagonal_block(char const* routine, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 0)
  {
    throw_runtime_info(routine, "found a singular diagonal block", info);
  }
}

inline void check_singular_triangular(char const* routine, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 0)
  {
    throw_runtime_info(routine, "found a singular triangular matrix", info);
  }
}

inline void check_convergence(char const* routine, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 0)
  {
    throw_runtime_info(routine, "failed to converge", info);
  }
}

inline void check_generalized_symmetric_eigensolver(char const* routine, blas_int n, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > n)
  {
    throw_runtime_info(routine, "found a metric matrix that is not positive definite", info);
  }
  if (info > 0)
  {
    throw_runtime_info(routine, "failed to converge", info);
  }
}

inline bool check_expert_solve(char const* routine, blas_int n, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 0 && info <= n)
  {
    throw_runtime_info(routine, "found a singular matrix", info);
  }
  if (info > n + 1)
  {
    throw_runtime_info(routine, "failed unexpectedly", info);
  }
  return info == n + 1;
}

inline bool check_symmetric_indefinite_expert_solve(char const* routine, blas_int n, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 0 && info <= n)
  {
    throw_runtime_info(routine, "found a singular diagonal block", info);
  }
  if (info > n + 1)
  {
    throw_runtime_info(routine, "failed unexpectedly", info);
  }
  return info == n + 1;
}

inline bool check_positive_definite_expert_solve(char const* routine, blas_int n, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 0 && info <= n)
  {
    throw_runtime_info(routine, "found a matrix that is not positive definite", info);
  }
  if (info > n + 1)
  {
    throw_runtime_info(routine, "failed unexpectedly", info);
  }
  return info == n + 1;
}

inline void check_equilibration(char const* routine, blas_int m, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 0 && info <= m)
  {
    throw_runtime_info(routine, "found an exactly zero row", info);
  }
  if (info > m)
  {
    throw_runtime_info(routine, "found an exactly zero column", info);
  }
}

inline bool check_sylvester(char const* routine, blas_int info)
{
  check_invalid_argument(routine, info);
  if (info > 1)
  {
    throw_runtime_info(routine, "failed unexpectedly", info);
  }
  return info == 1;
}

} // namespace detail

/// \brief Compute a dense real matrix norm through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
Scalar lange(char norm, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* work)
{
  return unchecked::lange(norm, m, n, a, lda, work);
}

/// \brief Compute a dense real symmetric matrix norm through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
Scalar lansy(char norm, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* work)
{
  return unchecked::lansy(norm, uplo, n, a, lda, work);
}

/// \brief Compute a dense real triangular or trapezoidal matrix norm through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
Scalar lantr(char norm, char uplo, char diag, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* work)
{
  return unchecked::lantr(norm, uplo, diag, m, n, a, lda, work);
}

/// \brief Compute a real general-band matrix norm through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
Scalar langb(char norm, blas_int n, blas_int kl, blas_int ku, Scalar* ab, blas_int ldab, Scalar* work)
{
  return unchecked::langb(norm, n, kl, ku, ab, ldab, work);
}

/// \brief Solve a real general-band linear system through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void gbsv(blas_int n, blas_int kl, blas_int ku, blas_int nrhs, Scalar* ab, blas_int ldab, blas_int* ipiv, Scalar* b,
          blas_int ldb)
{
  blas_int const info = unchecked::gbsv(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
  detail::check_singular("gbsv", info);
}

/// \brief Compute a real general-band LU factorization through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void gbtrf(blas_int m, blas_int n, blas_int kl, blas_int ku, Scalar* ab, blas_int ldab, blas_int* ipiv)
{
  blas_int const info = unchecked::gbtrf(m, n, kl, ku, ab, ldab, ipiv);
  detail::check_singular("gbtrf", info);
}

/// \brief Solve using an existing real general-band LU factorization.
template <uni20::LapackReal Scalar>
void gbtrs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, Scalar* ab, blas_int ldab,
           blas_int const* ipiv, Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::gbtrs(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
  detail::check_invalid_argument("gbtrs", info);
}

/// \brief Estimate a real general-band reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar gbcon(char norm, blas_int n, blas_int kl, blas_int ku, Scalar* ab, blas_int ldab, blas_int const* ipiv,
             Scalar anorm, Scalar* work, blas_int* iwork)
{
  Scalar rcond{};
  blas_int const info = unchecked::gbcon(norm, n, kl, ku, ab, ldab, ipiv, anorm, rcond, work, iwork);
  detail::check_invalid_argument("gbcon", info);
  return rcond;
}

/// \brief Refine a real general-band linear-system solve and return LAPACK diagnostics.
template <uni20::LapackReal Scalar>
void gbrfs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* afb,
           blas_int ldafb, blas_int const* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx,
           Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::gbrfs(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx,
                                         forward_error, backward_error, work, iwork);
  detail::check_invalid_argument("gbrfs", info);
}

/// \brief Solve a real general-band linear system with LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
bool gbsvx(char fact, char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, Scalar* ab, blas_int ldab,
           Scalar* afb, blas_int ldafb, blas_int* ipiv, char& equed, Scalar* row_scale, Scalar* column_scale, Scalar* b,
           blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond, Scalar* forward_error, Scalar* backward_error,
           Scalar* work, blas_int* iwork)
{
  blas_int const info =
      unchecked::gbsvx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, row_scale, column_scale, b, ldb,
                       x, ldx, rcond, forward_error, backward_error, work, iwork);
  return detail::check_expert_solve("gbsvx", n, info);
}

/// \brief Compute real general-band row/column equilibration factors.
template <uni20::LapackReal Scalar>
void gbequ(bool power_of_two, blas_int m, blas_int n, blas_int kl, blas_int ku, Scalar* ab, blas_int ldab,
           Scalar* row_scale, Scalar* column_scale, Scalar& row_condition, Scalar& column_condition, Scalar& max_abs)
{
  blas_int const info = unchecked::gbequ(power_of_two, m, n, kl, ku, ab, ldab, row_scale, column_scale, row_condition,
                                         column_condition, max_abs);
  detail::check_equilibration("gbequ", m, info);
}

/// \brief Solve a real symmetric positive-definite band linear system.
template <uni20::LapackReal Scalar>
void pbsv(char uplo, blas_int n, blas_int kd, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::pbsv(uplo, n, kd, nrhs, ab, ldab, b, ldb);
  detail::check_positive_definite("pbsv", info);
}

/// \brief Compute a real symmetric positive-definite band Cholesky factorization.
template <uni20::LapackReal Scalar> void pbtrf(char uplo, blas_int n, blas_int kd, Scalar* ab, blas_int ldab)
{
  blas_int const info = unchecked::pbtrf(uplo, n, kd, ab, ldab);
  detail::check_positive_definite("pbtrf", info);
}

/// \brief Solve using an existing real symmetric positive-definite band Cholesky factorization.
template <uni20::LapackReal Scalar>
void pbtrs(char uplo, blas_int n, blas_int kd, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::pbtrs(uplo, n, kd, nrhs, ab, ldab, b, ldb);
  detail::check_invalid_argument("pbtrs", info);
  if (info > 0)
  {
    detail::throw_runtime_info("pbtrs", "failed to solve the factored system", info);
  }
}

/// \brief Estimate a real symmetric positive-definite band reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar pbcon(char uplo, blas_int n, blas_int kd, Scalar* ab, blas_int ldab, Scalar anorm, Scalar* work, blas_int* iwork)
{
  Scalar rcond{};
  blas_int const info = unchecked::pbcon(uplo, n, kd, ab, ldab, anorm, rcond, work, iwork);
  detail::check_invalid_argument("pbcon", info);
  return rcond;
}

/// \brief Refine a real symmetric positive-definite band solve and return LAPACK diagnostics.
template <uni20::LapackReal Scalar>
void pbrfs(char uplo, blas_int n, blas_int kd, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* afb, blas_int ldafb,
           Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error, Scalar* backward_error,
           Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::pbrfs(uplo, n, kd, nrhs, ab, ldab, afb, ldafb, b, ldb, x, ldx, forward_error,
                                         backward_error, work, iwork);
  detail::check_invalid_argument("pbrfs", info);
}

/// \brief Solve a real symmetric positive-definite band system with LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
bool pbsvx(char fact, char uplo, blas_int n, blas_int kd, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* afb,
           blas_int ldafb, char& equed, Scalar* scale, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond,
           Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::pbsvx(fact, uplo, n, kd, nrhs, ab, ldab, afb, ldafb, equed, scale, b, ldb, x, ldx,
                                         rcond, forward_error, backward_error, work, iwork);
  return detail::check_positive_definite_expert_solve("pbsvx", n, info);
}

/// \brief Solve a real dense general linear system through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void gesv(blas_int n, blas_int nrhs, Scalar* a, blas_int lda, blas_int* ipiv, Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::gesv(n, nrhs, a, lda, ipiv, b, ldb);
  detail::check_singular("gesv", info);
}

/// \brief Solve a real dense general linear system with LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
bool gesvx(char fact, char trans, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* af, blas_int ldaf,
           blas_int* ipiv, char& equed, Scalar* row_scale, Scalar* column_scale, Scalar* b, blas_int ldb, Scalar* x,
           blas_int ldx, Scalar& rcond, Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::gesvx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, row_scale, column_scale,
                                         b, ldb, x, ldx, rcond, forward_error, backward_error, work, iwork);
  return detail::check_expert_solve("gesvx", n, info);
}

/// \brief Compute real dense general row/column equilibration factors.
template <uni20::LapackReal Scalar>
void geequ(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* row_scale, Scalar* column_scale,
           Scalar& row_condition, Scalar& column_condition, Scalar& max_abs)
{
  blas_int const info =
      unchecked::geequ(m, n, a, lda, row_scale, column_scale, row_condition, column_condition, max_abs);
  detail::check_equilibration("geequ", m, info);
}

/// \brief Compute a real dense general LU factorization through the configured LAPACK backend.
template <uni20::LapackReal Scalar> void getrf(blas_int m, blas_int n, Scalar* a, blas_int lda, blas_int* ipiv)
{
  blas_int const info = unchecked::getrf(m, n, a, lda, ipiv);
  detail::check_singular("getrf", info);
}

/// \brief Solve using an existing real dense general LU factorization.
template <uni20::LapackReal Scalar>
void getrs(char trans, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, blas_int const* ipiv, Scalar* b,
           blas_int ldb)
{
  blas_int const info = unchecked::getrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
  detail::check_invalid_argument("getrs", info);
}

/// \brief Refine a real dense general linear-system solve and return LAPACK diagnostics.
template <uni20::LapackReal Scalar>
void gerfs(char trans, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* factors, blas_int factor_lda,
           blas_int const* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error,
           Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::gerfs(trans, n, nrhs, a, lda, factors, factor_lda, ipiv, b, ldb, x, ldx,
                                         forward_error, backward_error, work, iwork);
  detail::check_invalid_argument("gerfs", info);
}

/// \brief Invert a real dense general matrix from an existing LU factorization.
template <uni20::LapackReal Scalar>
void getri(blas_int n, Scalar* a, blas_int lda, blas_int* ipiv, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::getri(n, a, lda, ipiv, work, lwork);
  detail::check_singular("getri", info);
}

/// \brief Estimate a real dense general reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar gecon(char norm, blas_int n, Scalar const* a, blas_int lda, Scalar anorm, Scalar* work, blas_int* iwork)
{
  Scalar rcond{};
  blas_int const info = unchecked::gecon(norm, n, a, lda, anorm, rcond, work, iwork);
  detail::check_invalid_argument("gecon", info);
  return rcond;
}

/// \brief Solve a real dense general least-squares problem through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void gels(char trans, blas_int m, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb,
          Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::gels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
  detail::check_invalid_argument("gels", info);
  if (info > 0)
  {
    detail::throw_runtime_info("gels", "found a rank-deficient triangular factor", info);
  }
}

/// \brief Solve a real dense general least-squares problem by SVD.
template <uni20::LapackReal Scalar>
void gelss(blas_int m, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* s,
           Scalar rcond, blas_int& rank, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::gelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork);
  detail::check_convergence("gelss", info);
}

/// \brief Solve a real dense general least-squares problem by divide-and-conquer SVD.
template <uni20::LapackReal Scalar>
void gelsd(blas_int m, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* s,
           Scalar rcond, blas_int& rank, Scalar* work, blas_int lwork, blas_int* iwork)
{
  blas_int const info = unchecked::gelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork);
  detail::check_convergence("gelsd", info);
}

/// \brief Solve a real dense general least-squares problem by complete orthogonal factorization.
template <uni20::LapackReal Scalar>
void gelsy(blas_int m, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, blas_int* jpvt,
           Scalar rcond, blas_int& rank, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::gelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank, work, lwork);
  detail::check_invalid_argument("gelsy", info);
  if (info > 0)
  {
    detail::throw_runtime_info("gelsy", "failed to compute a rank-revealing least-squares solution", info);
  }
}

/// \brief Compute a real dense QR factorization through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void geqrf(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::geqrf(m, n, a, lda, tau, work, lwork);
  detail::check_invalid_argument("geqrf", info);
}

/// \brief Compute a real dense LQ factorization through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void gelqf(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::gelqf(m, n, a, lda, tau, work, lwork);
  detail::check_invalid_argument("gelqf", info);
}

/// \brief Compute a real dense QL factorization through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void geqlf(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::geqlf(m, n, a, lda, tau, work, lwork);
  detail::check_invalid_argument("geqlf", info);
}

/// \brief Compute a real dense RQ factorization through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void gerqf(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::gerqf(m, n, a, lda, tau, work, lwork);
  detail::check_invalid_argument("gerqf", info);
}

/// \brief Generate an explicit real Q factor from a QR factorization.
template <uni20::LapackReal Scalar>
void orgqr(blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::orgqr(m, n, k, a, lda, tau, work, lwork);
  detail::check_invalid_argument("orgqr", info);
}

/// \brief Generate an explicit real Q factor from an LQ factorization.
template <uni20::LapackReal Scalar>
void orglq(blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::orglq(m, n, k, a, lda, tau, work, lwork);
  detail::check_invalid_argument("orglq", info);
}

/// \brief Generate an explicit real Q factor from a QL factorization.
template <uni20::LapackReal Scalar>
void orgql(blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::orgql(m, n, k, a, lda, tau, work, lwork);
  detail::check_invalid_argument("orgql", info);
}

/// \brief Generate an explicit real Q factor from an RQ factorization.
template <uni20::LapackReal Scalar>
void orgrq(blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::orgrq(m, n, k, a, lda, tau, work, lwork);
  detail::check_invalid_argument("orgrq", info);
}

/// \brief Multiply by the implicit real Q factor from a QR factorization.
template <uni20::LapackReal Scalar>
void ormqr(char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* c,
           blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::check_invalid_argument("ormqr", info);
}

/// \brief Multiply by the implicit real Q factor from an LQ factorization.
template <uni20::LapackReal Scalar>
void ormlq(char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* c,
           blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::ormlq(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::check_invalid_argument("ormlq", info);
}

/// \brief Multiply by the implicit real Q factor from a QL factorization.
template <uni20::LapackReal Scalar>
void ormql(char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* c,
           blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::ormql(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::check_invalid_argument("ormql", info);
}

/// \brief Multiply by the implicit real Q factor from an RQ factorization.
template <uni20::LapackReal Scalar>
void ormrq(char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* c,
           blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::ormrq(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::check_invalid_argument("ormrq", info);
}

/// \brief Reduce a real dense general matrix to bidiagonal form.
template <uni20::LapackReal Scalar>
void gebrd(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* d, Scalar* e, Scalar* tauq, Scalar* taup,
           Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::gebrd(m, n, a, lda, d, e, tauq, taup, work, lwork);
  detail::check_convergence("gebrd", info);
}

/// \brief Reduce a real dense general matrix to upper Hessenberg form.
template <uni20::LapackReal Scalar>
void gehrd(blas_int n, blas_int first, blas_int last, Scalar* a, blas_int lda, Scalar* tau, Scalar* work,
           blas_int lwork)
{
  blas_int const info = unchecked::gehrd(n, first, last, a, lda, tau, work, lwork);
  detail::check_invalid_argument("gehrd", info);
}

/// \brief Compute a real dense pivoted QR factorization.
template <uni20::LapackReal Scalar>
void geqp3(blas_int m, blas_int n, Scalar* a, blas_int lda, blas_int* jpvt, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::geqp3(m, n, a, lda, jpvt, tau, work, lwork);
  detail::check_invalid_argument("geqp3", info);
}

/// \brief Reduce a real dense symmetric matrix to tridiagonal form.
template <uni20::LapackReal Scalar>
void sytrd(char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* d, Scalar* e, Scalar* tau, Scalar* work,
           blas_int lwork)
{
  blas_int const info = unchecked::sytrd(uplo, n, a, lda, d, e, tau, work, lwork);
  detail::check_convergence("sytrd", info);
}

/// \brief Compute eigenvalues and optionally eigenvectors of a real symmetric matrix.
template <uni20::LapackReal Scalar>
void syev(char jobz, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* w, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::syev(jobz, uplo, n, a, lda, w, work, lwork);
  detail::check_convergence("syev", info);
}

/// \brief Compute eigenvalues and optionally eigenvectors of a real symmetric matrix by divide and conquer.
template <uni20::LapackReal Scalar>
void syevd(char jobz, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* w, Scalar* work, blas_int lwork,
           blas_int* iwork, blas_int liwork)
{
  blas_int const info = unchecked::syevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork);
  detail::check_convergence("syevd", info);
}

/// \brief Compute selected eigenvalues and optionally eigenvectors of a real symmetric matrix.
template <uni20::LapackReal Scalar>
void syevr(char jobz, char range, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar vl, Scalar vu, blas_int il,
           blas_int iu, Scalar abstol, blas_int& selected_count, Scalar* w, Scalar* z, blas_int ldz, blas_int* isuppz,
           Scalar* work, blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int const info = unchecked::syevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, selected_count, w, z,
                                         ldz, isuppz, work, lwork, iwork, liwork);
  detail::check_convergence("syevr", info);
}

/// \brief Compute eigenvalues and optionally eigenvectors of a real generalized symmetric-definite problem.
template <uni20::LapackReal Scalar>
void sygv(blas_int itype, char jobz, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* w,
          Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::sygv(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork);
  detail::check_generalized_symmetric_eigensolver("sygv", n, info);
}

/// \brief Compute eigenvalues and optionally eigenvectors of a real generalized symmetric-definite problem by divide
/// and
///        conquer.
template <uni20::LapackReal Scalar>
void sygvd(blas_int itype, char jobz, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb,
           Scalar* w, Scalar* work, blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int const info = unchecked::sygvd(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, iwork, liwork);
  detail::check_generalized_symmetric_eigensolver("sygvd", n, info);
}

/// \brief Compute selected eigenvalues and optionally eigenvectors of a real generalized symmetric-definite problem.
template <uni20::LapackReal Scalar>
void sygvx(blas_int itype, char jobz, char range, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar vl, Scalar vu, blas_int il, blas_int iu, Scalar abstol, blas_int& selected_count,
           Scalar* w, Scalar* z, blas_int ldz, Scalar* work, blas_int lwork, blas_int* iwork, blas_int* ifail)
{
  blas_int const info = unchecked::sygvx(itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol,
                                         selected_count, w, z, ldz, work, lwork, iwork, ifail);
  detail::check_generalized_symmetric_eigensolver("sygvx", n, info);
}

/// \brief Compute eigenvalues and optionally eigenvectors of a complex Hermitian matrix.
template <uni20::LapackComplexReal Real>
void heev(char jobz, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda, Real* w, uni20::complex<Real>* work,
          blas_int lwork, Real* rwork)
{
  blas_int const info = unchecked::heev(jobz, uplo, n, a, lda, w, work, lwork, rwork);
  detail::check_convergence("heev", info);
}

/// \brief Compute eigenvalues and optionally eigenvectors of a complex Hermitian matrix by divide and conquer.
template <uni20::LapackComplexReal Real>
void heevd(char jobz, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda, Real* w, uni20::complex<Real>* work,
           blas_int lwork, Real* rwork, blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  blas_int const info = unchecked::heevd(jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork);
  detail::check_convergence("heevd", info);
}

/// \brief Compute selected eigenvalues and optionally eigenvectors of a complex Hermitian matrix.
template <uni20::LapackComplexReal Real>
void heevr(char jobz, char range, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda, Real vl, Real vu,
           blas_int il, blas_int iu, Real abstol, blas_int& selected_count, Real* w, uni20::complex<Real>* z,
           blas_int ldz, blas_int* isuppz, uni20::complex<Real>* work, blas_int lwork, Real* rwork, blas_int lrwork,
           blas_int* iwork, blas_int liwork)
{
  blas_int const info = unchecked::heevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, selected_count, w, z,
                                         ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork);
  detail::check_convergence("heevr", info);
}

/// \brief Compute eigenvalues and optionally eigenvectors of a complex generalized Hermitian-definite problem.
template <uni20::LapackComplexReal Real>
void hegv(blas_int itype, char jobz, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda,
          uni20::complex<Real>* b, blas_int ldb, Real* w, uni20::complex<Real>* work, blas_int lwork, Real* rwork)
{
  blas_int const info = unchecked::hegv(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork);
  detail::check_generalized_symmetric_eigensolver("hegv", n, info);
}

/// \brief Compute eigenvalues and optionally eigenvectors of a complex generalized Hermitian-definite problem by divide
///        and conquer.
template <uni20::LapackComplexReal Real>
void hegvd(blas_int itype, char jobz, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda,
           uni20::complex<Real>* b, blas_int ldb, Real* w, uni20::complex<Real>* work, blas_int lwork, Real* rwork,
           blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  blas_int const info =
      unchecked::hegvd(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork, lrwork, iwork, liwork);
  detail::check_generalized_symmetric_eigensolver("hegvd", n, info);
}

/// \brief Compute selected eigenvalues and optionally eigenvectors of a complex generalized Hermitian-definite problem.
template <uni20::LapackComplexReal Real>
void hegvx(blas_int itype, char jobz, char range, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda,
           uni20::complex<Real>* b, blas_int ldb, Real vl, Real vu, blas_int il, blas_int iu, Real abstol,
           blas_int& selected_count, Real* w, uni20::complex<Real>* z, blas_int ldz, uni20::complex<Real>* work,
           blas_int lwork, Real* rwork, blas_int* iwork, blas_int* ifail)
{
  blas_int const info = unchecked::hegvx(itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol,
                                         selected_count, w, z, ldz, work, lwork, rwork, iwork, ifail);
  detail::check_generalized_symmetric_eigensolver("hegvx", n, info);
}

/// \brief Compute eigenvalues and optionally eigenvectors of a real nonsymmetric matrix.
template <uni20::LapackReal Scalar>
void geev(char jobvl, char jobvr, blas_int n, Scalar* a, blas_int lda, Scalar* wr, Scalar* wi, Scalar* vl,
          blas_int ldvl, Scalar* vr, blas_int ldvr, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::geev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork);
  detail::check_convergence("geev", info);
}

/// \brief Compute real nonsymmetric eigenpairs with balancing and condition estimates.
template <uni20::LapackReal Scalar>
void geevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, Scalar* a, blas_int lda, Scalar* wr, Scalar* wi,
           Scalar* vl, blas_int ldvl, Scalar* vr, blas_int ldvr, blas_int& ilo, blas_int& ihi, Scalar* scale,
           Scalar& abnrm, Scalar* rconde, Scalar* rcondv, Scalar* work, blas_int lwork, blas_int* iwork)
{
  blas_int const info = unchecked::geevx(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi,
                                         scale, abnrm, rconde, rcondv, work, lwork, iwork);
  detail::check_convergence("geevx", info);
}

/// \brief Balance a real nonsymmetric matrix before eigenanalysis.
template <uni20::LapackReal Scalar>
void gebal(char job, blas_int n, Scalar* a, blas_int lda, blas_int& first, blas_int& last, Scalar* scale)
{
  blas_int const info = unchecked::gebal(job, n, a, lda, first, last, scale);
  detail::check_invalid_argument("gebal", info);
  if (info > 0)
  {
    detail::throw_runtime_info("gebal", "failed unexpectedly", info);
  }
}

/// \brief Back-transform real nonsymmetric eigenvectors after balancing.
template <uni20::LapackReal Scalar>
void gebak(char job, char side, blas_int n, blas_int first, blas_int last, Scalar* scale, blas_int vector_count,
           Scalar* vectors, blas_int leading_dimension)
{
  blas_int const info = unchecked::gebak(job, side, n, first, last, scale, vector_count, vectors, leading_dimension);
  detail::check_invalid_argument("gebak", info);
  if (info > 0)
  {
    detail::throw_runtime_info("gebak", "failed unexpectedly", info);
  }
}

/// \brief Compute eigenvalues and optionally eigenvectors of a complex nonsymmetric matrix.
template <uni20::LapackComplexReal Real>
void geev(char jobvl, char jobvr, blas_int n, uni20::complex<Real>* a, blas_int lda, uni20::complex<Real>* w,
          uni20::complex<Real>* vl, blas_int ldvl, uni20::complex<Real>* vr, blas_int ldvr, uni20::complex<Real>* work,
          blas_int lwork, Real* rwork)
{
  blas_int const info = unchecked::geev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork);
  detail::check_convergence("geev", info);
}

/// \brief Compute the real Schur form of a real nonsymmetric matrix.
template <uni20::LapackReal Scalar>
void gees(char jobvs, char sort, blas_int n, Scalar* a, blas_int lda, blas_int& selected_dimension, Scalar* wr,
          Scalar* wi, Scalar* vs, blas_int ldvs, Scalar* work, blas_int lwork, blas_int* bwork)
{
  blas_int const info =
      unchecked::gees(jobvs, sort, n, a, lda, selected_dimension, wr, wi, vs, ldvs, work, lwork, bwork);
  detail::check_invalid_argument("gees", info);
  if (info > 0)
  {
    detail::throw_runtime_info("gees", "failed to compute a Schur form", info);
  }
}

/// \brief Compute the complex Schur form of a complex nonsymmetric matrix.
template <uni20::LapackComplexReal Real>
void gees(char jobvs, char sort, blas_int n, uni20::complex<Real>* a, blas_int lda, blas_int& selected_dimension,
          uni20::complex<Real>* w, uni20::complex<Real>* vs, blas_int ldvs, uni20::complex<Real>* work, blas_int lwork,
          Real* rwork, blas_int* bwork)
{
  blas_int const info =
      unchecked::gees(jobvs, sort, n, a, lda, selected_dimension, w, vs, ldvs, work, lwork, rwork, bwork);
  detail::check_invalid_argument("gees", info);
  if (info > 0)
  {
    detail::throw_runtime_info("gees", "failed to compute a Schur form", info);
  }
}

/// \brief Compute a Schur form of a real upper Hessenberg matrix.
template <uni20::LapackReal Scalar>
void hseqr(char job, char compz, blas_int n, blas_int first, blas_int last, Scalar* h, blas_int ldh, Scalar* wr,
           Scalar* wi, Scalar* z, blas_int ldz, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::hseqr(job, compz, n, first, last, h, ldh, wr, wi, z, ldz, work, lwork);
  detail::check_invalid_argument("hseqr", info);
  if (info > 0)
  {
    detail::throw_runtime_info("hseqr", "failed to compute a Schur form", info);
  }
}

/// \brief Reorder adjacent blocks in a real Schur form.
template <uni20::LapackReal Scalar>
void trexc(char compq, blas_int n, Scalar* t, blas_int ldt, Scalar* q, blas_int ldq, blas_int& first, blas_int& last,
           Scalar* work)
{
  blas_int const info = unchecked::trexc(compq, n, t, ldt, q, ldq, first, last, work);
  detail::check_invalid_argument("trexc", info);
  if (info > 0)
  {
    detail::throw_runtime_info("trexc", "failed to reorder adjacent Schur blocks", info);
  }
}

/// \brief Reorder adjacent entries in a complex Schur form.
template <uni20::LapackComplexReal Real>
void trexc(char compq, blas_int n, uni20::complex<Real>* t, blas_int ldt, uni20::complex<Real>* q, blas_int ldq,
           blas_int& first, blas_int& last)
{
  blas_int const info = unchecked::trexc(compq, n, t, ldt, q, ldq, first, last);
  detail::check_invalid_argument("trexc", info);
  if (info > 0)
  {
    detail::throw_runtime_info("trexc", "failed to reorder adjacent Schur blocks", info);
  }
}

/// \brief Reorder selected blocks in a real Schur form and estimate invariant subspace conditioning.
template <uni20::LapackReal Scalar>
void trsen(char job, char compq, blas_int* select, blas_int n, Scalar* t, blas_int ldt, Scalar* q, blas_int ldq,
           Scalar* wr, Scalar* wi, blas_int& selected_dimension, Scalar& reciprocal_eigenvalue_cluster_condition,
           Scalar& reciprocal_invariant_subspace_condition, Scalar* work, blas_int lwork, blas_int* iwork,
           blas_int liwork)
{
  blas_int const info = unchecked::trsen(job, compq, select, n, t, ldt, q, ldq, wr, wi, selected_dimension,
                                         reciprocal_eigenvalue_cluster_condition,
                                         reciprocal_invariant_subspace_condition, work, lwork, iwork, liwork);
  detail::check_invalid_argument("trsen", info);
  if (info == 1)
  {
    detail::throw_runtime_info("trsen", "could not reorder the selected Schur blocks", info);
  }
  if (info > 1)
  {
    detail::throw_runtime_info("trsen", "failed unexpectedly", info);
  }
}

/// \brief Compute eigenvectors of a real upper quasi-triangular Schur form.
template <uni20::LapackReal Scalar>
void trevc(char side, char howmny, blas_int* select, blas_int n, Scalar* t, blas_int ldt, Scalar* vl, blas_int ldvl,
           Scalar* vr, blas_int ldvr, blas_int mm, blas_int& computed_vectors, Scalar* work)
{
  blas_int const info =
      unchecked::trevc(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, computed_vectors, work);
  detail::check_invalid_argument("trevc", info);
  if (info > 0)
  {
    detail::throw_runtime_info("trevc", "failed to compute all requested Schur eigenvectors", info);
  }
}

/// \brief Estimate reciprocal condition numbers for real Schur eigenpairs.
template <uni20::LapackReal Scalar>
void trsna(char job, char howmny, blas_int* select, blas_int n, Scalar* t, blas_int ldt, Scalar* vl, blas_int ldvl,
           Scalar* vr, blas_int ldvr, Scalar* reciprocal_eigenvalue_condition_numbers,
           Scalar* reciprocal_eigenvector_condition_numbers, blas_int mm, blas_int& computed_estimates, Scalar* work,
           blas_int ldwork, blas_int* iwork)
{
  blas_int const info =
      unchecked::trsna(job, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, reciprocal_eigenvalue_condition_numbers,
                       reciprocal_eigenvector_condition_numbers, mm, computed_estimates, work, ldwork, iwork);
  detail::check_invalid_argument("trsna", info);
  if (info > 0)
  {
    detail::throw_runtime_info("trsna", "failed unexpectedly", info);
  }
}

/// \brief Compute generalized eigenvalues and optionally eigenvectors of a real nonsymmetric matrix pencil.
template <uni20::LapackReal Scalar>
void ggev(char jobvl, char jobvr, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* alphar,
          Scalar* alphai, Scalar* beta, Scalar* vl, blas_int ldvl, Scalar* vr, blas_int ldvr, Scalar* work,
          blas_int lwork)
{
  blas_int const info =
      unchecked::ggev(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork);
  detail::check_convergence("ggev", info);
}

/// \brief Compute generalized real nonsymmetric eigenpairs with balancing and condition estimates.
template <uni20::LapackReal Scalar>
void ggevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* alphar, Scalar* alphai, Scalar* beta, Scalar* vl, blas_int ldvl, Scalar* vr,
           blas_int ldvr, blas_int& ilo, blas_int& ihi, Scalar* lscale, Scalar* rscale, Scalar& abnrm, Scalar& bbnrm,
           Scalar* rconde, Scalar* rcondv, Scalar* work, blas_int lwork, blas_int* iwork, blas_int* bwork)
{
  blas_int const info =
      unchecked::ggevx(balanc, jobvl, jobvr, sense, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, ilo,
                       ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work, lwork, iwork, bwork);
  detail::check_convergence("ggevx", info);
}

/// \brief Balance a real nonsymmetric matrix pencil before generalized eigenanalysis.
template <uni20::LapackReal Scalar>
void ggbal(char job, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, blas_int& first, blas_int& last,
           Scalar* lscale, Scalar* rscale, Scalar* work)
{
  blas_int const info = unchecked::ggbal(job, n, a, lda, b, ldb, first, last, lscale, rscale, work);
  detail::check_invalid_argument("ggbal", info);
  if (info > 0)
  {
    detail::throw_runtime_info("ggbal", "failed unexpectedly", info);
  }
}

/// \brief Back-transform generalized nonsymmetric eigenvectors after pencil balancing.
template <uni20::LapackReal Scalar>
void ggbak(char job, char side, blas_int n, blas_int first, blas_int last, Scalar* lscale, Scalar* rscale,
           blas_int vector_count, Scalar* vectors, blas_int leading_dimension)
{
  blas_int const info =
      unchecked::ggbak(job, side, n, first, last, lscale, rscale, vector_count, vectors, leading_dimension);
  detail::check_invalid_argument("ggbak", info);
  if (info > 0)
  {
    detail::throw_runtime_info("ggbak", "failed unexpectedly", info);
  }
}

/// \brief Compute a generalized real Schur form of a real nonsymmetric matrix pencil.
template <uni20::LapackReal Scalar>
void gges(char jobvsl, char jobvsr, char sort, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb,
          blas_int& selected_dimension, Scalar* alphar, Scalar* alphai, Scalar* beta, Scalar* vsl, blas_int ldvsl,
          Scalar* vsr, blas_int ldvsr, Scalar* work, blas_int lwork, blas_int* bwork)
{
  blas_int const info = unchecked::gges(jobvsl, jobvsr, sort, n, a, lda, b, ldb, selected_dimension, alphar, alphai,
                                        beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork);
  detail::check_invalid_argument("gges", info);
  if (info > 0 && info <= n)
  {
    detail::throw_runtime_info("gges", "failed to compute a generalized Schur form", info);
  }
  if (info > n)
  {
    detail::throw_runtime_info("gges", "failed after generalized Schur convergence", info);
  }
}

/// \brief Reduce a real matrix pencil to generalized Hessenberg form.
template <uni20::LapackReal Scalar>
void gghrd(char compq, char compz, blas_int n, blas_int first, blas_int last, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* q, blas_int ldq, Scalar* z, blas_int ldz)
{
  blas_int const info = unchecked::gghrd(compq, compz, n, first, last, a, lda, b, ldb, q, ldq, z, ldz);
  detail::check_invalid_argument("gghrd", info);
  if (info > 0)
  {
    detail::throw_runtime_info("gghrd", "failed unexpectedly", info);
  }
}

/// \brief Compute a generalized Schur form from a real Hessenberg-triangular pencil.
template <uni20::LapackReal Scalar>
void hgeqz(char job, char compq, char compz, blas_int n, blas_int first, blas_int last, Scalar* h, blas_int ldh,
           Scalar* t, blas_int ldt, Scalar* alphar, Scalar* alphai, Scalar* beta, Scalar* q, blas_int ldq, Scalar* z,
           blas_int ldz, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::hgeqz(job, compq, compz, n, first, last, h, ldh, t, ldt, alphar, alphai, beta, q,
                                         ldq, z, ldz, work, lwork);
  detail::check_invalid_argument("hgeqz", info);
  if (info > 0)
  {
    detail::throw_runtime_info("hgeqz", "failed to compute a generalized Schur form", info);
  }
}

/// \brief Reorder adjacent blocks in a generalized real Schur form.
template <uni20::LapackReal Scalar>
void tgexc(bool wantq, bool wantz, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* q,
           blas_int ldq, Scalar* z, blas_int ldz, blas_int& first, blas_int& last, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::tgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, first, last, work, lwork);
  detail::check_invalid_argument("tgexc", info);
  if (info == 1)
  {
    detail::throw_runtime_info("tgexc", "could not swap adjacent generalized Schur blocks", info);
  }
  if (info > 1)
  {
    detail::throw_runtime_info("tgexc", "failed unexpectedly", info);
  }
}

/// \brief Reorder selected blocks in a generalized real Schur form and estimate conditioning.
template <uni20::LapackReal Scalar>
void tgsen(blas_int ijob, bool wantq, bool wantz, blas_int* select, blas_int n, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* alphar, Scalar* alphai, Scalar* beta, Scalar* q, blas_int ldq, Scalar* z, blas_int ldz,
           blas_int& selected_dimension, Scalar& pl, Scalar& pr, Scalar* dif, Scalar* work, blas_int lwork,
           blas_int* iwork, blas_int liwork)
{
  blas_int const info = unchecked::tgsen(ijob, wantq, wantz, select, n, a, lda, b, ldb, alphar, alphai, beta, q, ldq, z,
                                         ldz, selected_dimension, pl, pr, dif, work, lwork, iwork, liwork);
  detail::check_invalid_argument("tgsen", info);
  if (info == 1)
  {
    detail::throw_runtime_info("tgsen", "could not reorder the selected generalized Schur blocks", info);
  }
  if (info > 1)
  {
    detail::throw_runtime_info("tgsen", "failed unexpectedly", info);
  }
}

/// \brief Compute eigenvectors of a generalized real Schur form.
template <uni20::LapackReal Scalar>
void tgevc(char side, char howmny, blas_int* select, blas_int n, Scalar* s, blas_int lds, Scalar* p, blas_int ldp,
           Scalar* vl, blas_int ldvl, Scalar* vr, blas_int ldvr, blas_int mm, blas_int& computed_vectors, Scalar* work)
{
  blas_int const info =
      unchecked::tgevc(side, howmny, select, n, s, lds, p, ldp, vl, ldvl, vr, ldvr, mm, computed_vectors, work);
  detail::check_invalid_argument("tgevc", info);
  if (info > 0)
  {
    detail::throw_runtime_info("tgevc", "failed to compute generalized Schur eigenvectors", info);
  }
}

/// \brief Estimate reciprocal condition numbers for generalized real Schur eigenpairs.
template <uni20::LapackReal Scalar>
void tgsna(char job, char howmny, blas_int* select, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb,
           Scalar* vl, blas_int ldvl, Scalar* vr, blas_int ldvr, Scalar* reciprocal_eigenvalue_condition_numbers,
           Scalar* reciprocal_eigenvector_condition_numbers, blas_int mm, blas_int& computed_estimates, Scalar* work,
           blas_int lwork, blas_int* iwork)
{
  blas_int const info = unchecked::tgsna(
      job, howmny, select, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, reciprocal_eigenvalue_condition_numbers,
      reciprocal_eigenvector_condition_numbers, mm, computed_estimates, work, lwork, iwork);
  detail::check_invalid_argument("tgsna", info);
  if (info > 0)
  {
    detail::throw_runtime_info("tgsna", "failed unexpectedly", info);
  }
}

/// \brief Generate an explicit real orthogonal factor from bidiagonal reduction.
template <uni20::LapackReal Scalar>
void orgbr(char vect, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work,
           blas_int lwork)
{
  blas_int const info = unchecked::orgbr(vect, m, n, k, a, lda, tau, work, lwork);
  detail::check_invalid_argument("orgbr", info);
}

/// \brief Generate an explicit real orthogonal factor from Hessenberg reduction.
template <uni20::LapackReal Scalar>
void orghr(blas_int n, blas_int first, blas_int last, Scalar* a, blas_int lda, Scalar const* tau, Scalar* work,
           blas_int lwork)
{
  blas_int const info = unchecked::orghr(n, first, last, a, lda, tau, work, lwork);
  detail::check_invalid_argument("orghr", info);
}

/// \brief Generate an explicit real orthogonal factor from symmetric tridiagonal reduction.
template <uni20::LapackReal Scalar>
void orgtr(char uplo, blas_int n, Scalar* a, blas_int lda, Scalar const* tau, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::orgtr(uplo, n, a, lda, tau, work, lwork);
  detail::check_invalid_argument("orgtr", info);
}

/// \brief Multiply by an implicit real orthogonal factor from Hessenberg reduction.
template <uni20::LapackReal Scalar>
void ormhr(char side, char trans, blas_int m, blas_int n, blas_int first, blas_int last, Scalar* a, blas_int lda,
           Scalar const* tau, Scalar* c, blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::ormhr(side, trans, m, n, first, last, a, lda, tau, c, ldc, work, lwork);
  detail::check_invalid_argument("ormhr", info);
}

/// \brief Multiply by an implicit real orthogonal factor from symmetric tridiagonal reduction.
template <uni20::LapackReal Scalar>
void ormtr(char side, char uplo, char trans, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar const* tau,
           Scalar* c, blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::ormtr(side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork);
  detail::check_invalid_argument("ormtr", info);
}

/// \brief Multiply by an implicit real orthogonal factor from bidiagonal reduction.
template <uni20::LapackReal Scalar>
void ormbr(char vect, char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau,
           Scalar* c, blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::ormbr(vect, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::check_invalid_argument("ormbr", info);
}

/// \brief Compute a real dense singular value decomposition.
template <uni20::LapackReal Scalar>
void gesvd(char jobu, char jobvt, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* s, Scalar* u, blas_int ldu,
           Scalar* vt, blas_int ldvt, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::gesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
  detail::check_convergence("gesvd", info);
}

/// \brief Compute a real dense divide-and-conquer singular value decomposition.
template <uni20::LapackReal Scalar>
void gesdd(char jobz, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* s, Scalar* u, blas_int ldu, Scalar* vt,
           blas_int ldvt, Scalar* work, blas_int lwork, blas_int* iwork)
{
  blas_int const info = unchecked::gesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork);
  detail::check_convergence("gesdd", info);
}

/// \brief Compute selected singular values/vectors of a real dense matrix.
template <uni20::LapackReal Scalar>
void gesvdx(char jobu, char jobvt, char range, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar vl, Scalar vu,
            blas_int il, blas_int iu, blas_int& selected_count, Scalar* singular_values, Scalar* u, blas_int ldu,
            Scalar* vt, blas_int ldvt, Scalar* work, blas_int lwork, blas_int* iwork)
{
  blas_int const info = unchecked::gesvdx(jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, selected_count,
                                          singular_values, u, ldu, vt, ldvt, work, lwork, iwork);
  detail::check_convergence("gesvdx", info);
}

/// \brief Compute singular values of a real bidiagonal matrix.
template <uni20::LapackReal Scalar>
void bdsqr(char uplo, blas_int n, blas_int ncvt, blas_int nru, blas_int ncc, Scalar* d, Scalar* e, Scalar* vt,
           blas_int ldvt, Scalar* u, blas_int ldu, Scalar* c, blas_int ldc, Scalar* work)
{
  blas_int const info = unchecked::bdsqr(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work);
  detail::check_convergence("bdsqr", info);
}

/// \brief Compute the singular value decomposition of a real bidiagonal matrix.
template <uni20::LapackReal Scalar>
void bdsdc(char uplo, char compq, blas_int n, Scalar* d, Scalar* e, Scalar* u, blas_int ldu, Scalar* vt, blas_int ldvt,
           Scalar* q, blas_int* iq, Scalar* work, blas_int* iwork)
{
  if (compq != 'N' && compq != 'I')
  {
    throw std::invalid_argument("LAPACK bdsdc wrapper supports only singular values or explicit singular vectors");
  }
  blas_int const info = unchecked::bdsdc(uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, work, iwork);
  detail::check_convergence("bdsdc", info);
}

/// \brief Compute selected singular values/vectors of a real bidiagonal matrix.
template <uni20::LapackReal Scalar>
void bdsvdx(char uplo, char jobz, char range, blas_int n, Scalar* d, Scalar* e, Scalar vl, Scalar vu, blas_int il,
            blas_int iu, blas_int& selected_count, Scalar* singular_values, Scalar* z, blas_int ldz, Scalar* work,
            blas_int* iwork)
{
  blas_int const info = unchecked::bdsvdx(uplo, jobz, range, n, d, e, vl, vu, il, iu, selected_count, singular_values,
                                          z, ldz, work, iwork);
  detail::check_convergence("bdsvdx", info);
}

/// \brief Compute a real dense Cholesky factorization.
template <uni20::LapackReal Scalar> void potrf(char uplo, blas_int n, Scalar* a, blas_int lda)
{
  blas_int const info = unchecked::potrf(uplo, n, a, lda);
  detail::check_invalid_argument("potrf", info);
  if (info > 0)
  {
    detail::throw_runtime_info("potrf", "found a matrix that is not positive definite", info);
  }
}

/// \brief Compute a real dense pivoted Cholesky factorization.
template <uni20::LapackReal Scalar>
bool pstrf(char uplo, blas_int n, Scalar* a, blas_int lda, blas_int* pivots, blas_int& rank, Scalar tolerance,
           Scalar* work)
{
  blas_int const info = unchecked::pstrf(uplo, n, a, lda, pivots, rank, tolerance, work);
  detail::check_invalid_argument("pstrf", info);
  if (info > 1)
  {
    detail::throw_runtime_info("pstrf", "failed unexpectedly", info);
  }
  return info == 1;
}

/// \brief Invert a real dense matrix from an existing Cholesky factorization.
template <uni20::LapackReal Scalar> void potri(char uplo, blas_int n, Scalar* a, blas_int lda)
{
  blas_int const info = unchecked::potri(uplo, n, a, lda);
  detail::check_invalid_argument("potri", info);
  if (info > 0)
  {
    detail::throw_runtime_info("potri", "found a singular Cholesky factor", info);
  }
}

/// \brief Solve using an existing real dense Cholesky factorization.
template <uni20::LapackReal Scalar>
void potrs(char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::potrs(uplo, n, nrhs, a, lda, b, ldb);
  detail::check_invalid_argument("potrs", info);
}

/// \brief Refine a real dense SPD linear-system solve and return LAPACK diagnostics.
template <uni20::LapackReal Scalar>
void porfs(char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* factors, blas_int factor_lda,
           Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error, Scalar* backward_error,
           Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::porfs(uplo, n, nrhs, a, lda, factors, factor_lda, b, ldb, x, ldx, forward_error,
                                         backward_error, work, iwork);
  detail::check_invalid_argument("porfs", info);
}

/// \brief Solve a real dense SPD linear system with LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
bool posvx(char fact, char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* af, blas_int ldaf,
           char& equed, Scalar* scale, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond,
           Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::posvx(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, scale, b, ldb, x, ldx, rcond,
                                         forward_error, backward_error, work, iwork);
  detail::check_invalid_argument("posvx", info);
  if (info > 0 && info <= n)
  {
    detail::throw_runtime_info("posvx", "found a non-positive-definite leading minor", info);
  }
  if (info > n + 1)
  {
    detail::throw_runtime_info("posvx", "failed unexpectedly", info);
  }
  return info == n + 1;
}

/// \brief Compute real dense SPD equilibration factors.
template <uni20::LapackReal Scalar>
void poequ(blas_int n, Scalar* a, blas_int lda, Scalar* scale, Scalar& scale_condition, Scalar& max_abs)
{
  blas_int const info = unchecked::poequ(n, a, lda, scale, scale_condition, max_abs);
  detail::check_invalid_argument("poequ", info);
  if (info > 0)
  {
    detail::throw_runtime_info("poequ", "found a non-positive diagonal entry", info);
  }
}

/// \brief Estimate a real dense SPD reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar pocon(char uplo, blas_int n, Scalar* a, blas_int lda, Scalar matrix_one_norm, Scalar* work, blas_int* iwork)
{
  Scalar rcond{};
  blas_int const info = unchecked::pocon(uplo, n, a, lda, matrix_one_norm, rcond, work, iwork);
  detail::check_invalid_argument("pocon", info);
  return rcond;
}

/// \brief Compute a dense real symmetric-indefinite Bunch-Kaufman factorization.
template <uni20::LapackReal Scalar>
void sytrf(char uplo, blas_int n, Scalar* a, blas_int lda, blas_int* ipiv, Scalar* work, blas_int lwork)
{
  blas_int const info = unchecked::sytrf(uplo, n, a, lda, ipiv, work, lwork);
  detail::check_singular_diagonal_block("sytrf", info);
}

/// \brief Solve using an existing dense real symmetric-indefinite factorization.
template <uni20::LapackReal Scalar>
void sytrs(char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, blas_int const* ipiv, Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::sytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
  detail::check_invalid_argument("sytrs", info);
}

/// \brief Invert a dense real symmetric-indefinite matrix from its factorization.
template <uni20::LapackReal Scalar>
void sytri(char uplo, blas_int n, Scalar* a, blas_int lda, blas_int const* ipiv, Scalar* work)
{
  blas_int const info = unchecked::sytri(uplo, n, a, lda, ipiv, work);
  detail::check_singular_diagonal_block("sytri", info);
}

/// \brief Refine a dense real symmetric-indefinite solve and return LAPACK diagnostics.
template <uni20::LapackReal Scalar>
void syrfs(char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* factors, blas_int factor_lda,
           blas_int const* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error,
           Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::syrfs(uplo, n, nrhs, a, lda, factors, factor_lda, ipiv, b, ldb, x, ldx,
                                         forward_error, backward_error, work, iwork);
  detail::check_invalid_argument("syrfs", info);
}

/// \brief Solve a dense real symmetric-indefinite system with LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
bool sysvx(char fact, char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* af, blas_int ldaf,
           blas_int* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond, Scalar* forward_error,
           Scalar* backward_error, Scalar* work, blas_int lwork, blas_int* iwork)
{
  blas_int const info = unchecked::sysvx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, rcond,
                                         forward_error, backward_error, work, lwork, iwork);
  return detail::check_symmetric_indefinite_expert_solve("sysvx", n, info);
}

/// \brief Estimate a dense real symmetric-indefinite reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar sycon(char uplo, blas_int n, Scalar* a, blas_int lda, blas_int const* ipiv, Scalar matrix_one_norm, Scalar* work,
             blas_int* iwork)
{
  Scalar rcond{};
  blas_int const info = unchecked::sycon(uplo, n, a, lda, ipiv, matrix_one_norm, rcond, work, iwork);
  detail::check_invalid_argument("sycon", info);
  return rcond;
}

/// \brief Solve a dense real triangular system.
template <uni20::LapackReal Scalar>
void trtrs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb)
{
  blas_int const info = unchecked::trtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
  detail::check_singular_triangular("trtrs", info);
}

/// \brief Refine a dense real triangular solve and return LAPACK diagnostics.
template <uni20::LapackReal Scalar>
void trrfs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error, Scalar* backward_error, Scalar* work,
           blas_int* iwork)
{
  blas_int const info =
      unchecked::trrfs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, x, ldx, forward_error, backward_error, work, iwork);
  detail::check_invalid_argument("trrfs", info);
}

/// \brief Invert a dense real triangular matrix.
template <uni20::LapackReal Scalar> void trtri(char uplo, char diag, blas_int n, Scalar* a, blas_int lda)
{
  blas_int const info = unchecked::trtri(uplo, diag, n, a, lda);
  detail::check_singular_triangular("trtri", info);
}

/// \brief Estimate a dense real triangular reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar trcon(char norm, char uplo, char diag, blas_int n, Scalar* a, blas_int lda, Scalar* work, blas_int* iwork)
{
  Scalar rcond{};
  blas_int const info = unchecked::trcon(norm, uplo, diag, n, a, lda, rcond, work, iwork);
  detail::check_invalid_argument("trcon", info);
  return rcond;
}

/// \brief Solve a dense real Sylvester equation.
template <uni20::LapackReal Scalar>
bool trsyl(char trans_a, char trans_b, blas_int sign, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* c, blas_int ldc, Scalar& scale)
{
  blas_int const info = unchecked::trsyl(trans_a, trans_b, sign, m, n, a, lda, b, ldb, c, ldc, scale);
  return detail::check_sylvester("trsyl", info);
}

/// \brief Solve a real tridiagonal linear system through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void gtsv(blas_int n, blas_int nrhs, Scalar* dl, Scalar* d, Scalar* du, Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::gtsv(n, nrhs, dl, d, du, b, ldb);
  detail::check_singular("gtsv", info);
}

/// \brief Compute a real tridiagonal LU factorization through the configured LAPACK backend.
template <uni20::LapackReal Scalar>
void gttrf(blas_int n, Scalar* dl, Scalar* d, Scalar* du, Scalar* du2, blas_int* ipiv)
{
  blas_int const info = unchecked::gttrf(n, dl, d, du, du2, ipiv);
  detail::check_singular("gttrf", info);
}

/// \brief Solve using an existing real tridiagonal LU factorization.
template <uni20::LapackReal Scalar>
void gttrs(char trans, blas_int n, blas_int nrhs, Scalar* dl, Scalar* d, Scalar* du, Scalar* du2, blas_int const* ipiv,
           Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::gttrs(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
  detail::check_invalid_argument("gttrs", info);
}

/// \brief Estimate a real tridiagonal reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar gtcon(char norm, blas_int n, Scalar* dl, Scalar* d, Scalar* du, Scalar* du2, blas_int const* ipiv,
             Scalar matrix_norm, Scalar* work, blas_int* iwork)
{
  Scalar rcond{};
  blas_int const info = unchecked::gtcon(norm, n, dl, d, du, du2, ipiv, matrix_norm, rcond, work, iwork);
  detail::check_invalid_argument("gtcon", info);
  return rcond;
}

/// \brief Refine a real tridiagonal linear-system solve and return LAPACK diagnostics.
template <uni20::LapackReal Scalar>
void gtrfs(char trans, blas_int n, blas_int nrhs, Scalar* dl, Scalar* d, Scalar* du, Scalar* dlf, Scalar* df,
           Scalar* duf, Scalar* du2, blas_int const* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx,
           Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::gtrfs(trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx,
                                         forward_error, backward_error, work, iwork);
  detail::check_invalid_argument("gtrfs", info);
}

/// \brief Solve a real tridiagonal linear system with LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
bool gtsvx(char fact, char trans, blas_int n, blas_int nrhs, Scalar* dl, Scalar* d, Scalar* du, Scalar* dlf, Scalar* df,
           Scalar* duf, Scalar* du2, blas_int* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond,
           Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int const info = unchecked::gtsvx(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx,
                                         rcond, forward_error, backward_error, work, iwork);
  return detail::check_expert_solve("gtsvx", n, info);
}

/// \brief Solve a real symmetric positive-definite tridiagonal linear system.
template <uni20::LapackReal Scalar> void ptsv(blas_int n, blas_int nrhs, Scalar* d, Scalar* e, Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::ptsv(n, nrhs, d, e, b, ldb);
  detail::check_positive_definite("ptsv", info);
}

/// \brief Compute a real symmetric positive-definite tridiagonal factorization.
template <uni20::LapackReal Scalar> void pttrf(blas_int n, Scalar* d, Scalar* e)
{
  blas_int const info = unchecked::pttrf(n, d, e);
  detail::check_positive_definite("pttrf", info);
}

/// \brief Solve using an existing real symmetric positive-definite tridiagonal factorization.
template <uni20::LapackReal Scalar> void pttrs(blas_int n, blas_int nrhs, Scalar* d, Scalar* e, Scalar* b, blas_int ldb)
{
  blas_int const info = unchecked::pttrs(n, nrhs, d, e, b, ldb);
  detail::check_invalid_argument("pttrs", info);
  if (info > 0)
  {
    detail::throw_runtime_info("pttrs", "failed to solve the factored system", info);
  }
}

/// \brief Estimate a real symmetric positive-definite tridiagonal reciprocal condition number.
template <uni20::LapackReal Scalar> Scalar ptcon(blas_int n, Scalar* d, Scalar* e, Scalar matrix_norm, Scalar* work)
{
  Scalar rcond{};
  blas_int const info = unchecked::ptcon(n, d, e, matrix_norm, rcond, work);
  detail::check_invalid_argument("ptcon", info);
  return rcond;
}

/// \brief Refine a real symmetric positive-definite tridiagonal solve and return LAPACK diagnostics.
template <uni20::LapackReal Scalar>
void ptrfs(blas_int n, blas_int nrhs, Scalar* d, Scalar* e, Scalar* df, Scalar* ef, Scalar* b, blas_int ldb, Scalar* x,
           blas_int ldx, Scalar* forward_error, Scalar* backward_error, Scalar* work)
{
  blas_int const info = unchecked::ptrfs(n, nrhs, d, e, df, ef, b, ldb, x, ldx, forward_error, backward_error, work);
  detail::check_invalid_argument("ptrfs", info);
}

/// \brief Solve a real symmetric positive-definite tridiagonal system with LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
bool ptsvx(char fact, blas_int n, blas_int nrhs, Scalar* d, Scalar* e, Scalar* df, Scalar* ef, Scalar* b, blas_int ldb,
           Scalar* x, blas_int ldx, Scalar& rcond, Scalar* forward_error, Scalar* backward_error, Scalar* work)
{
  blas_int const info =
      unchecked::ptsvx(fact, n, nrhs, d, e, df, ef, b, ldb, x, ldx, rcond, forward_error, backward_error, work);
  return detail::check_positive_definite_expert_solve("ptsvx", n, info);
}

/// \brief Compute eigenvalues of a real symmetric tridiagonal matrix.
template <uni20::LapackReal Scalar> void sterf(blas_int n, Scalar* d, Scalar* e)
{
  blas_int const info = unchecked::sterf(n, d, e);
  detail::check_convergence("sterf", info);
}

/// \brief Compute a real symmetric tridiagonal eigensystem by QR iteration.
template <uni20::LapackReal Scalar>
void steqr(char compz, blas_int n, Scalar* d, Scalar* e, Scalar* z, blas_int ldz, Scalar* work)
{
  blas_int const info = unchecked::steqr(compz, n, d, e, z, ldz, work);
  detail::check_convergence("steqr", info);
}

/// \brief Compute a real symmetric tridiagonal eigensystem by divide-and-conquer.
template <uni20::LapackReal Scalar>
void stevd(char jobz, blas_int n, Scalar* d, Scalar* e, Scalar* z, blas_int ldz, Scalar* work, blas_int lwork,
           blas_int* iwork, blas_int liwork)
{
  blas_int const info = unchecked::stevd(jobz, n, d, e, z, ldz, work, lwork, iwork, liwork);
  detail::check_convergence("stevd", info);
}

/// \brief Compute selected eigenvalues/eigenvectors of a real symmetric tridiagonal matrix.
template <uni20::LapackReal Scalar>
void stevr(char jobz, char range, blas_int n, Scalar* d, Scalar* e, Scalar vl, Scalar vu, blas_int il, blas_int iu,
           Scalar abstol, blas_int& selected_count, Scalar* w, Scalar* z, blas_int ldz, blas_int* isuppz, Scalar* work,
           blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int const info = unchecked::stevr(jobz, range, n, d, e, vl, vu, il, iu, abstol, selected_count, w, z, ldz,
                                         isuppz, work, lwork, iwork, liwork);
  detail::check_convergence("stevr", info);
}

} // namespace uni20::lapack
