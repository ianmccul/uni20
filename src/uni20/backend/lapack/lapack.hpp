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

inline void check_convergence(char const* routine, blas_int info)
{
  check_invalid_argument(routine, info);
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

} // namespace uni20::lapack
