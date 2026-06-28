#pragma once

/**
 * \file tridiagonal.hpp
 * \ingroup backend_lapack_reference
 * \brief Reference LAPACK wrappers for real tridiagonal linear algebra.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/config.hpp>

namespace uni20::lapack::unchecked
{

namespace detail
{
extern "C"
{
  void sgtsv_(blas_int const* n, blas_int const* nrhs, float* dl, float* d, float* du, float* b, blas_int const* ldb,
              blas_int* info);

  void dgtsv_(blas_int const* n, blas_int const* nrhs, double* dl, double* d, double* du, double* b,
              blas_int const* ldb, blas_int* info);

  void sgttrf_(blas_int const* n, float* dl, float* d, float* du, float* du2, blas_int* ipiv, blas_int* info);

  void dgttrf_(blas_int const* n, double* dl, double* d, double* du, double* du2, blas_int* ipiv, blas_int* info);

  void sgttrs_(char const* trans, blas_int const* n, blas_int const* nrhs, float* dl, float* d, float* du, float* du2,
               blas_int const* ipiv, float* b, blas_int const* ldb, blas_int* info);

  void dgttrs_(char const* trans, blas_int const* n, blas_int const* nrhs, double* dl, double* d, double* du,
               double* du2, blas_int const* ipiv, double* b, blas_int const* ldb, blas_int* info);

  void sgtcon_(char const* norm, blas_int const* n, float* dl, float* d, float* du, float* du2, blas_int const* ipiv,
               float const* anorm, float* rcond, float* work, blas_int* iwork, blas_int* info);

  void dgtcon_(char const* norm, blas_int const* n, double* dl, double* d, double* du, double* du2,
               blas_int const* ipiv, double const* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info);

  void sgtrfs_(char const* trans, blas_int const* n, blas_int const* nrhs, float* dl, float* d, float* du, float* dlf,
               float* df, float* duf, float* du2, blas_int const* ipiv, float* b, blas_int const* ldb, float* x,
               blas_int const* ldx, float* ferr, float* berr, float* work, blas_int* iwork, blas_int* info);

  void dgtrfs_(char const* trans, blas_int const* n, blas_int const* nrhs, double* dl, double* d, double* du,
               double* dlf, double* df, double* duf, double* du2, blas_int const* ipiv, double* b, blas_int const* ldb,
               double* x, blas_int const* ldx, double* ferr, double* berr, double* work, blas_int* iwork,
               blas_int* info);

  void sgtsvx_(char const* fact, char const* trans, blas_int const* n, blas_int const* nrhs, float* dl, float* d,
               float* du, float* dlf, float* df, float* duf, float* du2, blas_int* ipiv, float* b, blas_int const* ldb,
               float* x, blas_int const* ldx, float* rcond, float* ferr, float* berr, float* work, blas_int* iwork,
               blas_int* info);

  void dgtsvx_(char const* fact, char const* trans, blas_int const* n, blas_int const* nrhs, double* dl, double* d,
               double* du, double* dlf, double* df, double* duf, double* du2, blas_int* ipiv, double* b,
               blas_int const* ldb, double* x, blas_int const* ldx, double* rcond, double* ferr, double* berr,
               double* work, blas_int* iwork, blas_int* info);
}
} // namespace detail

[[nodiscard]] inline blas_int gtsv(blas_int n, blas_int nrhs, float* dl, float* d, float* du, float* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgtsv_, n, nrhs, dl, d, du, b, ldb);
  detail::sgtsv_(&n, &nrhs, dl, d, du, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gtsv(blas_int n, blas_int nrhs, double* dl, double* d, double* du, double* b,
                                   blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgtsv_, n, nrhs, dl, d, du, b, ldb);
  detail::dgtsv_(&n, &nrhs, dl, d, du, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gttrf(blas_int n, float* dl, float* d, float* du, float* du2, blas_int* ipiv)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgttrf_, n, dl, d, du, du2, ipiv);
  detail::sgttrf_(&n, dl, d, du, du2, ipiv, &info);
  return info;
}

[[nodiscard]] inline blas_int gttrf(blas_int n, double* dl, double* d, double* du, double* du2, blas_int* ipiv)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgttrf_, n, dl, d, du, du2, ipiv);
  detail::dgttrf_(&n, dl, d, du, du2, ipiv, &info);
  return info;
}

[[nodiscard]] inline blas_int gttrs(char trans, blas_int n, blas_int nrhs, float* dl, float* d, float* du, float* du2,
                                    blas_int const* ipiv, float* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgttrs_, trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
  detail::sgttrs_(&trans, &n, &nrhs, dl, d, du, du2, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gttrs(char trans, blas_int n, blas_int nrhs, double* dl, double* d, double* du,
                                    double* du2, blas_int const* ipiv, double* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgttrs_, trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
  detail::dgttrs_(&trans, &n, &nrhs, dl, d, du, du2, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gtcon(char norm, blas_int n, float* dl, float* d, float* du, float* du2,
                                    blas_int const* ipiv, float matrix_norm, float& rcond, float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgtcon_, norm, n, dl, d, du, du2, ipiv, matrix_norm, work, iwork);
  detail::sgtcon_(&norm, &n, dl, d, du, du2, ipiv, &matrix_norm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gtcon(char norm, blas_int n, double* dl, double* d, double* du, double* du2,
                                    blas_int const* ipiv, double matrix_norm, double& rcond, double* work,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgtcon_, norm, n, dl, d, du, du2, ipiv, matrix_norm, work, iwork);
  detail::dgtcon_(&norm, &n, dl, d, du, du2, ipiv, &matrix_norm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gtrfs(char trans, blas_int n, blas_int nrhs, float* dl, float* d, float* du, float* dlf,
                                    float* df, float* duf, float* du2, blas_int const* ipiv, float* b, blas_int ldb,
                                    float* x, blas_int ldx, float* forward_error, float* backward_error, float* work,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgtrfs_, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  detail::sgtrfs_(&trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb, x, &ldx, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gtrfs(char trans, blas_int n, blas_int nrhs, double* dl, double* d, double* du,
                                    double* dlf, double* df, double* duf, double* du2, blas_int const* ipiv, double* b,
                                    blas_int ldb, double* x, blas_int ldx, double* forward_error,
                                    double* backward_error, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgtrfs_, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  detail::dgtrfs_(&trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb, x, &ldx, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gtsvx(char fact, char trans, blas_int n, blas_int nrhs, float* dl, float* d, float* du,
                                    float* dlf, float* df, float* duf, float* du2, blas_int* ipiv, float* b,
                                    blas_int ldb, float* x, blas_int ldx, float& rcond, float* forward_error,
                                    float* backward_error, float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgtsvx_, fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx,
                          work, iwork);
  detail::sgtsvx_(&fact, &trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb, x, &ldx, &rcond, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gtsvx(char fact, char trans, blas_int n, blas_int nrhs, double* dl, double* d, double* du,
                                    double* dlf, double* df, double* duf, double* du2, blas_int* ipiv, double* b,
                                    blas_int ldb, double* x, blas_int ldx, double& rcond, double* forward_error,
                                    double* backward_error, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgtsvx_, fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx,
                          work, iwork);
  detail::dgtsvx_(&fact, &trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb, x, &ldx, &rcond, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

} // namespace uni20::lapack::unchecked
