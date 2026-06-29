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

  void sptsv_(blas_int const* n, blas_int const* nrhs, float* d, float* e, float* b, blas_int const* ldb,
              blas_int* info);

  void dptsv_(blas_int const* n, blas_int const* nrhs, double* d, double* e, double* b, blas_int const* ldb,
              blas_int* info);

  void spttrf_(blas_int const* n, float* d, float* e, blas_int* info);

  void dpttrf_(blas_int const* n, double* d, double* e, blas_int* info);

  void spttrs_(blas_int const* n, blas_int const* nrhs, float* d, float* e, float* b, blas_int const* ldb,
               blas_int* info);

  void dpttrs_(blas_int const* n, blas_int const* nrhs, double* d, double* e, double* b, blas_int const* ldb,
               blas_int* info);

  void sptcon_(blas_int const* n, float* d, float* e, float const* anorm, float* rcond, float* work, blas_int* info);

  void dptcon_(blas_int const* n, double* d, double* e, double const* anorm, double* rcond, double* work,
               blas_int* info);

  void sptrfs_(blas_int const* n, blas_int const* nrhs, float* d, float* e, float* df, float* ef, float* b,
               blas_int const* ldb, float* x, blas_int const* ldx, float* ferr, float* berr, float* work,
               blas_int* info);

  void dptrfs_(blas_int const* n, blas_int const* nrhs, double* d, double* e, double* df, double* ef, double* b,
               blas_int const* ldb, double* x, blas_int const* ldx, double* ferr, double* berr, double* work,
               blas_int* info);

  void sptsvx_(char const* fact, blas_int const* n, blas_int const* nrhs, float* d, float* e, float* df, float* ef,
               float* b, blas_int const* ldb, float* x, blas_int const* ldx, float* rcond, float* ferr, float* berr,
               float* work, blas_int* info);

  void dptsvx_(char const* fact, blas_int const* n, blas_int const* nrhs, double* d, double* e, double* df, double* ef,
               double* b, blas_int const* ldb, double* x, blas_int const* ldx, double* rcond, double* ferr,
               double* berr, double* work, blas_int* info);

  void ssterf_(blas_int const* n, float* d, float* e, blas_int* info);

  void dsterf_(blas_int const* n, double* d, double* e, blas_int* info);

  void ssteqr_(char const* compz, blas_int const* n, float* d, float* e, float* z, blas_int const* ldz, float* work,
               blas_int* info);

  void dsteqr_(char const* compz, blas_int const* n, double* d, double* e, double* z, blas_int const* ldz, double* work,
               blas_int* info);

  void sstevd_(char const* jobz, blas_int const* n, float* d, float* e, float* z, blas_int const* ldz, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dstevd_(char const* jobz, blas_int const* n, double* d, double* e, double* z, blas_int const* ldz, double* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void sstevr_(char const* jobz, char const* range, blas_int const* n, float* d, float* e, float const* vl,
               float const* vu, blas_int const* il, blas_int const* iu, float const* abstol, blas_int* m, float* w,
               float* z, blas_int const* ldz, blas_int* isuppz, float* work, blas_int const* lwork, blas_int* iwork,
               blas_int const* liwork, blas_int* info);

  void dstevr_(char const* jobz, char const* range, blas_int const* n, double* d, double* e, double const* vl,
               double const* vu, blas_int const* il, blas_int const* iu, double const* abstol, blas_int* m, double* w,
               double* z, blas_int const* ldz, blas_int* isuppz, double* work, blas_int const* lwork, blas_int* iwork,
               blas_int const* liwork, blas_int* info);
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

[[nodiscard]] inline blas_int ptsv(blas_int n, blas_int nrhs, float* d, float* e, float* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sptsv_, n, nrhs, d, e, b, ldb);
  detail::sptsv_(&n, &nrhs, d, e, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int ptsv(blas_int n, blas_int nrhs, double* d, double* e, double* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dptsv_, n, nrhs, d, e, b, ldb);
  detail::dptsv_(&n, &nrhs, d, e, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int pttrf(blas_int n, float* d, float* e)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, spttrf_, n, d, e);
  detail::spttrf_(&n, d, e, &info);
  return info;
}

[[nodiscard]] inline blas_int pttrf(blas_int n, double* d, double* e)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dpttrf_, n, d, e);
  detail::dpttrf_(&n, d, e, &info);
  return info;
}

[[nodiscard]] inline blas_int pttrs(blas_int n, blas_int nrhs, float* d, float* e, float* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, spttrs_, n, nrhs, d, e, b, ldb);
  detail::spttrs_(&n, &nrhs, d, e, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int pttrs(blas_int n, blas_int nrhs, double* d, double* e, double* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dpttrs_, n, nrhs, d, e, b, ldb);
  detail::dpttrs_(&n, &nrhs, d, e, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int ptcon(blas_int n, float* d, float* e, float matrix_norm, float& rcond, float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sptcon_, n, d, e, matrix_norm, work);
  detail::sptcon_(&n, d, e, &matrix_norm, &rcond, work, &info);
  return info;
}

[[nodiscard]] inline blas_int ptcon(blas_int n, double* d, double* e, double matrix_norm, double& rcond, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dptcon_, n, d, e, matrix_norm, work);
  detail::dptcon_(&n, d, e, &matrix_norm, &rcond, work, &info);
  return info;
}

[[nodiscard]] inline blas_int ptrfs(blas_int n, blas_int nrhs, float* d, float* e, float* df, float* ef, float* b,
                                    blas_int ldb, float* x, blas_int ldx, float* forward_error, float* backward_error,
                                    float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sptrfs_, n, nrhs, d, e, df, ef, b, ldb, x, ldx, forward_error, backward_error, work);
  detail::sptrfs_(&n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, forward_error, backward_error, work, &info);
  return info;
}

[[nodiscard]] inline blas_int ptrfs(blas_int n, blas_int nrhs, double* d, double* e, double* df, double* ef, double* b,
                                    blas_int ldb, double* x, blas_int ldx, double* forward_error,
                                    double* backward_error, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dptrfs_, n, nrhs, d, e, df, ef, b, ldb, x, ldx, forward_error, backward_error, work);
  detail::dptrfs_(&n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, forward_error, backward_error, work, &info);
  return info;
}

[[nodiscard]] inline blas_int ptsvx(char fact, blas_int n, blas_int nrhs, float* d, float* e, float* df, float* ef,
                                    float* b, blas_int ldb, float* x, blas_int ldx, float& rcond, float* forward_error,
                                    float* backward_error, float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sptsvx_, fact, n, nrhs, d, e, df, ef, b, ldb, x, ldx, work);
  detail::sptsvx_(&fact, &n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, &rcond, forward_error, backward_error, work, &info);
  return info;
}

[[nodiscard]] inline blas_int ptsvx(char fact, blas_int n, blas_int nrhs, double* d, double* e, double* df, double* ef,
                                    double* b, blas_int ldb, double* x, blas_int ldx, double& rcond,
                                    double* forward_error, double* backward_error, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dptsvx_, fact, n, nrhs, d, e, df, ef, b, ldb, x, ldx, work);
  detail::dptsvx_(&fact, &n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, &rcond, forward_error, backward_error, work, &info);
  return info;
}

[[nodiscard]] inline blas_int sterf(blas_int n, float* d, float* e)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssterf_, n, d, e);
  detail::ssterf_(&n, d, e, &info);
  return info;
}

[[nodiscard]] inline blas_int sterf(blas_int n, double* d, double* e)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsterf_, n, d, e);
  detail::dsterf_(&n, d, e, &info);
  return info;
}

[[nodiscard]] inline blas_int steqr(char compz, blas_int n, float* d, float* e, float* z, blas_int ldz, float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssteqr_, compz, n, d, e, z, ldz, work);
  detail::ssteqr_(&compz, &n, d, e, z, &ldz, work, &info);
  return info;
}

[[nodiscard]] inline blas_int steqr(char compz, blas_int n, double* d, double* e, double* z, blas_int ldz, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsteqr_, compz, n, d, e, z, ldz, work);
  detail::dsteqr_(&compz, &n, d, e, z, &ldz, work, &info);
  return info;
}

[[nodiscard]] inline blas_int stevd(char jobz, blas_int n, float* d, float* e, float* z, blas_int ldz, float* work,
                                    blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sstevd_, jobz, n, d, e, z, ldz, work, lwork, iwork, liwork);
  detail::sstevd_(&jobz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int stevd(char jobz, blas_int n, double* d, double* e, double* z, blas_int ldz, double* work,
                                    blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dstevd_, jobz, n, d, e, z, ldz, work, lwork, iwork, liwork);
  detail::dstevd_(&jobz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int stevr(char jobz, char range, blas_int n, float* d, float* e, float vl, float vu,
                                    blas_int il, blas_int iu, float abstol, blas_int& selected_count, float* w,
                                    float* z, blas_int ldz, blas_int* isuppz, float* work, blas_int lwork,
                                    blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sstevr_, jobz, range, n, d, e, vl, vu, il, iu, abstol, selected_count, w, z, ldz,
                          isuppz, work, lwork, iwork, liwork);
  detail::sstevr_(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz, work,
                  &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int stevr(char jobz, char range, blas_int n, double* d, double* e, double vl, double vu,
                                    blas_int il, blas_int iu, double abstol, blas_int& selected_count, double* w,
                                    double* z, blas_int ldz, blas_int* isuppz, double* work, blas_int lwork,
                                    blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dstevr_, jobz, range, n, d, e, vl, vu, il, iu, abstol, selected_count, w, z, ldz,
                          isuppz, work, lwork, iwork, liwork);
  detail::dstevr_(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz, work,
                  &lwork, iwork, &liwork, &info);
  return info;
}

} // namespace uni20::lapack::unchecked
