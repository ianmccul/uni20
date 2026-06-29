#pragma once

/**
 * \file general.hpp
 * \ingroup backend_lapack_reference
 * \brief Reference LAPACK wrappers for real dense general linear systems.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/config.hpp>

namespace uni20::lapack::unchecked
{

namespace detail
{
extern "C"
{
  void sgesv_(blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, blas_int* ipiv, float* b,
              blas_int const* ldb, blas_int* info);

  void dgesv_(blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, blas_int* ipiv, double* b,
              blas_int const* ldb, blas_int* info);

  void sgesvx_(char const* fact, char const* trans, blas_int const* n, blas_int const* nrhs, float* a,
               blas_int const* lda, float* af, blas_int const* ldaf, blas_int* ipiv, char* equed, float* r, float* c,
               float* b, blas_int const* ldb, float* x, blas_int const* ldx, float* rcond, float* ferr, float* berr,
               float* work, blas_int* iwork, blas_int* info);

  void dgesvx_(char const* fact, char const* trans, blas_int const* n, blas_int const* nrhs, double* a,
               blas_int const* lda, double* af, blas_int const* ldaf, blas_int* ipiv, char* equed, double* r, double* c,
               double* b, blas_int const* ldb, double* x, blas_int const* ldx, double* rcond, double* ferr,
               double* berr, double* work, blas_int* iwork, blas_int* info);

  void sgeequ_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* r, float* c, float* rowcnd,
               float* colcnd, float* amax, blas_int* info);

  void dgeequ_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* r, double* c,
               double* rowcnd, double* colcnd, double* amax, blas_int* info);

  void sgetrf_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, blas_int* ipiv, blas_int* info);

  void dgetrf_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, blas_int* ipiv, blas_int* info);

  void sgetrs_(char const* trans, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda,
               blas_int const* ipiv, float* b, blas_int const* ldb, blas_int* info);

  void dgetrs_(char const* trans, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda,
               blas_int const* ipiv, double* b, blas_int const* ldb, blas_int* info);

  void sgerfs_(char const* trans, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* af,
               blas_int const* ldaf, blas_int const* ipiv, float* b, blas_int const* ldb, float* x, blas_int const* ldx,
               float* ferr, float* berr, float* work, blas_int* iwork, blas_int* info);

  void dgerfs_(char const* trans, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* af,
               blas_int const* ldaf, blas_int const* ipiv, double* b, blas_int const* ldb, double* x,
               blas_int const* ldx, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info);

  void sgetri_(blas_int const* n, float* a, blas_int const* lda, blas_int* ipiv, float* work, blas_int const* lwork,
               blas_int* info);

  void dgetri_(blas_int const* n, double* a, blas_int const* lda, blas_int* ipiv, double* work, blas_int const* lwork,
               blas_int* info);

  void sgecon_(char const* norm, blas_int const* n, float const* a, blas_int const* lda, float const* anorm,
               float* rcond, float* work, blas_int* iwork, blas_int* info);

  void dgecon_(char const* norm, blas_int const* n, double const* a, blas_int const* lda, double const* anorm,
               double* rcond, double* work, blas_int* iwork, blas_int* info);

  void sgels_(char const* trans, blas_int const* m, blas_int const* n, blas_int const* nrhs, float* a,
              blas_int const* lda, float* b, blas_int const* ldb, float* work, blas_int const* lwork, blas_int* info);

  void dgels_(char const* trans, blas_int const* m, blas_int const* n, blas_int const* nrhs, double* a,
              blas_int const* lda, double* b, blas_int const* ldb, double* work, blas_int const* lwork, blas_int* info);

  void sgelss_(blas_int const* m, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* b,
               blas_int const* ldb, float* s, float const* rcond, blas_int* rank, float* work, blas_int const* lwork,
               blas_int* info);

  void dgelss_(blas_int const* m, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* b,
               blas_int const* ldb, double* s, double const* rcond, blas_int* rank, double* work, blas_int const* lwork,
               blas_int* info);

  void sgelsd_(blas_int const* m, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* b,
               blas_int const* ldb, float* s, float const* rcond, blas_int* rank, float* work, blas_int const* lwork,
               blas_int* iwork, blas_int* info);

  void dgelsd_(blas_int const* m, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* b,
               blas_int const* ldb, double* s, double const* rcond, blas_int* rank, double* work, blas_int const* lwork,
               blas_int* iwork, blas_int* info);

  void sgelsy_(blas_int const* m, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* b,
               blas_int const* ldb, blas_int* jpvt, float const* rcond, blas_int* rank, float* work,
               blas_int const* lwork, blas_int* info);

  void dgelsy_(blas_int const* m, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* b,
               blas_int const* ldb, blas_int* jpvt, double const* rcond, blas_int* rank, double* work,
               blas_int const* lwork, blas_int* info);
}
} // namespace detail

[[nodiscard]] inline blas_int gesv(blas_int n, blas_int nrhs, float* a, blas_int lda, blas_int* ipiv, float* b,
                                   blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgesv_, n, nrhs, a, lda, ipiv, b, ldb);
  detail::sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gesv(blas_int n, blas_int nrhs, double* a, blas_int lda, blas_int* ipiv, double* b,
                                   blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgesv_, n, nrhs, a, lda, ipiv, b, ldb);
  detail::dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvx(char fact, char trans, blas_int n, blas_int nrhs, float* a, blas_int lda, float* af,
                                    blas_int ldaf, blas_int* ipiv, char& equed, float* row_scale, float* column_scale,
                                    float* b, blas_int ldb, float* x, blas_int ldx, float& rcond, float* forward_error,
                                    float* backward_error, float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgesvx_, fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, row_scale, column_scale,
                          b, ldb, x, ldx, work, iwork);
  detail::sgesvx_(&fact, &trans, &n, &nrhs, a, &lda, af, &ldaf, ipiv, &equed, row_scale, column_scale, b, &ldb, x, &ldx,
                  &rcond, forward_error, backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvx(char fact, char trans, blas_int n, blas_int nrhs, double* a, blas_int lda,
                                    double* af, blas_int ldaf, blas_int* ipiv, char& equed, double* row_scale,
                                    double* column_scale, double* b, blas_int ldb, double* x, blas_int ldx,
                                    double& rcond, double* forward_error, double* backward_error, double* work,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgesvx_, fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, row_scale, column_scale,
                          b, ldb, x, ldx, work, iwork);
  detail::dgesvx_(&fact, &trans, &n, &nrhs, a, &lda, af, &ldaf, ipiv, &equed, row_scale, column_scale, b, &ldb, x, &ldx,
                  &rcond, forward_error, backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geequ(blas_int m, blas_int n, float* a, blas_int lda, float* row_scale,
                                    float* column_scale, float& row_condition, float& column_condition, float& max_abs)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgeequ_, m, n, a, lda, row_scale, column_scale);
  detail::sgeequ_(&m, &n, a, &lda, row_scale, column_scale, &row_condition, &column_condition, &max_abs, &info);
  return info;
}

[[nodiscard]] inline blas_int geequ(blas_int m, blas_int n, double* a, blas_int lda, double* row_scale,
                                    double* column_scale, double& row_condition, double& column_condition,
                                    double& max_abs)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgeequ_, m, n, a, lda, row_scale, column_scale);
  detail::dgeequ_(&m, &n, a, &lda, row_scale, column_scale, &row_condition, &column_condition, &max_abs, &info);
  return info;
}

[[nodiscard]] inline blas_int getrf(blas_int m, blas_int n, float* a, blas_int lda, blas_int* ipiv)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgetrf_, m, n, a, lda, ipiv);
  detail::sgetrf_(&m, &n, a, &lda, ipiv, &info);
  return info;
}

[[nodiscard]] inline blas_int getrf(blas_int m, blas_int n, double* a, blas_int lda, blas_int* ipiv)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgetrf_, m, n, a, lda, ipiv);
  detail::dgetrf_(&m, &n, a, &lda, ipiv, &info);
  return info;
}

[[nodiscard]] inline blas_int getrs(char trans, blas_int n, blas_int nrhs, float* a, blas_int lda, blas_int const* ipiv,
                                    float* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgetrs_, trans, n, nrhs, a, lda, ipiv, b, ldb);
  detail::sgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int getrs(char trans, blas_int n, blas_int nrhs, double* a, blas_int lda,
                                    blas_int const* ipiv, double* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgetrs_, trans, n, nrhs, a, lda, ipiv, b, ldb);
  detail::dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gerfs(char trans, blas_int n, blas_int nrhs, float* a, blas_int lda, float* factors,
                                    blas_int factor_lda, blas_int const* ipiv, float* b, blas_int ldb, float* x,
                                    blas_int ldx, float* forward_error, float* backward_error, float* work,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgerfs_, trans, n, nrhs, a, lda, factors, factor_lda, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  detail::sgerfs_(&trans, &n, &nrhs, a, &lda, factors, &factor_lda, ipiv, b, &ldb, x, &ldx, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gerfs(char trans, blas_int n, blas_int nrhs, double* a, blas_int lda, double* factors,
                                    blas_int factor_lda, blas_int const* ipiv, double* b, blas_int ldb, double* x,
                                    blas_int ldx, double* forward_error, double* backward_error, double* work,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgerfs_, trans, n, nrhs, a, lda, factors, factor_lda, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  detail::dgerfs_(&trans, &n, &nrhs, a, &lda, factors, &factor_lda, ipiv, b, &ldb, x, &ldx, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int getri(blas_int n, float* a, blas_int lda, blas_int* ipiv, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgetri_, n, a, lda, ipiv, work, lwork);
  detail::sgetri_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int getri(blas_int n, double* a, blas_int lda, blas_int* ipiv, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgetri_, n, a, lda, ipiv, work, lwork);
  detail::dgetri_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gecon(char norm, blas_int n, float const* a, blas_int lda, float anorm, float& rcond,
                                    float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgecon_, norm, n, a, lda, anorm, work, iwork);
  detail::sgecon_(&norm, &n, a, &lda, &anorm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gecon(char norm, blas_int n, double const* a, blas_int lda, double anorm, double& rcond,
                                    double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgecon_, norm, n, a, lda, anorm, work, iwork);
  detail::dgecon_(&norm, &n, a, &lda, &anorm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gels(char trans, blas_int m, blas_int n, blas_int nrhs, float* a, blas_int lda, float* b,
                                   blas_int ldb, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgels_, trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
  detail::sgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gels(char trans, blas_int m, blas_int n, blas_int nrhs, double* a, blas_int lda,
                                   double* b, blas_int ldb, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgels_, trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
  detail::dgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gelss(blas_int m, blas_int n, blas_int nrhs, float* a, blas_int lda, float* b,
                                    blas_int ldb, float* s, float rcond, blas_int& rank, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgelss_, m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork);
  detail::sgelss_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gelss(blas_int m, blas_int n, blas_int nrhs, double* a, blas_int lda, double* b,
                                    blas_int ldb, double* s, double rcond, blas_int& rank, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgelss_, m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork);
  detail::dgelss_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gelsd(blas_int m, blas_int n, blas_int nrhs, float* a, blas_int lda, float* b,
                                    blas_int ldb, float* s, float rcond, blas_int& rank, float* work, blas_int lwork,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgelsd_, m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork);
  detail::sgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gelsd(blas_int m, blas_int n, blas_int nrhs, double* a, blas_int lda, double* b,
                                    blas_int ldb, double* s, double rcond, blas_int& rank, double* work, blas_int lwork,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgelsd_, m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork);
  detail::dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gelsy(blas_int m, blas_int n, blas_int nrhs, float* a, blas_int lda, float* b,
                                    blas_int ldb, blas_int* jpvt, float rcond, blas_int& rank, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgelsy_, m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank, work, lwork);
  detail::sgelsy_(&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, &rank, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gelsy(blas_int m, blas_int n, blas_int nrhs, double* a, blas_int lda, double* b,
                                    blas_int ldb, blas_int* jpvt, double rcond, blas_int& rank, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgelsy_, m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank, work, lwork);
  detail::dgelsy_(&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, &rank, work, &lwork, &info);
  return info;
}

} // namespace uni20::lapack::unchecked
