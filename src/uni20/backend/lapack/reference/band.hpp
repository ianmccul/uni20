#pragma once

/**
 * \file band.hpp
 * \ingroup backend_lapack_reference
 * \brief Reference LAPACK wrappers for real general-band linear algebra.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/config.hpp>

namespace uni20::lapack::unchecked
{

namespace detail
{
extern "C"
{
  void sgbsv_(blas_int const* n, blas_int const* kl, blas_int const* ku, blas_int const* nrhs, float* ab,
              blas_int const* ldab, blas_int* ipiv, float* b, blas_int const* ldb, blas_int* info);

  void dgbsv_(blas_int const* n, blas_int const* kl, blas_int const* ku, blas_int const* nrhs, double* ab,
              blas_int const* ldab, blas_int* ipiv, double* b, blas_int const* ldb, blas_int* info);

  void sgbtrf_(blas_int const* m, blas_int const* n, blas_int const* kl, blas_int const* ku, float* ab,
               blas_int const* ldab, blas_int* ipiv, blas_int* info);

  void dgbtrf_(blas_int const* m, blas_int const* n, blas_int const* kl, blas_int const* ku, double* ab,
               blas_int const* ldab, blas_int* ipiv, blas_int* info);

  void sgbtrs_(char const* trans, blas_int const* n, blas_int const* kl, blas_int const* ku, blas_int const* nrhs,
               float* ab, blas_int const* ldab, blas_int const* ipiv, float* b, blas_int const* ldb, blas_int* info);

  void dgbtrs_(char const* trans, blas_int const* n, blas_int const* kl, blas_int const* ku, blas_int const* nrhs,
               double* ab, blas_int const* ldab, blas_int const* ipiv, double* b, blas_int const* ldb, blas_int* info);

  void sgbcon_(char const* norm, blas_int const* n, blas_int const* kl, blas_int const* ku, float* ab,
               blas_int const* ldab, blas_int const* ipiv, float const* anorm, float* rcond, float* work,
               blas_int* iwork, blas_int* info);

  void dgbcon_(char const* norm, blas_int const* n, blas_int const* kl, blas_int const* ku, double* ab,
               blas_int const* ldab, blas_int const* ipiv, double const* anorm, double* rcond, double* work,
               blas_int* iwork, blas_int* info);

  void sgbrfs_(char const* trans, blas_int const* n, blas_int const* kl, blas_int const* ku, blas_int const* nrhs,
               float* ab, blas_int const* ldab, float* afb, blas_int const* ldafb, blas_int const* ipiv, float* b,
               blas_int const* ldb, float* x, blas_int const* ldx, float* ferr, float* berr, float* work,
               blas_int* iwork, blas_int* info);

  void dgbrfs_(char const* trans, blas_int const* n, blas_int const* kl, blas_int const* ku, blas_int const* nrhs,
               double* ab, blas_int const* ldab, double* afb, blas_int const* ldafb, blas_int const* ipiv, double* b,
               blas_int const* ldb, double* x, blas_int const* ldx, double* ferr, double* berr, double* work,
               blas_int* iwork, blas_int* info);

  void sgbsvx_(char const* fact, char const* trans, blas_int const* n, blas_int const* kl, blas_int const* ku,
               blas_int const* nrhs, float* ab, blas_int const* ldab, float* afb, blas_int const* ldafb, blas_int* ipiv,
               char* equed, float* row_scale, float* column_scale, float* b, blas_int const* ldb, float* x,
               blas_int const* ldx, float* rcond, float* ferr, float* berr, float* work, blas_int* iwork,
               blas_int* info);

  void dgbsvx_(char const* fact, char const* trans, blas_int const* n, blas_int const* kl, blas_int const* ku,
               blas_int const* nrhs, double* ab, blas_int const* ldab, double* afb, blas_int const* ldafb,
               blas_int* ipiv, char* equed, double* row_scale, double* column_scale, double* b, blas_int const* ldb,
               double* x, blas_int const* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork,
               blas_int* info);

  void sgbequ_(blas_int const* m, blas_int const* n, blas_int const* kl, blas_int const* ku, float* ab,
               blas_int const* ldab, float* row_scale, float* column_scale, float* row_condition,
               float* column_condition, float* max_abs, blas_int* info);

  void dgbequ_(blas_int const* m, blas_int const* n, blas_int const* kl, blas_int const* ku, double* ab,
               blas_int const* ldab, double* row_scale, double* column_scale, double* row_condition,
               double* column_condition, double* max_abs, blas_int* info);

  void sgbequb_(blas_int const* m, blas_int const* n, blas_int const* kl, blas_int const* ku, float* ab,
                blas_int const* ldab, float* row_scale, float* column_scale, float* row_condition,
                float* column_condition, float* max_abs, blas_int* info);

  void dgbequb_(blas_int const* m, blas_int const* n, blas_int const* kl, blas_int const* ku, double* ab,
                blas_int const* ldab, double* row_scale, double* column_scale, double* row_condition,
                double* column_condition, double* max_abs, blas_int* info);
}
} // namespace detail

[[nodiscard]] inline blas_int gbsv(blas_int n, blas_int kl, blas_int ku, blas_int nrhs, float* ab, blas_int ldab,
                                   blas_int* ipiv, float* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgbsv_, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
  detail::sgbsv_(&n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gbsv(blas_int n, blas_int kl, blas_int ku, blas_int nrhs, double* ab, blas_int ldab,
                                   blas_int* ipiv, double* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgbsv_, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
  detail::dgbsv_(&n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gbtrf(blas_int m, blas_int n, blas_int kl, blas_int ku, float* ab, blas_int ldab,
                                    blas_int* ipiv)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgbtrf_, m, n, kl, ku, ab, ldab, ipiv);
  detail::sgbtrf_(&m, &n, &kl, &ku, ab, &ldab, ipiv, &info);
  return info;
}

[[nodiscard]] inline blas_int gbtrf(blas_int m, blas_int n, blas_int kl, blas_int ku, double* ab, blas_int ldab,
                                    blas_int* ipiv)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgbtrf_, m, n, kl, ku, ab, ldab, ipiv);
  detail::dgbtrf_(&m, &n, &kl, &ku, ab, &ldab, ipiv, &info);
  return info;
}

[[nodiscard]] inline blas_int gbtrs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, float* ab,
                                    blas_int ldab, blas_int const* ipiv, float* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgbtrs_, trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
  detail::sgbtrs_(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gbtrs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, double* ab,
                                    blas_int ldab, blas_int const* ipiv, double* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgbtrs_, trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
  detail::dgbtrs_(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gbcon(char norm, blas_int n, blas_int kl, blas_int ku, float* ab, blas_int ldab,
                                    blas_int const* ipiv, float anorm, float& rcond, float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgbcon_, norm, n, kl, ku, ab, ldab, ipiv, anorm, work, iwork);
  detail::sgbcon_(&norm, &n, &kl, &ku, ab, &ldab, ipiv, &anorm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gbcon(char norm, blas_int n, blas_int kl, blas_int ku, double* ab, blas_int ldab,
                                    blas_int const* ipiv, double anorm, double& rcond, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgbcon_, norm, n, kl, ku, ab, ldab, ipiv, anorm, work, iwork);
  detail::dgbcon_(&norm, &n, &kl, &ku, ab, &ldab, ipiv, &anorm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gbrfs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, float* ab,
                                    blas_int ldab, float* afb, blas_int ldafb, blas_int const* ipiv, float* b,
                                    blas_int ldb, float* x, blas_int ldx, float* forward_error, float* backward_error,
                                    float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgbrfs_, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  detail::sgbrfs_(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, ipiv, b, &ldb, x, &ldx, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gbrfs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, double* ab,
                                    blas_int ldab, double* afb, blas_int ldafb, blas_int const* ipiv, double* b,
                                    blas_int ldb, double* x, blas_int ldx, double* forward_error,
                                    double* backward_error, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgbrfs_, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  detail::dgbrfs_(&trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, ipiv, b, &ldb, x, &ldx, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gbsvx(char fact, char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs,
                                    float* ab, blas_int ldab, float* afb, blas_int ldafb, blas_int* ipiv, char& equed,
                                    float* row_scale, float* column_scale, float* b, blas_int ldb, float* x,
                                    blas_int ldx, float& rcond, float* forward_error, float* backward_error,
                                    float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgbsvx_, fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, row_scale,
                          column_scale, b, ldb, x, ldx, work, iwork);
  detail::sgbsvx_(&fact, &trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, ipiv, &equed, row_scale, column_scale, b,
                  &ldb, x, &ldx, &rcond, forward_error, backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gbsvx(char fact, char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs,
                                    double* ab, blas_int ldab, double* afb, blas_int ldafb, blas_int* ipiv, char& equed,
                                    double* row_scale, double* column_scale, double* b, blas_int ldb, double* x,
                                    blas_int ldx, double& rcond, double* forward_error, double* backward_error,
                                    double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgbsvx_, fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, row_scale,
                          column_scale, b, ldb, x, ldx, work, iwork);
  detail::dgbsvx_(&fact, &trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, ipiv, &equed, row_scale, column_scale, b,
                  &ldb, x, &ldx, &rcond, forward_error, backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gbequ(bool power_of_two, blas_int m, blas_int n, blas_int kl, blas_int ku, float* ab,
                                    blas_int ldab, float* row_scale, float* column_scale, float& row_condition,
                                    float& column_condition, float& max_abs)
{
  blas_int info = 0;
  if (power_of_two)
  {
    UNI20_EXTERNAL_API_CALL(LAPACK, sgbequb_, m, n, kl, ku, ab, ldab, row_scale, column_scale);
    detail::sgbequb_(&m, &n, &kl, &ku, ab, &ldab, row_scale, column_scale, &row_condition, &column_condition, &max_abs,
                     &info);
  }
  else
  {
    UNI20_EXTERNAL_API_CALL(LAPACK, sgbequ_, m, n, kl, ku, ab, ldab, row_scale, column_scale);
    detail::sgbequ_(&m, &n, &kl, &ku, ab, &ldab, row_scale, column_scale, &row_condition, &column_condition, &max_abs,
                    &info);
  }
  return info;
}

[[nodiscard]] inline blas_int gbequ(bool power_of_two, blas_int m, blas_int n, blas_int kl, blas_int ku, double* ab,
                                    blas_int ldab, double* row_scale, double* column_scale, double& row_condition,
                                    double& column_condition, double& max_abs)
{
  blas_int info = 0;
  if (power_of_two)
  {
    UNI20_EXTERNAL_API_CALL(LAPACK, dgbequb_, m, n, kl, ku, ab, ldab, row_scale, column_scale);
    detail::dgbequb_(&m, &n, &kl, &ku, ab, &ldab, row_scale, column_scale, &row_condition, &column_condition, &max_abs,
                     &info);
  }
  else
  {
    UNI20_EXTERNAL_API_CALL(LAPACK, dgbequ_, m, n, kl, ku, ab, ldab, row_scale, column_scale);
    detail::dgbequ_(&m, &n, &kl, &ku, ab, &ldab, row_scale, column_scale, &row_condition, &column_condition, &max_abs,
                    &info);
  }
  return info;
}

} // namespace uni20::lapack::unchecked
