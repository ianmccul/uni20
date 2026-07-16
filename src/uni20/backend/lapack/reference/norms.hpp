#pragma once

/**
 * \defgroup backend_lapack_reference Reference LAPACK backend
 * \ingroup backend_lapack
 * \brief Header-only wrappers around the configured float/double LAPACK library.
 */

/**
 * \file norms.hpp
 * \ingroup backend_lapack_reference
 * \brief LAPACK norm wrappers for the reference float/double backend.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/config.hpp>

namespace uni20::lapack::unchecked
{

namespace detail
{
extern "C"
{
  float slange_(char const* norm, blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* work);

  double dlange_(char const* norm, blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* work);

  float clange_(char const* norm, blas_int const* m, blas_int const* n, uni20::complex<float>* a, blas_int const* lda,
                float* work);

  double zlange_(char const* norm, blas_int const* m, blas_int const* n, uni20::complex<double>* a, blas_int const* lda,
                 double* work);

  float slansy_(char const* norm, char const* uplo, blas_int const* n, float* a, blas_int const* lda, float* work);

  double dlansy_(char const* norm, char const* uplo, blas_int const* n, double* a, blas_int const* lda, double* work);

  float clanhe_(char const* norm, char const* uplo, blas_int const* n, uni20::complex<float>* a, blas_int const* lda,
                float* work);

  double zlanhe_(char const* norm, char const* uplo, blas_int const* n, uni20::complex<double>* a, blas_int const* lda,
                 double* work);

  float slantr_(char const* norm, char const* uplo, char const* diag, blas_int const* m, blas_int const* n, float* a,
                blas_int const* lda, float* work);

  double dlantr_(char const* norm, char const* uplo, char const* diag, blas_int const* m, blas_int const* n, double* a,
                 blas_int const* lda, double* work);

  float clantr_(char const* norm, char const* uplo, char const* diag, blas_int const* m, blas_int const* n,
                uni20::complex<float>* a, blas_int const* lda, float* work);

  double zlantr_(char const* norm, char const* uplo, char const* diag, blas_int const* m, blas_int const* n,
                 uni20::complex<double>* a, blas_int const* lda, double* work);

  float slangb_(char const* norm, blas_int const* n, blas_int const* kl, blas_int const* ku, float* ab,
                blas_int const* ldab, float* work);

  double dlangb_(char const* norm, blas_int const* n, blas_int const* kl, blas_int const* ku, double* ab,
                 blas_int const* ldab, double* work);
}
} // namespace detail

[[nodiscard]] inline float lange(char norm, blas_int m, blas_int n, float* a, blas_int lda, float* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, slange_, norm, m, n, a, lda, work);
  return detail::slange_(&norm, &m, &n, a, &lda, work);
}

[[nodiscard]] inline double lange(char norm, blas_int m, blas_int n, double* a, blas_int lda, double* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, dlange_, norm, m, n, a, lda, work);
  return detail::dlange_(&norm, &m, &n, a, &lda, work);
}

[[nodiscard]] inline float lange(char norm, blas_int m, blas_int n, uni20::complex<float>* a, blas_int lda, float* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, clange_, norm, m, n, a, lda, work);
  return detail::clange_(&norm, &m, &n, a, &lda, work);
}

[[nodiscard]] inline double lange(char norm, blas_int m, blas_int n, uni20::complex<double>* a, blas_int lda,
                                  double* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, zlange_, norm, m, n, a, lda, work);
  return detail::zlange_(&norm, &m, &n, a, &lda, work);
}

[[nodiscard]] inline float lansy(char norm, char uplo, blas_int n, float* a, blas_int lda, float* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, slansy_, norm, uplo, n, a, lda, work);
  return detail::slansy_(&norm, &uplo, &n, a, &lda, work);
}

[[nodiscard]] inline double lansy(char norm, char uplo, blas_int n, double* a, blas_int lda, double* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, dlansy_, norm, uplo, n, a, lda, work);
  return detail::dlansy_(&norm, &uplo, &n, a, &lda, work);
}

[[nodiscard]] inline float lanhe(char norm, char uplo, blas_int n, uni20::complex<float>* a, blas_int lda, float* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, clanhe_, norm, uplo, n, a, lda, work);
  return detail::clanhe_(&norm, &uplo, &n, a, &lda, work);
}

[[nodiscard]] inline double lanhe(char norm, char uplo, blas_int n, uni20::complex<double>* a, blas_int lda,
                                  double* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, zlanhe_, norm, uplo, n, a, lda, work);
  return detail::zlanhe_(&norm, &uplo, &n, a, &lda, work);
}

[[nodiscard]] inline float lantr(char norm, char uplo, char diag, blas_int m, blas_int n, float* a, blas_int lda,
                                 float* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, slantr_, norm, uplo, diag, m, n, a, lda, work);
  return detail::slantr_(&norm, &uplo, &diag, &m, &n, a, &lda, work);
}

[[nodiscard]] inline double lantr(char norm, char uplo, char diag, blas_int m, blas_int n, double* a, blas_int lda,
                                  double* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, dlantr_, norm, uplo, diag, m, n, a, lda, work);
  return detail::dlantr_(&norm, &uplo, &diag, &m, &n, a, &lda, work);
}

[[nodiscard]] inline float lantr(char norm, char uplo, char diag, blas_int m, blas_int n, uni20::complex<float>* a,
                                 blas_int lda, float* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, clantr_, norm, uplo, diag, m, n, a, lda, work);
  return detail::clantr_(&norm, &uplo, &diag, &m, &n, a, &lda, work);
}

[[nodiscard]] inline double lantr(char norm, char uplo, char diag, blas_int m, blas_int n, uni20::complex<double>* a,
                                  blas_int lda, double* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, zlantr_, norm, uplo, diag, m, n, a, lda, work);
  return detail::zlantr_(&norm, &uplo, &diag, &m, &n, a, &lda, work);
}

[[nodiscard]] inline float langb(char norm, blas_int n, blas_int kl, blas_int ku, float* ab, blas_int ldab, float* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, slangb_, norm, n, kl, ku, ab, ldab, work);
  return detail::slangb_(&norm, &n, &kl, &ku, ab, &ldab, work);
}

[[nodiscard]] inline double langb(char norm, blas_int n, blas_int kl, blas_int ku, double* ab, blas_int ldab,
                                  double* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, dlangb_, norm, n, kl, ku, ab, ldab, work);
  return detail::dlangb_(&norm, &n, &kl, &ku, ab, &ldab, work);
}

} // namespace uni20::lapack::unchecked
