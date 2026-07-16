#pragma once

/**
 * \file general.hpp
 * \ingroup backend_lapack_reference
 * \brief Reference LAPACK wrappers for real and complex dense general linear systems.
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

  void cgesv_(blas_int const* n, blas_int const* nrhs, uni20::complex<float>* a, blas_int const* lda, blas_int* ipiv,
              uni20::complex<float>* b, blas_int const* ldb, blas_int* info);

  void zgesv_(blas_int const* n, blas_int const* nrhs, uni20::complex<double>* a, blas_int const* lda, blas_int* ipiv,
              uni20::complex<double>* b, blas_int const* ldb, blas_int* info);

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

  void cgetrf_(blas_int const* m, blas_int const* n, uni20::complex<float>* a, blas_int const* lda, blas_int* ipiv,
               blas_int* info);

  void zgetrf_(blas_int const* m, blas_int const* n, uni20::complex<double>* a, blas_int const* lda, blas_int* ipiv,
               blas_int* info);

  void sgetrs_(char const* trans, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda,
               blas_int const* ipiv, float* b, blas_int const* ldb, blas_int* info);

  void dgetrs_(char const* trans, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda,
               blas_int const* ipiv, double* b, blas_int const* ldb, blas_int* info);

  void cgetrs_(char const* trans, blas_int const* n, blas_int const* nrhs, uni20::complex<float>* a,
               blas_int const* lda, blas_int const* ipiv, uni20::complex<float>* b, blas_int const* ldb,
               blas_int* info);

  void zgetrs_(char const* trans, blas_int const* n, blas_int const* nrhs, uni20::complex<double>* a,
               blas_int const* lda, blas_int const* ipiv, uni20::complex<double>* b, blas_int const* ldb,
               blas_int* info);

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

  void cgetri_(blas_int const* n, uni20::complex<float>* a, blas_int const* lda, blas_int* ipiv,
               uni20::complex<float>* work, blas_int const* lwork, blas_int* info);

  void zgetri_(blas_int const* n, uni20::complex<double>* a, blas_int const* lda, blas_int* ipiv,
               uni20::complex<double>* work, blas_int const* lwork, blas_int* info);

  void sgecon_(char const* norm, blas_int const* n, float const* a, blas_int const* lda, float const* anorm,
               float* rcond, float* work, blas_int* iwork, blas_int* info);

  void dgecon_(char const* norm, blas_int const* n, double const* a, blas_int const* lda, double const* anorm,
               double* rcond, double* work, blas_int* iwork, blas_int* info);

  void cgecon_(char const* norm, blas_int const* n, uni20::complex<float> const* a, blas_int const* lda,
               float const* anorm, float* rcond, uni20::complex<float>* work, float* rwork, blas_int* info);

  void zgecon_(char const* norm, blas_int const* n, uni20::complex<double> const* a, blas_int const* lda,
               double const* anorm, double* rcond, uni20::complex<double>* work, double* rwork, blas_int* info);

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

  void sgeqrf_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dgeqrf_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sgelqf_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dgelqf_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sgeqlf_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dgeqlf_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sgerqf_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dgerqf_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sorgqr_(blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dorgqr_(blas_int const* m, blas_int const* n, blas_int const* k, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sorglq_(blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dorglq_(blas_int const* m, blas_int const* n, blas_int const* k, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sorgql_(blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dorgql_(blas_int const* m, blas_int const* n, blas_int const* k, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sorgrq_(blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dorgrq_(blas_int const* m, blas_int const* n, blas_int const* k, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sormqr_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, float* a,
               blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormqr_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* c, blas_int const* ldc, double* work, blas_int const* lwork,
               blas_int* info);

  void sormlq_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, float* a,
               blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormlq_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* c, blas_int const* ldc, double* work, blas_int const* lwork,
               blas_int* info);

  void sormql_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, float* a,
               blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormql_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* c, blas_int const* ldc, double* work, blas_int const* lwork,
               blas_int* info);

  void sormrq_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, float* a,
               blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormrq_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* c, blas_int const* ldc, double* work, blas_int const* lwork,
               blas_int* info);

  void sgebrd_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* d, float* e, float* tauq,
               float* taup, float* work, blas_int const* lwork, blas_int* info);

  void dgebrd_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* d, double* e, double* tauq,
               double* taup, double* work, blas_int const* lwork, blas_int* info);

  void sgehrd_(blas_int const* n, blas_int const* ilo, blas_int const* ihi, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dgehrd_(blas_int const* n, blas_int const* ilo, blas_int const* ihi, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sgeqp3_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, blas_int* jpvt, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dgeqp3_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, blas_int* jpvt, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void ssytrd_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, float* d, float* e, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dsytrd_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, double* d, double* e, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void ssyev_(char const* jobz, char const* uplo, blas_int const* n, float* a, blas_int const* lda, float* w,
              float* work, blas_int const* lwork, blas_int* info);

  void dsyev_(char const* jobz, char const* uplo, blas_int const* n, double* a, blas_int const* lda, double* w,
              double* work, blas_int const* lwork, blas_int* info);

  void ssyevd_(char const* jobz, char const* uplo, blas_int const* n, float* a, blas_int const* lda, float* w,
               float* work, blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dsyevd_(char const* jobz, char const* uplo, blas_int const* n, double* a, blas_int const* lda, double* w,
               double* work, blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void ssyevr_(char const* jobz, char const* range, char const* uplo, blas_int const* n, float* a, blas_int const* lda,
               float const* vl, float const* vu, blas_int const* il, blas_int const* iu, float const* abstol,
               blas_int* m, float* w, float* z, blas_int const* ldz, blas_int* isuppz, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dsyevr_(char const* jobz, char const* range, char const* uplo, blas_int const* n, double* a, blas_int const* lda,
               double const* vl, double const* vu, blas_int const* il, blas_int const* iu, double const* abstol,
               blas_int* m, double* w, double* z, blas_int const* ldz, blas_int* isuppz, double* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void ssygv_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, float* a,
              blas_int const* lda, float* b, blas_int const* ldb, float* w, float* work, blas_int const* lwork,
              blas_int* info);

  void dsygv_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, double* a,
              blas_int const* lda, double* b, blas_int const* ldb, double* w, double* work, blas_int const* lwork,
              blas_int* info);

  void ssygvd_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, float* a,
               blas_int const* lda, float* b, blas_int const* ldb, float* w, float* work, blas_int const* lwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dsygvd_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, double* a,
               blas_int const* lda, double* b, blas_int const* ldb, double* w, double* work, blas_int const* lwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void ssygvx_(blas_int const* itype, char const* jobz, char const* range, char const* uplo, blas_int const* n,
               float* a, blas_int const* lda, float* b, blas_int const* ldb, float const* vl, float const* vu,
               blas_int const* il, blas_int const* iu, float const* abstol, blas_int* m, float* w, float* z,
               blas_int const* ldz, float* work, blas_int const* lwork, blas_int* iwork, blas_int* ifail,
               blas_int* info);

  void dsygvx_(blas_int const* itype, char const* jobz, char const* range, char const* uplo, blas_int const* n,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double const* vl, double const* vu,
               blas_int const* il, blas_int const* iu, double const* abstol, blas_int* m, double* w, double* z,
               blas_int const* ldz, double* work, blas_int const* lwork, blas_int* iwork, blas_int* ifail,
               blas_int* info);

  void cheev_(char const* jobz, char const* uplo, blas_int const* n, uni20::complex<float>* a, blas_int const* lda,
              float* w, uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int* info);

  void zheev_(char const* jobz, char const* uplo, blas_int const* n, uni20::complex<double>* a, blas_int const* lda,
              double* w, uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int* info);

  void cheevd_(char const* jobz, char const* uplo, blas_int const* n, uni20::complex<float>* a, blas_int const* lda,
               float* w, uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int const* lrwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void zheevd_(char const* jobz, char const* uplo, blas_int const* n, uni20::complex<double>* a, blas_int const* lda,
               double* w, uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int const* lrwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void cheevr_(char const* jobz, char const* range, char const* uplo, blas_int const* n, uni20::complex<float>* a,
               blas_int const* lda, float const* vl, float const* vu, blas_int const* il, blas_int const* iu,
               float const* abstol, blas_int* m, float* w, uni20::complex<float>* z, blas_int const* ldz,
               blas_int* isuppz, uni20::complex<float>* work, blas_int const* lwork, float* rwork,
               blas_int const* lrwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void zheevr_(char const* jobz, char const* range, char const* uplo, blas_int const* n, uni20::complex<double>* a,
               blas_int const* lda, double const* vl, double const* vu, blas_int const* il, blas_int const* iu,
               double const* abstol, blas_int* m, double* w, uni20::complex<double>* z, blas_int const* ldz,
               blas_int* isuppz, uni20::complex<double>* work, blas_int const* lwork, double* rwork,
               blas_int const* lrwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void chegv_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, uni20::complex<float>* a,
              blas_int const* lda, uni20::complex<float>* b, blas_int const* ldb, float* w, uni20::complex<float>* work,
              blas_int const* lwork, float* rwork, blas_int* info);

  void zhegv_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, uni20::complex<double>* a,
              blas_int const* lda, uni20::complex<double>* b, blas_int const* ldb, double* w,
              uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int* info);

  void chegvd_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, uni20::complex<float>* a,
               blas_int const* lda, uni20::complex<float>* b, blas_int const* ldb, float* w,
               uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int const* lrwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void zhegvd_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, uni20::complex<double>* a,
               blas_int const* lda, uni20::complex<double>* b, blas_int const* ldb, double* w,
               uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int const* lrwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void chegvx_(blas_int const* itype, char const* jobz, char const* range, char const* uplo, blas_int const* n,
               uni20::complex<float>* a, blas_int const* lda, uni20::complex<float>* b, blas_int const* ldb,
               float const* vl, float const* vu, blas_int const* il, blas_int const* iu, float const* abstol,
               blas_int* m, float* w, uni20::complex<float>* z, blas_int const* ldz, uni20::complex<float>* work,
               blas_int const* lwork, float* rwork, blas_int* iwork, blas_int* ifail, blas_int* info);

  void zhegvx_(blas_int const* itype, char const* jobz, char const* range, char const* uplo, blas_int const* n,
               uni20::complex<double>* a, blas_int const* lda, uni20::complex<double>* b, blas_int const* ldb,
               double const* vl, double const* vu, blas_int const* il, blas_int const* iu, double const* abstol,
               blas_int* m, double* w, uni20::complex<double>* z, blas_int const* ldz, uni20::complex<double>* work,
               blas_int const* lwork, double* rwork, blas_int* iwork, blas_int* ifail, blas_int* info);

  void sgeev_(char const* jobvl, char const* jobvr, blas_int const* n, float* a, blas_int const* lda, float* wr,
              float* wi, float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr, float* work,
              blas_int const* lwork, blas_int* info);

  void dgeev_(char const* jobvl, char const* jobvr, blas_int const* n, double* a, blas_int const* lda, double* wr,
              double* wi, double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr, double* work,
              blas_int const* lwork, blas_int* info);

  void sgeevx_(char const* balanc, char const* jobvl, char const* jobvr, char const* sense, blas_int const* n, float* a,
               blas_int const* lda, float* wr, float* wi, float* vl, blas_int const* ldvl, float* vr,
               blas_int const* ldvr, blas_int* ilo, blas_int* ihi, float* scale, float* abnrm, float* rconde,
               float* rcondv, float* work, blas_int const* lwork, blas_int* iwork, blas_int* info);

  void dgeevx_(char const* balanc, char const* jobvl, char const* jobvr, char const* sense, blas_int const* n,
               double* a, blas_int const* lda, double* wr, double* wi, double* vl, blas_int const* ldvl, double* vr,
               blas_int const* ldvr, blas_int* ilo, blas_int* ihi, double* scale, double* abnrm, double* rconde,
               double* rcondv, double* work, blas_int const* lwork, blas_int* iwork, blas_int* info);

  void sgebal_(char const* job, blas_int const* n, float* a, blas_int const* lda, blas_int* ilo, blas_int* ihi,
               float* scale, blas_int* info);

  void dgebal_(char const* job, blas_int const* n, double* a, blas_int const* lda, blas_int* ilo, blas_int* ihi,
               double* scale, blas_int* info);

  void sgebak_(char const* job, char const* side, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               float* scale, blas_int const* m, float* v, blas_int const* ldv, blas_int* info);

  void dgebak_(char const* job, char const* side, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               double* scale, blas_int const* m, double* v, blas_int const* ldv, blas_int* info);

  void cgeev_(char const* jobvl, char const* jobvr, blas_int const* n, uni20::complex<float>* a, blas_int const* lda,
              uni20::complex<float>* w, uni20::complex<float>* vl, blas_int const* ldvl, uni20::complex<float>* vr,
              blas_int const* ldvr, uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int* info);

  void zgeev_(char const* jobvl, char const* jobvr, blas_int const* n, uni20::complex<double>* a, blas_int const* lda,
              uni20::complex<double>* w, uni20::complex<double>* vl, blas_int const* ldvl, uni20::complex<double>* vr,
              blas_int const* ldvr, uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int* info);

  void sgees_(char const* jobvs, char const* sort, void* select, blas_int const* n, float* a, blas_int const* lda,
              blas_int* sdim, float* wr, float* wi, float* vs, blas_int const* ldvs, float* work, blas_int const* lwork,
              blas_int* bwork, blas_int* info);

  void dgees_(char const* jobvs, char const* sort, void* select, blas_int const* n, double* a, blas_int const* lda,
              blas_int* sdim, double* wr, double* wi, double* vs, blas_int const* ldvs, double* work,
              blas_int const* lwork, blas_int* bwork, blas_int* info);

  void cgees_(char const* jobvs, char const* sort, void* select, blas_int const* n, uni20::complex<float>* a,
              blas_int const* lda, blas_int* sdim, uni20::complex<float>* w, uni20::complex<float>* vs,
              blas_int const* ldvs, uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int* bwork,
              blas_int* info);

  void zgees_(char const* jobvs, char const* sort, void* select, blas_int const* n, uni20::complex<double>* a,
              blas_int const* lda, blas_int* sdim, uni20::complex<double>* w, uni20::complex<double>* vs,
              blas_int const* ldvs, uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int* bwork,
              blas_int* info);

  void shseqr_(char const* job, char const* compz, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               float* h, blas_int const* ldh, float* wr, float* wi, float* z, blas_int const* ldz, float* work,
               blas_int const* lwork, blas_int* info);

  void dhseqr_(char const* job, char const* compz, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               double* h, blas_int const* ldh, double* wr, double* wi, double* z, blas_int const* ldz, double* work,
               blas_int const* lwork, blas_int* info);

  void strexc_(char const* compq, blas_int const* n, float* t, blas_int const* ldt, float* q, blas_int const* ldq,
               blas_int* ifst, blas_int* ilst, float* work, blas_int* info);

  void dtrexc_(char const* compq, blas_int const* n, double* t, blas_int const* ldt, double* q, blas_int const* ldq,
               blas_int* ifst, blas_int* ilst, double* work, blas_int* info);

  void ctrexc_(char const* compq, blas_int const* n, uni20::complex<float>* t, blas_int const* ldt,
               uni20::complex<float>* q, blas_int const* ldq, blas_int* ifst, blas_int* ilst, blas_int* info);

  void ztrexc_(char const* compq, blas_int const* n, uni20::complex<double>* t, blas_int const* ldt,
               uni20::complex<double>* q, blas_int const* ldq, blas_int* ifst, blas_int* ilst, blas_int* info);

  void strsen_(char const* job, char const* compq, blas_int* select, blas_int const* n, float* t, blas_int const* ldt,
               float* q, blas_int const* ldq, float* wr, float* wi, blas_int* m, float* s, float* sep, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dtrsen_(char const* job, char const* compq, blas_int* select, blas_int const* n, double* t, blas_int const* ldt,
               double* q, blas_int const* ldq, double* wr, double* wi, blas_int* m, double* s, double* sep,
               double* work, blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void strevc_(char const* side, char const* howmny, blas_int* select, blas_int const* n, float* t, blas_int const* ldt,
               float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr, blas_int const* mm, blas_int* m,
               float* work, blas_int* info);

  void dtrevc_(char const* side, char const* howmny, blas_int* select, blas_int const* n, double* t,
               blas_int const* ldt, double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr,
               blas_int const* mm, blas_int* m, double* work, blas_int* info);

  void strsna_(char const* job, char const* howmny, blas_int* select, blas_int const* n, float* t, blas_int const* ldt,
               float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr, float* s, float* sep,
               blas_int const* mm, blas_int* m, float* work, blas_int const* ldwork, blas_int* iwork, blas_int* info);

  void dtrsna_(char const* job, char const* howmny, blas_int* select, blas_int const* n, double* t, blas_int const* ldt,
               double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr, double* s, double* sep,
               blas_int const* mm, blas_int* m, double* work, blas_int const* ldwork, blas_int* iwork, blas_int* info);

  void sggev_(char const* jobvl, char const* jobvr, blas_int const* n, float* a, blas_int const* lda, float* b,
              blas_int const* ldb, float* alphar, float* alphai, float* beta, float* vl, blas_int const* ldvl,
              float* vr, blas_int const* ldvr, float* work, blas_int const* lwork, blas_int* info);

  void dggev_(char const* jobvl, char const* jobvr, blas_int const* n, double* a, blas_int const* lda, double* b,
              blas_int const* ldb, double* alphar, double* alphai, double* beta, double* vl, blas_int const* ldvl,
              double* vr, blas_int const* ldvr, double* work, blas_int const* lwork, blas_int* info);

  void sggevx_(char const* balanc, char const* jobvl, char const* jobvr, char const* sense, blas_int const* n, float* a,
               blas_int const* lda, float* b, blas_int const* ldb, float* alphar, float* alphai, float* beta, float* vl,
               blas_int const* ldvl, float* vr, blas_int const* ldvr, blas_int* ilo, blas_int* ihi, float* lscale,
               float* rscale, float* abnrm, float* bbnrm, float* rconde, float* rcondv, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int* bwork, blas_int* info);

  void dggevx_(char const* balanc, char const* jobvl, char const* jobvr, char const* sense, blas_int const* n,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* alphar, double* alphai,
               double* beta, double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr, blas_int* ilo,
               blas_int* ihi, double* lscale, double* rscale, double* abnrm, double* bbnrm, double* rconde,
               double* rcondv, double* work, blas_int const* lwork, blas_int* iwork, blas_int* bwork, blas_int* info);

  void sggbal_(char const* job, blas_int const* n, float* a, blas_int const* lda, float* b, blas_int const* ldb,
               blas_int* ilo, blas_int* ihi, float* lscale, float* rscale, float* work, blas_int* info);

  void dggbal_(char const* job, blas_int const* n, double* a, blas_int const* lda, double* b, blas_int const* ldb,
               blas_int* ilo, blas_int* ihi, double* lscale, double* rscale, double* work, blas_int* info);

  void sggbak_(char const* job, char const* side, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               float* lscale, float* rscale, blas_int const* m, float* v, blas_int const* ldv, blas_int* info);

  void dggbak_(char const* job, char const* side, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               double* lscale, double* rscale, blas_int const* m, double* v, blas_int const* ldv, blas_int* info);

  void sgges_(char const* jobvsl, char const* jobvsr, char const* sort, void* select, blas_int const* n, float* a,
              blas_int const* lda, float* b, blas_int const* ldb, blas_int* selected_dimension, float* alphar,
              float* alphai, float* beta, float* vsl, blas_int const* ldvsl, float* vsr, blas_int const* ldvsr,
              float* work, blas_int const* lwork, blas_int* bwork, blas_int* info);

  void dgges_(char const* jobvsl, char const* jobvsr, char const* sort, void* select, blas_int const* n, double* a,
              blas_int const* lda, double* b, blas_int const* ldb, blas_int* selected_dimension, double* alphar,
              double* alphai, double* beta, double* vsl, blas_int const* ldvsl, double* vsr, blas_int const* ldvsr,
              double* work, blas_int const* lwork, blas_int* bwork, blas_int* info);

  void sgghrd_(char const* compq, char const* compz, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               float* a, blas_int const* lda, float* b, blas_int const* ldb, float* q, blas_int const* ldq, float* z,
               blas_int const* ldz, blas_int* info);

  void dgghrd_(char const* compq, char const* compz, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* q, blas_int const* ldq,
               double* z, blas_int const* ldz, blas_int* info);

  void shgeqz_(char const* job, char const* compq, char const* compz, blas_int const* n, blas_int const* ilo,
               blas_int const* ihi, float* h, blas_int const* ldh, float* t, blas_int const* ldt, float* alphar,
               float* alphai, float* beta, float* q, blas_int const* ldq, float* z, blas_int const* ldz, float* work,
               blas_int const* lwork, blas_int* info);

  void dhgeqz_(char const* job, char const* compq, char const* compz, blas_int const* n, blas_int const* ilo,
               blas_int const* ihi, double* h, blas_int const* ldh, double* t, blas_int const* ldt, double* alphar,
               double* alphai, double* beta, double* q, blas_int const* ldq, double* z, blas_int const* ldz,
               double* work, blas_int const* lwork, blas_int* info);

  void stgexc_(blas_int const* wantq, blas_int const* wantz, blas_int const* n, float* a, blas_int const* lda, float* b,
               blas_int const* ldb, float* q, blas_int const* ldq, float* z, blas_int const* ldz, blas_int* first,
               blas_int* last, float* work, blas_int const* lwork, blas_int* info);

  void dtgexc_(blas_int const* wantq, blas_int const* wantz, blas_int const* n, double* a, blas_int const* lda,
               double* b, blas_int const* ldb, double* q, blas_int const* ldq, double* z, blas_int const* ldz,
               blas_int* first, blas_int* last, double* work, blas_int const* lwork, blas_int* info);

  void stgsen_(blas_int const* ijob, blas_int const* wantq, blas_int const* wantz, blas_int* select, blas_int const* n,
               float* a, blas_int const* lda, float* b, blas_int const* ldb, float* alphar, float* alphai, float* beta,
               float* q, blas_int const* ldq, float* z, blas_int const* ldz, blas_int* selected_dimension, float* pl,
               float* pr, float* dif, float* work, blas_int const* lwork, blas_int* iwork, blas_int const* liwork,
               blas_int* info);

  void dtgsen_(blas_int const* ijob, blas_int const* wantq, blas_int const* wantz, blas_int* select, blas_int const* n,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* alphar, double* alphai,
               double* beta, double* q, blas_int const* ldq, double* z, blas_int const* ldz,
               blas_int* selected_dimension, double* pl, double* pr, double* dif, double* work, blas_int const* lwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void stgevc_(char const* side, char const* howmny, blas_int* select, blas_int const* n, float* s, blas_int const* lds,
               float* p, blas_int const* ldp, float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr,
               blas_int const* mm, blas_int* computed_vectors, float* work, blas_int* info);

  void dtgevc_(char const* side, char const* howmny, blas_int* select, blas_int const* n, double* s,
               blas_int const* lds, double* p, blas_int const* ldp, double* vl, blas_int const* ldvl, double* vr,
               blas_int const* ldvr, blas_int const* mm, blas_int* computed_vectors, double* work, blas_int* info);

  void stgsna_(char const* job, char const* howmny, blas_int* select, blas_int const* n, float* a, blas_int const* lda,
               float* b, blas_int const* ldb, float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr,
               float* s, float* dif, blas_int const* mm, blas_int* computed_estimates, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int* info);

  void dtgsna_(char const* job, char const* howmny, blas_int* select, blas_int const* n, double* a, blas_int const* lda,
               double* b, blas_int const* ldb, double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr,
               double* s, double* dif, blas_int const* mm, blas_int* computed_estimates, double* work,
               blas_int const* lwork, blas_int* iwork, blas_int* info);

  void sorgbr_(char const* vect, blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda,
               float* tau, float* work, blas_int const* lwork, blas_int* info);

  void dorgbr_(char const* vect, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* work, blas_int const* lwork, blas_int* info);

  void sorghr_(blas_int const* n, blas_int const* ilo, blas_int const* ihi, float* a, blas_int const* lda,
               float const* tau, float* work, blas_int const* lwork, blas_int* info);

  void dorghr_(blas_int const* n, blas_int const* ilo, blas_int const* ihi, double* a, blas_int const* lda,
               double const* tau, double* work, blas_int const* lwork, blas_int* info);

  void sorgtr_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, float const* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dorgtr_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, double const* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sormhr_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* ilo,
               blas_int const* ihi, float* a, blas_int const* lda, float const* tau, float* c, blas_int const* ldc,
               float* work, blas_int const* lwork, blas_int* info);

  void dormhr_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* ilo,
               blas_int const* ihi, double* a, blas_int const* lda, double const* tau, double* c, blas_int const* ldc,
               double* work, blas_int const* lwork, blas_int* info);

  void sormtr_(char const* side, char const* uplo, char const* trans, blas_int const* m, blas_int const* n, float* a,
               blas_int const* lda, float const* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormtr_(char const* side, char const* uplo, char const* trans, blas_int const* m, blas_int const* n, double* a,
               blas_int const* lda, double const* tau, double* c, blas_int const* ldc, double* work,
               blas_int const* lwork, blas_int* info);

  void sormbr_(char const* vect, char const* side, char const* trans, blas_int const* m, blas_int const* n,
               blas_int const* k, float* a, blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work,
               blas_int const* lwork, blas_int* info);

  void dormbr_(char const* vect, char const* side, char const* trans, blas_int const* m, blas_int const* n,
               blas_int const* k, double* a, blas_int const* lda, double* tau, double* c, blas_int const* ldc,
               double* work, blas_int const* lwork, blas_int* info);

  void sgesvd_(char const* jobu, char const* jobvt, blas_int const* m, blas_int const* n, float* a, blas_int const* lda,
               float* s, float* u, blas_int const* ldu, float* vt, blas_int const* ldvt, float* work,
               blas_int const* lwork, blas_int* info);

  void dgesvd_(char const* jobu, char const* jobvt, blas_int const* m, blas_int const* n, double* a,
               blas_int const* lda, double* s, double* u, blas_int const* ldu, double* vt, blas_int const* ldvt,
               double* work, blas_int const* lwork, blas_int* info);

  void cgesvd_(char const* jobu, char const* jobvt, blas_int const* m, blas_int const* n, uni20::complex<float>* a,
               blas_int const* lda, float* s, uni20::complex<float>* u, blas_int const* ldu, uni20::complex<float>* vt,
               blas_int const* ldvt, uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int* info);

  void zgesvd_(char const* jobu, char const* jobvt, blas_int const* m, blas_int const* n, uni20::complex<double>* a,
               blas_int const* lda, double* s, uni20::complex<double>* u, blas_int const* ldu,
               uni20::complex<double>* vt, blas_int const* ldvt, uni20::complex<double>* work, blas_int const* lwork,
               double* rwork, blas_int* info);

  void sgesdd_(char const* jobz, blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* s,
               float* u, blas_int const* ldu, float* vt, blas_int const* ldvt, float* work, blas_int const* lwork,
               blas_int* iwork, blas_int* info);

  void dgesdd_(char const* jobz, blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* s,
               double* u, blas_int const* ldu, double* vt, blas_int const* ldvt, double* work, blas_int const* lwork,
               blas_int* iwork, blas_int* info);

  void cgesdd_(char const* jobz, blas_int const* m, blas_int const* n, uni20::complex<float>* a, blas_int const* lda,
               float* s, uni20::complex<float>* u, blas_int const* ldu, uni20::complex<float>* vt, blas_int const* ldvt,
               uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int* iwork, blas_int* info);

  void zgesdd_(char const* jobz, blas_int const* m, blas_int const* n, uni20::complex<double>* a, blas_int const* lda,
               double* s, uni20::complex<double>* u, blas_int const* ldu, uni20::complex<double>* vt,
               blas_int const* ldvt, uni20::complex<double>* work, blas_int const* lwork, double* rwork,
               blas_int* iwork, blas_int* info);

  void sgesvdx_(char const* jobu, char const* jobvt, char const* range, blas_int const* m, blas_int const* n, float* a,
                blas_int const* lda, float const* vl, float const* vu, blas_int const* il, blas_int const* iu,
                blas_int* ns, float* s, float* u, blas_int const* ldu, float* vt, blas_int const* ldvt, float* work,
                blas_int const* lwork, blas_int* iwork, blas_int* info);

  void dgesvdx_(char const* jobu, char const* jobvt, char const* range, blas_int const* m, blas_int const* n, double* a,
                blas_int const* lda, double const* vl, double const* vu, blas_int const* il, blas_int const* iu,
                blas_int* ns, double* s, double* u, blas_int const* ldu, double* vt, blas_int const* ldvt, double* work,
                blas_int const* lwork, blas_int* iwork, blas_int* info);

  void cgesvdx_(char const* jobu, char const* jobvt, char const* range, blas_int const* m, blas_int const* n,
                uni20::complex<float>* a, blas_int const* lda, float const* vl, float const* vu, blas_int const* il,
                blas_int const* iu, blas_int* ns, float* s, uni20::complex<float>* u, blas_int const* ldu,
                uni20::complex<float>* vt, blas_int const* ldvt, uni20::complex<float>* work, blas_int const* lwork,
                float* rwork, blas_int* iwork, blas_int* info);

  void zgesvdx_(char const* jobu, char const* jobvt, char const* range, blas_int const* m, blas_int const* n,
                uni20::complex<double>* a, blas_int const* lda, double const* vl, double const* vu, blas_int const* il,
                blas_int const* iu, blas_int* ns, double* s, uni20::complex<double>* u, blas_int const* ldu,
                uni20::complex<double>* vt, blas_int const* ldvt, uni20::complex<double>* work, blas_int const* lwork,
                double* rwork, blas_int* iwork, blas_int* info);

  void sbdsqr_(char const* uplo, blas_int const* n, blas_int const* ncvt, blas_int const* nru, blas_int const* ncc,
               float* d, float* e, float* vt, blas_int const* ldvt, float* u, blas_int const* ldu, float* c,
               blas_int const* ldc, float* work, blas_int* info);

  void dbdsqr_(char const* uplo, blas_int const* n, blas_int const* ncvt, blas_int const* nru, blas_int const* ncc,
               double* d, double* e, double* vt, blas_int const* ldvt, double* u, blas_int const* ldu, double* c,
               blas_int const* ldc, double* work, blas_int* info);

  void sbdsdc_(char const* uplo, char const* compq, blas_int const* n, float* d, float* e, float* u,
               blas_int const* ldu, float* vt, blas_int const* ldvt, float* q, blas_int* iq, float* work,
               blas_int* iwork, blas_int* info);

  void dbdsdc_(char const* uplo, char const* compq, blas_int const* n, double* d, double* e, double* u,
               blas_int const* ldu, double* vt, blas_int const* ldvt, double* q, blas_int* iq, double* work,
               blas_int* iwork, blas_int* info);

  void sbdsvdx_(char const* uplo, char const* jobz, char const* range, blas_int const* n, float* d, float* e,
                float const* vl, float const* vu, blas_int const* il, blas_int const* iu, blas_int* ns, float* s,
                float* z, blas_int const* ldz, float* work, blas_int* iwork, blas_int* info);

  void dbdsvdx_(char const* uplo, char const* jobz, char const* range, blas_int const* n, double* d, double* e,
                double const* vl, double const* vu, blas_int const* il, blas_int const* iu, blas_int* ns, double* s,
                double* z, blas_int const* ldz, double* work, blas_int* iwork, blas_int* info);

  void spotrf_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int* info);

  void dpotrf_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int* info);

  void spstrf_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int* piv, blas_int* rank,
               float const* tol, float* work, blas_int* info);

  void dpstrf_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int* piv, blas_int* rank,
               double const* tol, double* work, blas_int* info);

  void spotri_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int* info);

  void dpotri_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int* info);

  void spotrs_(char const* uplo, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* b,
               blas_int const* ldb, blas_int* info);

  void dpotrs_(char const* uplo, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* b,
               blas_int const* ldb, blas_int* info);

  void sporfs_(char const* uplo, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* af,
               blas_int const* ldaf, float* b, blas_int const* ldb, float* x, blas_int const* ldx, float* ferr,
               float* berr, float* work, blas_int* iwork, blas_int* info);

  void dporfs_(char const* uplo, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* af,
               blas_int const* ldaf, double* b, blas_int const* ldb, double* x, blas_int const* ldx, double* ferr,
               double* berr, double* work, blas_int* iwork, blas_int* info);

  void sposvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* nrhs, float* a,
               blas_int const* lda, float* af, blas_int const* ldaf, char* equed, float* s, float* b,
               blas_int const* ldb, float* x, blas_int const* ldx, float* rcond, float* ferr, float* berr, float* work,
               blas_int* iwork, blas_int* info);

  void dposvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* nrhs, double* a,
               blas_int const* lda, double* af, blas_int const* ldaf, char* equed, double* s, double* b,
               blas_int const* ldb, double* x, blas_int const* ldx, double* rcond, double* ferr, double* berr,
               double* work, blas_int* iwork, blas_int* info);

  void spoequ_(blas_int const* n, float* a, blas_int const* lda, float* s, float* scond, float* amax, blas_int* info);

  void dpoequ_(blas_int const* n, double* a, blas_int const* lda, double* s, double* scond, double* amax,
               blas_int* info);

  void spocon_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, float const* anorm, float* rcond,
               float* work, blas_int* iwork, blas_int* info);

  void dpocon_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, double const* anorm, double* rcond,
               double* work, blas_int* iwork, blas_int* info);

  void ssytrf_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int* ipiv, float* work,
               blas_int const* lwork, blas_int* info);

  void dsytrf_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int* ipiv, double* work,
               blas_int const* lwork, blas_int* info);

  void ssytrs_(char const* uplo, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda,
               blas_int const* ipiv, float* b, blas_int const* ldb, blas_int* info);

  void dsytrs_(char const* uplo, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda,
               blas_int const* ipiv, double* b, blas_int const* ldb, blas_int* info);

  void ssytri_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int const* ipiv, float* work,
               blas_int* info);

  void dsytri_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int const* ipiv, double* work,
               blas_int* info);

  void ssyrfs_(char const* uplo, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* af,
               blas_int const* ldaf, blas_int* ipiv, float* b, blas_int const* ldb, float* x, blas_int const* ldx,
               float* ferr, float* berr, float* work, blas_int* iwork, blas_int* info);

  void dsyrfs_(char const* uplo, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* af,
               blas_int const* ldaf, blas_int* ipiv, double* b, blas_int const* ldb, double* x, blas_int const* ldx,
               double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info);

  void ssysvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* nrhs, float* a,
               blas_int const* lda, float* af, blas_int const* ldaf, blas_int* ipiv, float* b, blas_int const* ldb,
               float* x, blas_int const* ldx, float* rcond, float* ferr, float* berr, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int* info);

  void dsysvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* nrhs, double* a,
               blas_int const* lda, double* af, blas_int const* ldaf, blas_int* ipiv, double* b, blas_int const* ldb,
               double* x, blas_int const* ldx, double* rcond, double* ferr, double* berr, double* work,
               blas_int const* lwork, blas_int* iwork, blas_int* info);

  void ssycon_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int const* ipiv,
               float const* anorm, float* rcond, float* work, blas_int* iwork, blas_int* info);

  void dsycon_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int const* ipiv,
               double const* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info);

  void strtrs_(char const* uplo, char const* trans, char const* diag, blas_int const* n, blas_int const* nrhs, float* a,
               blas_int const* lda, float* b, blas_int const* ldb, blas_int* info);

  void dtrtrs_(char const* uplo, char const* trans, char const* diag, blas_int const* n, blas_int const* nrhs,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, blas_int* info);

  void strrfs_(char const* uplo, char const* trans, char const* diag, blas_int const* n, blas_int const* nrhs, float* a,
               blas_int const* lda, float* b, blas_int const* ldb, float* x, blas_int const* ldx, float* ferr,
               float* berr, float* work, blas_int* iwork, blas_int* info);

  void dtrrfs_(char const* uplo, char const* trans, char const* diag, blas_int const* n, blas_int const* nrhs,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* x, blas_int const* ldx,
               double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info);

  void strtri_(char const* uplo, char const* diag, blas_int const* n, float* a, blas_int const* lda, blas_int* info);

  void dtrtri_(char const* uplo, char const* diag, blas_int const* n, double* a, blas_int const* lda, blas_int* info);

  void strcon_(char const* norm, char const* uplo, char const* diag, blas_int const* n, float* a, blas_int const* lda,
               float* rcond, float* work, blas_int* iwork, blas_int* info);

  void dtrcon_(char const* norm, char const* uplo, char const* diag, blas_int const* n, double* a, blas_int const* lda,
               double* rcond, double* work, blas_int* iwork, blas_int* info);

  void strsyl_(char const* trana, char const* tranb, blas_int const* isgn, blas_int const* m, blas_int const* n,
               float* a, blas_int const* lda, float* b, blas_int const* ldb, float* c, blas_int const* ldc,
               float* scale, blas_int* info);

  void dtrsyl_(char const* trana, char const* tranb, blas_int const* isgn, blas_int const* m, blas_int const* n,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* c, blas_int const* ldc,
               double* scale, blas_int* info);
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

[[nodiscard]] inline blas_int gesv(blas_int n, blas_int nrhs, uni20::complex<float>* a, blas_int lda, blas_int* ipiv,
                                   uni20::complex<float>* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgesv_, n, nrhs, a, lda, ipiv, b, ldb);
  detail::cgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int gesv(blas_int n, blas_int nrhs, uni20::complex<double>* a, blas_int lda, blas_int* ipiv,
                                   uni20::complex<double>* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgesv_, n, nrhs, a, lda, ipiv, b, ldb);
  detail::zgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
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

[[nodiscard]] inline blas_int getrf(blas_int m, blas_int n, uni20::complex<float>* a, blas_int lda, blas_int* ipiv)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgetrf_, m, n, a, lda, ipiv);
  detail::cgetrf_(&m, &n, a, &lda, ipiv, &info);
  return info;
}

[[nodiscard]] inline blas_int getrf(blas_int m, blas_int n, uni20::complex<double>* a, blas_int lda, blas_int* ipiv)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgetrf_, m, n, a, lda, ipiv);
  detail::zgetrf_(&m, &n, a, &lda, ipiv, &info);
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

[[nodiscard]] inline blas_int getrs(char trans, blas_int n, blas_int nrhs, uni20::complex<float>* a, blas_int lda,
                                    blas_int const* ipiv, uni20::complex<float>* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgetrs_, trans, n, nrhs, a, lda, ipiv, b, ldb);
  detail::cgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int getrs(char trans, blas_int n, blas_int nrhs, uni20::complex<double>* a, blas_int lda,
                                    blas_int const* ipiv, uni20::complex<double>* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgetrs_, trans, n, nrhs, a, lda, ipiv, b, ldb);
  detail::zgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
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

[[nodiscard]] inline blas_int getri(blas_int n, uni20::complex<float>* a, blas_int lda, blas_int* ipiv,
                                    uni20::complex<float>* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgetri_, n, a, lda, ipiv, work, lwork);
  detail::cgetri_(&n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int getri(blas_int n, uni20::complex<double>* a, blas_int lda, blas_int* ipiv,
                                    uni20::complex<double>* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgetri_, n, a, lda, ipiv, work, lwork);
  detail::zgetri_(&n, a, &lda, ipiv, work, &lwork, &info);
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

[[nodiscard]] inline blas_int gecon(char norm, blas_int n, uni20::complex<float> const* a, blas_int lda, float anorm,
                                    float& rcond, uni20::complex<float>* work, float* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgecon_, norm, n, a, lda, anorm, work, rwork);
  detail::cgecon_(&norm, &n, a, &lda, &anorm, &rcond, work, rwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gecon(char norm, blas_int n, uni20::complex<double> const* a, blas_int lda, double anorm,
                                    double& rcond, uni20::complex<double>* work, double* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgecon_, norm, n, a, lda, anorm, work, rwork);
  detail::zgecon_(&norm, &n, a, &lda, &anorm, &rcond, work, rwork, &info);
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

[[nodiscard]] inline blas_int geqrf(blas_int m, blas_int n, float* a, blas_int lda, float* tau, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgeqrf_, m, n, a, lda, tau, work, lwork);
  detail::sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geqrf(blas_int m, blas_int n, double* a, blas_int lda, double* tau, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgeqrf_, m, n, a, lda, tau, work, lwork);
  detail::dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gelqf(blas_int m, blas_int n, float* a, blas_int lda, float* tau, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgelqf_, m, n, a, lda, tau, work, lwork);
  detail::sgelqf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gelqf(blas_int m, blas_int n, double* a, blas_int lda, double* tau, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgelqf_, m, n, a, lda, tau, work, lwork);
  detail::dgelqf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geqlf(blas_int m, blas_int n, float* a, blas_int lda, float* tau, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgeqlf_, m, n, a, lda, tau, work, lwork);
  detail::sgeqlf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geqlf(blas_int m, blas_int n, double* a, blas_int lda, double* tau, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgeqlf_, m, n, a, lda, tau, work, lwork);
  detail::dgeqlf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gerqf(blas_int m, blas_int n, float* a, blas_int lda, float* tau, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgerqf_, m, n, a, lda, tau, work, lwork);
  detail::sgerqf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gerqf(blas_int m, blas_int n, double* a, blas_int lda, double* tau, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgerqf_, m, n, a, lda, tau, work, lwork);
  detail::dgerqf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgqr(blas_int m, blas_int n, blas_int k, float* a, blas_int lda, float* tau, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sorgqr_, m, n, k, a, lda, tau, work, lwork);
  detail::sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgqr(blas_int m, blas_int n, blas_int k, double* a, blas_int lda, double* tau,
                                    double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dorgqr_, m, n, k, a, lda, tau, work, lwork);
  detail::dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orglq(blas_int m, blas_int n, blas_int k, float* a, blas_int lda, float* tau, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sorglq_, m, n, k, a, lda, tau, work, lwork);
  detail::sorglq_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orglq(blas_int m, blas_int n, blas_int k, double* a, blas_int lda, double* tau,
                                    double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dorglq_, m, n, k, a, lda, tau, work, lwork);
  detail::dorglq_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgql(blas_int m, blas_int n, blas_int k, float* a, blas_int lda, float* tau, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sorgql_, m, n, k, a, lda, tau, work, lwork);
  detail::sorgql_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgql(blas_int m, blas_int n, blas_int k, double* a, blas_int lda, double* tau,
                                    double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dorgql_, m, n, k, a, lda, tau, work, lwork);
  detail::dorgql_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgrq(blas_int m, blas_int n, blas_int k, float* a, blas_int lda, float* tau, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sorgrq_, m, n, k, a, lda, tau, work, lwork);
  detail::sorgrq_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgrq(blas_int m, blas_int n, blas_int k, double* a, blas_int lda, double* tau,
                                    double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dorgrq_, m, n, k, a, lda, tau, work, lwork);
  detail::dorgrq_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormqr(char side, char trans, blas_int m, blas_int n, blas_int k, float* a, blas_int lda,
                                    float* tau, float* c, blas_int ldc, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sormqr_, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormqr(char side, char trans, blas_int m, blas_int n, blas_int k, double* a, blas_int lda,
                                    double* tau, double* c, blas_int ldc, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dormqr_, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormlq(char side, char trans, blas_int m, blas_int n, blas_int k, float* a, blas_int lda,
                                    float* tau, float* c, blas_int ldc, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sormlq_, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::sormlq_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormlq(char side, char trans, blas_int m, blas_int n, blas_int k, double* a, blas_int lda,
                                    double* tau, double* c, blas_int ldc, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dormlq_, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::dormlq_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormql(char side, char trans, blas_int m, blas_int n, blas_int k, float* a, blas_int lda,
                                    float* tau, float* c, blas_int ldc, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sormql_, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::sormql_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormql(char side, char trans, blas_int m, blas_int n, blas_int k, double* a, blas_int lda,
                                    double* tau, double* c, blas_int ldc, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dormql_, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::dormql_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormrq(char side, char trans, blas_int m, blas_int n, blas_int k, float* a, blas_int lda,
                                    float* tau, float* c, blas_int ldc, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sormrq_, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::sormrq_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormrq(char side, char trans, blas_int m, blas_int n, blas_int k, double* a, blas_int lda,
                                    double* tau, double* c, blas_int ldc, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dormrq_, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::dormrq_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gebrd(blas_int m, blas_int n, float* a, blas_int lda, float* d, float* e, float* tauq,
                                    float* taup, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgebrd_, m, n, a, lda, d, e, tauq, taup, work, lwork);
  detail::sgebrd_(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gebrd(blas_int m, blas_int n, double* a, blas_int lda, double* d, double* e, double* tauq,
                                    double* taup, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgebrd_, m, n, a, lda, d, e, tauq, taup, work, lwork);
  detail::dgebrd_(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gehrd(blas_int n, blas_int first, blas_int last, float* a, blas_int lda, float* tau,
                                    float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgehrd_, n, first, last, a, lda, tau, work, lwork);
  detail::sgehrd_(&n, &first, &last, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gehrd(blas_int n, blas_int first, blas_int last, double* a, blas_int lda, double* tau,
                                    double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgehrd_, n, first, last, a, lda, tau, work, lwork);
  detail::dgehrd_(&n, &first, &last, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geqp3(blas_int m, blas_int n, float* a, blas_int lda, blas_int* jpvt, float* tau,
                                    float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgeqp3_, m, n, a, lda, jpvt, tau, work, lwork);
  detail::sgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geqp3(blas_int m, blas_int n, double* a, blas_int lda, blas_int* jpvt, double* tau,
                                    double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgeqp3_, m, n, a, lda, jpvt, tau, work, lwork);
  detail::dgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sytrd(char uplo, blas_int n, float* a, blas_int lda, float* d, float* e, float* tau,
                                    float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssytrd_, uplo, n, a, lda, d, e, tau, work, lwork);
  detail::ssytrd_(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sytrd(char uplo, blas_int n, double* a, blas_int lda, double* d, double* e, double* tau,
                                    double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsytrd_, uplo, n, a, lda, d, e, tau, work, lwork);
  detail::dsytrd_(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int syev(char jobz, char uplo, blas_int n, float* a, blas_int lda, float* w, float* work,
                                   blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssyev_, jobz, uplo, n, a, lda, w, work, lwork);
  detail::ssyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int syev(char jobz, char uplo, blas_int n, double* a, blas_int lda, double* w, double* work,
                                   blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsyev_, jobz, uplo, n, a, lda, w, work, lwork);
  detail::dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int syevd(char jobz, char uplo, blas_int n, float* a, blas_int lda, float* w, float* work,
                                    blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssyevd_, jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork);
  detail::ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int syevd(char jobz, char uplo, blas_int n, double* a, blas_int lda, double* w, double* work,
                                    blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsyevd_, jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork);
  detail::dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int syevr(char jobz, char range, char uplo, blas_int n, float* a, blas_int lda, float vl,
                                    float vu, blas_int il, blas_int iu, float abstol, blas_int& selected_count,
                                    float* w, float* z, blas_int ldz, blas_int* isuppz, float* work, blas_int lwork,
                                    blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssyevr_, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, selected_count, w, z,
                          ldz, isuppz, work, lwork, iwork, liwork);
  detail::ssyevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz,
                  work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int syevr(char jobz, char range, char uplo, blas_int n, double* a, blas_int lda, double vl,
                                    double vu, blas_int il, blas_int iu, double abstol, blas_int& selected_count,
                                    double* w, double* z, blas_int ldz, blas_int* isuppz, double* work, blas_int lwork,
                                    blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsyevr_, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, selected_count, w, z,
                          ldz, isuppz, work, lwork, iwork, liwork);
  detail::dsyevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz,
                  work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sygv(blas_int itype, char jobz, char uplo, blas_int n, float* a, blas_int lda, float* b,
                                   blas_int ldb, float* w, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssygv_, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork);
  detail::ssygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sygv(blas_int itype, char jobz, char uplo, blas_int n, double* a, blas_int lda, double* b,
                                   blas_int ldb, double* w, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsygv_, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork);
  detail::dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sygvd(blas_int itype, char jobz, char uplo, blas_int n, float* a, blas_int lda, float* b,
                                    blas_int ldb, float* w, float* work, blas_int lwork, blas_int* iwork,
                                    blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssygvd_, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, iwork, liwork);
  detail::ssygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sygvd(blas_int itype, char jobz, char uplo, blas_int n, double* a, blas_int lda,
                                    double* b, blas_int ldb, double* w, double* work, blas_int lwork, blas_int* iwork,
                                    blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsygvd_, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, iwork, liwork);
  detail::dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sygvx(blas_int itype, char jobz, char range, char uplo, blas_int n, float* a,
                                    blas_int lda, float* b, blas_int ldb, float vl, float vu, blas_int il, blas_int iu,
                                    float abstol, blas_int& selected_count, float* w, float* z, blas_int ldz,
                                    float* work, blas_int lwork, blas_int* iwork, blas_int* ifail)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssygvx_, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol,
                          selected_count, w, z, ldz, work, lwork, iwork, ifail);
  detail::ssygvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &selected_count, w,
                  z, &ldz, work, &lwork, iwork, ifail, &info);
  return info;
}

[[nodiscard]] inline blas_int sygvx(blas_int itype, char jobz, char range, char uplo, blas_int n, double* a,
                                    blas_int lda, double* b, blas_int ldb, double vl, double vu, blas_int il,
                                    blas_int iu, double abstol, blas_int& selected_count, double* w, double* z,
                                    blas_int ldz, double* work, blas_int lwork, blas_int* iwork, blas_int* ifail)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsygvx_, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol,
                          selected_count, w, z, ldz, work, lwork, iwork, ifail);
  detail::dsygvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &selected_count, w,
                  z, &ldz, work, &lwork, iwork, ifail, &info);
  return info;
}

[[nodiscard]] inline blas_int heev(char jobz, char uplo, blas_int n, uni20::complex<float>* a, blas_int lda, float* w,
                                   uni20::complex<float>* work, blas_int lwork, float* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cheev_, jobz, uplo, n, a, lda, w, work, lwork, rwork);
  detail::cheev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
  return info;
}

[[nodiscard]] inline blas_int heev(char jobz, char uplo, blas_int n, uni20::complex<double>* a, blas_int lda, double* w,
                                   uni20::complex<double>* work, blas_int lwork, double* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zheev_, jobz, uplo, n, a, lda, w, work, lwork, rwork);
  detail::zheev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
  return info;
}

[[nodiscard]] inline blas_int heevd(char jobz, char uplo, blas_int n, uni20::complex<float>* a, blas_int lda, float* w,
                                    uni20::complex<float>* work, blas_int lwork, float* rwork, blas_int lrwork,
                                    blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cheevd_, jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork);
  detail::cheevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int heevd(char jobz, char uplo, blas_int n, uni20::complex<double>* a, blas_int lda,
                                    double* w, uni20::complex<double>* work, blas_int lwork, double* rwork,
                                    blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zheevd_, jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork);
  detail::zheevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int heevr(char jobz, char range, char uplo, blas_int n, uni20::complex<float>* a,
                                    blas_int lda, float vl, float vu, blas_int il, blas_int iu, float abstol,
                                    blas_int& selected_count, float* w, uni20::complex<float>* z, blas_int ldz,
                                    blas_int* isuppz, uni20::complex<float>* work, blas_int lwork, float* rwork,
                                    blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cheevr_, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, selected_count, w, z,
                          ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork);
  detail::cheevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz,
                  work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int heevr(char jobz, char range, char uplo, blas_int n, uni20::complex<double>* a,
                                    blas_int lda, double vl, double vu, blas_int il, blas_int iu, double abstol,
                                    blas_int& selected_count, double* w, uni20::complex<double>* z, blas_int ldz,
                                    blas_int* isuppz, uni20::complex<double>* work, blas_int lwork, double* rwork,
                                    blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zheevr_, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, selected_count, w, z,
                          ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork);
  detail::zheevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz,
                  work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int hegv(blas_int itype, char jobz, char uplo, blas_int n, uni20::complex<float>* a,
                                   blas_int lda, uni20::complex<float>* b, blas_int ldb, float* w,
                                   uni20::complex<float>* work, blas_int lwork, float* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, chegv_, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork);
  detail::chegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info);
  return info;
}

[[nodiscard]] inline blas_int hegv(blas_int itype, char jobz, char uplo, blas_int n, uni20::complex<double>* a,
                                   blas_int lda, uni20::complex<double>* b, blas_int ldb, double* w,
                                   uni20::complex<double>* work, blas_int lwork, double* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zhegv_, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork);
  detail::zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info);
  return info;
}

[[nodiscard]] inline blas_int hegvd(blas_int itype, char jobz, char uplo, blas_int n, uni20::complex<float>* a,
                                    blas_int lda, uni20::complex<float>* b, blas_int ldb, float* w,
                                    uni20::complex<float>* work, blas_int lwork, float* rwork, blas_int lrwork,
                                    blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, chegvd_, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork, lrwork, iwork,
                          liwork);
  detail::chegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int hegvd(blas_int itype, char jobz, char uplo, blas_int n, uni20::complex<double>* a,
                                    blas_int lda, uni20::complex<double>* b, blas_int ldb, double* w,
                                    uni20::complex<double>* work, blas_int lwork, double* rwork, blas_int lrwork,
                                    blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zhegvd_, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork, lrwork, iwork,
                          liwork);
  detail::zhegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int hegvx(blas_int itype, char jobz, char range, char uplo, blas_int n,
                                    uni20::complex<float>* a, blas_int lda, uni20::complex<float>* b, blas_int ldb,
                                    float vl, float vu, blas_int il, blas_int iu, float abstol,
                                    blas_int& selected_count, float* w, uni20::complex<float>* z, blas_int ldz,
                                    uni20::complex<float>* work, blas_int lwork, float* rwork, blas_int* iwork,
                                    blas_int* ifail)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, chegvx_, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol,
                          selected_count, w, z, ldz, work, lwork, rwork, iwork, ifail);
  detail::chegvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &selected_count, w,
                  z, &ldz, work, &lwork, rwork, iwork, ifail, &info);
  return info;
}

[[nodiscard]] inline blas_int hegvx(blas_int itype, char jobz, char range, char uplo, blas_int n,
                                    uni20::complex<double>* a, blas_int lda, uni20::complex<double>* b, blas_int ldb,
                                    double vl, double vu, blas_int il, blas_int iu, double abstol,
                                    blas_int& selected_count, double* w, uni20::complex<double>* z, blas_int ldz,
                                    uni20::complex<double>* work, blas_int lwork, double* rwork, blas_int* iwork,
                                    blas_int* ifail)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zhegvx_, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol,
                          selected_count, w, z, ldz, work, lwork, rwork, iwork, ifail);
  detail::zhegvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &selected_count, w,
                  z, &ldz, work, &lwork, rwork, iwork, ifail, &info);
  return info;
}

[[nodiscard]] inline blas_int geev(char jobvl, char jobvr, blas_int n, float* a, blas_int lda, float* wr, float* wi,
                                   float* vl, blas_int ldvl, float* vr, blas_int ldvr, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgeev_, jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork);
  detail::sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geev(char jobvl, char jobvr, blas_int n, double* a, blas_int lda, double* wr, double* wi,
                                   double* vl, blas_int ldvl, double* vr, blas_int ldvr, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgeev_, jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork);
  detail::dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, float* a, blas_int lda,
                                    float* wr, float* wi, float* vl, blas_int ldvl, float* vr, blas_int ldvr,
                                    blas_int& ilo, blas_int& ihi, float* scale, float& abnrm, float* rconde,
                                    float* rcondv, float* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgeevx_, balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi,
                          scale, abnrm, rconde, rcondv, work, lwork, iwork);
  detail::sgeevx_(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, &ilo, &ihi, scale, &abnrm,
                  rconde, rcondv, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, double* a,
                                    blas_int lda, double* wr, double* wi, double* vl, blas_int ldvl, double* vr,
                                    blas_int ldvr, blas_int& ilo, blas_int& ihi, double* scale, double& abnrm,
                                    double* rconde, double* rcondv, double* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgeevx_, balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi,
                          scale, abnrm, rconde, rcondv, work, lwork, iwork);
  detail::dgeevx_(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, &ilo, &ihi, scale, &abnrm,
                  rconde, rcondv, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gebal(char job, blas_int n, float* a, blas_int lda, blas_int& first, blas_int& last,
                                    float* scale)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgebal_, job, n, a, lda, first, last, scale);
  detail::sgebal_(&job, &n, a, &lda, &first, &last, scale, &info);
  return info;
}

[[nodiscard]] inline blas_int gebal(char job, blas_int n, double* a, blas_int lda, blas_int& first, blas_int& last,
                                    double* scale)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgebal_, job, n, a, lda, first, last, scale);
  detail::dgebal_(&job, &n, a, &lda, &first, &last, scale, &info);
  return info;
}

[[nodiscard]] inline blas_int gebak(char job, char side, blas_int n, blas_int first, blas_int last, float* scale,
                                    blas_int vector_count, float* vectors, blas_int leading_dimension)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgebak_, job, side, n, first, last, scale, vector_count, vectors, leading_dimension);
  detail::sgebak_(&job, &side, &n, &first, &last, scale, &vector_count, vectors, &leading_dimension, &info);
  return info;
}

[[nodiscard]] inline blas_int gebak(char job, char side, blas_int n, blas_int first, blas_int last, double* scale,
                                    blas_int vector_count, double* vectors, blas_int leading_dimension)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgebak_, job, side, n, first, last, scale, vector_count, vectors, leading_dimension);
  detail::dgebak_(&job, &side, &n, &first, &last, scale, &vector_count, vectors, &leading_dimension, &info);
  return info;
}

[[nodiscard]] inline blas_int geev(char jobvl, char jobvr, blas_int n, uni20::complex<float>* a, blas_int lda,
                                   uni20::complex<float>* w, uni20::complex<float>* vl, blas_int ldvl,
                                   uni20::complex<float>* vr, blas_int ldvr, uni20::complex<float>* work,
                                   blas_int lwork, float* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgeev_, jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork);
  detail::cgeev_(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info);
  return info;
}

[[nodiscard]] inline blas_int geev(char jobvl, char jobvr, blas_int n, uni20::complex<double>* a, blas_int lda,
                                   uni20::complex<double>* w, uni20::complex<double>* vl, blas_int ldvl,
                                   uni20::complex<double>* vr, blas_int ldvr, uni20::complex<double>* work,
                                   blas_int lwork, double* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgeev_, jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork);
  detail::zgeev_(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gees(char jobvs, char sort, blas_int n, float* a, blas_int lda,
                                   blas_int& selected_dimension, float* wr, float* wi, float* vs, blas_int ldvs,
                                   float* work, blas_int lwork, blas_int* bwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgees_, jobvs, sort, n, a, lda, selected_dimension, wr, wi, vs, ldvs, work, lwork,
                          bwork);
  detail::sgees_(&jobvs, &sort, nullptr, &n, a, &lda, &selected_dimension, wr, wi, vs, &ldvs, work, &lwork, bwork,
                 &info);
  return info;
}

[[nodiscard]] inline blas_int gees(char jobvs, char sort, blas_int n, double* a, blas_int lda,
                                   blas_int& selected_dimension, double* wr, double* wi, double* vs, blas_int ldvs,
                                   double* work, blas_int lwork, blas_int* bwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgees_, jobvs, sort, n, a, lda, selected_dimension, wr, wi, vs, ldvs, work, lwork,
                          bwork);
  detail::dgees_(&jobvs, &sort, nullptr, &n, a, &lda, &selected_dimension, wr, wi, vs, &ldvs, work, &lwork, bwork,
                 &info);
  return info;
}

[[nodiscard]] inline blas_int gees(char jobvs, char sort, blas_int n, uni20::complex<float>* a, blas_int lda,
                                   blas_int& selected_dimension, uni20::complex<float>* w, uni20::complex<float>* vs,
                                   blas_int ldvs, uni20::complex<float>* work, blas_int lwork, float* rwork,
                                   blas_int* bwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgees_, jobvs, sort, n, a, lda, selected_dimension, w, vs, ldvs, work, lwork, rwork,
                          bwork);
  detail::cgees_(&jobvs, &sort, nullptr, &n, a, &lda, &selected_dimension, w, vs, &ldvs, work, &lwork, rwork, bwork,
                 &info);
  return info;
}

[[nodiscard]] inline blas_int gees(char jobvs, char sort, blas_int n, uni20::complex<double>* a, blas_int lda,
                                   blas_int& selected_dimension, uni20::complex<double>* w, uni20::complex<double>* vs,
                                   blas_int ldvs, uni20::complex<double>* work, blas_int lwork, double* rwork,
                                   blas_int* bwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgees_, jobvs, sort, n, a, lda, selected_dimension, w, vs, ldvs, work, lwork, rwork,
                          bwork);
  detail::zgees_(&jobvs, &sort, nullptr, &n, a, &lda, &selected_dimension, w, vs, &ldvs, work, &lwork, rwork, bwork,
                 &info);
  return info;
}

[[nodiscard]] inline blas_int hseqr(char job, char compz, blas_int n, blas_int first, blas_int last, float* h,
                                    blas_int ldh, float* wr, float* wi, float* z, blas_int ldz, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, shseqr_, job, compz, n, first, last, h, ldh, wr, wi, z, ldz, work, lwork);
  detail::shseqr_(&job, &compz, &n, &first, &last, h, &ldh, wr, wi, z, &ldz, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int hseqr(char job, char compz, blas_int n, blas_int first, blas_int last, double* h,
                                    blas_int ldh, double* wr, double* wi, double* z, blas_int ldz, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dhseqr_, job, compz, n, first, last, h, ldh, wr, wi, z, ldz, work, lwork);
  detail::dhseqr_(&job, &compz, &n, &first, &last, h, &ldh, wr, wi, z, &ldz, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int trexc(char compq, blas_int n, float* t, blas_int ldt, float* q, blas_int ldq,
                                    blas_int& first, blas_int& last, float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, strexc_, compq, n, t, ldt, q, ldq, first, last, work);
  detail::strexc_(&compq, &n, t, &ldt, q, &ldq, &first, &last, work, &info);
  return info;
}

[[nodiscard]] inline blas_int trexc(char compq, blas_int n, double* t, blas_int ldt, double* q, blas_int ldq,
                                    blas_int& first, blas_int& last, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtrexc_, compq, n, t, ldt, q, ldq, first, last, work);
  detail::dtrexc_(&compq, &n, t, &ldt, q, &ldq, &first, &last, work, &info);
  return info;
}

[[nodiscard]] inline blas_int trexc(char compq, blas_int n, uni20::complex<float>* t, blas_int ldt,
                                    uni20::complex<float>* q, blas_int ldq, blas_int& first, blas_int& last)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ctrexc_, compq, n, t, ldt, q, ldq, first, last);
  detail::ctrexc_(&compq, &n, t, &ldt, q, &ldq, &first, &last, &info);
  return info;
}

[[nodiscard]] inline blas_int trexc(char compq, blas_int n, uni20::complex<double>* t, blas_int ldt,
                                    uni20::complex<double>* q, blas_int ldq, blas_int& first, blas_int& last)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ztrexc_, compq, n, t, ldt, q, ldq, first, last);
  detail::ztrexc_(&compq, &n, t, &ldt, q, &ldq, &first, &last, &info);
  return info;
}

[[nodiscard]] inline blas_int trsen(char job, char compq, blas_int* select, blas_int n, float* t, blas_int ldt,
                                    float* q, blas_int ldq, float* wr, float* wi, blas_int& selected_dimension,
                                    float& reciprocal_eigenvalue_cluster_condition,
                                    float& reciprocal_invariant_subspace_condition, float* work, blas_int lwork,
                                    blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, strsen_, job, compq, select, n, t, ldt, q, ldq, wr, wi, selected_dimension,
                          reciprocal_eigenvalue_cluster_condition, reciprocal_invariant_subspace_condition, work, lwork,
                          iwork, liwork);
  detail::strsen_(&job, &compq, select, &n, t, &ldt, q, &ldq, wr, wi, &selected_dimension,
                  &reciprocal_eigenvalue_cluster_condition, &reciprocal_invariant_subspace_condition, work, &lwork,
                  iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int trsen(char job, char compq, blas_int* select, blas_int n, double* t, blas_int ldt,
                                    double* q, blas_int ldq, double* wr, double* wi, blas_int& selected_dimension,
                                    double& reciprocal_eigenvalue_cluster_condition,
                                    double& reciprocal_invariant_subspace_condition, double* work, blas_int lwork,
                                    blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtrsen_, job, compq, select, n, t, ldt, q, ldq, wr, wi, selected_dimension,
                          reciprocal_eigenvalue_cluster_condition, reciprocal_invariant_subspace_condition, work, lwork,
                          iwork, liwork);
  detail::dtrsen_(&job, &compq, select, &n, t, &ldt, q, &ldq, wr, wi, &selected_dimension,
                  &reciprocal_eigenvalue_cluster_condition, &reciprocal_invariant_subspace_condition, work, &lwork,
                  iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int trevc(char side, char howmny, blas_int* select, blas_int n, float* t, blas_int ldt,
                                    float* vl, blas_int ldvl, float* vr, blas_int ldvr, blas_int mm,
                                    blas_int& computed_vectors, float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, strevc_, side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, computed_vectors,
                          work);
  detail::strevc_(&side, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr, &mm, &computed_vectors, work, &info);
  return info;
}

[[nodiscard]] inline blas_int trevc(char side, char howmny, blas_int* select, blas_int n, double* t, blas_int ldt,
                                    double* vl, blas_int ldvl, double* vr, blas_int ldvr, blas_int mm,
                                    blas_int& computed_vectors, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtrevc_, side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, computed_vectors,
                          work);
  detail::dtrevc_(&side, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr, &mm, &computed_vectors, work, &info);
  return info;
}

[[nodiscard]] inline blas_int trsna(char job, char howmny, blas_int* select, blas_int n, float* t, blas_int ldt,
                                    float* vl, blas_int ldvl, float* vr, blas_int ldvr,
                                    float* reciprocal_eigenvalue_condition_numbers,
                                    float* reciprocal_eigenvector_condition_numbers, blas_int mm,
                                    blas_int& computed_estimates, float* work, blas_int ldwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, strsna_, job, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr,
                          reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers, mm,
                          computed_estimates, work, ldwork, iwork);
  detail::strsna_(&job, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr, reciprocal_eigenvalue_condition_numbers,
                  reciprocal_eigenvector_condition_numbers, &mm, &computed_estimates, work, &ldwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int trsna(char job, char howmny, blas_int* select, blas_int n, double* t, blas_int ldt,
                                    double* vl, blas_int ldvl, double* vr, blas_int ldvr,
                                    double* reciprocal_eigenvalue_condition_numbers,
                                    double* reciprocal_eigenvector_condition_numbers, blas_int mm,
                                    blas_int& computed_estimates, double* work, blas_int ldwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtrsna_, job, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr,
                          reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers, mm,
                          computed_estimates, work, ldwork, iwork);
  detail::dtrsna_(&job, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr, reciprocal_eigenvalue_condition_numbers,
                  reciprocal_eigenvector_condition_numbers, &mm, &computed_estimates, work, &ldwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ggev(char jobvl, char jobvr, blas_int n, float* a, blas_int lda, float* b, blas_int ldb,
                                   float* alphar, float* alphai, float* beta, float* vl, blas_int ldvl, float* vr,
                                   blas_int ldvr, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sggev_, jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr,
                          work, lwork);
  detail::sggev_(&jobvl, &jobvr, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ggev(char jobvl, char jobvr, blas_int n, double* a, blas_int lda, double* b, blas_int ldb,
                                   double* alphar, double* alphai, double* beta, double* vl, blas_int ldvl, double* vr,
                                   blas_int ldvr, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dggev_, jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr,
                          work, lwork);
  detail::dggev_(&jobvl, &jobvr, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ggevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, float* a, blas_int lda,
                                    float* b, blas_int ldb, float* alphar, float* alphai, float* beta, float* vl,
                                    blas_int ldvl, float* vr, blas_int ldvr, blas_int& ilo, blas_int& ihi,
                                    float* lscale, float* rscale, float& abnrm, float& bbnrm, float* rconde,
                                    float* rcondv, float* work, blas_int lwork, blas_int* iwork, blas_int* bwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sggevx_, balanc, jobvl, jobvr, sense, n, a, lda, b, ldb, alphar, alphai, beta, vl,
                          ldvl, vr, ldvr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work, lwork, iwork,
                          bwork);
  detail::sggevx_(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr,
                  &ilo, &ihi, lscale, rscale, &abnrm, &bbnrm, rconde, rcondv, work, &lwork, iwork, bwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ggevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, double* a,
                                    blas_int lda, double* b, blas_int ldb, double* alphar, double* alphai, double* beta,
                                    double* vl, blas_int ldvl, double* vr, blas_int ldvr, blas_int& ilo, blas_int& ihi,
                                    double* lscale, double* rscale, double& abnrm, double& bbnrm, double* rconde,
                                    double* rcondv, double* work, blas_int lwork, blas_int* iwork, blas_int* bwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dggevx_, balanc, jobvl, jobvr, sense, n, a, lda, b, ldb, alphar, alphai, beta, vl,
                          ldvl, vr, ldvr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work, lwork, iwork,
                          bwork);
  detail::dggevx_(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr,
                  &ilo, &ihi, lscale, rscale, &abnrm, &bbnrm, rconde, rcondv, work, &lwork, iwork, bwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ggbal(char job, blas_int n, float* a, blas_int lda, float* b, blas_int ldb,
                                    blas_int& first, blas_int& last, float* lscale, float* rscale, float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sggbal_, job, n, a, lda, b, ldb, first, last, lscale, rscale, work);
  detail::sggbal_(&job, &n, a, &lda, b, &ldb, &first, &last, lscale, rscale, work, &info);
  return info;
}

[[nodiscard]] inline blas_int ggbal(char job, blas_int n, double* a, blas_int lda, double* b, blas_int ldb,
                                    blas_int& first, blas_int& last, double* lscale, double* rscale, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dggbal_, job, n, a, lda, b, ldb, first, last, lscale, rscale, work);
  detail::dggbal_(&job, &n, a, &lda, b, &ldb, &first, &last, lscale, rscale, work, &info);
  return info;
}

[[nodiscard]] inline blas_int ggbak(char job, char side, blas_int n, blas_int first, blas_int last, float* lscale,
                                    float* rscale, blas_int vector_count, float* vectors, blas_int leading_dimension)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sggbak_, job, side, n, first, last, lscale, rscale, vector_count, vectors,
                          leading_dimension);
  detail::sggbak_(&job, &side, &n, &first, &last, lscale, rscale, &vector_count, vectors, &leading_dimension, &info);
  return info;
}

[[nodiscard]] inline blas_int ggbak(char job, char side, blas_int n, blas_int first, blas_int last, double* lscale,
                                    double* rscale, blas_int vector_count, double* vectors, blas_int leading_dimension)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dggbak_, job, side, n, first, last, lscale, rscale, vector_count, vectors,
                          leading_dimension);
  detail::dggbak_(&job, &side, &n, &first, &last, lscale, rscale, &vector_count, vectors, &leading_dimension, &info);
  return info;
}

[[nodiscard]] inline blas_int gges(char jobvsl, char jobvsr, char sort, blas_int n, float* a, blas_int lda, float* b,
                                   blas_int ldb, blas_int& selected_dimension, float* alphar, float* alphai,
                                   float* beta, float* vsl, blas_int ldvsl, float* vsr, blas_int ldvsr, float* work,
                                   blas_int lwork, blas_int* bwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgges_, jobvsl, jobvsr, sort, n, a, lda, b, ldb, selected_dimension, alphar, alphai,
                          beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork);
  detail::sgges_(&jobvsl, &jobvsr, &sort, nullptr, &n, a, &lda, b, &ldb, &selected_dimension, alphar, alphai, beta, vsl,
                 &ldvsl, vsr, &ldvsr, work, &lwork, bwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gges(char jobvsl, char jobvsr, char sort, blas_int n, double* a, blas_int lda, double* b,
                                   blas_int ldb, blas_int& selected_dimension, double* alphar, double* alphai,
                                   double* beta, double* vsl, blas_int ldvsl, double* vsr, blas_int ldvsr, double* work,
                                   blas_int lwork, blas_int* bwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgges_, jobvsl, jobvsr, sort, n, a, lda, b, ldb, selected_dimension, alphar, alphai,
                          beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork);
  detail::dgges_(&jobvsl, &jobvsr, &sort, nullptr, &n, a, &lda, b, &ldb, &selected_dimension, alphar, alphai, beta, vsl,
                 &ldvsl, vsr, &ldvsr, work, &lwork, bwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gghrd(char compq, char compz, blas_int n, blas_int first, blas_int last, float* a,
                                    blas_int lda, float* b, blas_int ldb, float* q, blas_int ldq, float* z,
                                    blas_int ldz)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgghrd_, compq, compz, n, first, last, a, lda, b, ldb, q, ldq, z, ldz);
  detail::sgghrd_(&compq, &compz, &n, &first, &last, a, &lda, b, &ldb, q, &ldq, z, &ldz, &info);
  return info;
}

[[nodiscard]] inline blas_int gghrd(char compq, char compz, blas_int n, blas_int first, blas_int last, double* a,
                                    blas_int lda, double* b, blas_int ldb, double* q, blas_int ldq, double* z,
                                    blas_int ldz)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgghrd_, compq, compz, n, first, last, a, lda, b, ldb, q, ldq, z, ldz);
  detail::dgghrd_(&compq, &compz, &n, &first, &last, a, &lda, b, &ldb, q, &ldq, z, &ldz, &info);
  return info;
}

[[nodiscard]] inline blas_int hgeqz(char job, char compq, char compz, blas_int n, blas_int first, blas_int last,
                                    float* h, blas_int ldh, float* t, blas_int ldt, float* alphar, float* alphai,
                                    float* beta, float* q, blas_int ldq, float* z, blas_int ldz, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, shgeqz_, job, compq, compz, n, first, last, h, ldh, t, ldt, alphar, alphai, beta, q,
                          ldq, z, ldz, work, lwork);
  detail::shgeqz_(&job, &compq, &compz, &n, &first, &last, h, &ldh, t, &ldt, alphar, alphai, beta, q, &ldq, z, &ldz,
                  work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int hgeqz(char job, char compq, char compz, blas_int n, blas_int first, blas_int last,
                                    double* h, blas_int ldh, double* t, blas_int ldt, double* alphar, double* alphai,
                                    double* beta, double* q, blas_int ldq, double* z, blas_int ldz, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dhgeqz_, job, compq, compz, n, first, last, h, ldh, t, ldt, alphar, alphai, beta, q,
                          ldq, z, ldz, work, lwork);
  detail::dhgeqz_(&job, &compq, &compz, &n, &first, &last, h, &ldh, t, &ldt, alphar, alphai, beta, q, &ldq, z, &ldz,
                  work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int tgexc(bool wantq, bool wantz, blas_int n, float* a, blas_int lda, float* b, blas_int ldb,
                                    float* q, blas_int ldq, float* z, blas_int ldz, blas_int& first, blas_int& last,
                                    float* work, blas_int lwork)
{
  blas_int info = 0;
  blas_int const fortran_wantq = wantq ? 1 : 0;
  blas_int const fortran_wantz = wantz ? 1 : 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, stgexc_, wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, first, last, work, lwork);
  detail::stgexc_(&fortran_wantq, &fortran_wantz, &n, a, &lda, b, &ldb, q, &ldq, z, &ldz, &first, &last, work, &lwork,
                  &info);
  return info;
}

[[nodiscard]] inline blas_int tgexc(bool wantq, bool wantz, blas_int n, double* a, blas_int lda, double* b,
                                    blas_int ldb, double* q, blas_int ldq, double* z, blas_int ldz, blas_int& first,
                                    blas_int& last, double* work, blas_int lwork)
{
  blas_int info = 0;
  blas_int const fortran_wantq = wantq ? 1 : 0;
  blas_int const fortran_wantz = wantz ? 1 : 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtgexc_, wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, first, last, work, lwork);
  detail::dtgexc_(&fortran_wantq, &fortran_wantz, &n, a, &lda, b, &ldb, q, &ldq, z, &ldz, &first, &last, work, &lwork,
                  &info);
  return info;
}

[[nodiscard]] inline blas_int tgsen(blas_int ijob, bool wantq, bool wantz, blas_int* select, blas_int n, float* a,
                                    blas_int lda, float* b, blas_int ldb, float* alphar, float* alphai, float* beta,
                                    float* q, blas_int ldq, float* z, blas_int ldz, blas_int& selected_dimension,
                                    float& pl, float& pr, float* dif, float* work, blas_int lwork, blas_int* iwork,
                                    blas_int liwork)
{
  blas_int info = 0;
  blas_int const fortran_wantq = wantq ? 1 : 0;
  blas_int const fortran_wantz = wantz ? 1 : 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, stgsen_, ijob, wantq, wantz, select, n, a, lda, b, ldb, alphar, alphai, beta, q, ldq,
                          z, ldz, selected_dimension, pl, pr, dif, work, lwork, iwork, liwork);
  detail::stgsen_(&ijob, &fortran_wantq, &fortran_wantz, select, &n, a, &lda, b, &ldb, alphar, alphai, beta, q, &ldq, z,
                  &ldz, &selected_dimension, &pl, &pr, dif, work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int tgsen(blas_int ijob, bool wantq, bool wantz, blas_int* select, blas_int n, double* a,
                                    blas_int lda, double* b, blas_int ldb, double* alphar, double* alphai, double* beta,
                                    double* q, blas_int ldq, double* z, blas_int ldz, blas_int& selected_dimension,
                                    double& pl, double& pr, double* dif, double* work, blas_int lwork, blas_int* iwork,
                                    blas_int liwork)
{
  blas_int info = 0;
  blas_int const fortran_wantq = wantq ? 1 : 0;
  blas_int const fortran_wantz = wantz ? 1 : 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtgsen_, ijob, wantq, wantz, select, n, a, lda, b, ldb, alphar, alphai, beta, q, ldq,
                          z, ldz, selected_dimension, pl, pr, dif, work, lwork, iwork, liwork);
  detail::dtgsen_(&ijob, &fortran_wantq, &fortran_wantz, select, &n, a, &lda, b, &ldb, alphar, alphai, beta, q, &ldq, z,
                  &ldz, &selected_dimension, &pl, &pr, dif, work, &lwork, iwork, &liwork, &info);
  return info;
}

[[nodiscard]] inline blas_int tgevc(char side, char howmny, blas_int* select, blas_int n, float* s, blas_int lds,
                                    float* p, blas_int ldp, float* vl, blas_int ldvl, float* vr, blas_int ldvr,
                                    blas_int mm, blas_int& computed_vectors, float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, stgevc_, side, howmny, select, n, s, lds, p, ldp, vl, ldvl, vr, ldvr, mm,
                          computed_vectors, work);
  detail::stgevc_(&side, &howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl, vr, &ldvr, &mm, &computed_vectors, work,
                  &info);
  return info;
}

[[nodiscard]] inline blas_int tgevc(char side, char howmny, blas_int* select, blas_int n, double* s, blas_int lds,
                                    double* p, blas_int ldp, double* vl, blas_int ldvl, double* vr, blas_int ldvr,
                                    blas_int mm, blas_int& computed_vectors, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtgevc_, side, howmny, select, n, s, lds, p, ldp, vl, ldvl, vr, ldvr, mm,
                          computed_vectors, work);
  detail::dtgevc_(&side, &howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl, vr, &ldvr, &mm, &computed_vectors, work,
                  &info);
  return info;
}

[[nodiscard]] inline blas_int tgsna(char job, char howmny, blas_int* select, blas_int n, float* a, blas_int lda,
                                    float* b, blas_int ldb, float* vl, blas_int ldvl, float* vr, blas_int ldvr,
                                    float* reciprocal_eigenvalue_condition_numbers,
                                    float* reciprocal_eigenvector_condition_numbers, blas_int mm,
                                    blas_int& computed_estimates, float* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, stgsna_, job, howmny, select, n, a, lda, b, ldb, vl, ldvl, vr, ldvr,
                          reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers, mm,
                          computed_estimates, work, lwork, iwork);
  detail::stgsna_(&job, &howmny, select, &n, a, &lda, b, &ldb, vl, &ldvl, vr, &ldvr,
                  reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers, &mm,
                  &computed_estimates, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int tgsna(char job, char howmny, blas_int* select, blas_int n, double* a, blas_int lda,
                                    double* b, blas_int ldb, double* vl, blas_int ldvl, double* vr, blas_int ldvr,
                                    double* reciprocal_eigenvalue_condition_numbers,
                                    double* reciprocal_eigenvector_condition_numbers, blas_int mm,
                                    blas_int& computed_estimates, double* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtgsna_, job, howmny, select, n, a, lda, b, ldb, vl, ldvl, vr, ldvr,
                          reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers, mm,
                          computed_estimates, work, lwork, iwork);
  detail::dtgsna_(&job, &howmny, select, &n, a, &lda, b, &ldb, vl, &ldvl, vr, &ldvr,
                  reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers, &mm,
                  &computed_estimates, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgbr(char vect, blas_int m, blas_int n, blas_int k, float* a, blas_int lda, float* tau,
                                    float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sorgbr_, vect, m, n, k, a, lda, tau, work, lwork);
  detail::sorgbr_(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgbr(char vect, blas_int m, blas_int n, blas_int k, double* a, blas_int lda, double* tau,
                                    double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dorgbr_, vect, m, n, k, a, lda, tau, work, lwork);
  detail::dorgbr_(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orghr(blas_int n, blas_int first, blas_int last, float* a, blas_int lda, float const* tau,
                                    float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sorghr_, n, first, last, a, lda, tau, work, lwork);
  detail::sorghr_(&n, &first, &last, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orghr(blas_int n, blas_int first, blas_int last, double* a, blas_int lda,
                                    double const* tau, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dorghr_, n, first, last, a, lda, tau, work, lwork);
  detail::dorghr_(&n, &first, &last, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgtr(char uplo, blas_int n, float* a, blas_int lda, float const* tau, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sorgtr_, uplo, n, a, lda, tau, work, lwork);
  detail::sorgtr_(&uplo, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int orgtr(char uplo, blas_int n, double* a, blas_int lda, double const* tau, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dorgtr_, uplo, n, a, lda, tau, work, lwork);
  detail::dorgtr_(&uplo, &n, a, &lda, tau, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormhr(char side, char trans, blas_int m, blas_int n, blas_int first, blas_int last,
                                    float* a, blas_int lda, float const* tau, float* c, blas_int ldc, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sormhr_, side, trans, m, n, first, last, a, lda, tau, c, ldc, work, lwork);
  detail::sormhr_(&side, &trans, &m, &n, &first, &last, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormhr(char side, char trans, blas_int m, blas_int n, blas_int first, blas_int last,
                                    double* a, blas_int lda, double const* tau, double* c, blas_int ldc, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dormhr_, side, trans, m, n, first, last, a, lda, tau, c, ldc, work, lwork);
  detail::dormhr_(&side, &trans, &m, &n, &first, &last, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormtr(char side, char uplo, char trans, blas_int m, blas_int n, float* a, blas_int lda,
                                    float const* tau, float* c, blas_int ldc, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sormtr_, side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork);
  detail::sormtr_(&side, &uplo, &trans, &m, &n, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormtr(char side, char uplo, char trans, blas_int m, blas_int n, double* a, blas_int lda,
                                    double const* tau, double* c, blas_int ldc, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dormtr_, side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork);
  detail::dormtr_(&side, &uplo, &trans, &m, &n, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormbr(char vect, char side, char trans, blas_int m, blas_int n, blas_int k, float* a,
                                    blas_int lda, float* tau, float* c, blas_int ldc, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sormbr_, vect, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::sormbr_(&vect, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int ormbr(char vect, char side, char trans, blas_int m, blas_int n, blas_int k, double* a,
                                    blas_int lda, double* tau, double* c, blas_int ldc, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dormbr_, vect, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  detail::dormbr_(&vect, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvd(char jobu, char jobvt, blas_int m, blas_int n, float* a, blas_int lda, float* s,
                                    float* u, blas_int ldu, float* vt, blas_int ldvt, float* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgesvd_, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
  detail::sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvd(char jobu, char jobvt, blas_int m, blas_int n, double* a, blas_int lda, double* s,
                                    double* u, blas_int ldu, double* vt, blas_int ldvt, double* work, blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgesvd_, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
  detail::dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvd(char jobu, char jobvt, blas_int m, blas_int n, uni20::complex<float>* a,
                                    blas_int lda, float* s, uni20::complex<float>* u, blas_int ldu,
                                    uni20::complex<float>* vt, blas_int ldvt, uni20::complex<float>* work,
                                    blas_int lwork, float* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgesvd_, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork);
  detail::cgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvd(char jobu, char jobvt, blas_int m, blas_int n, uni20::complex<double>* a,
                                    blas_int lda, double* s, uni20::complex<double>* u, blas_int ldu,
                                    uni20::complex<double>* vt, blas_int ldvt, uni20::complex<double>* work,
                                    blas_int lwork, double* rwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgesvd_, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork);
  detail::zgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesdd(char jobz, blas_int m, blas_int n, float* a, blas_int lda, float* s, float* u,
                                    blas_int ldu, float* vt, blas_int ldvt, float* work, blas_int lwork,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgesdd_, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork);
  detail::sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesdd(char jobz, blas_int m, blas_int n, double* a, blas_int lda, double* s, double* u,
                                    blas_int ldu, double* vt, blas_int ldvt, double* work, blas_int lwork,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgesdd_, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork);
  detail::dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesdd(char jobz, blas_int m, blas_int n, uni20::complex<float>* a, blas_int lda, float* s,
                                    uni20::complex<float>* u, blas_int ldu, uni20::complex<float>* vt, blas_int ldvt,
                                    uni20::complex<float>* work, blas_int lwork, float* rwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgesdd_, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork);
  detail::cgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesdd(char jobz, blas_int m, blas_int n, uni20::complex<double>* a, blas_int lda,
                                    double* s, uni20::complex<double>* u, blas_int ldu, uni20::complex<double>* vt,
                                    blas_int ldvt, uni20::complex<double>* work, blas_int lwork, double* rwork,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgesdd_, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork);
  detail::zgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvdx(char jobu, char jobvt, char range, blas_int m, blas_int n, float* a, blas_int lda,
                                     float vl, float vu, blas_int il, blas_int iu, blas_int& selected_count,
                                     float* singular_values, float* u, blas_int ldu, float* vt, blas_int ldvt,
                                     float* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sgesvdx_, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, selected_count,
                          singular_values, u, ldu, vt, ldvt, work, lwork, iwork);
  detail::sgesvdx_(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, &selected_count, singular_values, u,
                   &ldu, vt, &ldvt, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvdx(char jobu, char jobvt, char range, blas_int m, blas_int n, double* a, blas_int lda,
                                     double vl, double vu, blas_int il, blas_int iu, blas_int& selected_count,
                                     double* singular_values, double* u, blas_int ldu, double* vt, blas_int ldvt,
                                     double* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dgesvdx_, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, selected_count,
                          singular_values, u, ldu, vt, ldvt, work, lwork, iwork);
  detail::dgesvdx_(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, &selected_count, singular_values, u,
                   &ldu, vt, &ldvt, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvdx(char jobu, char jobvt, char range, blas_int m, blas_int n,
                                     uni20::complex<float>* a, blas_int lda, float vl, float vu, blas_int il,
                                     blas_int iu, blas_int& selected_count, float* singular_values,
                                     uni20::complex<float>* u, blas_int ldu, uni20::complex<float>* vt, blas_int ldvt,
                                     uni20::complex<float>* work, blas_int lwork, float* rwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, cgesvdx_, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, selected_count,
                          singular_values, u, ldu, vt, ldvt, work, lwork, rwork, iwork);
  detail::cgesvdx_(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, &selected_count, singular_values, u,
                   &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int gesvdx(char jobu, char jobvt, char range, blas_int m, blas_int n,
                                     uni20::complex<double>* a, blas_int lda, double vl, double vu, blas_int il,
                                     blas_int iu, blas_int& selected_count, double* singular_values,
                                     uni20::complex<double>* u, blas_int ldu, uni20::complex<double>* vt, blas_int ldvt,
                                     uni20::complex<double>* work, blas_int lwork, double* rwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, zgesvdx_, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, selected_count,
                          singular_values, u, ldu, vt, ldvt, work, lwork, rwork, iwork);
  detail::zgesvdx_(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, &selected_count, singular_values, u,
                   &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int bdsqr(char uplo, blas_int n, blas_int ncvt, blas_int nru, blas_int ncc, float* d,
                                    float* e, float* vt, blas_int ldvt, float* u, blas_int ldu, float* c, blas_int ldc,
                                    float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sbdsqr_, uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work);
  detail::sbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt, &ldvt, u, &ldu, c, &ldc, work, &info);
  return info;
}

[[nodiscard]] inline blas_int bdsqr(char uplo, blas_int n, blas_int ncvt, blas_int nru, blas_int ncc, double* d,
                                    double* e, double* vt, blas_int ldvt, double* u, blas_int ldu, double* c,
                                    blas_int ldc, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dbdsqr_, uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work);
  detail::dbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt, &ldvt, u, &ldu, c, &ldc, work, &info);
  return info;
}

[[nodiscard]] inline blas_int bdsdc(char uplo, char compq, blas_int n, float* d, float* e, float* u, blas_int ldu,
                                    float* vt, blas_int ldvt, float* q, blas_int* iq, float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sbdsdc_, uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, work, iwork);
  detail::sbdsdc_(&uplo, &compq, &n, d, e, u, &ldu, vt, &ldvt, q, iq, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int bdsdc(char uplo, char compq, blas_int n, double* d, double* e, double* u, blas_int ldu,
                                    double* vt, blas_int ldvt, double* q, blas_int* iq, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dbdsdc_, uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, work, iwork);
  detail::dbdsdc_(&uplo, &compq, &n, d, e, u, &ldu, vt, &ldvt, q, iq, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int bdsvdx(char uplo, char jobz, char range, blas_int n, float* d, float* e, float vl,
                                     float vu, blas_int il, blas_int iu, blas_int& selected_count,
                                     float* singular_values, float* z, blas_int ldz, float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sbdsvdx_, uplo, jobz, range, n, d, e, vl, vu, il, iu, selected_count, singular_values,
                          z, ldz, work, iwork);
  detail::sbdsvdx_(&uplo, &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &selected_count, singular_values, z, &ldz, work,
                   iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int bdsvdx(char uplo, char jobz, char range, blas_int n, double* d, double* e, double vl,
                                     double vu, blas_int il, blas_int iu, blas_int& selected_count,
                                     double* singular_values, double* z, blas_int ldz, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dbdsvdx_, uplo, jobz, range, n, d, e, vl, vu, il, iu, selected_count, singular_values,
                          z, ldz, work, iwork);
  detail::dbdsvdx_(&uplo, &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &selected_count, singular_values, z, &ldz, work,
                   iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int potrf(char uplo, blas_int n, float* a, blas_int lda)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, spotrf_, uplo, n, a, lda);
  detail::spotrf_(&uplo, &n, a, &lda, &info);
  return info;
}

[[nodiscard]] inline blas_int potrf(char uplo, blas_int n, double* a, blas_int lda)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dpotrf_, uplo, n, a, lda);
  detail::dpotrf_(&uplo, &n, a, &lda, &info);
  return info;
}

[[nodiscard]] inline blas_int pstrf(char uplo, blas_int n, float* a, blas_int lda, blas_int* pivots, blas_int& rank,
                                    float tolerance, float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, spstrf_, uplo, n, a, lda, pivots, rank, tolerance, work);
  detail::spstrf_(&uplo, &n, a, &lda, pivots, &rank, &tolerance, work, &info);
  return info;
}

[[nodiscard]] inline blas_int pstrf(char uplo, blas_int n, double* a, blas_int lda, blas_int* pivots, blas_int& rank,
                                    double tolerance, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dpstrf_, uplo, n, a, lda, pivots, rank, tolerance, work);
  detail::dpstrf_(&uplo, &n, a, &lda, pivots, &rank, &tolerance, work, &info);
  return info;
}

[[nodiscard]] inline blas_int potri(char uplo, blas_int n, float* a, blas_int lda)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, spotri_, uplo, n, a, lda);
  detail::spotri_(&uplo, &n, a, &lda, &info);
  return info;
}

[[nodiscard]] inline blas_int potri(char uplo, blas_int n, double* a, blas_int lda)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dpotri_, uplo, n, a, lda);
  detail::dpotri_(&uplo, &n, a, &lda, &info);
  return info;
}

[[nodiscard]] inline blas_int potrs(char uplo, blas_int n, blas_int nrhs, float* a, blas_int lda, float* b,
                                    blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, spotrs_, uplo, n, nrhs, a, lda, b, ldb);
  detail::spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int potrs(char uplo, blas_int n, blas_int nrhs, double* a, blas_int lda, double* b,
                                    blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dpotrs_, uplo, n, nrhs, a, lda, b, ldb);
  detail::dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int porfs(char uplo, blas_int n, blas_int nrhs, float* a, blas_int lda, float* factors,
                                    blas_int factor_lda, float* b, blas_int ldb, float* x, blas_int ldx,
                                    float* forward_error, float* backward_error, float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sporfs_, uplo, n, nrhs, a, lda, factors, factor_lda, b, ldb, x, ldx, forward_error,
                          backward_error, work, iwork);
  detail::sporfs_(&uplo, &n, &nrhs, a, &lda, factors, &factor_lda, b, &ldb, x, &ldx, forward_error, backward_error,
                  work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int porfs(char uplo, blas_int n, blas_int nrhs, double* a, blas_int lda, double* factors,
                                    blas_int factor_lda, double* b, blas_int ldb, double* x, blas_int ldx,
                                    double* forward_error, double* backward_error, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dporfs_, uplo, n, nrhs, a, lda, factors, factor_lda, b, ldb, x, ldx, forward_error,
                          backward_error, work, iwork);
  detail::dporfs_(&uplo, &n, &nrhs, a, &lda, factors, &factor_lda, b, &ldb, x, &ldx, forward_error, backward_error,
                  work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int posvx(char fact, char uplo, blas_int n, blas_int nrhs, float* a, blas_int lda, float* af,
                                    blas_int ldaf, char& equed, float* scale, float* b, blas_int ldb, float* x,
                                    blas_int ldx, float& rcond, float* forward_error, float* backward_error,
                                    float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, sposvx_, fact, uplo, n, nrhs, a, lda, af, ldaf, equed, scale, b, ldb, x, ldx, work,
                          iwork);
  detail::sposvx_(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, &equed, scale, b, &ldb, x, &ldx, &rcond, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int posvx(char fact, char uplo, blas_int n, blas_int nrhs, double* a, blas_int lda,
                                    double* af, blas_int ldaf, char& equed, double* scale, double* b, blas_int ldb,
                                    double* x, blas_int ldx, double& rcond, double* forward_error,
                                    double* backward_error, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dposvx_, fact, uplo, n, nrhs, a, lda, af, ldaf, equed, scale, b, ldb, x, ldx, work,
                          iwork);
  detail::dposvx_(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, &equed, scale, b, &ldb, x, &ldx, &rcond, forward_error,
                  backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int poequ(blas_int n, float* a, blas_int lda, float* scale, float& scale_condition,
                                    float& max_abs)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, spoequ_, n, a, lda, scale);
  detail::spoequ_(&n, a, &lda, scale, &scale_condition, &max_abs, &info);
  return info;
}

[[nodiscard]] inline blas_int poequ(blas_int n, double* a, blas_int lda, double* scale, double& scale_condition,
                                    double& max_abs)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dpoequ_, n, a, lda, scale);
  detail::dpoequ_(&n, a, &lda, scale, &scale_condition, &max_abs, &info);
  return info;
}

[[nodiscard]] inline blas_int pocon(char uplo, blas_int n, float* a, blas_int lda, float matrix_one_norm, float& rcond,
                                    float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, spocon_, uplo, n, a, lda, matrix_one_norm, work, iwork);
  detail::spocon_(&uplo, &n, a, &lda, &matrix_one_norm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int pocon(char uplo, blas_int n, double* a, blas_int lda, double matrix_one_norm,
                                    double& rcond, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dpocon_, uplo, n, a, lda, matrix_one_norm, work, iwork);
  detail::dpocon_(&uplo, &n, a, &lda, &matrix_one_norm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sytrf(char uplo, blas_int n, float* a, blas_int lda, blas_int* ipiv, float* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssytrf_, uplo, n, a, lda, ipiv, work, lwork);
  detail::ssytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sytrf(char uplo, blas_int n, double* a, blas_int lda, blas_int* ipiv, double* work,
                                    blas_int lwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsytrf_, uplo, n, a, lda, ipiv, work, lwork);
  detail::dsytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sytrs(char uplo, blas_int n, blas_int nrhs, float* a, blas_int lda, blas_int const* ipiv,
                                    float* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssytrs_, uplo, n, nrhs, a, lda, ipiv, b, ldb);
  detail::ssytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int sytrs(char uplo, blas_int n, blas_int nrhs, double* a, blas_int lda, blas_int const* ipiv,
                                    double* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsytrs_, uplo, n, nrhs, a, lda, ipiv, b, ldb);
  detail::dsytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int sytri(char uplo, blas_int n, float* a, blas_int lda, blas_int const* ipiv, float* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssytri_, uplo, n, a, lda, ipiv, work);
  detail::ssytri_(&uplo, &n, a, &lda, ipiv, work, &info);
  return info;
}

[[nodiscard]] inline blas_int sytri(char uplo, blas_int n, double* a, blas_int lda, blas_int const* ipiv, double* work)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsytri_, uplo, n, a, lda, ipiv, work);
  detail::dsytri_(&uplo, &n, a, &lda, ipiv, work, &info);
  return info;
}

[[nodiscard]] inline blas_int syrfs(char uplo, blas_int n, blas_int nrhs, float* a, blas_int lda, float* factors,
                                    blas_int factor_lda, blas_int const* ipiv, float* b, blas_int ldb, float* x,
                                    blas_int ldx, float* forward_error, float* backward_error, float* work,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssyrfs_, uplo, n, nrhs, a, lda, factors, factor_lda, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  detail::ssyrfs_(&uplo, &n, &nrhs, a, &lda, factors, &factor_lda, const_cast<blas_int*>(ipiv), b, &ldb, x, &ldx,
                  forward_error, backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int syrfs(char uplo, blas_int n, blas_int nrhs, double* a, blas_int lda, double* factors,
                                    blas_int factor_lda, blas_int const* ipiv, double* b, blas_int ldb, double* x,
                                    blas_int ldx, double* forward_error, double* backward_error, double* work,
                                    blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsyrfs_, uplo, n, nrhs, a, lda, factors, factor_lda, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  detail::dsyrfs_(&uplo, &n, &nrhs, a, &lda, factors, &factor_lda, const_cast<blas_int*>(ipiv), b, &ldb, x, &ldx,
                  forward_error, backward_error, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sysvx(char fact, char uplo, blas_int n, blas_int nrhs, float* a, blas_int lda, float* af,
                                    blas_int ldaf, blas_int* ipiv, float* b, blas_int ldb, float* x, blas_int ldx,
                                    float& rcond, float* forward_error, float* backward_error, float* work,
                                    blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssysvx_, fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, work, lwork,
                          iwork);
  detail::ssysvx_(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b, &ldb, x, &ldx, &rcond, forward_error,
                  backward_error, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sysvx(char fact, char uplo, blas_int n, blas_int nrhs, double* a, blas_int lda,
                                    double* af, blas_int ldaf, blas_int* ipiv, double* b, blas_int ldb, double* x,
                                    blas_int ldx, double& rcond, double* forward_error, double* backward_error,
                                    double* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsysvx_, fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, work, lwork,
                          iwork);
  detail::dsysvx_(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b, &ldb, x, &ldx, &rcond, forward_error,
                  backward_error, work, &lwork, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sycon(char uplo, blas_int n, float* a, blas_int lda, blas_int const* ipiv,
                                    float matrix_one_norm, float& rcond, float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, ssycon_, uplo, n, a, lda, ipiv, matrix_one_norm, work, iwork);
  detail::ssycon_(&uplo, &n, a, &lda, ipiv, &matrix_one_norm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int sycon(char uplo, blas_int n, double* a, blas_int lda, blas_int const* ipiv,
                                    double matrix_one_norm, double& rcond, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dsycon_, uplo, n, a, lda, ipiv, matrix_one_norm, work, iwork);
  detail::dsycon_(&uplo, &n, a, &lda, ipiv, &matrix_one_norm, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int trtrs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, float* a, blas_int lda,
                                    float* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, strtrs_, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
  detail::strtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int trtrs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, double* a,
                                    blas_int lda, double* b, blas_int ldb)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtrtrs_, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
  detail::dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
  return info;
}

[[nodiscard]] inline blas_int trrfs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, float* a, blas_int lda,
                                    float* b, blas_int ldb, float* x, blas_int ldx, float* forward_error,
                                    float* backward_error, float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, strrfs_, uplo, trans, diag, n, nrhs, a, lda, b, ldb, x, ldx, forward_error,
                          backward_error, work, iwork);
  detail::strrfs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, x, &ldx, forward_error, backward_error, work,
                  iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int trrfs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, double* a,
                                    blas_int lda, double* b, blas_int ldb, double* x, blas_int ldx,
                                    double* forward_error, double* backward_error, double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtrrfs_, uplo, trans, diag, n, nrhs, a, lda, b, ldb, x, ldx, forward_error,
                          backward_error, work, iwork);
  detail::dtrrfs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, x, &ldx, forward_error, backward_error, work,
                  iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int trtri(char uplo, char diag, blas_int n, float* a, blas_int lda)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, strtri_, uplo, diag, n, a, lda);
  detail::strtri_(&uplo, &diag, &n, a, &lda, &info);
  return info;
}

[[nodiscard]] inline blas_int trtri(char uplo, char diag, blas_int n, double* a, blas_int lda)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtrtri_, uplo, diag, n, a, lda);
  detail::dtrtri_(&uplo, &diag, &n, a, &lda, &info);
  return info;
}

[[nodiscard]] inline blas_int trcon(char norm, char uplo, char diag, blas_int n, float* a, blas_int lda, float& rcond,
                                    float* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, strcon_, norm, uplo, diag, n, a, lda, work, iwork);
  detail::strcon_(&norm, &uplo, &diag, &n, a, &lda, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int trcon(char norm, char uplo, char diag, blas_int n, double* a, blas_int lda, double& rcond,
                                    double* work, blas_int* iwork)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtrcon_, norm, uplo, diag, n, a, lda, work, iwork);
  detail::dtrcon_(&norm, &uplo, &diag, &n, a, &lda, &rcond, work, iwork, &info);
  return info;
}

[[nodiscard]] inline blas_int trsyl(char trans_a, char trans_b, blas_int sign, blas_int m, blas_int n, float* a,
                                    blas_int lda, float* b, blas_int ldb, float* c, blas_int ldc, float& scale)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, strsyl_, trans_a, trans_b, sign, m, n, a, lda, b, ldb, c, ldc);
  detail::strsyl_(&trans_a, &trans_b, &sign, &m, &n, a, &lda, b, &ldb, c, &ldc, &scale, &info);
  return info;
}

[[nodiscard]] inline blas_int trsyl(char trans_a, char trans_b, blas_int sign, blas_int m, blas_int n, double* a,
                                    blas_int lda, double* b, blas_int ldb, double* c, blas_int ldc, double& scale)
{
  blas_int info = 0;
  UNI20_EXTERNAL_API_CALL(LAPACK, dtrsyl_, trans_a, trans_b, sign, m, n, a, lda, b, ldb, c, ldc);
  detail::dtrsyl_(&trans_a, &trans_b, &sign, &m, &n, a, &lda, b, &ldb, c, &ldc, &scale, &info);
  return info;
}

} // namespace uni20::lapack::unchecked
