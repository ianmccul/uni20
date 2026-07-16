#pragma once

/**
 * \file general.hpp
 * \ingroup backend_lapack_mplapack
 * \brief MPLAPACK binary128 wrappers for real and complex dense general linear systems.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/backend/lapack/common.hpp>
#include <uni20/backend/lapack/mplapack/common.hpp>
#include <uni20/config.hpp>
#include <uni20/core/types.hpp>

#include <memory>

#if !(UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK)
#error "uni20/backend/lapack/mplapack/general.hpp requires MPLAPACK float128 support"
#endif

// clang-format off
#include <mplapack_config.h>
#include <mplapack_binary128.h>
// clang-format on

namespace uni20::lapack::unchecked
{

[[nodiscard]] inline blas_int gesv(blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda, blas_int* ipiv,
                                   uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgesv, n, nrhs, a, lda, ipiv, b, ldb);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgesv(static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda),
        mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gesv(blas_int n, blas_int nrhs, uni20::complex<uni20::float128>* a, blas_int lda,
                                   blas_int* ipiv, uni20::complex<uni20::float128>* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgesv, n, nrhs, a, lda, ipiv, b, ldb);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Cgesv(static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda),
        mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gesvx(char fact, char trans, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* af, blas_int ldaf, blas_int* ipiv, char& equed,
                                    uni20::float128* row_scale, uni20::float128* column_scale, uni20::float128* b,
                                    blas_int ldb, uni20::float128* x, blas_int ldx, uni20::float128& rcond,
                                    uni20::float128* forward_error, uni20::float128* backward_error,
                                    uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgesvx, fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, row_scale, column_scale,
                          b, ldb, x, ldx, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgesvx(&fact, &trans, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda),
         af, static_cast<mplapackint>(ldaf), mplapack_ipiv.data(), &equed, row_scale, column_scale, b,
         static_cast<mplapackint>(ldb), x, static_cast<mplapackint>(ldx), rcond, forward_error, backward_error, work,
         mplapack_iwork.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int geequ(blas_int m, blas_int n, uni20::float128* a, blas_int lda,
                                    uni20::float128* row_scale, uni20::float128* column_scale,
                                    uni20::float128& row_condition, uni20::float128& column_condition,
                                    uni20::float128& max_abs)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgeequ, m, n, a, lda, row_scale, column_scale);
  mplapackint mplapack_info = 0;
  Rgeequ(static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), row_scale,
         column_scale, row_condition, column_condition, max_abs, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int getrf(blas_int m, blas_int n, uni20::float128* a, blas_int lda, blas_int* ipiv)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgetrf, m, n, a, lda, ipiv);
  mplapackint mplapack_info = 0;
  blas_int const pivot_count = std::min(m, n);
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::make_mplapack_int_work(pivot_count);
  Rgetrf(static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda),
         mplapack_ipiv.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, pivot_count);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int getrf(blas_int m, blas_int n, uni20::complex<uni20::float128>* a, blas_int lda,
                                    blas_int* ipiv)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgetrf, m, n, a, lda, ipiv);
  mplapackint mplapack_info = 0;
  blas_int const pivot_count = std::min(m, n);
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::make_mplapack_int_work(pivot_count);
  Cgetrf(static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda),
         mplapack_ipiv.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, pivot_count);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int getrs(char trans, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    blas_int const* ipiv, uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgetrs, trans, n, nrhs, a, lda, ipiv, b, ldb);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  Rgetrs(&trans, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda),
         mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int getrs(char trans, blas_int n, blas_int nrhs, uni20::complex<uni20::float128>* a,
                                    blas_int lda, blas_int const* ipiv, uni20::complex<uni20::float128>* b,
                                    blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgetrs, trans, n, nrhs, a, lda, ipiv, b, ldb);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  Cgetrs(&trans, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda),
         mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gerfs(char trans, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* factors, blas_int factor_lda, blas_int const* ipiv,
                                    uni20::float128* b, blas_int ldb, uni20::float128* x, blas_int ldx,
                                    uni20::float128* forward_error, uni20::float128* backward_error,
                                    uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgerfs, trans, n, nrhs, a, lda, factors, factor_lda, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgerfs(&trans, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda), factors,
         static_cast<mplapackint>(factor_lda), mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), x,
         static_cast<mplapackint>(ldx), forward_error, backward_error, work, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int getri(blas_int n, uni20::float128* a, blas_int lda, blas_int* ipiv, uni20::float128* work,
                                    blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgetri, n, a, lda, ipiv, work, lwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  Rgetri(static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_ipiv.data(), work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int getri(blas_int n, uni20::complex<uni20::float128>* a, blas_int lda, blas_int* ipiv,
                                    uni20::complex<uni20::float128>* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgetri, n, a, lda, ipiv, work, lwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  Cgetri(static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_ipiv.data(), work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gecon(char norm, blas_int n, uni20::float128 const* a, blas_int lda,
                                    uni20::float128 anorm, uni20::float128& rcond, uni20::float128* work,
                                    blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgecon, norm, n, a, lda, anorm, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgecon(&norm, static_cast<mplapackint>(n), const_cast<uni20::float128*>(a), static_cast<mplapackint>(lda), anorm,
         rcond, work, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gecon(char norm, blas_int n, uni20::complex<uni20::float128> const* a, blas_int lda,
                                    uni20::float128 anorm, uni20::float128& rcond,
                                    uni20::complex<uni20::float128>* work, uni20::float128* rwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgecon, norm, n, a, lda, anorm, work, rwork);
  mplapackint mplapack_info = 0;
  Cgecon(&norm, static_cast<mplapackint>(n), const_cast<uni20::complex<uni20::float128>*>(a),
         static_cast<mplapackint>(lda), anorm, rcond, work, rwork, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gels(char trans, blas_int m, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                   uni20::float128* b, blas_int ldb, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgels, trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
  mplapackint mplapack_info = 0;
  Rgels(&trans, static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a,
        static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), work, static_cast<mplapackint>(lwork),
        mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gelss(blas_int m, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* b, blas_int ldb, uni20::float128* s, uni20::float128 rcond,
                                    blas_int& rank, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgelss, m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork);
  mplapackint mplapack_rank = 0;
  mplapackint mplapack_info = 0;
  Rgelss(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a,
         static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), s, rcond, mplapack_rank, work,
         static_cast<mplapackint>(lwork), mplapack_info);
  rank = static_cast<blas_int>(mplapack_rank);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gelsd(blas_int m, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* b, blas_int ldb, uni20::float128* s, uni20::float128 rcond,
                                    blas_int& rank, uni20::float128* work, blas_int lwork, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgelsd, m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork);
  mplapackint mplapack_rank = 0;
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = uni20::lapack::mplapack::detail::gelsd_iwork_size(m, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Rgelsd(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a,
         static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), s, rcond, mplapack_rank, work,
         static_cast<mplapackint>(lwork), mplapack_iwork.data(), mplapack_info);
  rank = static_cast<blas_int>(mplapack_rank);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gelsy(blas_int m, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* b, blas_int ldb, blas_int* jpvt, uni20::float128 rcond,
                                    blas_int& rank, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgelsy, m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank, work, lwork);
  mplapackint mplapack_rank = 0;
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_jpvt = uni20::lapack::mplapack::detail::to_mplapack_ints(jpvt, n);
  Rgelsy(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a,
         static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), mplapack_jpvt.data(), rcond, mplapack_rank,
         work, static_cast<mplapackint>(lwork), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_jpvt, jpvt, n);
  rank = static_cast<blas_int>(mplapack_rank);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int geqrf(blas_int m, blas_int n, uni20::float128* a, blas_int lda, uni20::float128* tau,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgeqrf, m, n, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rgeqrf(static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), tau, work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gelqf(blas_int m, blas_int n, uni20::float128* a, blas_int lda, uni20::float128* tau,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgelqf, m, n, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rgelqf(static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), tau, work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int geqlf(blas_int m, blas_int n, uni20::float128* a, blas_int lda, uni20::float128* tau,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgeqlf, m, n, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rgeqlf(static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), tau, work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gerqf(blas_int m, blas_int n, uni20::float128* a, blas_int lda, uni20::float128* tau,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgerqf, m, n, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rgerqf(static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), tau, work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int orgqr(blas_int m, blas_int n, blas_int k, uni20::float128* a, blas_int lda,
                                    uni20::float128* tau, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rorgqr, m, n, k, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rorgqr(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, work, static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int orglq(blas_int m, blas_int n, blas_int k, uni20::float128* a, blas_int lda,
                                    uni20::float128* tau, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rorglq, m, n, k, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rorglq(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, work, static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int orgql(blas_int m, blas_int n, blas_int k, uni20::float128* a, blas_int lda,
                                    uni20::float128* tau, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rorgql, m, n, k, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rorgql(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, work, static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int orgrq(blas_int m, blas_int n, blas_int k, uni20::float128* a, blas_int lda,
                                    uni20::float128* tau, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rorgrq, m, n, k, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rorgrq(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, work, static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ormqr(char side, char trans, blas_int m, blas_int n, blas_int k, uni20::float128* a,
                                    blas_int lda, uni20::float128* tau, uni20::float128* c, blas_int ldc,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rormqr, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  mplapackint mplapack_info = 0;
  Rormqr(&side, &trans, static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, c, static_cast<mplapackint>(ldc), work, static_cast<mplapackint>(lwork),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ormlq(char side, char trans, blas_int m, blas_int n, blas_int k, uni20::float128* a,
                                    blas_int lda, uni20::float128* tau, uni20::float128* c, blas_int ldc,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rormlq, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  mplapackint mplapack_info = 0;
  Rormlq(&side, &trans, static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, c, static_cast<mplapackint>(ldc), work, static_cast<mplapackint>(lwork),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ormql(char side, char trans, blas_int m, blas_int n, blas_int k, uni20::float128* a,
                                    blas_int lda, uni20::float128* tau, uni20::float128* c, blas_int ldc,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rormql, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  mplapackint mplapack_info = 0;
  Rormql(&side, &trans, static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, c, static_cast<mplapackint>(ldc), work, static_cast<mplapackint>(lwork),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ormrq(char side, char trans, blas_int m, blas_int n, blas_int k, uni20::float128* a,
                                    blas_int lda, uni20::float128* tau, uni20::float128* c, blas_int ldc,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rormrq, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  mplapackint mplapack_info = 0;
  Rormrq(&side, &trans, static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, c, static_cast<mplapackint>(ldc), work, static_cast<mplapackint>(lwork),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gebrd(blas_int m, blas_int n, uni20::float128* a, blas_int lda, uni20::float128* d,
                                    uni20::float128* e, uni20::float128* tauq, uni20::float128* taup,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgebrd, m, n, a, lda, d, e, tauq, taup, work, lwork);
  mplapackint mplapack_info = 0;
  Rgebrd(static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), d, e, tauq, taup,
         work, static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gehrd(blas_int n, blas_int first, blas_int last, uni20::float128* a, blas_int lda,
                                    uni20::float128* tau, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgehrd, n, first, last, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rgehrd(static_cast<mplapackint>(n), static_cast<mplapackint>(first), static_cast<mplapackint>(last), a,
         static_cast<mplapackint>(lda), tau, work, static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int geqp3(blas_int m, blas_int n, uni20::float128* a, blas_int lda, blas_int* jpvt,
                                    uni20::float128* tau, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgeqp3, m, n, a, lda, jpvt, tau, work, lwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_jpvt = uni20::lapack::mplapack::detail::to_mplapack_ints(jpvt, n);
  Rgeqp3(static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda),
         mplapack_jpvt.data(), tau, work, static_cast<mplapackint>(lwork), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_jpvt, jpvt, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sytrd(char uplo, blas_int n, uni20::float128* a, blas_int lda, uni20::float128* d,
                                    uni20::float128* e, uni20::float128* tau, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsytrd, uplo, n, a, lda, d, e, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rsytrd(&uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), d, e, tau, work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int syev(char jobz, char uplo, blas_int n, uni20::float128* a, blas_int lda,
                                   uni20::float128* w, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsyev, jobz, uplo, n, a, lda, w, work, lwork);
  mplapackint mplapack_info = 0;
  Rsyev(&jobz, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), w, work,
        static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int syevd(char jobz, char uplo, blas_int n, uni20::float128* a, blas_int lda,
                                    uni20::float128* w, uni20::float128* work, blas_int lwork, blas_int* iwork,
                                    blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsyevd, jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Rsyevd(&jobz, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), w, work,
         static_cast<mplapackint>(lwork), mplapack_iwork.data(), static_cast<mplapackint>(liwork), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_query_int_work(mplapack_iwork, iwork, liwork);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int syevr(char jobz, char range, char uplo, blas_int n, uni20::float128* a, blas_int lda,
                                    uni20::float128 vl, uni20::float128 vu, blas_int il, blas_int iu,
                                    uni20::float128 abstol, blas_int& selected_count, uni20::float128* w,
                                    uni20::float128* z, blas_int ldz, blas_int* isuppz, uni20::float128* work,
                                    blas_int lwork, blas_int* iwork, blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsyevr, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, selected_count, w, z,
                          ldz, isuppz, work, lwork, iwork, liwork);
  mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
  mplapackint mplapack_info = 0;
  blas_int const support_size = 2 * std::max<blas_int>(1, n);
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::vector<mplapackint> mplapack_isuppz;
  if (isuppz != nullptr)
  {
    mplapack_isuppz = uni20::lapack::mplapack::detail::make_mplapack_int_work(support_size);
  }
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Rsyevr(&jobz, &range, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), vl, vu,
         static_cast<mplapackint>(il), static_cast<mplapackint>(iu), abstol, mplapack_selected_count, w, z,
         static_cast<mplapackint>(ldz), isuppz == nullptr ? nullptr : mplapack_isuppz.data(), work,
         static_cast<mplapackint>(lwork), mplapack_iwork.data(), static_cast<mplapackint>(liwork), mplapack_info);
  selected_count = static_cast<blas_int>(mplapack_selected_count);
  if (isuppz != nullptr)
  {
    uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_isuppz, isuppz, support_size);
  }
  uni20::lapack::mplapack::detail::copy_from_mplapack_query_int_work(mplapack_iwork, iwork, liwork);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sygv(blas_int itype, char jobz, char uplo, blas_int n, uni20::float128* a, blas_int lda,
                                   uni20::float128* b, blas_int ldb, uni20::float128* w, uni20::float128* work,
                                   blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsygv, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork);
  mplapackint mplapack_info = 0;
  Rsygv(static_cast<mplapackint>(itype), &jobz, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), b,
        static_cast<mplapackint>(ldb), w, work, static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sygvd(blas_int itype, char jobz, char uplo, blas_int n, uni20::float128* a, blas_int lda,
                                    uni20::float128* b, blas_int ldb, uni20::float128* w, uni20::float128* work,
                                    blas_int lwork, blas_int* iwork, blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsygvd, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, iwork, liwork);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Rsygvd(static_cast<mplapackint>(itype), &jobz, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda),
         b, static_cast<mplapackint>(ldb), w, work, static_cast<mplapackint>(lwork), mplapack_iwork.data(),
         static_cast<mplapackint>(liwork), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_query_int_work(mplapack_iwork, iwork, liwork);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sygvx(blas_int itype, char jobz, char range, char uplo, blas_int n, uni20::float128* a,
                                    blas_int lda, uni20::float128* b, blas_int ldb, uni20::float128 vl,
                                    uni20::float128 vu, blas_int il, blas_int iu, uni20::float128 abstol,
                                    blas_int& selected_count, uni20::float128* w, uni20::float128* z, blas_int ldz,
                                    uni20::float128* work, blas_int lwork, blas_int* iwork, blas_int* ifail)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsygvx, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol,
                          selected_count, w, z, ldz, work, lwork, iwork, ifail);
  mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = 5 * std::max<blas_int>(1, n);
  blas_int const fail_size = std::max<blas_int>(1, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  std::vector<mplapackint> mplapack_ifail;
  if (ifail != nullptr)
  {
    mplapack_ifail = uni20::lapack::mplapack::detail::make_mplapack_int_work(fail_size);
  }
  Rsygvx(static_cast<mplapackint>(itype), &jobz, &range, &uplo, static_cast<mplapackint>(n), a,
         static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), vl, vu, static_cast<mplapackint>(il),
         static_cast<mplapackint>(iu), abstol, mplapack_selected_count, w, z, static_cast<mplapackint>(ldz), work,
         static_cast<mplapackint>(lwork), mplapack_iwork.data(), ifail == nullptr ? nullptr : mplapack_ifail.data(),
         mplapack_info);
  selected_count = static_cast<blas_int>(mplapack_selected_count);
  if (ifail != nullptr)
  {
    uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ifail, ifail, fail_size);
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int heev(char jobz, char uplo, blas_int n, uni20::complex<uni20::float128>* a, blas_int lda,
                                   uni20::float128* w, uni20::complex<uni20::float128>* work, blas_int lwork,
                                   uni20::float128* rwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cheev, jobz, uplo, n, a, lda, w, work, lwork, rwork);
  mplapackint mplapack_info = 0;
  Cheev(&jobz, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), w, work,
        static_cast<mplapackint>(lwork), rwork, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int heevd(char jobz, char uplo, blas_int n, uni20::complex<uni20::float128>* a, blas_int lda,
                                    uni20::float128* w, uni20::complex<uni20::float128>* work, blas_int lwork,
                                    uni20::float128* rwork, blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cheevd, jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Cheevd(&jobz, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), w, work,
         static_cast<mplapackint>(lwork), rwork, static_cast<mplapackint>(lrwork), mplapack_iwork.data(),
         static_cast<mplapackint>(liwork), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_query_int_work(mplapack_iwork, iwork, liwork);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int heevr(char jobz, char range, char uplo, blas_int n, uni20::complex<uni20::float128>* a,
                                    blas_int lda, uni20::float128 vl, uni20::float128 vu, blas_int il, blas_int iu,
                                    uni20::float128 abstol, blas_int& selected_count, uni20::float128* w,
                                    uni20::complex<uni20::float128>* z, blas_int ldz, blas_int* isuppz,
                                    uni20::complex<uni20::float128>* work, blas_int lwork, uni20::float128* rwork,
                                    blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cheevr, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, selected_count, w, z,
                          ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork);
  mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
  mplapackint mplapack_info = 0;
  blas_int const support_size = 2 * std::max<blas_int>(1, n);
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::vector<mplapackint> mplapack_isuppz;
  if (isuppz != nullptr)
  {
    mplapack_isuppz = uni20::lapack::mplapack::detail::make_mplapack_int_work(support_size);
  }
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Cheevr(&jobz, &range, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), vl, vu,
         static_cast<mplapackint>(il), static_cast<mplapackint>(iu), abstol, mplapack_selected_count, w, z,
         static_cast<mplapackint>(ldz), isuppz == nullptr ? nullptr : mplapack_isuppz.data(), work,
         static_cast<mplapackint>(lwork), rwork, static_cast<mplapackint>(lrwork), mplapack_iwork.data(),
         static_cast<mplapackint>(liwork), mplapack_info);
  selected_count = static_cast<blas_int>(mplapack_selected_count);
  if (isuppz != nullptr)
  {
    uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_isuppz, isuppz, support_size);
  }
  uni20::lapack::mplapack::detail::copy_from_mplapack_query_int_work(mplapack_iwork, iwork, liwork);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int hegv(blas_int itype, char jobz, char uplo, blas_int n, uni20::complex<uni20::float128>* a,
                                   blas_int lda, uni20::complex<uni20::float128>* b, blas_int ldb, uni20::float128* w,
                                   uni20::complex<uni20::float128>* work, blas_int lwork, uni20::float128* rwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Chegv, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork);
  mplapackint mplapack_info = 0;
  Chegv(static_cast<mplapackint>(itype), &jobz, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), b,
        static_cast<mplapackint>(ldb), w, work, static_cast<mplapackint>(lwork), rwork, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int hegvd(blas_int itype, char jobz, char uplo, blas_int n,
                                    uni20::complex<uni20::float128>* a, blas_int lda,
                                    uni20::complex<uni20::float128>* b, blas_int ldb, uni20::float128* w,
                                    uni20::complex<uni20::float128>* work, blas_int lwork, uni20::float128* rwork,
                                    blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Chegvd, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork, lrwork, iwork,
                          liwork);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Chegvd(static_cast<mplapackint>(itype), &jobz, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda),
         b, static_cast<mplapackint>(ldb), w, work, static_cast<mplapackint>(lwork), rwork,
         static_cast<mplapackint>(lrwork), mplapack_iwork.data(), static_cast<mplapackint>(liwork), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_query_int_work(mplapack_iwork, iwork, liwork);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int hegvx(blas_int itype, char jobz, char range, char uplo, blas_int n,
                                    uni20::complex<uni20::float128>* a, blas_int lda,
                                    uni20::complex<uni20::float128>* b, blas_int ldb, uni20::float128 vl,
                                    uni20::float128 vu, blas_int il, blas_int iu, uni20::float128 abstol,
                                    blas_int& selected_count, uni20::float128* w, uni20::complex<uni20::float128>* z,
                                    blas_int ldz, uni20::complex<uni20::float128>* work, blas_int lwork,
                                    uni20::float128* rwork, blas_int* iwork, blas_int* ifail)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Chegvx, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol,
                          selected_count, w, z, ldz, work, lwork, rwork, iwork, ifail);
  mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = 5 * std::max<blas_int>(1, n);
  blas_int const fail_size = std::max<blas_int>(1, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  std::vector<mplapackint> mplapack_ifail;
  if (ifail != nullptr)
  {
    mplapack_ifail = uni20::lapack::mplapack::detail::make_mplapack_int_work(fail_size);
  }
  Chegvx(static_cast<mplapackint>(itype), &jobz, &range, &uplo, static_cast<mplapackint>(n), a,
         static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), vl, vu, static_cast<mplapackint>(il),
         static_cast<mplapackint>(iu), abstol, mplapack_selected_count, w, z, static_cast<mplapackint>(ldz), work,
         static_cast<mplapackint>(lwork), rwork, mplapack_iwork.data(),
         ifail == nullptr ? nullptr : mplapack_ifail.data(), mplapack_info);
  selected_count = static_cast<blas_int>(mplapack_selected_count);
  if (ifail != nullptr)
  {
    uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ifail, ifail, fail_size);
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int geev(char jobvl, char jobvr, blas_int n, uni20::float128* a, blas_int lda,
                                   uni20::float128* wr, uni20::float128* wi, uni20::float128* vl, blas_int ldvl,
                                   uni20::float128* vr, blas_int ldvr, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgeev, jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork);
  mplapackint mplapack_info = 0;
  Rgeev(&jobvl, &jobvr, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), wr, wi, vl,
        static_cast<mplapackint>(ldvl), vr, static_cast<mplapackint>(ldvr), work, static_cast<mplapackint>(lwork),
        mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int geevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, uni20::float128* a,
                                    blas_int lda, uni20::float128* wr, uni20::float128* wi, uni20::float128* vl,
                                    blas_int ldvl, uni20::float128* vr, blas_int ldvr, blas_int& ilo, blas_int& ihi,
                                    uni20::float128* scale, uni20::float128& abnrm, uni20::float128* rconde,
                                    uni20::float128* rcondv, uni20::float128* work, blas_int lwork, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgeevx, balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi,
                          scale, abnrm, rconde, rcondv, work, lwork, iwork);
  mplapackint mplapack_ilo = static_cast<mplapackint>(ilo);
  mplapackint mplapack_ihi = static_cast<mplapackint>(ihi);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = std::max<blas_int>(1, 2 * n - 2);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Rgeevx(&balanc, &jobvl, &jobvr, &sense, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), wr, wi, vl,
         static_cast<mplapackint>(ldvl), vr, static_cast<mplapackint>(ldvr), mplapack_ilo, mplapack_ihi, scale, abnrm,
         rconde, rcondv, work, static_cast<mplapackint>(lwork), mplapack_iwork.data(), mplapack_info);
  ilo = static_cast<blas_int>(mplapack_ilo);
  ihi = static_cast<blas_int>(mplapack_ihi);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gebal(char job, blas_int n, uni20::float128* a, blas_int lda, blas_int& first,
                                    blas_int& last, uni20::float128* scale)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgebal, job, n, a, lda, first, last, scale);
  mplapackint mplapack_first = static_cast<mplapackint>(first);
  mplapackint mplapack_last = static_cast<mplapackint>(last);
  mplapackint mplapack_info = 0;
  Rgebal(&job, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_first, mplapack_last, scale,
         mplapack_info);
  first = static_cast<blas_int>(mplapack_first);
  last = static_cast<blas_int>(mplapack_last);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gebak(char job, char side, blas_int n, blas_int first, blas_int last,
                                    uni20::float128* scale, blas_int vector_count, uni20::float128* vectors,
                                    blas_int leading_dimension)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgebak, job, side, n, first, last, scale, vector_count, vectors, leading_dimension);
  mplapackint mplapack_info = 0;
  Rgebak(&job, &side, static_cast<mplapackint>(n), static_cast<mplapackint>(first), static_cast<mplapackint>(last),
         scale, static_cast<mplapackint>(vector_count), vectors, static_cast<mplapackint>(leading_dimension),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int geev(char jobvl, char jobvr, blas_int n, uni20::complex<uni20::float128>* a, blas_int lda,
                                   uni20::complex<uni20::float128>* w, uni20::complex<uni20::float128>* vl,
                                   blas_int ldvl, uni20::complex<uni20::float128>* vr, blas_int ldvr,
                                   uni20::complex<uni20::float128>* work, blas_int lwork, uni20::float128* rwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgeev, jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork);
  mplapackint mplapack_info = 0;
  Cgeev(&jobvl, &jobvr, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), w, vl,
        static_cast<mplapackint>(ldvl), vr, static_cast<mplapackint>(ldvr), work, static_cast<mplapackint>(lwork),
        rwork, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gees(char jobvs, char sort, blas_int n, uni20::float128* a, blas_int lda,
                                   blas_int& selected_dimension, uni20::float128* wr, uni20::float128* wi,
                                   uni20::float128* vs, blas_int ldvs, uni20::float128* work, blas_int lwork,
                                   blas_int* bwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgees, jobvs, sort, n, a, lda, selected_dimension, wr, wi, vs, ldvs, work, lwork,
                          bwork);
  mplapackint mplapack_selected_dimension = static_cast<mplapackint>(selected_dimension);
  mplapackint mplapack_info = 0;
  blas_int const bwork_size = std::max<blas_int>(1, n);
  std::unique_ptr<bool[]> mplapack_bwork;
  if (bwork != nullptr)
  {
    mplapack_bwork = std::make_unique<bool[]>(static_cast<std::size_t>(bwork_size));
    for (blas_int i = 0; i < n; ++i)
    {
      mplapack_bwork[static_cast<std::size_t>(i)] = bwork[i] != 0;
    }
  }
  Rgees(&jobvs, &sort, nullptr, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda),
        mplapack_selected_dimension, wr, wi, vs, static_cast<mplapackint>(ldvs), work, static_cast<mplapackint>(lwork),
        mplapack_bwork.get(), mplapack_info);
  selected_dimension = static_cast<blas_int>(mplapack_selected_dimension);
  if (bwork != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      bwork[i] = mplapack_bwork[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gees(char jobvs, char sort, blas_int n, uni20::complex<uni20::float128>* a, blas_int lda,
                                   blas_int& selected_dimension, uni20::complex<uni20::float128>* w,
                                   uni20::complex<uni20::float128>* vs, blas_int ldvs,
                                   uni20::complex<uni20::float128>* work, blas_int lwork, uni20::float128* rwork,
                                   blas_int* bwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgees, jobvs, sort, n, a, lda, selected_dimension, w, vs, ldvs, work, lwork, rwork,
                          bwork);
  mplapackint mplapack_selected_dimension = static_cast<mplapackint>(selected_dimension);
  mplapackint mplapack_info = 0;
  blas_int const bwork_size = std::max<blas_int>(1, n);
  std::unique_ptr<bool[]> mplapack_bwork;
  if (bwork != nullptr)
  {
    mplapack_bwork = std::make_unique<bool[]>(static_cast<std::size_t>(bwork_size));
    for (blas_int i = 0; i < n; ++i)
    {
      mplapack_bwork[static_cast<std::size_t>(i)] = bwork[i] != 0;
    }
  }
  Cgees(&jobvs, &sort, nullptr, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda),
        mplapack_selected_dimension, w, vs, static_cast<mplapackint>(ldvs), work, static_cast<mplapackint>(lwork),
        rwork, mplapack_bwork.get(), mplapack_info);
  selected_dimension = static_cast<blas_int>(mplapack_selected_dimension);
  if (bwork != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      bwork[i] = mplapack_bwork[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int hseqr(char job, char compz, blas_int n, blas_int first, blas_int last, uni20::float128* h,
                                    blas_int ldh, uni20::float128* wr, uni20::float128* wi, uni20::float128* z,
                                    blas_int ldz, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rhseqr, job, compz, n, first, last, h, ldh, wr, wi, z, ldz, work, lwork);
  mplapackint mplapack_info = 0;
  Rhseqr(&job, &compz, static_cast<mplapackint>(n), static_cast<mplapackint>(first), static_cast<mplapackint>(last), h,
         static_cast<mplapackint>(ldh), wr, wi, z, static_cast<mplapackint>(ldz), work, static_cast<mplapackint>(lwork),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trexc(char compq, blas_int n, uni20::float128* t, blas_int ldt, uni20::float128* q,
                                    blas_int ldq, blas_int& first, blas_int& last, uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtrexc, compq, n, t, ldt, q, ldq, first, last, work);
  mplapackint mplapack_first = static_cast<mplapackint>(first);
  mplapackint mplapack_last = static_cast<mplapackint>(last);
  mplapackint mplapack_info = 0;
  Rtrexc(&compq, static_cast<mplapackint>(n), t, static_cast<mplapackint>(ldt), q, static_cast<mplapackint>(ldq),
         mplapack_first, mplapack_last, work, mplapack_info);
  first = static_cast<blas_int>(mplapack_first);
  last = static_cast<blas_int>(mplapack_last);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trexc(char compq, blas_int n, uni20::complex<uni20::float128>* t, blas_int ldt,
                                    uni20::complex<uni20::float128>* q, blas_int ldq, blas_int& first, blas_int& last)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Ctrexc, compq, n, t, ldt, q, ldq, first, last);
  mplapackint mplapack_first = static_cast<mplapackint>(first);
  mplapackint mplapack_last = static_cast<mplapackint>(last);
  mplapackint mplapack_info = 0;
  Ctrexc(&compq, static_cast<mplapackint>(n), t, static_cast<mplapackint>(ldt), q, static_cast<mplapackint>(ldq),
         mplapack_first, mplapack_last, mplapack_info);
  first = static_cast<blas_int>(mplapack_first);
  last = static_cast<blas_int>(mplapack_last);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trsen(char job, char compq, blas_int* select, blas_int n, uni20::float128* t,
                                    blas_int ldt, uni20::float128* q, blas_int ldq, uni20::float128* wr,
                                    uni20::float128* wi, blas_int& selected_dimension,
                                    uni20::float128& reciprocal_eigenvalue_cluster_condition,
                                    uni20::float128& reciprocal_invariant_subspace_condition, uni20::float128* work,
                                    blas_int lwork, blas_int* iwork, blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtrsen, job, compq, select, n, t, ldt, q, ldq, wr, wi, selected_dimension,
                          reciprocal_eigenvalue_cluster_condition, reciprocal_invariant_subspace_condition, work, lwork,
                          iwork, liwork);
  mplapackint mplapack_selected_dimension = static_cast<mplapackint>(selected_dimension);
  mplapackint mplapack_info = 0;
  blas_int const select_size = std::max<blas_int>(1, n);
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::unique_ptr<bool[]> mplapack_select;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  if (select != nullptr)
  {
    mplapack_select = std::make_unique<bool[]>(static_cast<std::size_t>(select_size));
    for (blas_int i = 0; i < n; ++i)
    {
      mplapack_select[static_cast<std::size_t>(i)] = select[i] != 0;
    }
  }
  Rtrsen(&job, &compq, mplapack_select.get(), static_cast<mplapackint>(n), t, static_cast<mplapackint>(ldt), q,
         static_cast<mplapackint>(ldq), wr, wi, mplapack_selected_dimension, reciprocal_eigenvalue_cluster_condition,
         reciprocal_invariant_subspace_condition, work, static_cast<mplapackint>(lwork), mplapack_iwork.data(),
         static_cast<mplapackint>(liwork), mplapack_info);
  selected_dimension = static_cast<blas_int>(mplapack_selected_dimension);
  if (select != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      select[i] = mplapack_select[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  uni20::lapack::mplapack::detail::copy_from_mplapack_query_int_work(mplapack_iwork, iwork, liwork);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trevc(char side, char howmny, blas_int* select, blas_int n, uni20::float128* t,
                                    blas_int ldt, uni20::float128* vl, blas_int ldvl, uni20::float128* vr,
                                    blas_int ldvr, blas_int mm, blas_int& computed_vectors, uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtrevc, side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, computed_vectors,
                          work);
  mplapackint mplapack_computed_vectors = static_cast<mplapackint>(computed_vectors);
  mplapackint mplapack_info = 0;
  blas_int const select_size = std::max<blas_int>(1, n);
  std::unique_ptr<bool[]> mplapack_select;
  if (select != nullptr)
  {
    mplapack_select = std::make_unique<bool[]>(static_cast<std::size_t>(select_size));
    for (blas_int i = 0; i < n; ++i)
    {
      mplapack_select[static_cast<std::size_t>(i)] = select[i] != 0;
    }
  }
  Rtrevc(&side, &howmny, mplapack_select.get(), static_cast<mplapackint>(n), t, static_cast<mplapackint>(ldt), vl,
         static_cast<mplapackint>(ldvl), vr, static_cast<mplapackint>(ldvr), static_cast<mplapackint>(mm),
         mplapack_computed_vectors, work, mplapack_info);
  computed_vectors = static_cast<blas_int>(mplapack_computed_vectors);
  if (select != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      select[i] = mplapack_select[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trsna(char job, char howmny, blas_int* select, blas_int n, uni20::float128* t,
                                    blas_int ldt, uni20::float128* vl, blas_int ldvl, uni20::float128* vr,
                                    blas_int ldvr, uni20::float128* reciprocal_eigenvalue_condition_numbers,
                                    uni20::float128* reciprocal_eigenvector_condition_numbers, blas_int mm,
                                    blas_int& computed_estimates, uni20::float128* work, blas_int ldwork,
                                    blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtrsna, job, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr,
                          reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers, mm,
                          computed_estimates, work, ldwork, iwork);
  mplapackint mplapack_computed_estimates = static_cast<mplapackint>(computed_estimates);
  mplapackint mplapack_info = 0;
  blas_int const select_size = std::max<blas_int>(1, n);
  blas_int const iwork_size = std::max<blas_int>(1, 2 * std::max<blas_int>(1, n) - 2);
  std::unique_ptr<bool[]> mplapack_select;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  if (select != nullptr)
  {
    mplapack_select = std::make_unique<bool[]>(static_cast<std::size_t>(select_size));
    for (blas_int i = 0; i < n; ++i)
    {
      mplapack_select[static_cast<std::size_t>(i)] = select[i] != 0;
    }
  }
  Rtrsna(&job, &howmny, mplapack_select.get(), static_cast<mplapackint>(n), t, static_cast<mplapackint>(ldt), vl,
         static_cast<mplapackint>(ldvl), vr, static_cast<mplapackint>(ldvr), reciprocal_eigenvalue_condition_numbers,
         reciprocal_eigenvector_condition_numbers, static_cast<mplapackint>(mm), mplapack_computed_estimates, work,
         static_cast<mplapackint>(ldwork), mplapack_iwork.data(), mplapack_info);
  computed_estimates = static_cast<blas_int>(mplapack_computed_estimates);
  if (select != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      select[i] = mplapack_select[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ggev(char jobvl, char jobvr, blas_int n, uni20::float128* a, blas_int lda,
                                   uni20::float128* b, blas_int ldb, uni20::float128* alphar, uni20::float128* alphai,
                                   uni20::float128* beta, uni20::float128* vl, blas_int ldvl, uni20::float128* vr,
                                   blas_int ldvr, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rggev, jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr,
                          work, lwork);
  mplapackint mplapack_info = 0;
  Rggev(&jobvl, &jobvr, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb),
        alphar, alphai, beta, vl, static_cast<mplapackint>(ldvl), vr, static_cast<mplapackint>(ldvr), work,
        static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ggevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, uni20::float128* a,
                                    blas_int lda, uni20::float128* b, blas_int ldb, uni20::float128* alphar,
                                    uni20::float128* alphai, uni20::float128* beta, uni20::float128* vl, blas_int ldvl,
                                    uni20::float128* vr, blas_int ldvr, blas_int& ilo, blas_int& ihi,
                                    uni20::float128* lscale, uni20::float128* rscale, uni20::float128& abnrm,
                                    uni20::float128& bbnrm, uni20::float128* rconde, uni20::float128* rcondv,
                                    uni20::float128* work, blas_int lwork, blas_int* iwork, blas_int* bwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rggevx, balanc, jobvl, jobvr, sense, n, a, lda, b, ldb, alphar, alphai, beta, vl,
                          ldvl, vr, ldvr, ilo, ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work, lwork, iwork,
                          bwork);
  mplapackint mplapack_ilo = static_cast<mplapackint>(ilo);
  mplapackint mplapack_ihi = static_cast<mplapackint>(ihi);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = std::max<blas_int>(1, n + 6);
  blas_int const bwork_size = std::max<blas_int>(1, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  std::unique_ptr<bool[]> mplapack_bwork;
  if (bwork != nullptr)
  {
    mplapack_bwork = std::make_unique<bool[]>(static_cast<std::size_t>(bwork_size));
    for (blas_int i = 0; i < n; ++i)
    {
      mplapack_bwork[static_cast<std::size_t>(i)] = bwork[i] != 0;
    }
  }
  Rggevx(&balanc, &jobvl, &jobvr, &sense, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), b,
         static_cast<mplapackint>(ldb), alphar, alphai, beta, vl, static_cast<mplapackint>(ldvl), vr,
         static_cast<mplapackint>(ldvr), mplapack_ilo, mplapack_ihi, lscale, rscale, abnrm, bbnrm, rconde, rcondv, work,
         static_cast<mplapackint>(lwork), mplapack_iwork.data(), mplapack_bwork.get(), mplapack_info);
  ilo = static_cast<blas_int>(mplapack_ilo);
  ihi = static_cast<blas_int>(mplapack_ihi);
  if (bwork != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      bwork[i] = mplapack_bwork[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ggbal(char job, blas_int n, uni20::float128* a, blas_int lda, uni20::float128* b,
                                    blas_int ldb, blas_int& first, blas_int& last, uni20::float128* lscale,
                                    uni20::float128* rscale, uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rggbal, job, n, a, lda, b, ldb, first, last, lscale, rscale, work);
  mplapackint mplapack_first = static_cast<mplapackint>(first);
  mplapackint mplapack_last = static_cast<mplapackint>(last);
  mplapackint mplapack_info = 0;
  Rggbal(&job, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb),
         mplapack_first, mplapack_last, lscale, rscale, work, mplapack_info);
  first = static_cast<blas_int>(mplapack_first);
  last = static_cast<blas_int>(mplapack_last);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ggbak(char job, char side, blas_int n, blas_int first, blas_int last,
                                    uni20::float128* lscale, uni20::float128* rscale, blas_int vector_count,
                                    uni20::float128* vectors, blas_int leading_dimension)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rggbak, job, side, n, first, last, lscale, rscale, vector_count, vectors,
                          leading_dimension);
  mplapackint mplapack_info = 0;
  Rggbak(&job, &side, static_cast<mplapackint>(n), static_cast<mplapackint>(first), static_cast<mplapackint>(last),
         lscale, rscale, static_cast<mplapackint>(vector_count), vectors, static_cast<mplapackint>(leading_dimension),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gges(char jobvsl, char jobvsr, char sort, blas_int n, uni20::float128* a, blas_int lda,
                                   uni20::float128* b, blas_int ldb, blas_int& selected_dimension,
                                   uni20::float128* alphar, uni20::float128* alphai, uni20::float128* beta,
                                   uni20::float128* vsl, blas_int ldvsl, uni20::float128* vsr, blas_int ldvsr,
                                   uni20::float128* work, blas_int lwork, blas_int* bwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgges, jobvsl, jobvsr, sort, n, a, lda, b, ldb, selected_dimension, alphar, alphai,
                          beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork);
  mplapackint mplapack_selected_dimension = static_cast<mplapackint>(selected_dimension);
  mplapackint mplapack_info = 0;
  blas_int const bwork_size = std::max<blas_int>(1, n);
  std::unique_ptr<bool[]> mplapack_bwork;
  if (bwork != nullptr)
  {
    mplapack_bwork = std::make_unique<bool[]>(static_cast<std::size_t>(bwork_size));
  }
  Rgges(&jobvsl, &jobvsr, &sort, nullptr, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), b,
        static_cast<mplapackint>(ldb), mplapack_selected_dimension, alphar, alphai, beta, vsl,
        static_cast<mplapackint>(ldvsl), vsr, static_cast<mplapackint>(ldvsr), work, static_cast<mplapackint>(lwork),
        mplapack_bwork.get(), mplapack_info);
  selected_dimension = static_cast<blas_int>(mplapack_selected_dimension);
  if (bwork != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      bwork[i] = mplapack_bwork[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gghrd(char compq, char compz, blas_int n, blas_int first, blas_int last,
                                    uni20::float128* a, blas_int lda, uni20::float128* b, blas_int ldb,
                                    uni20::float128* q, blas_int ldq, uni20::float128* z, blas_int ldz)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgghrd, compq, compz, n, first, last, a, lda, b, ldb, q, ldq, z, ldz);
  mplapackint mplapack_info = 0;
  Rgghrd(&compq, &compz, static_cast<mplapackint>(n), static_cast<mplapackint>(first), static_cast<mplapackint>(last),
         a, static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), q, static_cast<mplapackint>(ldq), z,
         static_cast<mplapackint>(ldz), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int hgeqz(char job, char compq, char compz, blas_int n, blas_int first, blas_int last,
                                    uni20::float128* h, blas_int ldh, uni20::float128* t, blas_int ldt,
                                    uni20::float128* alphar, uni20::float128* alphai, uni20::float128* beta,
                                    uni20::float128* q, blas_int ldq, uni20::float128* z, blas_int ldz,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rhgeqz, job, compq, compz, n, first, last, h, ldh, t, ldt, alphar, alphai, beta, q,
                          ldq, z, ldz, work, lwork);
  mplapackint mplapack_info = 0;
  Rhgeqz(&job, &compq, &compz, static_cast<mplapackint>(n), static_cast<mplapackint>(first),
         static_cast<mplapackint>(last), h, static_cast<mplapackint>(ldh), t, static_cast<mplapackint>(ldt), alphar,
         alphai, beta, q, static_cast<mplapackint>(ldq), z, static_cast<mplapackint>(ldz), work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int tgexc(bool wantq, bool wantz, blas_int n, uni20::float128* a, blas_int lda,
                                    uni20::float128* b, blas_int ldb, uni20::float128* q, blas_int ldq,
                                    uni20::float128* z, blas_int ldz, blas_int& first, blas_int& last,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtgexc, wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, first, last, work, lwork);
  mplapackint mplapack_first = static_cast<mplapackint>(first);
  mplapackint mplapack_last = static_cast<mplapackint>(last);
  mplapackint mplapack_info = 0;
  Rtgexc(wantq, wantz, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb),
         q, static_cast<mplapackint>(ldq), z, static_cast<mplapackint>(ldz), mplapack_first, mplapack_last, work,
         static_cast<mplapackint>(lwork), mplapack_info);
  first = static_cast<blas_int>(mplapack_first);
  last = static_cast<blas_int>(mplapack_last);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int tgsen(blas_int ijob, bool wantq, bool wantz, blas_int* select, blas_int n,
                                    uni20::float128* a, blas_int lda, uni20::float128* b, blas_int ldb,
                                    uni20::float128* alphar, uni20::float128* alphai, uni20::float128* beta,
                                    uni20::float128* q, blas_int ldq, uni20::float128* z, blas_int ldz,
                                    blas_int& selected_dimension, uni20::float128& pl, uni20::float128& pr,
                                    uni20::float128* dif, uni20::float128* work, blas_int lwork, blas_int* iwork,
                                    blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtgsen, ijob, wantq, wantz, select, n, a, lda, b, ldb, alphar, alphai, beta, q, ldq,
                          z, ldz, selected_dimension, pl, pr, dif, work, lwork, iwork, liwork);
  mplapackint mplapack_selected_dimension = static_cast<mplapackint>(selected_dimension);
  mplapackint mplapack_info = 0;
  blas_int const select_size = std::max<blas_int>(1, n);
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::unique_ptr<bool[]> mplapack_select;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  if (select != nullptr)
  {
    mplapack_select = std::make_unique<bool[]>(static_cast<std::size_t>(select_size));
    for (blas_int i = 0; i < n; ++i)
    {
      mplapack_select[static_cast<std::size_t>(i)] = select[i] != 0;
    }
  }
  Rtgsen(static_cast<mplapackint>(ijob), wantq, wantz, mplapack_select.get(), static_cast<mplapackint>(n), a,
         static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), alphar, alphai, beta, q,
         static_cast<mplapackint>(ldq), z, static_cast<mplapackint>(ldz), mplapack_selected_dimension, pl, pr, dif,
         work, static_cast<mplapackint>(lwork), mplapack_iwork.data(), static_cast<mplapackint>(liwork), mplapack_info);
  selected_dimension = static_cast<blas_int>(mplapack_selected_dimension);
  if (select != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      select[i] = mplapack_select[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  uni20::lapack::mplapack::detail::copy_from_mplapack_query_int_work(mplapack_iwork, iwork, liwork);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int tgevc(char side, char howmny, blas_int* select, blas_int n, uni20::float128* s,
                                    blas_int lds, uni20::float128* p, blas_int ldp, uni20::float128* vl, blas_int ldvl,
                                    uni20::float128* vr, blas_int ldvr, blas_int mm, blas_int& computed_vectors,
                                    uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtgevc, side, howmny, select, n, s, lds, p, ldp, vl, ldvl, vr, ldvr, mm,
                          computed_vectors, work);
  mplapackint mplapack_computed_vectors = static_cast<mplapackint>(computed_vectors);
  mplapackint mplapack_info = 0;
  blas_int const select_size = std::max<blas_int>(1, n);
  std::unique_ptr<bool[]> mplapack_select;
  if (select != nullptr)
  {
    mplapack_select = std::make_unique<bool[]>(static_cast<std::size_t>(select_size));
    for (blas_int i = 0; i < n; ++i)
    {
      mplapack_select[static_cast<std::size_t>(i)] = select[i] != 0;
    }
  }
  Rtgevc(&side, &howmny, mplapack_select.get(), static_cast<mplapackint>(n), s, static_cast<mplapackint>(lds), p,
         static_cast<mplapackint>(ldp), vl, static_cast<mplapackint>(ldvl), vr, static_cast<mplapackint>(ldvr),
         static_cast<mplapackint>(mm), mplapack_computed_vectors, work, mplapack_info);
  computed_vectors = static_cast<blas_int>(mplapack_computed_vectors);
  if (select != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      select[i] = mplapack_select[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int tgsna(char job, char howmny, blas_int* select, blas_int n, uni20::float128* a,
                                    blas_int lda, uni20::float128* b, blas_int ldb, uni20::float128* vl, blas_int ldvl,
                                    uni20::float128* vr, blas_int ldvr,
                                    uni20::float128* reciprocal_eigenvalue_condition_numbers,
                                    uni20::float128* reciprocal_eigenvector_condition_numbers, blas_int mm,
                                    blas_int& computed_estimates, uni20::float128* work, blas_int lwork,
                                    blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtgsna, job, howmny, select, n, a, lda, b, ldb, vl, ldvl, vr, ldvr,
                          reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers, mm,
                          computed_estimates, work, lwork, iwork);
  mplapackint mplapack_computed_estimates = static_cast<mplapackint>(computed_estimates);
  mplapackint mplapack_info = 0;
  blas_int const select_size = std::max<blas_int>(1, n);
  blas_int const iwork_size = std::max<blas_int>(1, n + 6);
  std::unique_ptr<bool[]> mplapack_select;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  if (select != nullptr)
  {
    mplapack_select = std::make_unique<bool[]>(static_cast<std::size_t>(select_size));
    for (blas_int i = 0; i < n; ++i)
    {
      mplapack_select[static_cast<std::size_t>(i)] = select[i] != 0;
    }
  }
  Rtgsna(&job, &howmny, mplapack_select.get(), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), b,
         static_cast<mplapackint>(ldb), vl, static_cast<mplapackint>(ldvl), vr, static_cast<mplapackint>(ldvr),
         reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers,
         static_cast<mplapackint>(mm), mplapack_computed_estimates, work, static_cast<mplapackint>(lwork),
         mplapack_iwork.data(), mplapack_info);
  computed_estimates = static_cast<blas_int>(mplapack_computed_estimates);
  if (select != nullptr)
  {
    for (blas_int i = 0; i < n; ++i)
    {
      select[i] = mplapack_select[static_cast<std::size_t>(i)] ? 1 : 0;
    }
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int orgbr(char vect, blas_int m, blas_int n, blas_int k, uni20::float128* a, blas_int lda,
                                    uni20::float128* tau, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rorgbr, vect, m, n, k, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rorgbr(&vect, static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, work, static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int orghr(blas_int n, blas_int first, blas_int last, uni20::float128* a, blas_int lda,
                                    uni20::float128 const* tau, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rorghr, n, first, last, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rorghr(static_cast<mplapackint>(n), static_cast<mplapackint>(first), static_cast<mplapackint>(last), a,
         static_cast<mplapackint>(lda), const_cast<uni20::float128*>(tau), work, static_cast<mplapackint>(lwork),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int orgtr(char uplo, blas_int n, uni20::float128* a, blas_int lda, uni20::float128 const* tau,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rorgtr, uplo, n, a, lda, tau, work, lwork);
  mplapackint mplapack_info = 0;
  Rorgtr(&uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), const_cast<uni20::float128*>(tau), work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ormhr(char side, char trans, blas_int m, blas_int n, blas_int first, blas_int last,
                                    uni20::float128* a, blas_int lda, uni20::float128 const* tau, uni20::float128* c,
                                    blas_int ldc, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rormhr, side, trans, m, n, first, last, a, lda, tau, c, ldc, work, lwork);
  mplapackint mplapack_info = 0;
  Rormhr(&side, &trans, static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(first),
         static_cast<mplapackint>(last), a, static_cast<mplapackint>(lda), const_cast<uni20::float128*>(tau), c,
         static_cast<mplapackint>(ldc), work, static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ormtr(char side, char uplo, char trans, blas_int m, blas_int n, uni20::float128* a,
                                    blas_int lda, uni20::float128 const* tau, uni20::float128* c, blas_int ldc,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rormtr, side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork);
  mplapackint mplapack_info = 0;
  Rormtr(&side, &uplo, &trans, static_cast<mplapackint>(m), static_cast<mplapackint>(n), a,
         static_cast<mplapackint>(lda), const_cast<uni20::float128*>(tau), c, static_cast<mplapackint>(ldc), work,
         static_cast<mplapackint>(lwork), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ormbr(char vect, char side, char trans, blas_int m, blas_int n, blas_int k,
                                    uni20::float128* a, blas_int lda, uni20::float128* tau, uni20::float128* c,
                                    blas_int ldc, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rormbr, vect, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
  mplapackint mplapack_info = 0;
  Rormbr(&vect, &side, &trans, static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(k), a,
         static_cast<mplapackint>(lda), tau, c, static_cast<mplapackint>(ldc), work, static_cast<mplapackint>(lwork),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gesvd(char jobu, char jobvt, blas_int m, blas_int n, uni20::float128* a, blas_int lda,
                                    uni20::float128* s, uni20::float128* u, blas_int ldu, uni20::float128* vt,
                                    blas_int ldvt, uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgesvd, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
  mplapackint mplapack_info = 0;
  Rgesvd(&jobu, &jobvt, static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), s,
         u, static_cast<mplapackint>(ldu), vt, static_cast<mplapackint>(ldvt), work, static_cast<mplapackint>(lwork),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gesvd(char jobu, char jobvt, blas_int m, blas_int n, uni20::complex<uni20::float128>* a,
                                    blas_int lda, uni20::float128* s, uni20::complex<uni20::float128>* u, blas_int ldu,
                                    uni20::complex<uni20::float128>* vt, blas_int ldvt,
                                    uni20::complex<uni20::float128>* work, blas_int lwork, uni20::float128* rwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgesvd, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork);
  mplapackint mplapack_info = 0;
  Cgesvd(&jobu, &jobvt, static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), s,
         u, static_cast<mplapackint>(ldu), vt, static_cast<mplapackint>(ldvt), work, static_cast<mplapackint>(lwork),
         rwork, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gesdd(char jobz, blas_int m, blas_int n, uni20::float128* a, blas_int lda,
                                    uni20::float128* s, uni20::float128* u, blas_int ldu, uni20::float128* vt,
                                    blas_int ldvt, uni20::float128* work, blas_int lwork, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgesdd, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(8 * std::min(m, n));
  Rgesdd(&jobz, static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), s, u,
         static_cast<mplapackint>(ldu), vt, static_cast<mplapackint>(ldvt), work, static_cast<mplapackint>(lwork),
         mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gesdd(char jobz, blas_int m, blas_int n, uni20::complex<uni20::float128>* a, blas_int lda,
                                    uni20::float128* s, uni20::complex<uni20::float128>* u, blas_int ldu,
                                    uni20::complex<uni20::float128>* vt, blas_int ldvt,
                                    uni20::complex<uni20::float128>* work, blas_int lwork, uni20::float128* rwork,
                                    blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgesdd, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(8 * std::min(m, n));
  Cgesdd(&jobz, static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), s, u,
         static_cast<mplapackint>(ldu), vt, static_cast<mplapackint>(ldvt), work, static_cast<mplapackint>(lwork),
         rwork, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gesvdx(char jobu, char jobvt, char range, blas_int m, blas_int n, uni20::float128* a,
                                     blas_int lda, uni20::float128 vl, uni20::float128 vu, blas_int il, blas_int iu,
                                     blas_int& selected_count, uni20::float128* singular_values, uni20::float128* u,
                                     blas_int ldu, uni20::float128* vt, blas_int ldvt, uni20::float128* work,
                                     blas_int lwork, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgesvdx, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, selected_count,
                          singular_values, u, ldu, vt, ldvt, work, lwork, iwork);
  mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = 12 * std::max<blas_int>(1, std::min(m, n));
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Rgesvdx(&jobu, &jobvt, &range, static_cast<mplapackint>(m), static_cast<mplapackint>(n), a,
          static_cast<mplapackint>(lda), vl, vu, static_cast<mplapackint>(il), static_cast<mplapackint>(iu),
          mplapack_selected_count, singular_values, u, static_cast<mplapackint>(ldu), vt,
          static_cast<mplapackint>(ldvt), work, static_cast<mplapackint>(lwork), mplapack_iwork.data(), mplapack_info);
  selected_count = static_cast<blas_int>(mplapack_selected_count);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gesvdx(char jobu, char jobvt, char range, blas_int m, blas_int n,
                                     uni20::complex<uni20::float128>* a, blas_int lda, uni20::float128 vl,
                                     uni20::float128 vu, blas_int il, blas_int iu, blas_int& selected_count,
                                     uni20::float128* singular_values, uni20::complex<uni20::float128>* u, blas_int ldu,
                                     uni20::complex<uni20::float128>* vt, blas_int ldvt,
                                     uni20::complex<uni20::float128>* work, blas_int lwork, uni20::float128* rwork,
                                     blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Cgesvdx, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, selected_count,
                          singular_values, u, ldu, vt, ldvt, work, lwork, rwork, iwork);
  mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = 12 * std::max<blas_int>(1, std::min(m, n));
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Cgesvdx(&jobu, &jobvt, &range, static_cast<mplapackint>(m), static_cast<mplapackint>(n), a,
          static_cast<mplapackint>(lda), vl, vu, static_cast<mplapackint>(il), static_cast<mplapackint>(iu),
          mplapack_selected_count, singular_values, u, static_cast<mplapackint>(ldu), vt,
          static_cast<mplapackint>(ldvt), work, static_cast<mplapackint>(lwork), rwork, mplapack_iwork.data(),
          mplapack_info);
  selected_count = static_cast<blas_int>(mplapack_selected_count);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int bdsqr(char uplo, blas_int n, blas_int ncvt, blas_int nru, blas_int ncc,
                                    uni20::float128* d, uni20::float128* e, uni20::float128* vt, blas_int ldvt,
                                    uni20::float128* u, blas_int ldu, uni20::float128* c, blas_int ldc,
                                    uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rbdsqr, uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work);
  mplapackint mplapack_info = 0;
  Rbdsqr(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(ncvt), static_cast<mplapackint>(nru),
         static_cast<mplapackint>(ncc), d, e, vt, static_cast<mplapackint>(ldvt), u, static_cast<mplapackint>(ldu), c,
         static_cast<mplapackint>(ldc), work, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int bdsdc(char uplo, char compq, blas_int n, uni20::float128* d, uni20::float128* e,
                                    uni20::float128* u, blas_int ldu, uni20::float128* vt, blas_int ldvt,
                                    uni20::float128* q, blas_int* iq, uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rbdsdc, uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iq = uni20::lapack::mplapack::detail::to_mplapack_ints(iq, 1);
  blas_int const iwork_size = 8 * std::max<blas_int>(1, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Rbdsdc(&uplo, &compq, static_cast<mplapackint>(n), d, e, u, static_cast<mplapackint>(ldu), vt,
         static_cast<mplapackint>(ldvt), q, mplapack_iq.data(), work, mplapack_iwork.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_iq, iq, 1);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int bdsvdx(char uplo, char jobz, char range, blas_int n, uni20::float128* d,
                                     uni20::float128* e, uni20::float128 vl, uni20::float128 vu, blas_int il,
                                     blas_int iu, blas_int& selected_count, uni20::float128* singular_values,
                                     uni20::float128* z, blas_int ldz, uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rbdsvdx, uplo, jobz, range, n, d, e, vl, vu, il, iu, selected_count, singular_values,
                          z, ldz, work, iwork);
  mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = 12 * std::max<blas_int>(1, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(iwork_size);
  Rbdsvdx(&uplo, &jobz, &range, static_cast<mplapackint>(n), d, e, vl, vu, static_cast<mplapackint>(il),
          static_cast<mplapackint>(iu), mplapack_selected_count, singular_values, z, static_cast<mplapackint>(ldz),
          work, mplapack_iwork.data(), mplapack_info);
  selected_count = static_cast<blas_int>(mplapack_selected_count);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int potrf(char uplo, blas_int n, uni20::float128* a, blas_int lda)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpotrf, uplo, n, a, lda);
  mplapackint mplapack_info = 0;
  Rpotrf(&uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pstrf(char uplo, blas_int n, uni20::float128* a, blas_int lda, blas_int* pivots,
                                    blas_int& rank, uni20::float128 tolerance, uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpstrf, uplo, n, a, lda, pivots, rank, tolerance, work);
  mplapackint mplapack_rank = static_cast<mplapackint>(rank);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_pivots = uni20::lapack::mplapack::detail::to_mplapack_ints(pivots, n);
  Rpstrf(&uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_pivots.data(), mplapack_rank,
         tolerance, work, mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_pivots, pivots, n);
  rank = static_cast<blas_int>(mplapack_rank);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int potri(char uplo, blas_int n, uni20::float128* a, blas_int lda)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpotri, uplo, n, a, lda);
  mplapackint mplapack_info = 0;
  Rpotri(&uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int potrs(char uplo, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpotrs, uplo, n, nrhs, a, lda, b, ldb);
  mplapackint mplapack_info = 0;
  Rpotrs(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda), b,
         static_cast<mplapackint>(ldb), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int porfs(char uplo, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* factors, blas_int factor_lda, uni20::float128* b, blas_int ldb,
                                    uni20::float128* x, blas_int ldx, uni20::float128* forward_error,
                                    uni20::float128* backward_error, uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rporfs, uplo, n, nrhs, a, lda, factors, factor_lda, b, ldb, x, ldx, forward_error,
                          backward_error, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rporfs(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda), factors,
         static_cast<mplapackint>(factor_lda), b, static_cast<mplapackint>(ldb), x, static_cast<mplapackint>(ldx),
         forward_error, backward_error, work, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int posvx(char fact, char uplo, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* af, blas_int ldaf, char& equed, uni20::float128* scale,
                                    uni20::float128* b, blas_int ldb, uni20::float128* x, blas_int ldx,
                                    uni20::float128& rcond, uni20::float128* forward_error,
                                    uni20::float128* backward_error, uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rposvx, fact, uplo, n, nrhs, a, lda, af, ldaf, equed, scale, b, ldb, x, ldx, work,
                          iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rposvx(&fact, &uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda),
         af, static_cast<mplapackint>(ldaf), &equed, scale, b, static_cast<mplapackint>(ldb), x,
         static_cast<mplapackint>(ldx), rcond, forward_error, backward_error, work, mplapack_iwork.data(),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int poequ(blas_int n, uni20::float128* a, blas_int lda, uni20::float128* scale,
                                    uni20::float128& scale_condition, uni20::float128& max_abs)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpoequ, n, a, lda, scale);
  mplapackint mplapack_info = 0;
  Rpoequ(static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), scale, scale_condition, max_abs, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pocon(char uplo, blas_int n, uni20::float128* a, blas_int lda,
                                    uni20::float128 matrix_one_norm, uni20::float128& rcond, uni20::float128* work,
                                    blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpocon, uplo, n, a, lda, matrix_one_norm, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rpocon(&uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), matrix_one_norm, rcond, work,
         mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sytrf(char uplo, blas_int n, uni20::float128* a, blas_int lda, blas_int* ipiv,
                                    uni20::float128* work, blas_int lwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsytrf, uplo, n, a, lda, ipiv, work, lwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rsytrf(&uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_ipiv.data(), work,
         static_cast<mplapackint>(lwork), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sytrs(char uplo, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    blas_int const* ipiv, uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsytrs, uplo, n, nrhs, a, lda, ipiv, b, ldb);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  Rsytrs(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda),
         mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sytri(char uplo, blas_int n, uni20::float128* a, blas_int lda, blas_int const* ipiv,
                                    uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsytri, uplo, n, a, lda, ipiv, work);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  Rsytri(&uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_ipiv.data(), work,
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int syrfs(char uplo, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* factors, blas_int factor_lda, blas_int const* ipiv,
                                    uni20::float128* b, blas_int ldb, uni20::float128* x, blas_int ldx,
                                    uni20::float128* forward_error, uni20::float128* backward_error,
                                    uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsyrfs, uplo, n, nrhs, a, lda, factors, factor_lda, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rsyrfs(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda), factors,
         static_cast<mplapackint>(factor_lda), mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), x,
         static_cast<mplapackint>(ldx), forward_error, backward_error, work, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sysvx(char fact, char uplo, blas_int n, blas_int nrhs, uni20::float128* a, blas_int lda,
                                    uni20::float128* af, blas_int ldaf, blas_int* ipiv, uni20::float128* b,
                                    blas_int ldb, uni20::float128* x, blas_int ldx, uni20::float128& rcond,
                                    uni20::float128* forward_error, uni20::float128* backward_error,
                                    uni20::float128* work, blas_int lwork, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsysvx, fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, work, lwork,
                          iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rsysvx(&fact, &uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a, static_cast<mplapackint>(lda),
         af, static_cast<mplapackint>(ldaf), mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), x,
         static_cast<mplapackint>(ldx), rcond, forward_error, backward_error, work, static_cast<mplapackint>(lwork),
         mplapack_iwork.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sycon(char uplo, blas_int n, uni20::float128* a, blas_int lda, blas_int const* ipiv,
                                    uni20::float128 matrix_one_norm, uni20::float128& rcond, uni20::float128* work,
                                    blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsycon, uplo, n, a, lda, ipiv, matrix_one_norm, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rsycon(&uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_ipiv.data(), matrix_one_norm,
         rcond, work, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trtrs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, uni20::float128* a,
                                    blas_int lda, uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtrtrs, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
  mplapackint mplapack_info = 0;
  Rtrtrs(&uplo, &trans, &diag, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a,
         static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trrfs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, uni20::float128* a,
                                    blas_int lda, uni20::float128* b, blas_int ldb, uni20::float128* x, blas_int ldx,
                                    uni20::float128* forward_error, uni20::float128* backward_error,
                                    uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtrrfs, uplo, trans, diag, n, nrhs, a, lda, b, ldb, x, ldx, forward_error,
                          backward_error, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rtrrfs(&uplo, &trans, &diag, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), a,
         static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), x, static_cast<mplapackint>(ldx),
         forward_error, backward_error, work, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trtri(char uplo, char diag, blas_int n, uni20::float128* a, blas_int lda)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtrtri, uplo, diag, n, a, lda);
  mplapackint mplapack_info = 0;
  Rtrtri(&uplo, &diag, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trcon(char norm, char uplo, char diag, blas_int n, uni20::float128* a, blas_int lda,
                                    uni20::float128& rcond, uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtrcon, norm, uplo, diag, n, a, lda, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rtrcon(&norm, &uplo, &diag, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), rcond, work,
         mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int trsyl(char trans_a, char trans_b, blas_int sign, blas_int m, blas_int n,
                                    uni20::float128* a, blas_int lda, uni20::float128* b, blas_int ldb,
                                    uni20::float128* c, blas_int ldc, uni20::float128& scale)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rtrsyl, trans_a, trans_b, sign, m, n, a, lda, b, ldb, c, ldc);
  mplapackint mplapack_info = 0;
  Rtrsyl(&trans_a, &trans_b, static_cast<mplapackint>(sign), static_cast<mplapackint>(m), static_cast<mplapackint>(n),
         a, static_cast<mplapackint>(lda), b, static_cast<mplapackint>(ldb), c, static_cast<mplapackint>(ldc), scale,
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

} // namespace uni20::lapack::unchecked
