#pragma once

/**
 * \file tridiagonal.hpp
 * \ingroup backend_lapack_mplapack
 * \brief MPLAPACK binary128 wrappers for real tridiagonal linear algebra.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/backend/lapack/common.hpp>
#include <uni20/backend/lapack/mplapack/common.hpp>
#include <uni20/config.hpp>
#include <uni20/core/types.hpp>

#if !(UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK)
#error "uni20/backend/lapack/mplapack/tridiagonal.hpp requires MPLAPACK float128 support"
#endif

// clang-format off
#include <mplapack_config.h>
#include <mplapack_binary128.h>
// clang-format on

namespace uni20::lapack::unchecked
{

[[nodiscard]] inline blas_int gtsv(blas_int n, blas_int nrhs, uni20::float128* dl, uni20::float128* d,
                                   uni20::float128* du, uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgtsv, n, nrhs, dl, d, du, b, ldb);
  mplapackint mplapack_info = 0;
  Rgtsv(static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), dl, d, du, b, static_cast<mplapackint>(ldb),
        mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gttrf(blas_int n, uni20::float128* dl, uni20::float128* d, uni20::float128* du,
                                    uni20::float128* du2, blas_int* ipiv)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgttrf, n, dl, d, du, du2, ipiv);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgttrf(static_cast<mplapackint>(n), dl, d, du, du2, mplapack_ipiv.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gttrs(char trans, blas_int n, blas_int nrhs, uni20::float128* dl, uni20::float128* d,
                                    uni20::float128* du, uni20::float128* du2, blas_int const* ipiv, uni20::float128* b,
                                    blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgttrs, trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  Rgttrs(&trans, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), dl, d, du, du2, mplapack_ipiv.data(), b,
         static_cast<mplapackint>(ldb), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gtcon(char norm, blas_int n, uni20::float128* dl, uni20::float128* d, uni20::float128* du,
                                    uni20::float128* du2, blas_int const* ipiv, uni20::float128 matrix_norm,
                                    uni20::float128& rcond, uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgtcon, norm, n, dl, d, du, du2, ipiv, matrix_norm, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgtcon(&norm, static_cast<mplapackint>(n), dl, d, du, du2, mplapack_ipiv.data(), matrix_norm, rcond, work,
         mplapack_iwork.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_iwork, iwork, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gtrfs(char trans, blas_int n, blas_int nrhs, uni20::float128* dl, uni20::float128* d,
                                    uni20::float128* du, uni20::float128* dlf, uni20::float128* df,
                                    uni20::float128* duf, uni20::float128* du2, blas_int const* ipiv,
                                    uni20::float128* b, blas_int ldb, uni20::float128* x, blas_int ldx,
                                    uni20::float128* forward_error, uni20::float128* backward_error,
                                    uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgtrfs, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgtrfs(&trans, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), dl, d, du, dlf, df, duf, du2,
         mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), x, static_cast<mplapackint>(ldx), forward_error,
         backward_error, work, mplapack_iwork.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_iwork, iwork, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gtsvx(char fact, char trans, blas_int n, blas_int nrhs, uni20::float128* dl,
                                    uni20::float128* d, uni20::float128* du, uni20::float128* dlf, uni20::float128* df,
                                    uni20::float128* duf, uni20::float128* du2, blas_int* ipiv, uni20::float128* b,
                                    blas_int ldb, uni20::float128* x, blas_int ldx, uni20::float128& rcond,
                                    uni20::float128* forward_error, uni20::float128* backward_error,
                                    uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgtsvx, fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx,
                          work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgtsvx(&fact, &trans, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), dl, d, du, dlf, df, duf, du2,
         mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), x, static_cast<mplapackint>(ldx), rcond, forward_error,
         backward_error, work, mplapack_iwork.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, n);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_iwork, iwork, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ptsv(blas_int n, blas_int nrhs, uni20::float128* d, uni20::float128* e,
                                   uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rptsv, n, nrhs, d, e, b, ldb);
  mplapackint mplapack_info = 0;
  Rptsv(static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), d, e, b, static_cast<mplapackint>(ldb),
        mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pttrf(blas_int n, uni20::float128* d, uni20::float128* e)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpttrf, n, d, e);
  mplapackint mplapack_info = 0;
  Rpttrf(static_cast<mplapackint>(n), d, e, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pttrs(blas_int n, blas_int nrhs, uni20::float128* d, uni20::float128* e,
                                    uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpttrs, n, nrhs, d, e, b, ldb);
  mplapackint mplapack_info = 0;
  Rpttrs(static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), d, e, b, static_cast<mplapackint>(ldb),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ptcon(blas_int n, uni20::float128* d, uni20::float128* e, uni20::float128 matrix_norm,
                                    uni20::float128& rcond, uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rptcon, n, d, e, matrix_norm, work);
  mplapackint mplapack_info = 0;
  Rptcon(static_cast<mplapackint>(n), d, e, matrix_norm, rcond, work, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ptrfs(blas_int n, blas_int nrhs, uni20::float128* d, uni20::float128* e,
                                    uni20::float128* df, uni20::float128* ef, uni20::float128* b, blas_int ldb,
                                    uni20::float128* x, blas_int ldx, uni20::float128* forward_error,
                                    uni20::float128* backward_error, uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rptrfs, n, nrhs, d, e, df, ef, b, ldb, x, ldx, forward_error, backward_error, work);
  mplapackint mplapack_info = 0;
  Rptrfs(static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), d, e, df, ef, b, static_cast<mplapackint>(ldb), x,
         static_cast<mplapackint>(ldx), forward_error, backward_error, work, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int ptsvx(char fact, blas_int n, blas_int nrhs, uni20::float128* d, uni20::float128* e,
                                    uni20::float128* df, uni20::float128* ef, uni20::float128* b, blas_int ldb,
                                    uni20::float128* x, blas_int ldx, uni20::float128& rcond,
                                    uni20::float128* forward_error, uni20::float128* backward_error,
                                    uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rptsvx, fact, n, nrhs, d, e, df, ef, b, ldb, x, ldx, work);
  mplapackint mplapack_info = 0;
  Rptsvx(&fact, static_cast<mplapackint>(n), static_cast<mplapackint>(nrhs), d, e, df, ef, b,
         static_cast<mplapackint>(ldb), x, static_cast<mplapackint>(ldx), rcond, forward_error, backward_error, work,
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int sterf(blas_int n, uni20::float128* d, uni20::float128* e)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsterf, n, d, e);
  mplapackint mplapack_info = 0;
  Rsterf(static_cast<mplapackint>(n), d, e, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int steqr(char compz, blas_int n, uni20::float128* d, uni20::float128* e, uni20::float128* z,
                                    blas_int ldz, uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rsteqr, compz, n, d, e, z, ldz, work);
  mplapackint mplapack_info = 0;
  Rsteqr(&compz, static_cast<mplapackint>(n), d, e, z, static_cast<mplapackint>(ldz), work, mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int stevd(char jobz, blas_int n, uni20::float128* d, uni20::float128* e, uni20::float128* z,
                                    blas_int ldz, uni20::float128* work, blas_int lwork, blas_int* iwork,
                                    blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rstevd, jobz, n, d, e, z, ldz, work, lwork, iwork, liwork);
  mplapackint mplapack_info = 0;
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::to_mplapack_ints(iwork, iwork_size);
  Rstevd(&jobz, static_cast<mplapackint>(n), d, e, z, static_cast<mplapackint>(ldz), work,
         static_cast<mplapackint>(lwork), mplapack_iwork.data(), static_cast<mplapackint>(liwork), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_iwork, iwork, iwork_size);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int stevr(char jobz, char range, blas_int n, uni20::float128* d, uni20::float128* e,
                                    uni20::float128 vl, uni20::float128 vu, blas_int il, blas_int iu,
                                    uni20::float128 abstol, blas_int& selected_count, uni20::float128* w,
                                    uni20::float128* z, blas_int ldz, blas_int* isuppz, uni20::float128* work,
                                    blas_int lwork, blas_int* iwork, blas_int liwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rstevr, jobz, range, n, d, e, vl, vu, il, iu, abstol, selected_count, w, z, ldz,
                          isuppz, work, lwork, iwork, liwork);
  mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
  mplapackint mplapack_info = 0;
  blas_int const support_size = 2 * std::max<blas_int>(1, n);
  blas_int const iwork_size = std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork);
  std::vector<mplapackint> mplapack_isuppz = uni20::lapack::mplapack::detail::to_mplapack_ints(isuppz, support_size);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::to_mplapack_ints(iwork, iwork_size);
  Rstevr(&jobz, &range, static_cast<mplapackint>(n), d, e, vl, vu, static_cast<mplapackint>(il),
         static_cast<mplapackint>(iu), abstol, mplapack_selected_count, w, z, static_cast<mplapackint>(ldz),
         mplapack_isuppz.data(), work, static_cast<mplapackint>(lwork), mplapack_iwork.data(),
         static_cast<mplapackint>(liwork), mplapack_info);
  selected_count = static_cast<blas_int>(mplapack_selected_count);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_isuppz, isuppz, support_size);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_iwork, iwork, iwork_size);
  return static_cast<blas_int>(mplapack_info);
}

} // namespace uni20::lapack::unchecked
