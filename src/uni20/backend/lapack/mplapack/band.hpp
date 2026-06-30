#pragma once

/**
 * \file band.hpp
 * \ingroup backend_lapack_mplapack
 * \brief MPLAPACK binary128 wrappers for real band linear algebra.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/backend/lapack/common.hpp>
#include <uni20/backend/lapack/mplapack/common.hpp>
#include <uni20/config.hpp>
#include <uni20/core/types.hpp>

#if !(UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK)
#error "uni20/backend/lapack/mplapack/band.hpp requires MPLAPACK float128 support"
#endif

// clang-format off
#include <mplapack_config.h>
#include <mplapack_binary128.h>
// clang-format on

namespace uni20::lapack::unchecked
{

[[nodiscard]] inline blas_int gbsv(blas_int n, blas_int kl, blas_int ku, blas_int nrhs, uni20::float128* ab,
                                   blas_int ldab, blas_int* ipiv, uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgbsv, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgbsv(static_cast<mplapackint>(n), static_cast<mplapackint>(kl), static_cast<mplapackint>(ku),
        static_cast<mplapackint>(nrhs), ab, static_cast<mplapackint>(ldab), mplapack_ipiv.data(), b,
        static_cast<mplapackint>(ldb), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gbtrf(blas_int m, blas_int n, blas_int kl, blas_int ku, uni20::float128* ab,
                                    blas_int ldab, blas_int* ipiv)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgbtrf, m, n, kl, ku, ab, ldab, ipiv);
  mplapackint mplapack_info = 0;
  blas_int const pivot_count = std::min(m, n);
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::make_mplapack_int_work(pivot_count);
  Rgbtrf(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(kl),
         static_cast<mplapackint>(ku), ab, static_cast<mplapackint>(ldab), mplapack_ipiv.data(), mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, pivot_count);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gbtrs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs,
                                    uni20::float128* ab, blas_int ldab, blas_int const* ipiv, uni20::float128* b,
                                    blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgbtrs, trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  Rgbtrs(&trans, static_cast<mplapackint>(n), static_cast<mplapackint>(kl), static_cast<mplapackint>(ku),
         static_cast<mplapackint>(nrhs), ab, static_cast<mplapackint>(ldab), mplapack_ipiv.data(), b,
         static_cast<mplapackint>(ldb), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gbcon(char norm, blas_int n, blas_int kl, blas_int ku, uni20::float128* ab, blas_int ldab,
                                    blas_int const* ipiv, uni20::float128 anorm, uni20::float128& rcond,
                                    uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgbcon, norm, n, kl, ku, ab, ldab, ipiv, anorm, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgbcon(&norm, static_cast<mplapackint>(n), static_cast<mplapackint>(kl), static_cast<mplapackint>(ku), ab,
         static_cast<mplapackint>(ldab), mplapack_ipiv.data(), anorm, rcond, work, mplapack_iwork.data(),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gbrfs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs,
                                    uni20::float128* ab, blas_int ldab, uni20::float128* afb, blas_int ldafb,
                                    blas_int const* ipiv, uni20::float128* b, blas_int ldb, uni20::float128* x,
                                    blas_int ldx, uni20::float128* forward_error, uni20::float128* backward_error,
                                    uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgbrfs, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx,
                          forward_error, backward_error, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgbrfs(&trans, static_cast<mplapackint>(n), static_cast<mplapackint>(kl), static_cast<mplapackint>(ku),
         static_cast<mplapackint>(nrhs), ab, static_cast<mplapackint>(ldab), afb, static_cast<mplapackint>(ldafb),
         mplapack_ipiv.data(), b, static_cast<mplapackint>(ldb), x, static_cast<mplapackint>(ldx), forward_error,
         backward_error, work, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gbsvx(char fact, char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs,
                                    uni20::float128* ab, blas_int ldab, uni20::float128* afb, blas_int ldafb,
                                    blas_int* ipiv, char& equed, uni20::float128* row_scale,
                                    uni20::float128* column_scale, uni20::float128* b, blas_int ldb, uni20::float128* x,
                                    blas_int ldx, uni20::float128& rcond, uni20::float128* forward_error,
                                    uni20::float128* backward_error, uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rgbsvx, fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, row_scale,
                          column_scale, b, ldb, x, ldx, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_ipiv = uni20::lapack::mplapack::detail::to_mplapack_ints(ipiv, n);
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rgbsvx(&fact, &trans, static_cast<mplapackint>(n), static_cast<mplapackint>(kl), static_cast<mplapackint>(ku),
         static_cast<mplapackint>(nrhs), ab, static_cast<mplapackint>(ldab), afb, static_cast<mplapackint>(ldafb),
         mplapack_ipiv.data(), &equed, row_scale, column_scale, b, static_cast<mplapackint>(ldb), x,
         static_cast<mplapackint>(ldx), rcond, forward_error, backward_error, work, mplapack_iwork.data(),
         mplapack_info);
  uni20::lapack::mplapack::detail::copy_from_mplapack_ints(mplapack_ipiv, ipiv, n);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int gbequ(bool power_of_two, blas_int m, blas_int n, blas_int kl, blas_int ku,
                                    uni20::float128* ab, blas_int ldab, uni20::float128* row_scale,
                                    uni20::float128* column_scale, uni20::float128& row_condition,
                                    uni20::float128& column_condition, uni20::float128& max_abs)
{
  mplapackint mplapack_info = 0;
  if (power_of_two)
  {
    UNI20_EXTERNAL_API_CALL(LAPACK, Rgbequb, m, n, kl, ku, ab, ldab, row_scale, column_scale);
    Rgbequb(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(kl),
            static_cast<mplapackint>(ku), ab, static_cast<mplapackint>(ldab), row_scale, column_scale, row_condition,
            column_condition, max_abs, mplapack_info);
  }
  else
  {
    UNI20_EXTERNAL_API_CALL(LAPACK, Rgbequ, m, n, kl, ku, ab, ldab, row_scale, column_scale);
    Rgbequ(static_cast<mplapackint>(m), static_cast<mplapackint>(n), static_cast<mplapackint>(kl),
           static_cast<mplapackint>(ku), ab, static_cast<mplapackint>(ldab), row_scale, column_scale, row_condition,
           column_condition, max_abs, mplapack_info);
  }
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pbsv(char uplo, blas_int n, blas_int kd, blas_int nrhs, uni20::float128* ab,
                                   blas_int ldab, uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpbsv, uplo, n, kd, nrhs, ab, ldab, b, ldb);
  mplapackint mplapack_info = 0;
  Rpbsv(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(kd), static_cast<mplapackint>(nrhs), ab,
        static_cast<mplapackint>(ldab), b, static_cast<mplapackint>(ldb), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pbtrf(char uplo, blas_int n, blas_int kd, uni20::float128* ab, blas_int ldab)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpbtrf, uplo, n, kd, ab, ldab);
  mplapackint mplapack_info = 0;
  Rpbtrf(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(kd), ab, static_cast<mplapackint>(ldab),
         mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pbtrs(char uplo, blas_int n, blas_int kd, blas_int nrhs, uni20::float128* ab,
                                    blas_int ldab, uni20::float128* b, blas_int ldb)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpbtrs, uplo, n, kd, nrhs, ab, ldab, b, ldb);
  mplapackint mplapack_info = 0;
  Rpbtrs(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(kd), static_cast<mplapackint>(nrhs), ab,
         static_cast<mplapackint>(ldab), b, static_cast<mplapackint>(ldb), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pbcon(char uplo, blas_int n, blas_int kd, uni20::float128* ab, blas_int ldab,
                                    uni20::float128 anorm, uni20::float128& rcond, uni20::float128* work,
                                    blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpbcon, uplo, n, kd, ab, ldab, anorm, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rpbcon(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(kd), ab, static_cast<mplapackint>(ldab), anorm,
         rcond, work, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pbrfs(char uplo, blas_int n, blas_int kd, blas_int nrhs, uni20::float128* ab,
                                    blas_int ldab, uni20::float128* afb, blas_int ldafb, uni20::float128* b,
                                    blas_int ldb, uni20::float128* x, blas_int ldx, uni20::float128* forward_error,
                                    uni20::float128* backward_error, uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpbrfs, uplo, n, kd, nrhs, ab, ldab, afb, ldafb, b, ldb, x, ldx, forward_error,
                          backward_error, work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rpbrfs(&uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(kd), static_cast<mplapackint>(nrhs), ab,
         static_cast<mplapackint>(ldab), afb, static_cast<mplapackint>(ldafb), b, static_cast<mplapackint>(ldb), x,
         static_cast<mplapackint>(ldx), forward_error, backward_error, work, mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

[[nodiscard]] inline blas_int pbsvx(char fact, char uplo, blas_int n, blas_int kd, blas_int nrhs, uni20::float128* ab,
                                    blas_int ldab, uni20::float128* afb, blas_int ldafb, char& equed,
                                    uni20::float128* scale, uni20::float128* b, blas_int ldb, uni20::float128* x,
                                    blas_int ldx, uni20::float128& rcond, uni20::float128* forward_error,
                                    uni20::float128* backward_error, uni20::float128* work, blas_int* iwork)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rpbsvx, fact, uplo, n, kd, nrhs, ab, ldab, afb, ldafb, equed, scale, b, ldb, x, ldx,
                          work, iwork);
  mplapackint mplapack_info = 0;
  std::vector<mplapackint> mplapack_iwork = uni20::lapack::mplapack::detail::make_mplapack_int_work(n);
  Rpbsvx(&fact, &uplo, static_cast<mplapackint>(n), static_cast<mplapackint>(kd), static_cast<mplapackint>(nrhs), ab,
         static_cast<mplapackint>(ldab), afb, static_cast<mplapackint>(ldafb), &equed, scale, b,
         static_cast<mplapackint>(ldb), x, static_cast<mplapackint>(ldx), rcond, forward_error, backward_error, work,
         mplapack_iwork.data(), mplapack_info);
  return static_cast<blas_int>(mplapack_info);
}

} // namespace uni20::lapack::unchecked
