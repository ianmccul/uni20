#pragma once

/**
 * \defgroup backend_lapack_mplapack MPLAPACK backend
 * \ingroup backend_lapack
 * \brief Header-only wrappers around configured MPLAPACK extension-precision routines.
 */

/**
 * \file norms.hpp
 * \ingroup backend_lapack_mplapack
 * \brief MPLAPACK binary128 norm wrappers.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/backend/lapack/common.hpp>
#include <uni20/config.hpp>
#include <uni20/core/types.hpp>

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
// clang-format off
#include <mplapack_config.h>
#include <mplapack_binary128.h>
// clang-format on
#endif

namespace uni20::lapack::unchecked
{

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK

[[nodiscard]] inline uni20::float128 lange(char norm, blas_int m, blas_int n, uni20::float128* a, blas_int lda,
                                           uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rlange, norm, m, n, a, lda, work);
  return Rlange(&norm, static_cast<mplapackint>(m), static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda),
                work);
}

[[nodiscard]] inline uni20::float128 lansy(char norm, char uplo, blas_int n, uni20::float128* a, blas_int lda,
                                           uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rlansy, norm, uplo, n, a, lda, work);
  return Rlansy(&norm, &uplo, static_cast<mplapackint>(n), a, static_cast<mplapackint>(lda), work);
}

[[nodiscard]] inline uni20::float128 lantr(char norm, char uplo, char diag, blas_int m, blas_int n, uni20::float128* a,
                                           blas_int lda, uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rlantr, norm, uplo, diag, m, n, a, lda, work);
  return Rlantr(&norm, &uplo, &diag, static_cast<mplapackint>(m), static_cast<mplapackint>(n), a,
                static_cast<mplapackint>(lda), work);
}

[[nodiscard]] inline uni20::float128 langb(char norm, blas_int n, blas_int kl, blas_int ku, uni20::float128* ab,
                                           blas_int ldab, uni20::float128* work)
{
  UNI20_EXTERNAL_API_CALL(LAPACK, Rlangb, norm, n, kl, ku, ab, ldab, work);
  return Rlangb(&norm, static_cast<mplapackint>(n), static_cast<mplapackint>(kl), static_cast<mplapackint>(ku), ab,
                static_cast<mplapackint>(ldab), work);
}

#endif

} // namespace uni20::lapack::unchecked
