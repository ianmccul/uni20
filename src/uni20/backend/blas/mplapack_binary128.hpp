#pragma once

/**
 * \file mplapack_binary128.hpp
 * \ingroup backend_blas
 * \brief MPBLAS binary128 overloads for the Uni20 BLAS wrapper surface.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/config.hpp>
#include <uni20/core/types.hpp>

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
#include <mplapack_config.h>
#endif

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK && defined(MPLAPACK_BINARY128_MODE) &&                      \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
#include <mpblas_binary128.h>

namespace uni20::blas
{

namespace mplapack_binary128_detail
{

inline char const* char_string(char value, char (&buffer)[2])
{
  buffer[0] = value;
  buffer[1] = '\0';
  return buffer;
}

} // namespace mplapack_binary128_detail

inline void gemm(char transa, char transb, blas_int m, blas_int n, blas_int k, uni20::float128 alpha,
                 uni20::float128 const* A, blas_int lda, uni20::float128 const* B, blas_int ldb, uni20::float128 beta,
                 uni20::float128* C, blas_int ldc)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Rgemm, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  char transa_string[2]{};
  char transb_string[2]{};
  Rgemm(mplapack_binary128_detail::char_string(transa, transa_string),
        mplapack_binary128_detail::char_string(transb, transb_string), static_cast<mplapackint>(m),
        static_cast<mplapackint>(n), static_cast<mplapackint>(k), alpha, const_cast<uni20::float128*>(A),
        static_cast<mplapackint>(lda), const_cast<uni20::float128*>(B), static_cast<mplapackint>(ldb), beta, C,
        static_cast<mplapackint>(ldc));
}

inline void gemv(char trans, blas_int m, blas_int n, uni20::float128 alpha, uni20::float128 const* A, blas_int lda,
                 uni20::float128 const* x, blas_int incx, uni20::float128 beta, uni20::float128* y, blas_int incy)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Rgemv, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  char trans_string[2]{};
  Rgemv(mplapack_binary128_detail::char_string(trans, trans_string), static_cast<mplapackint>(m),
        static_cast<mplapackint>(n), alpha, const_cast<uni20::float128*>(A), static_cast<mplapackint>(lda),
        const_cast<uni20::float128*>(x), static_cast<mplapackint>(incx), beta, y, static_cast<mplapackint>(incy));
}

inline void ger(blas_int m, blas_int n, uni20::float128 alpha, uni20::float128 const* x, blas_int incx,
                uni20::float128 const* y, blas_int incy, uni20::float128* A, blas_int lda)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Rger, m, n, alpha, x, incx, y, incy, A, lda);
  Rger(static_cast<mplapackint>(m), static_cast<mplapackint>(n), alpha, const_cast<uni20::float128*>(x),
       static_cast<mplapackint>(incx), const_cast<uni20::float128*>(y), static_cast<mplapackint>(incy), A,
       static_cast<mplapackint>(lda));
}

inline void syrk(char uplo, char trans, blas_int n, blas_int k, uni20::float128 alpha, uni20::float128 const* A,
                 blas_int lda, uni20::float128 beta, uni20::float128* C, blas_int ldc)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Rsyrk, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  char uplo_string[2]{};
  char trans_string[2]{};
  Rsyrk(mplapack_binary128_detail::char_string(uplo, uplo_string),
        mplapack_binary128_detail::char_string(trans, trans_string), static_cast<mplapackint>(n),
        static_cast<mplapackint>(k), alpha, const_cast<uni20::float128*>(A), static_cast<mplapackint>(lda), beta, C,
        static_cast<mplapackint>(ldc));
}

inline void syr2k(char uplo, char trans, blas_int n, blas_int k, uni20::float128 alpha, uni20::float128 const* A,
                  blas_int lda, uni20::float128 const* B, blas_int ldb, uni20::float128 beta, uni20::float128* C,
                  blas_int ldc)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Rsyr2k, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  char uplo_string[2]{};
  char trans_string[2]{};
  Rsyr2k(mplapack_binary128_detail::char_string(uplo, uplo_string),
         mplapack_binary128_detail::char_string(trans, trans_string), static_cast<mplapackint>(n),
         static_cast<mplapackint>(k), alpha, const_cast<uni20::float128*>(A), static_cast<mplapackint>(lda),
         const_cast<uni20::float128*>(B), static_cast<mplapackint>(ldb), beta, C, static_cast<mplapackint>(ldc));
}

inline void gemm(char transa, char transb, blas_int m, blas_int n, blas_int k, uni20::complex<uni20::float128> alpha,
                 uni20::complex<uni20::float128> const* A, blas_int lda, uni20::complex<uni20::float128> const* B,
                 blas_int ldb, uni20::complex<uni20::float128> beta, uni20::complex<uni20::float128>* C, blas_int ldc)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Cgemm, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  char transa_string[2]{};
  char transb_string[2]{};
  Cgemm(mplapack_binary128_detail::char_string(transa, transa_string),
        mplapack_binary128_detail::char_string(transb, transb_string), static_cast<mplapackint>(m),
        static_cast<mplapackint>(n), static_cast<mplapackint>(k), alpha,
        const_cast<uni20::complex<uni20::float128>*>(A), static_cast<mplapackint>(lda),
        const_cast<uni20::complex<uni20::float128>*>(B), static_cast<mplapackint>(ldb), beta, C,
        static_cast<mplapackint>(ldc));
}

inline void gemv(char trans, blas_int m, blas_int n, uni20::complex<uni20::float128> alpha,
                 uni20::complex<uni20::float128> const* A, blas_int lda, uni20::complex<uni20::float128> const* x,
                 blas_int incx, uni20::complex<uni20::float128> beta, uni20::complex<uni20::float128>* y, blas_int incy)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Cgemv, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  char trans_string[2]{};
  Cgemv(mplapack_binary128_detail::char_string(trans, trans_string), static_cast<mplapackint>(m),
        static_cast<mplapackint>(n), alpha, const_cast<uni20::complex<uni20::float128>*>(A),
        static_cast<mplapackint>(lda), const_cast<uni20::complex<uni20::float128>*>(x), static_cast<mplapackint>(incx),
        beta, y, static_cast<mplapackint>(incy));
}

inline void geru(blas_int m, blas_int n, uni20::complex<uni20::float128> alpha,
                 uni20::complex<uni20::float128> const* x, blas_int incx, uni20::complex<uni20::float128> const* y,
                 blas_int incy, uni20::complex<uni20::float128>* A, blas_int lda)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Cgeru, m, n, alpha, x, incx, y, incy, A, lda);
  Cgeru(static_cast<mplapackint>(m), static_cast<mplapackint>(n), alpha,
        const_cast<uni20::complex<uni20::float128>*>(x), static_cast<mplapackint>(incx),
        const_cast<uni20::complex<uni20::float128>*>(y), static_cast<mplapackint>(incy), A,
        static_cast<mplapackint>(lda));
}

inline void gerc(blas_int m, blas_int n, uni20::complex<uni20::float128> alpha,
                 uni20::complex<uni20::float128> const* x, blas_int incx, uni20::complex<uni20::float128> const* y,
                 blas_int incy, uni20::complex<uni20::float128>* A, blas_int lda)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Cgerc, m, n, alpha, x, incx, y, incy, A, lda);
  Cgerc(static_cast<mplapackint>(m), static_cast<mplapackint>(n), alpha,
        const_cast<uni20::complex<uni20::float128>*>(x), static_cast<mplapackint>(incx),
        const_cast<uni20::complex<uni20::float128>*>(y), static_cast<mplapackint>(incy), A,
        static_cast<mplapackint>(lda));
}

inline void herk(char uplo, char trans, blas_int n, blas_int k, uni20::float128 alpha,
                 uni20::complex<uni20::float128> const* A, blas_int lda, uni20::float128 beta,
                 uni20::complex<uni20::float128>* C, blas_int ldc)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Cherk, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  char uplo_string[2]{};
  char trans_string[2]{};
  Cherk(mplapack_binary128_detail::char_string(uplo, uplo_string),
        mplapack_binary128_detail::char_string(trans, trans_string), static_cast<mplapackint>(n),
        static_cast<mplapackint>(k), alpha, const_cast<uni20::complex<uni20::float128>*>(A),
        static_cast<mplapackint>(lda), beta, C, static_cast<mplapackint>(ldc));
}

inline void her2k(char uplo, char trans, blas_int n, blas_int k, uni20::complex<uni20::float128> alpha,
                  uni20::complex<uni20::float128> const* A, blas_int lda, uni20::complex<uni20::float128> const* B,
                  blas_int ldb, uni20::float128 beta, uni20::complex<uni20::float128>* C, blas_int ldc)
{
  UNI20_EXTERNAL_API_CALL(BLAS, Cher2k, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  char uplo_string[2]{};
  char trans_string[2]{};
  Cher2k(mplapack_binary128_detail::char_string(uplo, uplo_string),
         mplapack_binary128_detail::char_string(trans, trans_string), static_cast<mplapackint>(n),
         static_cast<mplapackint>(k), alpha, const_cast<uni20::complex<uni20::float128>*>(A),
         static_cast<mplapackint>(lda), const_cast<uni20::complex<uni20::float128>*>(B), static_cast<mplapackint>(ldb),
         beta, C, static_cast<mplapackint>(ldc));
}

} // namespace uni20::blas

#endif
