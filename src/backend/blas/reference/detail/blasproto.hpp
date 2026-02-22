/// \file blasproto.hpp
/// \brief Templated BLAS interface declarations and wrappers
/// \ingroup internal
///
/// \details
/// This header defines overloads for BLAS level 2 and 3 routines
/// such as `gemm`, `gemv`, `syrk`, `herk`, etc., using Fortran-style
/// symbols like `dgemm_`, `zher2k_`, and so on.
///
/// It is designed to be included **multiple times**, with the following
/// preprocessor macros defined before each inclusion:
///
/// - `BLASCHAR`: One of `s`, `d`, `c`, or `z`, indicating the BLAS prefix.
/// - `BLASTYPE`: The C++ scalar type corresponding to the BLAS prefix.
/// - `BLASREALTYPE`: The corresponding real type, used by `herk`/`her2k`.
/// - `BLASCOMPLEX`: Either `0` (for real types) or `1` (for complex types).
///
/// Each inclusion generates extern "C" declarations and inline C++ overloads,
/// optionally including trace hooks via `UNI20_API_CALL`.
///
/// \note This file must not use `#pragma once` or include guards.
///
/// \par Valid combinations of macros:
///
/// | Fortran Type     | `BLASCHAR` | `BLASTYPE`     | `BLASREALTYPE` | `BLASCOMPLEX`  |
/// |------------------|------------|----------------|----------------|----------------|
/// | REAL             | `s`        | `float32`      | `float32`      | `0`            |
/// | DOUBLE PRECISION | `d`        | `float64`      | `float64`      | `0`            |
/// | COMPLEX          | `c`        | `complex64`    | `float32`      | `1`            |
/// | COMPLEX*16       | `z`        | `complex128`   | `float64`      | `1`            |

#ifndef BLASCHAR
#error "BLASCHAR must be defined before including blasproto.hpp"
#endif

#ifndef BLASTYPE
#error "BLASTYPE must be defined before including blasproto.hpp"
#endif

#ifndef BLASCOMPLEX
#error "BLASCOMPLEX must be defined before including blasproto.hpp"
#endif

// Concatenation helpers
/// \brief Internal macro that concatenates two tokens without macro expansion on \p x.
/// \ingroup internal
#define UNI20_INTERNAL_CONCAT2(x, y) x##y
/// \brief Internal macro that concatenates two tokens after expanding both arguments.
/// \ingroup internal
#define UNI20_INTERNAL_CONCAT(x, y) UNI20_INTERNAL_CONCAT2(x, y)

// Form full Fortran symbol name
/// \brief Builds the mangled Fortran symbol name for the current BLAS instantiation.
/// \ingroup internal
#define UNI20_INTERNAL_BLAS_FN(NAME) UNI20_INTERNAL_CONCAT(UNI20_INTERNAL_CONCAT(BLASCHAR, NAME), _)

/// \addtogroup backend_blas_reference
/// \{

namespace uni20::blas
{
namespace detail
{

extern "C"
{

  // Level 3
  void UNI20_INTERNAL_BLAS_FN(gemm)(char const* transa, char const* transb, blas_int const* m, blas_int const* n,
                                    blas_int const* k, BLASTYPE const* alpha, BLASTYPE const* A, blas_int const* lda,
                                    BLASTYPE const* B, blas_int const* ldb, BLASTYPE const* beta, BLASTYPE* C,
                                    blas_int const* ldc);

  // Level 2
  void UNI20_INTERNAL_BLAS_FN(gemv)(char const* trans, blas_int const* m, blas_int const* n, BLASTYPE const* alpha,
                                    BLASTYPE const* A, blas_int const* lda, BLASTYPE const* x, blas_int const* incx,
                                    BLASTYPE const* beta, BLASTYPE* y, blas_int const* incy);

#if BLASCOMPLEX

#ifndef BLASREALTYPE
#error "BLASREALTYPE must be defined before including blasproto.hpp for BLASCOMPLEX functions"
#endif

  void UNI20_INTERNAL_BLAS_FN(geru)(blas_int const* m, blas_int const* n, BLASTYPE const* alpha, BLASTYPE const* x,
                                    blas_int const* incx, BLASTYPE const* y, blas_int const* incy, BLASTYPE* A,
                                    blas_int const* lda);

  void UNI20_INTERNAL_BLAS_FN(gerc)(blas_int const* m, blas_int const* n, BLASTYPE const* alpha, BLASTYPE const* x,
                                    blas_int const* incx, BLASTYPE const* y, blas_int const* incy, BLASTYPE* A,
                                    blas_int const* lda);

  void UNI20_INTERNAL_BLAS_FN(herk)(char const* uplo, char const* trans, blas_int const* n, blas_int const* k,
                                    BLASREALTYPE const* alpha, BLASTYPE const* A, blas_int const* lda,
                                    BLASREALTYPE const* beta, BLASTYPE* C, blas_int const* ldc);

  void UNI20_INTERNAL_BLAS_FN(her2k)(char const* uplo, char const* trans, blas_int const* n, blas_int const* k,
                                     BLASTYPE const* alpha, BLASTYPE const* A, blas_int const* lda, BLASTYPE const* B,
                                     blas_int const* ldb, BLASREALTYPE const* beta, BLASTYPE* C, blas_int const* ldc);

#else

  void UNI20_INTERNAL_BLAS_FN(ger)(blas_int const* m, blas_int const* n, BLASTYPE const* alpha, BLASTYPE const* x,
                                   blas_int const* incx, BLASTYPE const* y, blas_int const* incy, BLASTYPE* A,
                                   blas_int const* lda);

  void UNI20_INTERNAL_BLAS_FN(syrk)(char const* uplo, char const* trans, blas_int const* n, blas_int const* k,
                                    BLASTYPE const* alpha, BLASTYPE const* A, blas_int const* lda, BLASTYPE const* beta,
                                    BLASTYPE* C, blas_int const* ldc);

  void UNI20_INTERNAL_BLAS_FN(syr2k)(char const* uplo, char const* trans, blas_int const* n, blas_int const* k,
                                     BLASTYPE const* alpha, BLASTYPE const* A, blas_int const* lda, BLASTYPE const* B,
                                     blas_int const* ldb, BLASTYPE const* beta, BLASTYPE* C, blas_int const* ldc);
#endif

} // extern "C"
} // namespace detail

// ----------- Wrappers -------------

inline void gemm(char transa, char transb, blas_int m, blas_int n, blas_int k, BLASTYPE alpha, BLASTYPE const* A,
                 blas_int lda, BLASTYPE const* B, blas_int ldb, BLASTYPE beta, BLASTYPE* C, blas_int ldc)
{
  UNI20_API_CALL(BLAS, UNI20_INTERNAL_BLAS_FN(gemm), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  detail::UNI20_INTERNAL_BLAS_FN(gemm)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void gemv(char trans, blas_int m, blas_int n, BLASTYPE alpha, BLASTYPE const* A, blas_int lda, BLASTYPE const* x,
                 blas_int incx, BLASTYPE beta, BLASTYPE* y, blas_int incy)
{
  UNI20_API_CALL(BLAS, UNI20_INTERNAL_BLAS_FN(gemv), trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  detail::UNI20_INTERNAL_BLAS_FN(gemv)(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

#if BLASCOMPLEX

inline void geru(blas_int m, blas_int n, BLASTYPE alpha, BLASTYPE const* x, blas_int incx, BLASTYPE const* y,
                 blas_int incy, BLASTYPE* A, blas_int lda)
{
  UNI20_API_CALL(BLAS, UNI20_INTERNAL_BLAS_FN(geru), m, n, alpha, x, incx, y, incy, A, lda);
  detail::UNI20_INTERNAL_BLAS_FN(geru)(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
}

inline void gerc(blas_int m, blas_int n, BLASTYPE alpha, BLASTYPE const* x, blas_int incx, BLASTYPE const* y,
                 blas_int incy, BLASTYPE* A, blas_int lda)
{
  UNI20_API_CALL(BLAS, UNI20_INTERNAL_BLAS_FN(gerc), m, n, alpha, x, incx, y, incy, A, lda);
  detail::UNI20_INTERNAL_BLAS_FN(gerc)(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
}

inline void herk(char uplo, char trans, blas_int n, blas_int k, BLASREALTYPE alpha, BLASTYPE const* A, blas_int lda,
                 BLASREALTYPE beta, BLASTYPE* C, blas_int ldc)
{
  UNI20_API_CALL(BLAS, UNI20_INTERNAL_BLAS_FN(herk), uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  detail::UNI20_INTERNAL_BLAS_FN(herk)(&uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
}

inline void her2k(char uplo, char trans, blas_int n, blas_int k, BLASTYPE alpha, BLASTYPE const* A, blas_int lda,
                  BLASTYPE const* B, blas_int ldb, BLASREALTYPE beta, BLASTYPE* C, blas_int ldc)
{
  UNI20_API_CALL(BLAS, UNI20_INTERNAL_BLAS_FN(her2k), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  detail::UNI20_INTERNAL_BLAS_FN(her2k)(&uplo, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

#else

inline void ger(blas_int m, blas_int n, BLASTYPE alpha, BLASTYPE const* x, blas_int incx, BLASTYPE const* y,
                blas_int incy, BLASTYPE* A, blas_int lda)
{
  UNI20_API_CALL(BLAS, UNI20_INTERNAL_BLAS_FN(ger), m, n, alpha, x, incx, y, incy, A, lda);
  detail::UNI20_INTERNAL_BLAS_FN(ger)(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
}

inline void syrk(char uplo, char trans, blas_int n, blas_int k, BLASTYPE alpha, BLASTYPE const* A, blas_int lda,
                 BLASTYPE beta, BLASTYPE* C, blas_int ldc)
{
  UNI20_API_CALL(BLAS, UNI20_INTERNAL_BLAS_FN(syrk), uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  detail::UNI20_INTERNAL_BLAS_FN(syrk)(&uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
}

inline void syr2k(char uplo, char trans, blas_int n, blas_int k, BLASTYPE alpha, BLASTYPE const* A, blas_int lda,
                  BLASTYPE const* B, blas_int ldb, BLASTYPE beta, BLASTYPE* C, blas_int ldc)
{
  UNI20_API_CALL(BLAS, UNI20_INTERNAL_BLAS_FN(syr2k), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  detail::UNI20_INTERNAL_BLAS_FN(syr2k)(&uplo, &trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
#endif

} // namespace uni20::blas

// Clean up internal macros
#undef UNI20_INTERNAL_CONCAT
#undef UNI20_INTERNAL_CONCAT2
#undef UNI20_INTERNAL_BLAS_FN

/// \}
