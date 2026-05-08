#pragma once

#include <uni20/config.hpp>
#include <uni20/tags/blas.hpp>

#include <mkl.h>
#include <string>

/**
 * \defgroup backend_blas_mkl Intel MKL BLAS backend
 * \ingroup backend_blas
 * \brief Tag types and helpers for integrating Intel MKL with Uni20 BLAS abstractions.
 */

#if !UNI20_BACKEND_MKL
#error "backend_mkl.hpp requires UNI20_BACKEND_MKL"
#endif

namespace uni20
{
/// \brief Tag type that selects the Intel MKL-backed BLAS implementation.
/// \ingroup backend_blas_mkl
struct mkl_sequential_tag : blas_tag
{};

/// \brief Tag type that selects the internally threaded Intel MKL-backed BLAS implementation.
/// \ingroup backend_blas_mkl
struct mkl_threaded_tag : blas_tag
{};

namespace blas::mkl
{
/// \brief Return whether this build links the sequential MKL backend.
/// \return True when `UNI20_BACKEND_MKL_SEQUENTIAL` is enabled.
/// \ingroup backend_blas_mkl
constexpr bool sequential_backend() { return UNI20_BACKEND_MKL_SEQUENTIAL; }

/// \brief Return whether this build links the threaded MKL backend.
/// \return True when `UNI20_BACKEND_MKL_THREADED` is enabled.
/// \ingroup backend_blas_mkl
constexpr bool threaded_backend() { return UNI20_BACKEND_MKL_THREADED; }

/// \brief MKL computation domains accepted by domain-specific thread controls.
/// \ingroup backend_blas_mkl
enum class Domain : int
{
  all = MKL_DOMAIN_ALL,
  blas = MKL_DOMAIN_BLAS,
  fft = MKL_DOMAIN_FFT,
  vml = MKL_DOMAIN_VML,
  pardiso = MKL_DOMAIN_PARDISO,
  lapack = MKL_DOMAIN_LAPACK
};

/// \brief Return the oneMKL version string.
/// \return Human-readable oneMKL version text.
/// \ingroup backend_blas_mkl
inline std::string version_string()
{
  char buffer[198]{};
  MKL_Get_Version_String(buffer, static_cast<int>(sizeof(buffer)));
  return std::string(buffer);
}

/// \brief Set the global maximum number of oneMKL threads.
/// \param threads Requested thread count.
/// \ingroup backend_blas_mkl
inline void set_num_threads(int threads) { MKL_Set_Num_Threads(threads); }

/// \brief Return the current global maximum number of oneMKL threads.
/// \return Current maximum thread count.
/// \ingroup backend_blas_mkl
inline int max_threads() { return MKL_Get_Max_Threads(); }

/// \brief Set the maximum number of oneMKL threads for a specific domain.
/// \param threads Requested thread count.
/// \param domain oneMKL computation domain to configure.
/// \return oneMKL status code from `MKL_Domain_Set_Num_Threads`.
/// \ingroup backend_blas_mkl
inline int set_num_threads(int threads, Domain domain)
{
  return MKL_Domain_Set_Num_Threads(threads, static_cast<int>(domain));
}

/// \brief Return the maximum oneMKL thread count for a specific domain.
/// \param domain oneMKL computation domain to query.
/// \return Current maximum thread count for the requested domain.
/// \ingroup backend_blas_mkl
inline int max_threads(Domain domain) { return MKL_Domain_Get_Max_Threads(static_cast<int>(domain)); }
} // namespace blas::mkl
} // namespace uni20
