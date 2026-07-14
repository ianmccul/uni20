#pragma once

#include <uni20/config.hpp>

#include <string>

/**
 * \defgroup backend_blas_openblas OpenBLAS backend
 * \ingroup backend_blas
 * \brief Runtime helpers for integrating OpenBLAS extensions with Uni20 BLAS abstractions.
 */

#if !UNI20_BACKEND_OPENBLAS
#error "backend_openblas.hpp requires UNI20_BACKEND_OPENBLAS"
#endif

namespace uni20
{
namespace blas::openblas
{
namespace detail
{
extern "C"
{
  void openblas_set_num_threads(int num_threads);
  int openblas_get_num_threads();
  int openblas_get_num_procs();
  char* openblas_get_config();
  char* openblas_get_corename();
  int openblas_get_parallel();
}

/// \brief Convert an OpenBLAS-owned C string to `std::string`.
/// \param value Pointer returned by an OpenBLAS query routine.
/// \return Copied string, or an empty string when OpenBLAS returns null.
/// \ingroup internal
inline std::string copy_string(char const* value) { return value == nullptr ? std::string{} : std::string(value); }
} // namespace detail

/// \brief OpenBLAS parallel runtime mode.
/// \ingroup backend_blas_openblas
enum class ParallelMode : int
{
  sequential = 0,
  thread = 1,
  openmp = 2
};

/// \brief Return the OpenBLAS build configuration string.
/// \return OpenBLAS-owned configuration text copied into `std::string`.
/// \ingroup backend_blas_openblas
inline std::string config() { return detail::copy_string(detail::openblas_get_config()); }

/// \brief Return the OpenBLAS CPU core name selected by the library.
/// \return OpenBLAS-owned core-name text copied into `std::string`.
/// \ingroup backend_blas_openblas
inline std::string core_name() { return detail::copy_string(detail::openblas_get_corename()); }

/// \brief Set the number of OpenBLAS worker threads.
/// \param threads Requested thread count.
/// \ingroup backend_blas_openblas
inline void set_num_threads(int threads) { detail::openblas_set_num_threads(threads); }

/// \brief Return the current number of OpenBLAS worker threads.
/// \return Current OpenBLAS thread count.
/// \ingroup backend_blas_openblas
inline int num_threads() { return detail::openblas_get_num_threads(); }

/// \brief Return the processor count reported by OpenBLAS.
/// \return OpenBLAS processor count.
/// \ingroup backend_blas_openblas
inline int num_procs() { return detail::openblas_get_num_procs(); }

/// \brief Return the OpenBLAS parallel runtime mode.
/// \return Runtime mode reported by `openblas_get_parallel`.
/// \ingroup backend_blas_openblas
inline ParallelMode parallel_mode() { return static_cast<ParallelMode>(detail::openblas_get_parallel()); }
} // namespace blas::openblas
} // namespace uni20
