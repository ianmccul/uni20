#pragma once

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
struct mkl_tag : blas_tag
{};
} // namespace uni20
