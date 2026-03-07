#pragma once

/// \file backend_blas.hpp
/// \brief Central include that selects the configured BLAS backend wrappers.
/// \ingroup backend_blas

#include <uni20/backend/backend.hpp>
#include <uni20/config.hpp>

#if !UNI20_BACKEND_BLAS
#error "backend_blas.hpp requires UNI20_BACKEND_BLAS"
#endif

#if (UNI20_BACKEND_MKL && UNI20_MKL_REPLACES_BLAS)
#include <uni20/backend/blas/mkl/reference_blas_mkl_direct.hpp>
#else
#include <uni20/backend/blas/reference/reference_blas.hpp>
#endif

#if UNI20_BACKEND_MKL
#include <uni20/backend/blas/mkl/backend_mkl.hpp>
#endif

#if UNI20_BACKEND_OPENBLAS
#include <uni20/backend/blas/openblas/backend_openblas.hpp>
#endif
