#pragma once

/// \file reference_blas_mkl_direct.hpp
/// \brief Enables MKL direct-call support for Uni20 reference BLAS wrappers.
/// \ingroup internal

/// \brief Marks that MKL direct-call entry points should be used instead of dispatcher stubs.
/// \ingroup internal
#define MKL_DIRECT_CALL
/// \brief Ensures the sequential (non-threaded) MKL direct-call path is selected.
/// \ingroup internal
#define MKL_DIRECT_CALL_SEQ
#include <mkl.h>

// wrapper functions here...
