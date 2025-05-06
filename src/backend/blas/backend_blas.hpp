#pragma once

#include "backend/backend.hpp"
#include "config.hpp"

#if !UNI20_BACKEND_BLAS
#error "backend_blas.hpp requires UNI20_BACKEND_BLAS"
#endif

namespace uni20
{

struct blas_tag : cpu_tag
{};

} // namespace uni20

#if (UNI20_BACKEND_MKL && UNI20_MKL_REPLACES_BLAS)
#include "mkl/reference_blas_mkl_direct.hpp"
#else
#include "reference/reference_blas.hpp"
#endif

#if UNI20_BACKEND_MKL
#include "mkl/backend_mkl.hpp"
#endif

#if UNI20_BACKEND_OPENBLAS
#include "openblas/backend_openblas.hpp"
#endif
