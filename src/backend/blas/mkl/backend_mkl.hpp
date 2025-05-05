#pragma once

#if !UNI20_BACKEND_MKL
#error "backend_mkl.hpp requires UNI20_BACKEND_MKL"
#endif

namespace uni20
{
struct mkl_tag : blas_tag
{};
} // namespace uni20
