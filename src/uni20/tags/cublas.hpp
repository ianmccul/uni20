#pragma once

#include <uni20/tags/cuda.hpp>

/// \file cublas.hpp
/// \brief Tag for selecting cuBLAS-backed kernels.

namespace uni20
{

/// \brief Tag selecting cuBLAS-backed CUDA kernels.
struct cublas_tag : cuda_tag
{};

} // namespace uni20
