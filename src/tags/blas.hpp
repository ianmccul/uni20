#pragma once

#include "tags/cpu.hpp"

/// \file blas.hpp
/// \brief Tag for selecting BLAS-backed kernels.

namespace uni20
{

/// \brief Tag selecting BLAS-backed kernels layered on CPU primitives.
struct blas_tag : cpu_tag
{};

} // namespace uni20

