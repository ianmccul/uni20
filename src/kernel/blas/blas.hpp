#pragma once

#include "kernel/cpu/cpu.hpp"

/**
 * \defgroup kernel_blas BLAS kernel backends
 * \ingroup kernel
 * \brief Tag types for kernels implemented on top of BLAS libraries.
 */

namespace uni20
{

/// \brief Tag selecting BLAS-backed tensor-kernel implementations.
/// \ingroup kernel_blas
struct blas_tag : cpu_tag
{};

} // namespace uni20
