#pragma once

#include "kernel/cpu/backend.hpp"

namespace uni20
{

/// \brief Tag selecting BLAS-backed tensor-kernel implementations.
/// \ingroup kernel_blas
struct blas_tag : cpu_tag
{};

} // namespace uni20
