#pragma once

#include "tags/blas.hpp"

/// \file lapack.hpp
/// \brief Tag for selecting LAPACK-backed kernels.

namespace uni20
{

/// \brief Tag selecting LAPACK-backed kernels layered on BLAS primitives.
struct lapack_tag : blas_tag
{};

} // namespace uni20

