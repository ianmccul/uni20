#pragma once

#include "tags/cublas.hpp"

/// \file cusolver.hpp
/// \brief Tag for selecting cuSOLVER-backed kernels.

namespace uni20
{

/// \brief Tag selecting cuSOLVER-backed CUDA kernels.
struct cusolver_tag : cublas_tag
{};

} // namespace uni20

