#pragma once

/// \file tags.hpp
/// \brief Tag hierarchy for selecting Uni20 backend implementations.

namespace uni20
{

/// \brief Tag selecting CPU-based tensor and linear-algebra kernels.
struct cpu_tag
{};

/// \brief Tag selecting BLAS-backed kernels layered on CPU primitives.
struct blas_tag : cpu_tag
{};

/// \brief Tag selecting LAPACK-backed kernels layered on BLAS primitives.
struct lapack_tag : blas_tag
{};

/// \brief Tag selecting CUDA-capable kernels.
struct cuda_tag
{};

/// \brief Tag selecting cuBLAS-backed CUDA kernels.
struct cublas_tag : cuda_tag
{};

/// \brief Tag selecting cuSOLVER-backed CUDA kernels.
struct cusolver_tag : cublas_tag
{};

/// \brief Alias for the default backend tag when none is explicitly provided.
using default_tag = cpu_tag;

} // namespace uni20

