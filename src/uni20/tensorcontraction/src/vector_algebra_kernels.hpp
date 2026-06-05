#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace uni20::tensorcontraction::detail
{

/// \brief Launch an elementwise in-place scale over a coalesced slab.
/// \param values Device pointer to the slab values.
/// \param size Number of scalar values in the slab.
/// \param alpha Scaling factor.
/// \param stream CUDA stream used for the launch.
void launch_slab_scale_kernel(double* values, std::size_t size, double alpha, cudaStream_t stream);

/// \brief Launch an elementwise AXPY update over coalesced slabs.
/// \param x Device pointer to the input slab.
/// \param y Device pointer to the output slab, updated as `y += alpha * x`.
/// \param size Number of scalar values in each slab.
/// \param alpha Scaling factor applied to `x`.
/// \param stream CUDA stream used for the launch.
void launch_slab_axpy_kernel(double const* x, double* y, std::size_t size, double alpha, cudaStream_t stream);

} // namespace uni20::tensorcontraction::detail
