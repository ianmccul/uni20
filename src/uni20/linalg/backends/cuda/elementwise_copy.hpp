#pragma once

/**
 * \file elementwise_copy.hpp
 * \ingroup linalg
 * \brief Typed launch boundary for CUDA accessor-respecting element copies.
 */

#include <uni20/core/types.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>

namespace uni20::linalg::detail::cuda_reference
{

enum class ElementwiseCopyTransform
{
  identity,
  conjugate
};

void enqueue_elementwise_copy(float* output, float const* input, std::size_t count, ElementwiseCopyTransform transform,
                              cudaStream_t stream, int device);
void enqueue_elementwise_copy(double* output, double const* input, std::size_t count,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(uni20::cfloat* output, uni20::cfloat const* input, std::size_t count,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(uni20::cdouble* output, uni20::cdouble const* input, std::size_t count,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);

template <class Scalar>
inline constexpr bool supports_elementwise_copy =
    std::same_as<Scalar, float> || std::same_as<Scalar, double> || std::same_as<Scalar, uni20::cfloat> ||
    std::same_as<Scalar, uni20::cdouble>;

} // namespace uni20::linalg::detail::cuda_reference
