#pragma once

/**
 * \file elementwise_copy.hpp
 * \ingroup linalg
 * \brief Typed launch boundary for CUDA accessor-respecting element copies.
 */

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>

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

inline constexpr std::size_t elementwise_copy_maximum_rank = 8;
using ElementwiseCopyPlan32 = StridedElementwisePlan32<2, elementwise_copy_maximum_rank>;
using ElementwiseCopyPlan64 = StridedElementwisePlan64<2, elementwise_copy_maximum_rank>;
using LoweredElementwiseCopyPlan = LoweredStridedElementwisePlan<2, elementwise_copy_maximum_rank>;

void enqueue_elementwise_copy(float* output, float const* input, ElementwiseCopyPlan32 const& plan,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(float* output, float const* input, ElementwiseCopyPlan64 const& plan,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(double* output, double const* input, ElementwiseCopyPlan32 const& plan,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(double* output, double const* input, ElementwiseCopyPlan64 const& plan,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(uni20::cfloat* output, uni20::cfloat const* input, ElementwiseCopyPlan32 const& plan,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(uni20::cfloat* output, uni20::cfloat const* input, ElementwiseCopyPlan64 const& plan,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(uni20::cdouble* output, uni20::cdouble const* input, ElementwiseCopyPlan32 const& plan,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(uni20::cdouble* output, uni20::cdouble const* input, ElementwiseCopyPlan64 const& plan,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);

template <class Scalar>
inline constexpr bool supports_elementwise_copy =
    std::same_as<Scalar, float> || std::same_as<Scalar, double> || std::same_as<Scalar, uni20::cfloat> ||
    std::same_as<Scalar, uni20::cdouble>;

} // namespace uni20::linalg::detail::cuda_reference
