#pragma once

/**
 * \file elementwise_negate.hpp
 * \ingroup linalg
 * \brief Compiled CUDA entry points for elementwise negation.
 */

#include <uni20/core/types.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace uni20::linalg::detail::cuda_reference
{

inline constexpr std::size_t elementwise_negate_maximum_rank = 8;
using ElementwiseNegatePlan32 = StridedElementwisePlan32<2, elementwise_negate_maximum_rank>;
using ElementwiseNegatePlan64 = StridedElementwisePlan64<2, elementwise_negate_maximum_rank>;
using LoweredElementwiseNegatePlan = LoweredStridedElementwisePlan<2, elementwise_negate_maximum_rank>;

void enqueue_elementwise_negate(float* output, float const* input, ElementwiseNegatePlan32 const& plan,
                                cudaStream_t stream, int device);
void enqueue_elementwise_negate(float* output, float const* input, ElementwiseNegatePlan64 const& plan,
                                cudaStream_t stream, int device);
void enqueue_elementwise_negate(double* output, double const* input, ElementwiseNegatePlan32 const& plan,
                                cudaStream_t stream, int device);
void enqueue_elementwise_negate(double* output, double const* input, ElementwiseNegatePlan64 const& plan,
                                cudaStream_t stream, int device);
void enqueue_elementwise_negate(uni20::cfloat* output, uni20::cfloat const* input, ElementwiseNegatePlan32 const& plan,
                                cudaStream_t stream, int device);
void enqueue_elementwise_negate(uni20::cfloat* output, uni20::cfloat const* input, ElementwiseNegatePlan64 const& plan,
                                cudaStream_t stream, int device);
void enqueue_elementwise_negate(uni20::cdouble* output, uni20::cdouble const* input,
                                ElementwiseNegatePlan32 const& plan, cudaStream_t stream, int device);
void enqueue_elementwise_negate(uni20::cdouble* output, uni20::cdouble const* input,
                                ElementwiseNegatePlan64 const& plan, cudaStream_t stream, int device);

template <class Scalar>
inline constexpr bool supports_elementwise_negate =
    std::same_as<Scalar, float> || std::same_as<Scalar, double> || std::same_as<Scalar, uni20::cfloat> ||
    std::same_as<Scalar, uni20::cdouble>;

} // namespace uni20::linalg::detail::cuda_reference
