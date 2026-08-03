#pragma once

/**
 * \file elementwise_conjugate.hpp
 * \ingroup linalg
 * \brief Typed launch boundary for CUDA accessor-respecting in-place conjugation.
 */

#include <uni20/core/types.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>

namespace uni20::linalg::detail::cuda_reference
{

inline constexpr std::size_t elementwise_conjugate_maximum_rank = 8;
using ElementwiseConjugatePlan32 = StridedElementwisePlan32<1, elementwise_conjugate_maximum_rank>;
using ElementwiseConjugatePlan64 = StridedElementwisePlan64<1, elementwise_conjugate_maximum_rank>;
using LoweredElementwiseConjugatePlan = LoweredStridedElementwisePlan<1, elementwise_conjugate_maximum_rank>;

void enqueue_elementwise_conjugate(uni20::cfloat* output, ElementwiseConjugatePlan32 const& plan, cudaStream_t stream,
                                   int device);
void enqueue_elementwise_conjugate(uni20::cfloat* output, ElementwiseConjugatePlan64 const& plan, cudaStream_t stream,
                                   int device);
void enqueue_elementwise_conjugate(uni20::cdouble* output, ElementwiseConjugatePlan32 const& plan, cudaStream_t stream,
                                   int device);
void enqueue_elementwise_conjugate(uni20::cdouble* output, ElementwiseConjugatePlan64 const& plan, cudaStream_t stream,
                                   int device);

template <class Scalar>
inline constexpr bool supports_elementwise_conjugate =
    std::same_as<Scalar, uni20::cfloat> || std::same_as<Scalar, uni20::cdouble>;

} // namespace uni20::linalg::detail::cuda_reference
