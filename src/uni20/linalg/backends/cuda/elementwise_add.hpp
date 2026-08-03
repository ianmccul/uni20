#pragma once

/**
 * \file elementwise_add.hpp
 * \ingroup linalg
 * \brief Compiled CUDA entry points for elementwise binary addition.
 */

#include <uni20/core/types.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>

namespace uni20::linalg::detail::cuda_reference
{

inline constexpr std::size_t elementwise_add_maximum_rank = 8;
using ElementwiseAddPlan32 = StridedElementwisePlan32<3, elementwise_add_maximum_rank>;
using ElementwiseAddPlan64 = StridedElementwisePlan64<3, elementwise_add_maximum_rank>;
using LoweredElementwiseAddPlan = LoweredStridedElementwisePlan<3, elementwise_add_maximum_rank>;

#define UNI20_DECLARE_ELEMENTWISE_ADD(Scalar)                                                                          \
  void enqueue_elementwise_add(Scalar* output, Scalar const* lhs, Scalar const* rhs, ElementwiseAddPlan32 const& plan, \
                               cudaStream_t stream, int device);                                                       \
  void enqueue_elementwise_add(Scalar* output, Scalar const* lhs, Scalar const* rhs, ElementwiseAddPlan64 const& plan, \
                               cudaStream_t stream, int device)

UNI20_DECLARE_ELEMENTWISE_ADD(float);
UNI20_DECLARE_ELEMENTWISE_ADD(double);
UNI20_DECLARE_ELEMENTWISE_ADD(uni20::cfloat);
UNI20_DECLARE_ELEMENTWISE_ADD(uni20::cdouble);

#undef UNI20_DECLARE_ELEMENTWISE_ADD

template <class Scalar>
inline constexpr bool supports_elementwise_add =
    std::same_as<Scalar, float> || std::same_as<Scalar, double> || std::same_as<Scalar, uni20::cfloat> ||
    std::same_as<Scalar, uni20::cdouble>;

} // namespace uni20::linalg::detail::cuda_reference
