#pragma once

/**
 * \file elementwise_scale.hpp
 * \ingroup linalg
 * \brief Compiled CUDA entry points for elementwise scaling.
 */

#include <uni20/core/types.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace uni20::linalg::detail::cuda_reference
{

inline constexpr std::size_t elementwise_scale_maximum_rank = 8;
using ElementwiseScalePlan32 = StridedElementwisePlan32<2, elementwise_scale_maximum_rank>;
using ElementwiseScalePlan64 = StridedElementwisePlan64<2, elementwise_scale_maximum_rank>;
using LoweredElementwiseScalePlan = LoweredStridedElementwisePlan<2, elementwise_scale_maximum_rank>;

#define UNI20_DECLARE_ELEMENTWISE_SCALE(Scalar, Factor)                                                                \
  void enqueue_elementwise_scale(Scalar* output, Scalar const* input, Factor factor,                                   \
                                 ElementwiseScalePlan32 const& plan, cudaStream_t stream, int device);                 \
  void enqueue_elementwise_scale(Scalar* output, Scalar const* input, Factor factor,                                   \
                                 ElementwiseScalePlan64 const& plan, cudaStream_t stream, int device)

UNI20_DECLARE_ELEMENTWISE_SCALE(float, float);
UNI20_DECLARE_ELEMENTWISE_SCALE(double, double);
UNI20_DECLARE_ELEMENTWISE_SCALE(uni20::cfloat, float);
UNI20_DECLARE_ELEMENTWISE_SCALE(uni20::cfloat, uni20::cfloat);
UNI20_DECLARE_ELEMENTWISE_SCALE(uni20::cdouble, double);
UNI20_DECLARE_ELEMENTWISE_SCALE(uni20::cdouble, uni20::cdouble);

#undef UNI20_DECLARE_ELEMENTWISE_SCALE

template <class Scalar, class Factor>
inline constexpr bool supports_elementwise_scale =
    (std::same_as<Scalar, float> && std::same_as<Factor, float>) ||
    (std::same_as<Scalar, double> && std::same_as<Factor, double>) ||
    (std::same_as<Scalar, uni20::cfloat> && (std::same_as<Factor, float> || std::same_as<Factor, uni20::cfloat>)) ||
    (std::same_as<Scalar, uni20::cdouble> && (std::same_as<Factor, double> || std::same_as<Factor, uni20::cdouble>));

} // namespace uni20::linalg::detail::cuda_reference
