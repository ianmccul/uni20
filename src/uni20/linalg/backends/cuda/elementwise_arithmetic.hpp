#pragma once

/**
 * \file elementwise_arithmetic.hpp
 * \ingroup linalg
 * \brief Compiled CUDA entry points for named stateless arithmetic transforms.
 */

#include <uni20/core/types.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>
#include <uni20/linalg/elementwise_functions.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace uni20::linalg::detail::cuda_reference
{

inline constexpr std::size_t elementwise_arithmetic_maximum_rank = 8;
using ElementwiseUnaryPlan32 = StridedElementwisePlan32<2, elementwise_arithmetic_maximum_rank>;
using ElementwiseUnaryPlan64 = StridedElementwisePlan64<2, elementwise_arithmetic_maximum_rank>;
using LoweredElementwiseUnaryPlan = LoweredStridedElementwisePlan<2, elementwise_arithmetic_maximum_rank>;
using ElementwiseBinaryPlan32 = StridedElementwisePlan32<3, elementwise_arithmetic_maximum_rank>;
using ElementwiseBinaryPlan64 = StridedElementwisePlan64<3, elementwise_arithmetic_maximum_rank>;
using LoweredElementwiseBinaryPlan = LoweredStridedElementwisePlan<3, elementwise_arithmetic_maximum_rank>;

template <class Function>
concept RegisteredStatelessUnary =
    std::same_as<Function, negate> || std::same_as<Function, square> || std::same_as<Function, reciprocal>;

template <class Function>
concept RegisteredStatelessBinary = std::same_as<Function, add> || std::same_as<Function, subtract> ||
                                    std::same_as<Function, multiply> || std::same_as<Function, divide>;

template <class Scalar>
inline constexpr bool supports_elementwise_arithmetic =
    std::same_as<Scalar, float> || std::same_as<Scalar, double> || std::same_as<Scalar, uni20::cfloat> ||
    std::same_as<Scalar, uni20::cdouble>;

template <RegisteredStatelessUnary Function, class Scalar>
void enqueue_elementwise_unary(Scalar* output, Scalar const* input, Function function,
                               ElementwiseUnaryPlan32 const& plan, cudaStream_t stream, int device);

template <RegisteredStatelessUnary Function, class Scalar>
void enqueue_elementwise_unary(Scalar* output, Scalar const* input, Function function,
                               ElementwiseUnaryPlan64 const& plan, cudaStream_t stream, int device);

template <RegisteredStatelessBinary Function, class Scalar>
void enqueue_elementwise_binary(Scalar* output, Scalar const* lhs, Scalar const* rhs, Function function,
                                ElementwiseBinaryPlan32 const& plan, cudaStream_t stream, int device);

template <RegisteredStatelessBinary Function, class Scalar>
void enqueue_elementwise_binary(Scalar* output, Scalar const* lhs, Scalar const* rhs, Function function,
                                ElementwiseBinaryPlan64 const& plan, cudaStream_t stream, int device);

} // namespace uni20::linalg::detail::cuda_reference
