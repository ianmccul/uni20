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
using ElementwiseFillPlan32 = StridedElementwisePlan32<1, elementwise_arithmetic_maximum_rank>;
using ElementwiseFillPlan64 = StridedElementwisePlan64<1, elementwise_arithmetic_maximum_rank>;
using LoweredElementwiseFillPlan = LoweredStridedElementwisePlan<1, elementwise_arithmetic_maximum_rank>;
using ElementwiseUnaryPlan32 = StridedElementwisePlan32<2, elementwise_arithmetic_maximum_rank>;
using ElementwiseUnaryPlan64 = StridedElementwisePlan64<2, elementwise_arithmetic_maximum_rank>;
using LoweredElementwiseUnaryPlan = LoweredStridedElementwisePlan<2, elementwise_arithmetic_maximum_rank>;
using ElementwiseBinaryPlan32 = StridedElementwisePlan32<3, elementwise_arithmetic_maximum_rank>;
using ElementwiseBinaryPlan64 = StridedElementwisePlan64<3, elementwise_arithmetic_maximum_rank>;
using LoweredElementwiseBinaryPlan = LoweredStridedElementwisePlan<3, elementwise_arithmetic_maximum_rank>;
using ElementwiseInplaceUnaryPlan32 = ElementwiseFillPlan32;
using ElementwiseInplaceUnaryPlan64 = ElementwiseFillPlan64;
using ElementwiseInplaceBinaryPlan32 = ElementwiseUnaryPlan32;
using ElementwiseInplaceBinaryPlan64 = ElementwiseUnaryPlan64;

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

template <class Scalar>
void enqueue_elementwise_fill(Scalar* output, Scalar value, ElementwiseFillPlan32 const& plan, cudaStream_t stream,
                              int device);

template <class Scalar>
void enqueue_elementwise_fill(Scalar* output, Scalar value, ElementwiseFillPlan64 const& plan, cudaStream_t stream,
                              int device);

template <class Function, class Scalar>
void enqueue_elementwise_unary(Scalar* output, Scalar const* input, Function function,
                               ElementwiseUnaryPlan32 const& plan, cudaStream_t stream, int device);

template <class Function, class Scalar>
void enqueue_elementwise_unary(Scalar* output, Scalar const* input, Function function,
                               ElementwiseUnaryPlan64 const& plan, cudaStream_t stream, int device);

template <class Function, class Scalar>
void enqueue_elementwise_binary(Scalar* output, Scalar const* lhs, Scalar const* rhs, Function function,
                                ElementwiseBinaryPlan32 const& plan, cudaStream_t stream, int device);

template <class Function, class Scalar>
void enqueue_elementwise_binary(Scalar* output, Scalar const* lhs, Scalar const* rhs, Function function,
                                ElementwiseBinaryPlan64 const& plan, cudaStream_t stream, int device);

template <class Scalar, class Factor>
void enqueue_elementwise_inplace_scale(Scalar* output, Factor factor, ElementwiseInplaceUnaryPlan32 const& plan,
                                       cudaStream_t stream, int device);

template <class Scalar, class Factor>
void enqueue_elementwise_inplace_scale(Scalar* output, Factor factor, ElementwiseInplaceUnaryPlan64 const& plan,
                                       cudaStream_t stream, int device);

template <class Function, class Scalar>
void enqueue_elementwise_inplace_binary(Scalar* output, Scalar const* input, Function function,
                                        ElementwiseInplaceBinaryPlan32 const& plan, cudaStream_t stream, int device);

template <class Function, class Scalar>
void enqueue_elementwise_inplace_binary(Scalar* output, Scalar const* input, Function function,
                                        ElementwiseInplaceBinaryPlan64 const& plan, cudaStream_t stream, int device);

template <class Scalar, class Factor>
void enqueue_elementwise_add_scaled(Scalar* output, Scalar const* input, Factor factor,
                                    ElementwiseInplaceBinaryPlan32 const& plan, cudaStream_t stream, int device);

template <class Scalar, class Factor>
void enqueue_elementwise_add_scaled(Scalar* output, Scalar const* input, Factor factor,
                                    ElementwiseInplaceBinaryPlan64 const& plan, cudaStream_t stream, int device);

} // namespace uni20::linalg::detail::cuda_reference
