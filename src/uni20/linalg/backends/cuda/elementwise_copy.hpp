#pragma once

/**
 * \file elementwise_copy.hpp
 * \ingroup linalg
 * \brief Typed launch boundary for CUDA accessor-respecting element copies.
 */

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/core/types.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <concepts>
#include <cstddef>

namespace uni20::linalg::detail::cuda_reference
{

enum class ElementwiseCopyTransform
{
  identity,
  conjugate
};

/// \brief Physical output and input offsets for one logical element.
struct ElementwiseCopyOffsets
{
    std::size_t output = 0;
    std::size_t input = 0;
};

/// \brief Compact logical-index plan for the precompiled CUDA copy executor.
struct ElementwiseCopyLayout
{
    static constexpr std::size_t maximum_rank = 8;

    std::size_t rank = 0;
    std::size_t element_count = 0;
    std::array<std::size_t, maximum_rank> extents{};
    std::array<std::size_t, maximum_rank> output_strides{};
    std::array<std::size_t, maximum_rank> input_strides{};

    /// \brief Decode one logical linear index into independent physical offsets.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr ElementwiseCopyOffsets offsets(std::size_t logical_index) const noexcept
    {
      ElementwiseCopyOffsets result;
      for (std::size_t axis = rank; axis > 0; --axis)
      {
        std::size_t const dimension = axis - 1;
        std::size_t const coordinate = logical_index % extents[dimension];
        logical_index /= extents[dimension];
        result.output += coordinate * output_strides[dimension];
        result.input += coordinate * input_strides[dimension];
      }
      return result;
    }
};

void enqueue_elementwise_copy(float* output, float const* input, ElementwiseCopyLayout const& layout,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(double* output, double const* input, ElementwiseCopyLayout const& layout,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(uni20::cfloat* output, uni20::cfloat const* input, ElementwiseCopyLayout const& layout,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);
void enqueue_elementwise_copy(uni20::cdouble* output, uni20::cdouble const* input, ElementwiseCopyLayout const& layout,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device);

template <class Scalar>
inline constexpr bool supports_elementwise_copy =
    std::same_as<Scalar, float> || std::same_as<Scalar, double> || std::same_as<Scalar, uni20::cfloat> ||
    std::same_as<Scalar, uni20::cdouble>;

} // namespace uni20::linalg::detail::cuda_reference
