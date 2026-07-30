#pragma once

/**
 * \file matrix_exponential.hpp
 * \ingroup linalg
 * \brief Fixed-output dense matrix exponential operation.
 */

#include <uni20/linalg/backends/cpu/matrix_exponential.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <utility>

namespace uni20::linalg
{

/// \brief Compute a fixed-output matrix exponential through an explicit selector.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> InputTensor, class TimeScalar>
void matrix_exponential(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input, TimeScalar time)
{
  auto output_descriptor = uni20::device_mdspan_of(output);
  auto input_descriptor = uni20::device_mdspan_of(input);
  dispatch_kernel(std::forward<BackendSelector>(selector), matrix_exponential_op{}, output_descriptor, input_descriptor,
                  time);
}

/// \brief Compute a fixed-output matrix exponential using tensor storage policy.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::RankedDeviceTensorView<2> InputTensor,
          class TimeScalar>
void matrix_exponential(OutputTensor&& output, InputTensor const& input, TimeScalar time)
{
  auto selector = select_backend(matrix_exponential_op{}, output, input);
  matrix_exponential(selector, std::forward<OutputTensor>(output), input, time);
}

} // namespace uni20::linalg
