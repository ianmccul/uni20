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
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> InputTensor,
          class TimeScalar>
void matrix_exponential(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input, TimeScalar time)
{
  dispatch_kernel(std::forward<BackendSelector>(selector), matrix_exponential_op{}, output.mdspan(), input.mdspan(),
                  time);
}

/// \brief Compute a fixed-output matrix exponential using tensor storage policy.
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> InputTensor, class TimeScalar>
void matrix_exponential(OutputTensor&& output, InputTensor const& input, TimeScalar time)
{
  auto selector = select_backend(matrix_exponential_op{}, output, input);
  matrix_exponential(selector, std::forward<OutputTensor>(output), input, time);
}

} // namespace uni20::linalg
