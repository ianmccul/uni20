#pragma once

/**
 * \file gemv.hpp
 * \ingroup linalg
 * \brief Fixed-output Tensor GEMV front end.
 */

#include <uni20/linalg/backends/cpu/gemv.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#if UNI20_BACKEND_BLAS
#include <uni20/linalg/backends/blas/gemv.hpp>
#endif

#include <utility>

namespace uni20::linalg
{

/// \brief Run tensor GEMV through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<1> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> MatrixTensor, uni20::RankedDeviceTensorView<1> InputTensor>
void gemv(BackendSelector&& selector, OutputTensor&& output, Scalar alpha, MatrixTensor const& matrix,
          InputTensor const& input, Scalar beta)
{
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto matrix_span = uni20::detail::tensor_device_mdspan(matrix);
  auto input_span = uni20::detail::tensor_device_mdspan(input);
  dispatch_kernel(std::forward<BackendSelector>(selector), gemv_op{}, output_span, alpha, matrix_span, input_span,
                  beta);
}

/// \brief Run tensor GEMV using the operands' default backend selector.
template <uni20::MutableRankedDeviceTensorView<1> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> MatrixTensor, uni20::RankedDeviceTensorView<1> InputTensor>
void gemv(OutputTensor&& output, Scalar alpha, MatrixTensor const& matrix, InputTensor const& input, Scalar beta)
{
  auto selector = select_backend(gemv_op{}, output, matrix, input);
  gemv(selector, std::forward<OutputTensor>(output), alpha, matrix, input, beta);
}

} // namespace uni20::linalg
