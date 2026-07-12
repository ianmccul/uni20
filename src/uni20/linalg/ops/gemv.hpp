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

/// \brief Run fixed-storage tensor GEMV through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedTensorView<1> OutputTensor, class Scalar,
          uni20::RankedTensorView<2> MatrixTensor, uni20::RankedTensorView<1> InputTensor>
void gemv(BackendSelector&& selector, OutputTensor&& output, Scalar alpha, MatrixTensor const& matrix,
          InputTensor const& input, Scalar beta)
{
  dispatch_kernel(std::forward<BackendSelector>(selector), gemv_op{}, output.mdspan(), alpha, matrix.mdspan(),
                  input.mdspan(), beta);
}

/// \brief Run fixed-storage tensor GEMV using the operands' default backend selector.
template <uni20::MutableRankedTensorView<1> OutputTensor, class Scalar, uni20::RankedTensorView<2> MatrixTensor,
          uni20::RankedTensorView<1> InputTensor>
void gemv(OutputTensor&& output, Scalar alpha, MatrixTensor const& matrix, InputTensor const& input, Scalar beta)
{
  auto selector = select_backend(gemv_op{}, output, matrix, input);
  gemv(selector, std::forward<OutputTensor>(output), alpha, matrix, input, beta);
}

} // namespace uni20::linalg
