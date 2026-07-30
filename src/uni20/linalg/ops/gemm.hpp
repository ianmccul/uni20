#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Fixed-output Tensor GEMM front end.
 */

#include <uni20/linalg/backends/cpu/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#if UNI20_BACKEND_BLAS
#include <uni20/linalg/backends/blas/gemm.hpp>
#endif

#if UNI20_BACKEND_CUBLAS
#include <uni20/linalg/backends/cublas/gemm.hpp>
#endif

#include <utility>

namespace uni20::linalg
{
/// \brief Run fixed-storage tensor GEMM through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
void gemm(BackendSelector&& selector, OutputTensor&& output, Scalar alpha, LhsTensor const& lhs, RhsTensor const& rhs,
          Scalar beta)
{
  auto output_span = uni20::device_mdspan_of(output);
  auto lhs_span = uni20::device_mdspan_of(lhs);
  auto rhs_span = uni20::device_mdspan_of(rhs);
  dispatch_kernel(std::forward<BackendSelector>(selector), gemm_op{}, output_span, alpha, lhs_span, rhs_span, beta);
}

/// \brief Run fixed-storage tensor GEMM using the operands' default backend selector.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
void gemm(OutputTensor&& output, Scalar alpha, LhsTensor const& lhs, RhsTensor const& rhs, Scalar beta)
{
  auto selector = select_backend(gemm_op{}, output, lhs, rhs);
  gemm(selector, std::forward<OutputTensor>(output), alpha, lhs, rhs, beta);
}

} // namespace uni20::linalg
