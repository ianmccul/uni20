#pragma once

/**
 * \file gemv.hpp
 * \ingroup linalg
 * \brief Async Tensor wrapper for fixed-output matrix-vector multiplication.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/concepts.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/linalg/async/dispatch.hpp>
#include <uni20/linalg/ops/gemv.hpp>

#include <memory>
#include <tuple>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::RankedTensorView<2> MatrixTensor,
          uni20::RankedTensorView<1> InputTensor, class AlphaAwaiter, class BetaAwaiter>
async::AsyncTask co_gemv(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                         async::ReadBuffer<MatrixTensor> matrix, async::ReadBuffer<InputTensor> input,
                         AlphaAwaiter alpha, BetaAwaiter beta)
{
  auto output_awaiter = uni20::detail::mutable_async_tensor_awaiter(output);
  auto awaited = co_await async::all(output_awaiter, matrix, input, alpha, beta);
  decltype(auto) output_value = uni20::detail::mutable_async_tensor_value<OutputTensor>(std::get<0>(awaited));
  auto const& matrix_value = std::get<1>(awaited);
  auto const& input_value = std::get<2>(awaited);
  using scalar_type = uni20::tensor_element_t<OutputTensor>;
  scalar_type const alpha_value = static_cast<scalar_type>(std::get<3>(awaited));
  scalar_type const beta_value = static_cast<scalar_type>(std::get<4>(awaited));

  auto output_descriptor = uni20::mdspec_of(output_value);
  auto matrix_descriptor = uni20::mdspec_of(matrix_value);
  auto input_descriptor = uni20::mdspec_of(input_value);
  co_await co_dispatch_kernel(selector, gemv_op{}, output_descriptor, alpha_value, matrix_descriptor, input_descriptor,
                              beta_value);
  co_return;
}

template <class BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::RankedTensorView<2> MatrixTensor,
          uni20::RankedTensorView<1> InputTensor, class AlphaAwaiter, class BetaAwaiter>
void schedule_async_gemv(BackendSelector selector, async::Async<OutputTensor>& output,
                         async::Async<MatrixTensor> const& matrix, async::Async<InputTensor> const& input,
                         AlphaAwaiter alpha, BetaAwaiter beta)
{
  auto const* const output_queue = std::addressof(output.queue());
  ERROR_IF(output_queue == std::addressof(matrix.queue()) || output_queue == std::addressof(input.queue()),
           "async gemv output must not share an epoch queue with an input");
  auto task =
      co_gemv(std::move(selector), output.write(), matrix.read(), input.read(), std::move(alpha), std::move(beta));
  task.debug_name("gemv");
  async::schedule(std::move(task));
}
} // namespace detail

/// \brief Schedule fixed-output GEMV through an explicit backend selector.
/// \details Tensor operands are asynchronous; `alpha` and `beta` may be immediate or Async scalars.
/// \pre The output is constructed, has a distinct epoch queue from both inputs, and has compatible shape.
template <KernelBackendSelector BackendSelector, uni20::AsyncTensorOutput OutputTensor, class Alpha,
          uni20::RankedTensorView<2> MatrixTensor, uni20::RankedTensorView<1> InputTensor, class Beta>
  requires uni20::MutableRankedTensorView<OutputTensor, 1> &&
           AsyncOperationScalar<Alpha, uni20::tensor_element_t<OutputTensor>> &&
           AsyncOperationScalar<Beta, uni20::tensor_element_t<OutputTensor>>
void gemv(BackendSelector selector, async::Async<OutputTensor>& output, Alpha&& alpha,
          async::Async<MatrixTensor> const& matrix, async::Async<InputTensor> const& input, Beta&& beta)
{
  detail::schedule_async_gemv(std::move(selector), output, matrix, input, async::read(std::forward<Alpha>(alpha)),
                              async::read(std::forward<Beta>(beta)));
}

/// \brief Schedule fixed-output GEMV using static Tensor policy.
template <uni20::AsyncTensorOutput OutputTensor, class Alpha, uni20::RankedTensorView<2> MatrixTensor,
          uni20::RankedTensorView<1> InputTensor, class Beta>
  requires uni20::MutableRankedTensorView<OutputTensor, 1> &&
           AsyncOperationScalar<Alpha, uni20::tensor_element_t<OutputTensor>> &&
           AsyncOperationScalar<Beta, uni20::tensor_element_t<OutputTensor>>
void gemv(async::Async<OutputTensor>& output, Alpha&& alpha, async::Async<MatrixTensor> const& matrix,
          async::Async<InputTensor> const& input, Beta&& beta)
{
  auto selector = select_backend_for<OutputTensor, MatrixTensor, InputTensor>(gemv_op{});
  gemv(std::move(selector), output, std::forward<Alpha>(alpha), matrix, input, std::forward<Beta>(beta));
}

} // namespace uni20::linalg
