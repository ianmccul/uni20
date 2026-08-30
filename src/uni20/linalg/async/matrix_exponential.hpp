#pragma once

/**
 * \file matrix_exponential.hpp
 * \ingroup linalg
 * \brief Async Tensor wrapper for fixed-output matrix exponentiation.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/linalg/async/dispatch.hpp>
#include <uni20/linalg/ops/matrix_exponential.hpp>

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::RankedTensorView<2> InputTensor,
          class TimeAwaiter>
async::AsyncTask co_matrix_exponential(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                                       async::ReadBuffer<InputTensor> input, TimeAwaiter time)
{
  auto output_awaiter = uni20::detail::mutable_async_tensor_awaiter(output);
  auto awaited = co_await async::all(output_awaiter, input, time);
  decltype(auto) output_value = uni20::detail::mutable_async_tensor_value<OutputTensor>(std::get<0>(awaited));
  auto const& input_value = std::get<1>(awaited);
  using time_type = std::remove_cvref_t<decltype(std::get<2>(awaited))>;
  time_type const time_value = std::get<2>(awaited);

  auto output_descriptor = uni20::mdspec_of(output_value);
  auto input_descriptor = uni20::mdspec_of(input_value);
  co_await co_dispatch_kernel(selector, matrix_exponential_op{}, output_descriptor, input_descriptor, time_value);
  co_return;
}

template <class BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::RankedTensorView<2> InputTensor,
          class TimeAwaiter>
void schedule_async_matrix_exponential(BackendSelector selector, async::Async<OutputTensor>& output,
                                       async::Async<InputTensor> const& input, TimeAwaiter time)
{
  ERROR_IF(std::addressof(output.queue()) == std::addressof(input.queue()),
           "async matrix exponential output must not share an epoch queue with its input");
  auto task = co_matrix_exponential(std::move(selector), output.write(), input.read(), std::move(time));
  task.debug_name("matrix_exponential");
  async::schedule(std::move(task));
}
} // namespace detail

/// \brief Schedule fixed-output matrix exponentiation through an explicit selector.
/// \details Tensor operands are asynchronous; `time` may be an immediate or Async scalar.
/// \pre The output is constructed, has a distinct epoch queue from the input, and has compatible square shape.
template <KernelBackendSelector BackendSelector, uni20::AsyncTensorOutput OutputTensor,
          uni20::RankedTensorView<2> InputTensor, class TimeScalar>
  requires uni20::MutableRankedTensorView<OutputTensor, 2> &&
           requires(TimeScalar&& time) { async::read(std::forward<TimeScalar>(time)); }
void matrix_exponential(BackendSelector selector, async::Async<OutputTensor>& output,
                        async::Async<InputTensor> const& input, TimeScalar&& time)
{
  detail::schedule_async_matrix_exponential(std::move(selector), output, input,
                                            async::read(std::forward<TimeScalar>(time)));
}

/// \brief Schedule fixed-output matrix exponentiation using static Tensor policy.
template <uni20::AsyncTensorOutput OutputTensor, uni20::RankedTensorView<2> InputTensor, class TimeScalar>
  requires uni20::MutableRankedTensorView<OutputTensor, 2> &&
           requires(TimeScalar&& time) { async::read(std::forward<TimeScalar>(time)); }
void matrix_exponential(async::Async<OutputTensor>& output, async::Async<InputTensor> const& input, TimeScalar&& time)
{
  auto selector = select_backend_for<OutputTensor, InputTensor>(matrix_exponential_op{});
  matrix_exponential(std::move(selector), output, input, std::forward<TimeScalar>(time));
}

} // namespace uni20::linalg
