#pragma once

/**
 * \file conjugate_inplace.hpp
 * \ingroup linalg
 * \brief Async Tensor wrapper for eager in-place conjugation.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/linalg/async/dispatch.hpp>
#include <uni20/tensor/conjugate_inplace.hpp>

#include <tuple>
#include <utility>

namespace uni20
{
namespace detail
{
template <class BackendSelector, AsyncTensorOutput Tensor>
async::AsyncTask co_conjugate_inplace(BackendSelector const selector, async::WriteBuffer<Tensor> tensor)
{
  auto tensor_awaiter = mutable_async_tensor_awaiter(tensor);
  auto awaited = co_await async::all(tensor_awaiter);
  decltype(auto) tensor_value = mutable_async_tensor_value<Tensor>(std::get<0>(awaited));
  auto descriptor = mdspec_of(tensor_value);
  co_await linalg::co_dispatch_kernel(selector, linalg::conjugate_inplace_op{}, descriptor);
  co_return;
}

template <class BackendSelector, AsyncTensorOutput Tensor>
void schedule_async_conjugate_inplace(BackendSelector selector, async::Async<Tensor>& tensor)
{
  auto task = co_conjugate_inplace(std::move(selector), tensor.write());
  task.debug_name("conjugate_inplace");
  async::schedule(std::move(task));
}
} // namespace detail

/// \brief Schedule eager in-place conjugation through an explicit selector.
template <class BackendSelector, AsyncTensorOutput Tensor>
void conjugate_inplace(BackendSelector selector, async::Async<Tensor>& tensor)
{
  detail::schedule_async_conjugate_inplace(std::move(selector), tensor);
}

/// \brief Schedule eager in-place conjugation using static Tensor policy.
template <AsyncTensorOutput Tensor> void conjugate_inplace(async::Async<Tensor>& tensor)
{
  auto selector = linalg::select_backend_for<Tensor>(linalg::conjugate_inplace_op{});
  detail::schedule_async_conjugate_inplace(std::move(selector), tensor);
}

} // namespace uni20
