#pragma once

/**
 * \file copy.hpp
 * \ingroup linalg
 * \brief Async Tensor copy and placement-materialization wrappers.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/linalg/async/dispatch.hpp>
#include <uni20/tensor/copy.hpp>

#if UNI20_BACKEND_CUDA
#include <uni20/linalg/backends/cuda/copy_task.hpp>
#include <uni20/storage/cuda_storage.hpp>
#endif

#include <concepts>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class OutputTensor, class InputTensor>
concept AsyncCopyTensors = AsyncTensorOutput<OutputTensor> && TensorView<InputTensor> &&
                           (tensor_mdspec_t<OutputTensor>::rank() == tensor_mdspec_t<InputTensor>::rank()) &&
                           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>>;

#if UNI20_BACKEND_CUDA
template <cuda::BufferMdspec InputMdspan>
[[nodiscard]] cuda::DeviceResources& cuda_copy_resources(InputMdspan const& input)
{
  return input.data_descriptor().buffer().resources();
}
#endif

template <AsyncTensorOutput OutputTensor, TensorView InputTensor>
[[nodiscard]] OutputTensor& prepare_async_copy_output(async::shared_storage<OutputTensor>& storage,
                                                      InputTensor const& input)
{
  if (storage.constructed())
  {
    prepare_output(*storage, input.extents());
    return *storage;
  }

  using extents_type = tensor_extents_t<OutputTensor>;
  auto const extents = convert_tensor_extents<extents_type>(input.extents());
  if constexpr (std::constructible_from<OutputTensor, extents_type const&>)
  {
    return storage.emplace(extents);
  }
#if UNI20_BACKEND_CUDA
  else if constexpr (std::same_as<tensor_storage_policy_t<OutputTensor>, CudaStorage> &&
                     cuda::BufferMdspec<tensor_mdspec_t<InputTensor>>)
  {
    auto input_descriptor = mdspec_of(input);
    return storage.emplace(cuda_copy_resources(input_descriptor), extents);
  }
#endif
  else
  {
    throw async::buffer_write_uninitialized{};
  }
}

template <class BackendSelector, AsyncTensorOutput OutputTensor, TensorView InputTensor>
async::AsyncTask co_copy(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                         async::ReadBuffer<InputTensor> input)
{
  if constexpr (async::is_async_alias_v<OutputTensor>)
  {
    AsyncAliasWriteDescriptorAwaiter output_descriptor_awaiter(output);
    auto awaited = co_await async::all(output_descriptor_awaiter, input);
    auto output_value = std::get<0>(awaited);
    auto const& input_value = std::get<1>(awaited);
    prepare_output(output_value, input_value.extents());
    auto output_descriptor = mdspec_of(output_value);
    auto input_descriptor = mdspec_of(input_value);
    co_await linalg::co_dispatch_kernel(selector, linalg::copy_op{}, output_descriptor, input_descriptor);
  }
  else
  {
    auto output_storage = output.storage();
    auto awaited = co_await async::all(output_storage, input);
    auto& storage = std::get<0>(awaited);
    auto const& input_value = std::get<1>(awaited);
    auto& output_value = prepare_async_copy_output(storage, input_value);
    auto output_descriptor = mdspec_of(output_value);
    auto input_descriptor = mdspec_of(input_value);
    co_await linalg::co_dispatch_kernel(selector, linalg::copy_op{}, output_descriptor, input_descriptor);
  }
  co_return;
}

template <class BackendSelector, AsyncTensorOutput OutputTensor, TensorView InputTensor>
void schedule_async_copy(BackendSelector selector, async::Async<OutputTensor>& output,
                         async::Async<InputTensor> const& input)
{
  ERROR_IF(std::addressof(output.queue()) == std::addressof(input.queue()),
           "async copy output must not share an epoch queue with its input");
  auto task = co_copy(std::move(selector), output.write(), input.read());
  task.debug_name("copy");
  async::schedule(std::move(task));
}

template <TensorView InputTensor>
using async_materialized_tensor_t =
    std::remove_cvref_t<decltype(uni20::make_tensor(std::declval<InputTensor const&>()))>;

template <class BackendSelector, class ResultTensor, TensorView InputTensor>
async::AsyncTask co_make_tensor(BackendSelector const selector, async::WriteBuffer<ResultTensor> output,
                                async::ReadBuffer<InputTensor> input)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, input);
  auto& storage = std::get<0>(awaited);
  auto const& input_value = std::get<1>(awaited);
  ResultTensor result(convert_tensor_extents<typename ResultTensor::extents_type>(input_value.extents()));
  uni20::copy(selector, result, input_value);
  storage.emplace(std::move(result));
  co_return;
}

#if UNI20_BACKEND_CUDA
template <TensorView InputTensor>
using async_device_tensor_t = std::remove_cvref_t<decltype(uni20::to_device(std::declval<InputTensor const&>(),
                                                                            std::declval<cuda::DeviceResources&>()))>;

template <class ResultTensor, TensorView InputTensor>
async::AsyncTask co_to_device(async::WriteBuffer<ResultTensor> output, async::ReadBuffer<InputTensor> input,
                              cuda::DeviceResources* resources)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, input);
  auto& storage = std::get<0>(awaited);
  auto const& input_value = std::get<1>(awaited);
  ResultTensor result(*resources, convert_tensor_extents<typename ResultTensor::extents_type>(input_value.extents()));
  auto output_descriptor = mdspec_of(result);
  auto input_descriptor = mdspec_of(input_value);
  co_await linalg::co_dispatch_kernel(linalg::CudaReferenceBackend{}, linalg::copy_op{}, output_descriptor,
                                      input_descriptor);
  storage.emplace(std::move(result));
  co_return;
}
#endif

} // namespace detail

/// \brief Schedule a Tensor copy through an explicit backend selector.
/// \details CUDA-to-CUDA copies may complete their outer epoch after submission;
///          the destination buffer ledger then carries device completion. A
///          pageable host transfer runs its necessarily blocking CUDA call on
///          the scheduler task.
/// \pre The output has a distinct epoch queue from the input. An unconstructed
///      context-bound output can be prepared only when placement is available
///      from the input descriptor.
template <class BackendSelector, class OutputTensor, class InputTensor>
  requires detail::AsyncCopyTensors<OutputTensor, InputTensor>
void copy(BackendSelector selector, async::Async<OutputTensor>& output, async::Async<InputTensor> const& input)
{
  detail::schedule_async_copy(std::move(selector), output, input);
}

/// \brief Schedule a Tensor copy using the storage backend list.
template <class OutputTensor, class InputTensor>
  requires detail::AsyncCopyTensors<OutputTensor, InputTensor>
void copy(async::Async<OutputTensor>& output, async::Async<InputTensor> const& input)
{
#if UNI20_BACKEND_CUDA
  if constexpr (detail::is_pageable_cuda_transfer<OutputTensor, InputTensor>)
  {
    detail::schedule_async_copy(linalg::CudaReferenceBackend{}, output, input);
  }
  else
#endif
  {
    auto selector = linalg::select_backend_for<OutputTensor, InputTensor>(linalg::copy_op{});
    detail::schedule_async_copy(std::move(selector), output, input);
  }
}

/// \brief Materialize an async Tensor view as an independent owning host Tensor.
/// \details The result type and canonical layout match synchronous `make_tensor`.
template <TensorView InputTensor> [[nodiscard]] auto make_tensor(async::Async<InputTensor> const& input)
{
  using result_type = detail::async_materialized_tensor_t<InputTensor>;
  async::Async<result_type> output;
  output.debug_name("make_tensor.result");
#if UNI20_BACKEND_CUDA
  auto selector = [&] {
    if constexpr (detail::is_pageable_cuda_transfer<result_type, InputTensor>)
      return linalg::CudaReferenceBackend{};
    else
      return linalg::select_backend_for<result_type, InputTensor>(linalg::copy_op{});
  }();
#else
  auto selector = linalg::select_backend_for<result_type, InputTensor>(linalg::copy_op{});
#endif
  auto task = detail::co_make_tensor(std::move(selector), output.write(), input.read());
  task.debug_name("make_tensor");
  async::schedule(std::move(task));
  return output;
}

#if UNI20_BACKEND_CUDA
/// \brief Materialize an async CUDA Tensor as an owning pageable host Tensor.
template <TensorView InputTensor>
  requires std::same_as<tensor_storage_policy_t<InputTensor>, CudaStorage>
[[nodiscard]] auto to_host(async::Async<InputTensor> const& input)
{
  return uni20::make_tensor(input);
}

/// \brief Materialize an async Tensor on an explicit CUDA device resource domain.
/// \details The result owns an independent epoch and retains CUDA completion in
///          its buffer ledger. Pageable host transfer blocks only the scheduled task.
template <TensorView InputTensor>
  requires detail::CanonicalReshapeLayout<typename tensor_mdspec_t<InputTensor>::layout_type>
[[nodiscard]] auto to_device(async::Async<InputTensor> const& input, cuda::DeviceResources& resources)
{
  using result_type = detail::async_device_tensor_t<InputTensor>;
  async::Async<result_type> output;
  output.debug_name("to_device.result");
  auto task = detail::co_to_device(output.write(), input.read(), std::addressof(resources));
  task.debug_name("to_device");
  async::schedule(std::move(task));
  return output;
}

/// \brief Materialize an async Tensor on an enrolled CUDA device ordinal.
template <TensorView InputTensor>
  requires detail::CanonicalReshapeLayout<typename tensor_mdspec_t<InputTensor>::layout_type>
[[nodiscard]] auto to_device(async::Async<InputTensor> const& input, int device)
{
  return uni20::to_device(input, cuda::device_resources(device));
}

/// \brief Materialize an async Tensor on an enrolled CUDA device.
template <TensorView InputTensor>
  requires detail::CanonicalReshapeLayout<typename tensor_mdspec_t<InputTensor>::layout_type>
[[nodiscard]] auto to_device(async::Async<InputTensor> const& input, cuda::Device device)
{
  return uni20::to_device(input, device.ordinal());
}
#endif

} // namespace uni20
