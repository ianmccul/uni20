#pragma once

/**
 * \file copy.hpp
 * \ingroup linalg
 * \brief Async CUDA-to-CUDA Tensor copy wrapper.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/dispatch.hpp>
#include <uni20/linalg/backends/cuda/copy_task.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/tensor/copy_into.hpp>

#include <concepts>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class Tensor>
concept OwningCudaTensor = OwningTensor<Tensor> && std::same_as<tensor_storage_policy_t<Tensor>, CudaStorage>;

template <class Tensor>
concept CudaCopyInputTensor = DeviceTensorView<Tensor> && CudaBufferDeviceMdspan<device_tensor_mdspan_t<Tensor>>;

template <class OutputTensor, class InputTensor>
concept AsyncCudaCopyTensors =
    OwningCudaTensor<OutputTensor> && CudaCopyInputTensor<InputTensor> &&
    (device_tensor_mdspan_t<OutputTensor>::rank() == device_tensor_mdspan_t<InputTensor>::rank()) &&
    std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>>;

template <CudaBufferDeviceMdspan InputMdspan>
[[nodiscard]] cuda::DeviceResources& cuda_copy_resources(InputMdspan const& input)
{
  return input.data_descriptor().buffer().resources();
}

template <OwningCudaTensor OutputTensor, CudaBufferDeviceMdspan InputMdspan>
[[nodiscard]] OutputTensor& prepare_async_copy_output(async::shared_storage<OutputTensor>& storage,
                                                      InputMdspan const& input)
{
  if (!storage.constructed())
  {
    using extents_type = tensor_extents_t<OutputTensor>;
    auto const extents = convert_tensor_extents<extents_type>(input.extents());
    return storage.emplace(cuda_copy_resources(input), extents);
  }

  prepare_output(*storage, input.extents());
  return *storage;
}

template <class BackendSelector, OwningCudaTensor OutputTensor, CudaCopyInputTensor InputTensor>
async::AsyncTask co_cuda_copy(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                              async::ReadBuffer<InputTensor> input)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, input);
  auto& storage = std::get<0>(awaited);
  auto const& input_value = std::get<1>(awaited);
  auto input_span = device_mdspan_of(input_value);
  auto& output_value = prepare_async_copy_output(storage, input_span);
  auto output_span = device_mdspan_of(output_value);
  co_await linalg::co_dispatch_kernel(selector, linalg::copy_op{}, output_span, input_span);
  co_return;
}

template <class BackendSelector, OwningCudaTensor OutputTensor, CudaCopyInputTensor InputTensor>
void schedule_async_cuda_copy(BackendSelector selector, async::Async<OutputTensor>& output,
                              async::Async<InputTensor> const& input)
{
  ERROR_IF(std::addressof(output.queue()) == std::addressof(input.queue()),
           "async CUDA copy output must not share an epoch queue with its input");
  auto task = co_cuda_copy(std::move(selector), output.write(), input.read());
  task.debug_name("copy");
  async::schedule(std::move(task));
}

} // namespace detail

/// \brief Schedule a CUDA-to-CUDA Tensor copy with an explicit backend selector.
/// \details The outer Async epoch completes after the CUDA copy is submitted;
///          the destination buffer ledger carries device completion into later
///          CUDA operations. Pageable host transfers are intentionally not
///          exposed through this non-blocking overload.
template <class BackendSelector, class OutputTensor, class InputTensor>
  requires detail::AsyncCudaCopyTensors<OutputTensor, InputTensor>
void copy(BackendSelector selector, async::Async<OutputTensor>& output, async::Async<InputTensor> const& input)
{
  detail::schedule_async_cuda_copy(std::move(selector), output, input);
}

/// \brief Schedule a CUDA-to-CUDA Tensor copy using the storage backend list.
template <class OutputTensor, class InputTensor>
  requires detail::AsyncCudaCopyTensors<OutputTensor, InputTensor>
void copy(async::Async<OutputTensor>& output, async::Async<InputTensor> const& input)
{
  auto selector = linalg::select_backend_for<OutputTensor, InputTensor>(linalg::copy_op{});
  detail::schedule_async_cuda_copy(std::move(selector), output, input);
}

} // namespace uni20
