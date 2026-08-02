#pragma once

/**
 * \file transform.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for elementwise overwrite and update operations.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/transform.hpp>

#include <concepts>
#include <cstddef>
#include <tuple>
#include <utility>

namespace uni20
{
namespace detail
{

template <class OutputTensor, class... InputTensors>
void validate_async_transform_aliasing(async::Async<OutputTensor> const& output,
                                       async::Async<InputTensors> const&... inputs)
{
  if constexpr (sizeof...(InputTensors) > 0)
  {
    auto const* const output_queue = std::addressof(output.queue());
    auto validate_input = [&](auto const& input) {
      ERROR_IF(output_queue == std::addressof(input.queue()),
               "async elementwise transform output must not share an epoch queue with an input");
    };
    (validate_input(inputs), ...);
  }
}

template <AsyncTensorOutput OutputTensor, TensorView FirstInputTensor>
[[nodiscard]] OutputTensor& prepare_async_assign_transform_output(async::shared_storage<OutputTensor>& storage,
                                                                  FirstInputTensor const& first_input)
{
  if (storage.constructed()) return *storage;

  using extents_type = tensor_extents_t<OutputTensor>;
  if constexpr (std::constructible_from<OutputTensor, extents_type const&>)
  {
    auto const extents = convert_tensor_extents<extents_type>(first_input.extents());
    return storage.emplace(extents);
  }
  else
  {
    throw async::buffer_write_uninitialized{};
  }
}

template <class BackendSelector, class Function, AsyncTensorOutput OutputTensor, class Awaited, std::size_t... Input>
void invoke_async_assign_transform(BackendSelector const& selector, linalg::transform_op<Function>& operation,
                                   OutputTensor& output, Awaited& awaited, std::index_sequence<Input...>)
{
  uni20::assign_transform(selector, output, std::move(operation.function), std::get<Input + 1>(awaited)...);
}

template <class BackendSelector, class Function, AsyncTensorOutput OutputTensor, class Awaited, std::size_t... Input>
void invoke_async_transform_inplace(BackendSelector const& selector, linalg::transform_inplace_op<Function>& operation,
                                    OutputTensor& output, Awaited& awaited, std::index_sequence<Input...>)
{
  uni20::transform_inplace(selector, output, std::move(operation.function), std::get<Input + 1>(awaited)...);
}

template <class BackendSelector, AsyncTensorOutput OutputTensor, class Function, TensorView... InputTensors>
async::AsyncTask co_assign_transform(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                                     linalg::transform_op<Function> operation,
                                     async::ReadBuffer<InputTensors>... inputs)
{
  if constexpr (async::is_async_alias_v<OutputTensor>)
  {
    AsyncAliasWriteDescriptorAwaiter output_descriptor(output);
    auto awaited = co_await async::all(output_descriptor, inputs...);
    auto mutable_output = std::get<0>(awaited);
    require_output(mutable_output, std::get<1>(awaited).extents());
    invoke_async_assign_transform(selector, operation, mutable_output, awaited,
                                  std::index_sequence_for<InputTensors...>{});
  }
  else
  {
    auto output_storage = output.storage();
    auto awaited = co_await async::all(output_storage, inputs...);
    auto& storage = std::get<0>(awaited);
    auto& output_value = prepare_async_assign_transform_output(storage, std::get<1>(awaited));
    invoke_async_assign_transform(selector, operation, output_value, awaited,
                                  std::index_sequence_for<InputTensors...>{});
  }
  co_return;
}

template <class BackendSelector, AsyncTensorOutput OutputTensor, class Function, TensorView... InputTensors>
async::AsyncTask co_transform_inplace(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                                      linalg::transform_inplace_op<Function> operation,
                                      async::ReadBuffer<InputTensors>... inputs)
{
  if constexpr (async::is_async_alias_v<OutputTensor>)
  {
    AsyncAliasWriteDescriptorAwaiter output_descriptor(output);
    auto awaited = co_await async::all(output_descriptor, inputs...);
    auto mutable_output = std::get<0>(awaited);
    invoke_async_transform_inplace(selector, operation, mutable_output, awaited,
                                   std::index_sequence_for<InputTensors...>{});
  }
  else
  {
    auto output_storage = output.storage();
    auto awaited = co_await async::all(output_storage, inputs...);
    auto& storage = std::get<0>(awaited);
    if (!storage.constructed()) throw async::buffer_write_uninitialized{};
    invoke_async_transform_inplace(selector, operation, *storage, awaited, std::index_sequence_for<InputTensors...>{});
  }
  co_return;
}

template <class BackendSelector, AsyncTensorOutput OutputTensor, class Function, TensorView... InputTensors>
void schedule_async_assign_transform(BackendSelector selector, async::Async<OutputTensor>& output,
                                     linalg::transform_op<Function> operation,
                                     async::Async<InputTensors> const&... inputs)
{
  validate_async_transform_aliasing(output, inputs...);
  auto task = co_assign_transform(std::move(selector), output.write(), std::move(operation), inputs.read()...);
  task.debug_name("assign_transform");
  async::schedule(std::move(task));
}

template <class BackendSelector, AsyncTensorOutput OutputTensor, class Function, TensorView... InputTensors>
void schedule_async_transform_inplace(BackendSelector selector, async::Async<OutputTensor>& output,
                                      linalg::transform_inplace_op<Function> operation,
                                      async::Async<InputTensors> const&... inputs)
{
  validate_async_transform_aliasing(output, inputs...);
  auto task = co_transform_inplace(std::move(selector), output.write(), std::move(operation), inputs.read()...);
  task.debug_name("transform_inplace");
  async::schedule(std::move(task));
}

} // namespace detail

/// \brief Schedule a variadic elementwise overwrite with an explicit backend selector.
/// \details Every Tensor operand is asynchronous. The callable is owned by the
///          coroutine and invoked as const by the selected synchronous backend.
///          An unconstructed output is initialized when its Tensor type can be
///          constructed from the first input's extents.
/// \pre The output must not share an epoch queue with any input.
template <class BackendSelector, detail::AsyncTensorOutput OutputTensor, class Function, TensorView FirstInputTensor,
          TensorView... RestInputTensors>
  requires detail::OverwriteTransformTensors<OutputTensor, FirstInputTensor, RestInputTensors...>
void assign_transform(BackendSelector selector, async::Async<OutputTensor>& output, Function&& function,
                      async::Async<FirstInputTensor> const& first_input,
                      async::Async<RestInputTensors> const&... rest_inputs)
{
  auto operation = linalg::transform_op{std::forward<Function>(function)};
  detail::schedule_async_assign_transform(std::move(selector), output, std::move(operation), first_input,
                                          rest_inputs...);
}

/// \brief Schedule a variadic elementwise overwrite using the static Tensor selector.
/// \details Selector resolution depends only on Tensor and callable types and
///          occurs before scheduling. Unhandled failures propagate to the
///          output epoch.
/// \pre The output must not share an epoch queue with any input.
template <detail::AsyncTensorOutput OutputTensor, class Function, TensorView FirstInputTensor,
          TensorView... RestInputTensors>
  requires detail::OverwriteTransformTensors<OutputTensor, FirstInputTensor, RestInputTensors...>
void assign_transform(async::Async<OutputTensor>& output, Function&& function,
                      async::Async<FirstInputTensor> const& first_input,
                      async::Async<RestInputTensors> const&... rest_inputs)
{
  auto operation = linalg::transform_op{std::forward<Function>(function)};
  auto selector = linalg::select_backend_for<OutputTensor, FirstInputTensor, RestInputTensors...>(operation);
  detail::schedule_async_assign_transform(std::move(selector), output, std::move(operation), first_input,
                                          rest_inputs...);
}

/// \brief Schedule a variadic elementwise update with an explicit backend selector.
/// \details The output must already contain a Tensor. Its old element value is
///          supplied through the single output writer and precedes all input
///          values in the callable argument list.
/// \pre The output must not share an epoch queue with any input.
template <class BackendSelector, detail::AsyncTensorOutput OutputTensor, class Function, TensorView... InputTensors>
  requires detail::UpdateTransformTensors<OutputTensor, InputTensors...>
void transform_inplace(BackendSelector selector, async::Async<OutputTensor>& output, Function&& function,
                       async::Async<InputTensors> const&... inputs)
{
  auto operation = linalg::transform_inplace_op{std::forward<Function>(function)};
  detail::schedule_async_transform_inplace(std::move(selector), output, std::move(operation), inputs...);
}

/// \brief Schedule a variadic elementwise update using the static Tensor selector.
/// \details Selector resolution occurs before scheduling. The output contributes
///          one writer and is never enrolled again as a read-only input.
/// \pre The output must not share an epoch queue with any input.
template <detail::AsyncTensorOutput OutputTensor, class Function, TensorView... InputTensors>
  requires detail::UpdateTransformTensors<OutputTensor, InputTensors...>
void transform_inplace(async::Async<OutputTensor>& output, Function&& function,
                       async::Async<InputTensors> const&... inputs)
{
  auto operation = linalg::transform_inplace_op{std::forward<Function>(function)};
  auto selector = linalg::select_backend_for<OutputTensor, InputTensors...>(operation);
  detail::schedule_async_transform_inplace(std::move(selector), output, std::move(operation), inputs...);
}

} // namespace uni20
