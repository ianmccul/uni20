#pragma once

/**
 * \file reductions.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for full and axis-selective sums.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/tensor/reductions.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>

namespace uni20
{
namespace detail
{

template <class OutputTensor, class InputTensor>
void validate_async_reduction_aliasing(async::Async<OutputTensor> const& output, async::Async<InputTensor> const& input)
{
  ERROR_IF(std::addressof(output.queue()) == std::addressof(input.queue()),
           "async reduction output must not share an epoch queue with its input");
}

template <AsyncTensorOutput OutputTensor, TensorView InputTensor, std::size_t InputRank, std::size_t ReducedRank>
[[nodiscard]] OutputTensor& prepare_async_sum_output(async::shared_storage<OutputTensor>& storage,
                                                     InputTensor const& input,
                                                     linalg::ReductionAxes<InputRank, ReducedRank> const& axes)
{
  auto const required = reduction_output_extents(input, axes);
  if (storage.constructed())
  {
    prepare_output(*storage, required);
    return *storage;
  }

  using extents_type = tensor_extents_t<OutputTensor>;
  if constexpr (std::constructible_from<OutputTensor, extents_type const&>)
  {
    auto const extents = convert_tensor_extents<extents_type>(required);
    return storage.emplace(extents);
  }
  else
  {
    throw async::buffer_write_uninitialized{};
  }
}

template <class BackendSelector, AsyncTensorOutput OutputTensor, TensorView InputTensor, std::size_t InputRank,
          std::size_t ReducedRank>
async::AsyncTask co_sum(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                        async::ReadBuffer<InputTensor> input, linalg::ReductionAxes<InputRank, ReducedRank> axes)
{
  if constexpr (async::is_async_alias_v<OutputTensor>)
  {
    AsyncAliasWriteDescriptorAwaiter output_descriptor_awaiter(output);
    auto awaited = co_await async::all(output_descriptor_awaiter, input);
    auto mutable_output = std::get<0>(awaited);
    auto const& input_value = std::get<1>(awaited);
    prepare_output(mutable_output, reduction_output_extents(input_value, axes));
    auto output_descriptor = mdspec_of(mutable_output);
    auto input_descriptor = mdspec_of(input_value);
    dispatch_sum(selector, output_descriptor, input_descriptor, std::move(axes));
  }
  else
  {
    auto output_storage = output.storage();
    auto awaited = co_await async::all(output_storage, input);
    auto& storage = std::get<0>(awaited);
    auto const& input_value = std::get<1>(awaited);
    auto& output_value = prepare_async_sum_output(storage, input_value, axes);
    auto output_descriptor = mdspec_of(output_value);
    auto input_descriptor = mdspec_of(input_value);
    dispatch_sum(selector, output_descriptor, input_descriptor, std::move(axes));
  }
  co_return;
}

template <class BackendSelector, TensorView InputTensor>
async::AsyncTask co_sum_host(BackendSelector const selector, async::WriteBuffer<tensor_element_t<InputTensor>> output,
                             async::ReadBuffer<InputTensor> input)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, input);
  auto& storage = std::get<0>(awaited);
  auto const& input_value = std::get<1>(awaited);
  auto result = uni20::sum_host(selector, input_value);
  storage.emplace(std::move(result));
  co_return;
}

template <class BackendSelector, AsyncTensorOutput OutputTensor, TensorView InputTensor, std::size_t InputRank,
          std::size_t ReducedRank>
void schedule_async_sum(BackendSelector selector, async::Async<OutputTensor>& output,
                        async::Async<InputTensor> const& input, linalg::ReductionAxes<InputRank, ReducedRank> axes)
{
  validate_async_reduction_aliasing(output, input);
  auto task = co_sum(std::move(selector), output.write(), input.read(), std::move(axes));
  task.debug_name("sum");
  async::schedule(std::move(task));
}

template <class BackendSelector, TensorView InputTensor>
[[nodiscard]] auto schedule_async_sum_host(BackendSelector selector, async::Async<InputTensor> const& input)
{
  async::Async<tensor_element_t<InputTensor>> output;
  output.debug_name("sum_host.result");
  auto task = co_sum_host(std::move(selector), output.write(), input.read());
  task.debug_name("sum_host");
  async::schedule(std::move(task));
  return output;
}

} // namespace detail

/// \brief Schedule a full sum into an explicit async scalar tensor.
/// \details Axis normalization is compile-time for the full reduction. Shape
///          preparation and backend dispatch occur after the input is readable.
/// \pre The output must not share an epoch queue with the input.
template <linalg::KernelBackendSelector BackendSelector, detail::AsyncTensorOutput OutputTensor, TensorView InputTensor>
  requires MutableScalarTensorView<OutputTensor> && RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>>
void sum(BackendSelector selector, async::Async<OutputTensor>& output, async::Async<InputTensor> const& input)
{
  constexpr std::size_t input_rank = tensor_mdspec_t<InputTensor>::rank();
  detail::schedule_async_sum(std::move(selector), output, input, linalg::all_reduction_axes<input_rank>());
}

/// \brief Schedule a full sum into an explicit async scalar tensor using storage policy.
/// \pre The output must not share an epoch queue with the input.
template <detail::AsyncTensorOutput OutputTensor, TensorView InputTensor>
  requires MutableScalarTensorView<OutputTensor> && RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>>
void sum(async::Async<OutputTensor>& output, async::Async<InputTensor> const& input)
{
  constexpr std::size_t input_rank = tensor_mdspec_t<InputTensor>::rank();
  auto axes = linalg::all_reduction_axes<input_rank>();
  auto operation = linalg::sum_reduction_op<input_rank, input_rank>{.axes = axes};
  auto selector = linalg::select_backend_for<OutputTensor, InputTensor>(operation);
  detail::schedule_async_sum(std::move(selector), output, input, std::move(axes));
}

/// \brief Schedule an axis-selective sum into an explicit async output.
/// \details Negative axes are accepted. Axis errors are reported before task
///          submission; output shape validation or resizing occurs in the task.
/// \pre The output must not share an epoch queue with the input.
template <linalg::KernelBackendSelector BackendSelector, detail::AsyncTensorOutput OutputTensor, TensorView InputTensor,
          linalg::ReductionAxis FirstAxis, linalg::ReductionAxis... RestAxes>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>> &&
           (1 + sizeof...(RestAxes) <= tensor_mdspec_t<InputTensor>::rank()) &&
           (mutable_tensor_mdspec_t<OutputTensor>::rank() + 1 + sizeof...(RestAxes) ==
            tensor_mdspec_t<InputTensor>::rank())
void sum(BackendSelector selector, async::Async<OutputTensor>& output, async::Async<InputTensor> const& input,
         FirstAxis first_axis, RestAxes... rest_axes)
{
  constexpr std::size_t input_rank = tensor_mdspec_t<InputTensor>::rank();
  auto axes = linalg::make_reduction_axes<input_rank>(first_axis, rest_axes...);
  detail::schedule_async_sum(std::move(selector), output, input, std::move(axes));
}

/// \brief Schedule an axis-selective sum into an explicit async output using storage policy.
/// \pre The output must not share an epoch queue with the input.
template <detail::AsyncTensorOutput OutputTensor, TensorView InputTensor, linalg::ReductionAxis FirstAxis,
          linalg::ReductionAxis... RestAxes>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>> &&
           (1 + sizeof...(RestAxes) <= tensor_mdspec_t<InputTensor>::rank()) &&
           (mutable_tensor_mdspec_t<OutputTensor>::rank() + 1 + sizeof...(RestAxes) ==
            tensor_mdspec_t<InputTensor>::rank())
void sum(async::Async<OutputTensor>& output, async::Async<InputTensor> const& input, FirstAxis first_axis,
         RestAxes... rest_axes)
{
  constexpr std::size_t input_rank = tensor_mdspec_t<InputTensor>::rank();
  constexpr std::size_t reduced_rank = 1 + sizeof...(RestAxes);
  auto axes = linalg::make_reduction_axes<input_rank>(first_axis, rest_axes...);
  auto operation = linalg::sum_reduction_op<input_rank, reduced_rank>{.axes = axes};
  auto selector = linalg::select_backend_for<OutputTensor, InputTensor>(operation);
  detail::schedule_async_sum(std::move(selector), output, input, std::move(axes));
}

/// \brief Return an async storage-preserving full Tensor sum through an explicit selector.
template <linalg::KernelBackendSelector BackendSelector, TensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           detail::ReductionResultAvailable<tensor_element_t<InputTensor>, InputTensor>
[[nodiscard]] auto sum(BackendSelector selector, async::Async<InputTensor> const& input)
{
  constexpr std::size_t input_rank = tensor_mdspec_t<InputTensor>::rank();
  using result_type = detail::sum_reduction_result_t<InputTensor, input_rank>;
  async::Async<result_type> output;
  output.debug_name("sum.result");
  detail::schedule_async_sum(std::move(selector), output, input, linalg::all_reduction_axes<input_rank>());
  return output;
}

/// \brief Return an async storage-preserving full Tensor sum using storage policy.
template <TensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           detail::ReductionResultAvailable<tensor_element_t<InputTensor>, InputTensor>
[[nodiscard]] auto sum(async::Async<InputTensor> const& input)
{
  constexpr std::size_t input_rank = tensor_mdspec_t<InputTensor>::rank();
  auto axes = linalg::all_reduction_axes<input_rank>();
  auto operation = linalg::sum_reduction_op<input_rank, input_rank>{.axes = axes};
  auto selector = linalg::select_backend_for<InputTensor>(operation);
  return sum(std::move(selector), input);
}

/// \brief Return an async storage-preserving sum over selected axes.
/// \details Negative axes are accepted and surviving axes retain logical order.
template <linalg::KernelBackendSelector BackendSelector, TensorView InputTensor, linalg::ReductionAxis FirstAxis,
          linalg::ReductionAxis... RestAxes>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           (1 + sizeof...(RestAxes) <= tensor_mdspec_t<InputTensor>::rank()) &&
           detail::ReductionResultAvailable<tensor_element_t<InputTensor>, InputTensor>
[[nodiscard]] auto sum(BackendSelector selector, async::Async<InputTensor> const& input, FirstAxis first_axis,
                       RestAxes... rest_axes)
{
  constexpr std::size_t input_rank = tensor_mdspec_t<InputTensor>::rank();
  constexpr std::size_t reduced_rank = 1 + sizeof...(RestAxes);
  auto axes = linalg::make_reduction_axes<input_rank>(first_axis, rest_axes...);
  using result_type = detail::sum_reduction_result_t<InputTensor, reduced_rank>;
  async::Async<result_type> output;
  output.debug_name("sum.result");
  detail::schedule_async_sum(std::move(selector), output, input, std::move(axes));
  return output;
}

/// \brief Return an async storage-preserving sum over selected axes using storage policy.
template <TensorView InputTensor, linalg::ReductionAxis FirstAxis, linalg::ReductionAxis... RestAxes>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           (1 + sizeof...(RestAxes) <= tensor_mdspec_t<InputTensor>::rank()) &&
           detail::ReductionResultAvailable<tensor_element_t<InputTensor>, InputTensor>
[[nodiscard]] auto sum(async::Async<InputTensor> const& input, FirstAxis first_axis, RestAxes... rest_axes)
{
  constexpr std::size_t input_rank = tensor_mdspec_t<InputTensor>::rank();
  constexpr std::size_t reduced_rank = 1 + sizeof...(RestAxes);
  auto axes = linalg::make_reduction_axes<input_rank>(first_axis, rest_axes...);
  auto operation = linalg::sum_reduction_op<input_rank, reduced_rank>{.axes = axes};
  auto selector = linalg::select_backend_for<InputTensor>(operation);
  return sum(std::move(selector), input, first_axis, rest_axes...);
}

/// \brief Return a nonblocking async host C++ scalar sum through an explicit selector.
template <linalg::KernelBackendSelector BackendSelector, TensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>>
[[nodiscard]] auto sum_host(BackendSelector selector, async::Async<InputTensor> const& input)
{
  return detail::schedule_async_sum_host(std::move(selector), input);
}

/// \brief Return a nonblocking async host C++ scalar sum using storage policy.
template <TensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>>
[[nodiscard]] auto sum_host(async::Async<InputTensor> const& input)
{
  constexpr std::size_t input_rank = tensor_mdspec_t<InputTensor>::rank();
  auto operation = linalg::sum_reduction_op<input_rank, input_rank>{.axes = linalg::all_reduction_axes<input_rank>()};
  auto selector = linalg::select_backend_for<InputTensor>(operation);
  return detail::schedule_async_sum_host(std::move(selector), input);
}

} // namespace uni20
