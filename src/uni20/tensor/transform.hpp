#pragma once

/**
 * \file transform.hpp
 * \ingroup tensor
 * \brief Backend-dispatched elementwise overwrite and update operations.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/linalg/backends/cpu/transform.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/output.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class Output, class... Inputs>
concept OverwriteTransformSpans =
    MutableMdspanLike<Output> && (sizeof...(Inputs) >= 1) && (MdspanLike<Inputs> && ...) &&
    ((std::remove_cvref_t<Output>::rank() == std::remove_cvref_t<Inputs>::rank()) && ...);

template <class Output, class... Inputs>
concept UpdateTransformSpans = MutableMdspanLike<Output> && (MdspanLike<Inputs> && ...) &&
                               ((std::remove_cvref_t<Output>::rank() == std::remove_cvref_t<Inputs>::rank()) && ...);

template <class Output, class... Inputs>
concept OverwriteTransformTensors =
    MutableDeviceTensorView<Output> && (sizeof...(Inputs) >= 1) && (DeviceTensorView<Inputs> && ...) &&
    ((device_tensor_mdspan_t<Output>::rank() == device_tensor_mdspan_t<Inputs>::rank()) && ...);

template <class Output, class... Inputs>
concept UpdateTransformTensors =
    MutableDeviceTensorView<Output> && (DeviceTensorView<Inputs> && ...) &&
    ((device_tensor_mdspan_t<Output>::rank() == device_tensor_mdspan_t<Inputs>::rank()) && ...);

template <class Reference, class... Others>
void require_transform_extents(Reference const& reference, Others const&... others)
{
  if constexpr (sizeof...(Others) > 0)
  {
    constexpr std::size_t Rank = [] {
      if constexpr (DeviceTensorView<Reference>)
        return device_tensor_mdspan_t<Reference>::rank();
      else
        return std::remove_cvref_t<Reference>::rank();
    }();
    if constexpr (Rank > 0)
    {
      auto require_equal = [&](auto const& other) {
        for (std::size_t axis = 0; axis < Rank; ++axis)
          ERROR_IF(reference.extent(axis) != other.extent(axis),
                   "elementwise transform operands have different extents");
      };
      (require_equal(others), ...);
    }
  }
}

template <class BackendSelector, class Operation, class OutputMdspan, class... InputMdspans>
void dispatch_transform(BackendSelector&& selector, Operation operation, OutputMdspan&& output,
                        InputMdspans&&... inputs)
{
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), std::move(operation),
                          std::forward<OutputMdspan>(output), std::forward<InputMdspans>(inputs)...);
}

} // namespace detail

/// \brief Overwrite a fixed-shape mdspan through an explicit backend selector.
/// \details Computes `output[i...] = function(inputs[i...]...)`. Input elements
///          are passed as values even when an input mdspan is mutable.
/// \pre All operands have equal extents and output does not overlap an input.
template <class BackendSelector, class OutputMdspan, class Function, class... InputMdspans>
  requires detail::OverwriteTransformSpans<OutputMdspan, InputMdspans...>
void assign_transform(BackendSelector&& selector, OutputMdspan&& output, Function&& function, InputMdspans&&... inputs)
{
  detail::require_transform_extents(output, inputs...);
  auto operation = linalg::transform_op{std::forward<Function>(function)};
  auto output_descriptor = std::forward<OutputMdspan>(output);
  auto input_descriptors = std::tuple{make_const_mdspan(inputs)...};
  std::apply(
      [&](auto&... input_descriptor) {
        detail::dispatch_transform(std::forward<BackendSelector>(selector), std::move(operation), output_descriptor,
                                   input_descriptor...);
      },
      input_descriptors);
}

/// \brief Update a fixed-shape mdspan through an explicit backend selector.
/// \details Computes `output[i...] = function(output[i...], inputs[i...]...)`.
///          The old output and input elements are passed as values.
/// \pre All operands have equal extents and output does not overlap an input.
template <class BackendSelector, class OutputMdspan, class Function, class... InputMdspans>
  requires detail::UpdateTransformSpans<OutputMdspan, InputMdspans...>
void transform_inplace(BackendSelector&& selector, OutputMdspan&& output, Function&& function, InputMdspans&&... inputs)
{
  detail::require_transform_extents(output, inputs...);
  auto operation = linalg::transform_inplace_op{std::forward<Function>(function)};
  auto output_descriptor = std::forward<OutputMdspan>(output);
  auto input_descriptors = std::tuple{make_const_mdspan(inputs)...};
  std::apply(
      [&](auto&... input_descriptor) {
        detail::dispatch_transform(std::forward<BackendSelector>(selector), std::move(operation), output_descriptor,
                                   input_descriptor...);
      },
      input_descriptors);
}

/// \brief Overwrite a Tensor output through an explicit backend selector.
/// \details A resizable output adopts the first input's extents before mdspans
///          are resolved. Fixed outputs validate the same shape. Input elements
///          are passed to the callable as values.
/// \pre Input operands have equal extents and output does not overlap an input.
template <class BackendSelector, class OutputTensor, class Function, class FirstInputTensor, class... RestInputTensors>
  requires detail::OverwriteTransformTensors<OutputTensor, FirstInputTensor, RestInputTensors...>
void assign_transform(BackendSelector&& selector, OutputTensor&& output, Function&& function,
                      FirstInputTensor const& first_input, RestInputTensors const&... rest_inputs)
{
  detail::require_transform_extents(first_input, rest_inputs...);
  prepare_output(output, first_input.extents());
  auto operation = linalg::transform_op{std::forward<Function>(function)};
  auto output_descriptor = device_mdspan_of(output);
  auto input_descriptors = std::tuple{device_mdspan_of(first_input), device_mdspan_of(rest_inputs)...};
  std::apply(
      [&](auto&... inputs) {
        detail::dispatch_transform(std::forward<BackendSelector>(selector), std::move(operation), output_descriptor,
                                   inputs...);
      },
      input_descriptors);
}

/// \brief Overwrite a Tensor output using its operands' default backend selector.
template <class OutputTensor, class Function, class FirstInputTensor, class... RestInputTensors>
  requires detail::OverwriteTransformTensors<OutputTensor, FirstInputTensor, RestInputTensors...>
void assign_transform(OutputTensor&& output, Function&& function, FirstInputTensor const& first_input,
                      RestInputTensors const&... rest_inputs)
{
  auto operation = linalg::transform_op{std::forward<Function>(function)};
  auto selector = linalg::select_backend(operation, output, first_input, rest_inputs...);
  detail::require_transform_extents(first_input, rest_inputs...);
  prepare_output(output, first_input.extents());
  auto output_descriptor = device_mdspan_of(output);
  auto input_descriptors = std::tuple{device_mdspan_of(first_input), device_mdspan_of(rest_inputs)...};
  std::apply(
      [&](auto&... inputs) {
        detail::dispatch_transform(selector, std::move(operation), output_descriptor, inputs...);
      },
      input_descriptors);
}

/// \brief Update a Tensor output through an explicit backend selector.
/// \details Existing output values participate and the output shape is fixed.
///          The old output and input elements are passed as values.
/// \pre All operands have equal extents and output does not overlap an input.
template <class BackendSelector, class OutputTensor, class Function, class... InputTensors>
  requires detail::UpdateTransformTensors<OutputTensor, InputTensors...>
void transform_inplace(BackendSelector&& selector, OutputTensor&& output, Function&& function,
                       InputTensors const&... inputs)
{
  detail::require_transform_extents(output, inputs...);
  auto operation = linalg::transform_inplace_op{std::forward<Function>(function)};
  auto output_descriptor = device_mdspan_of(output);
  auto input_descriptors = std::tuple{device_mdspan_of(inputs)...};
  std::apply(
      [&](auto&... input_descriptors) {
        detail::dispatch_transform(std::forward<BackendSelector>(selector), std::move(operation), output_descriptor,
                                   input_descriptors...);
      },
      input_descriptors);
}

/// \brief Update a Tensor output using its operands' default backend selector.
template <class OutputTensor, class Function, class... InputTensors>
  requires detail::UpdateTransformTensors<OutputTensor, InputTensors...>
void transform_inplace(OutputTensor&& output, Function&& function, InputTensors const&... inputs)
{
  auto operation = linalg::transform_inplace_op{std::forward<Function>(function)};
  auto selector = linalg::select_backend(operation, output, inputs...);
  detail::require_transform_extents(output, inputs...);
  auto output_descriptor = device_mdspan_of(output);
  auto input_descriptors = std::tuple{device_mdspan_of(inputs)...};
  std::apply(
      [&](auto&... input_descriptors) {
        detail::dispatch_transform(selector, std::move(operation), output_descriptor, input_descriptors...);
      },
      input_descriptors);
}

} // namespace uni20
