#pragma once

/**
 * \file copy_into.hpp
 * \ingroup tensor
 * \brief Backend-dispatched copies into existing mdspan and tensor outputs.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backends/cpu/copy.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/output.hpp>

#if UNI20_BACKEND_CUDA
#include <uni20/linalg/backends/cuda/copy.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/storage/vectorstorage.hpp>
#endif

#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{
template <class Output, class Input>
concept CopySpans = MutableDeviceSpanLike<Output> && DeviceSpanLike<Input> &&
                    (std::remove_cvref_t<Output>::rank() == std::remove_cvref_t<Input>::rank());

template <class Output, class Input>
concept CopyTensors = MutableDeviceTensorView<Output> && DeviceTensorView<Input> &&
                      (device_tensor_mdspan_t<Output>::rank() == device_tensor_mdspan_t<Input>::rank());

#if UNI20_BACKEND_CUDA
template <class Output, class Input>
inline constexpr bool is_pageable_cuda_transfer = (std::same_as<tensor_storage_policy_t<Output>, CudaStorage> &&
                                                   std::same_as<tensor_storage_policy_t<Input>, VectorStorage>) ||
                                                  (std::same_as<tensor_storage_policy_t<Output>, VectorStorage> &&
                                                   std::same_as<tensor_storage_policy_t<Input>, CudaStorage>);
#endif

template <DeviceSpanLike Output, DeviceSpanLike Input>
[[nodiscard]] constexpr bool copy_extents_match(Output const& output, Input const& input) noexcept
{
  if constexpr (std::remove_cvref_t<Output>::rank() != std::remove_cvref_t<Input>::rank())
  {
    return false;
  }
  else
  {
    constexpr std::size_t rank = std::remove_cvref_t<Output>::rank();
    if constexpr (rank > 0)
    {
      for (std::size_t axis = 0; axis < rank; ++axis)
      {
        if (output.extent(axis) != input.extent(axis)) return false;
      }
    }
    return true;
  }
}
} // namespace detail

/// \brief Copy between fixed-shape mdspan-like operands through an explicit selector.
/// \details Accessor semantics are observed by the selected backend, so a
///          conjugating input remains a lazy view until this explicit copy.
/// \pre Input and output do not destructively overlap.
template <class BackendSelector, class OutputMdspan, class InputMdspan>
  requires detail::CopySpans<OutputMdspan, InputMdspan>
void copy(BackendSelector&& selector, OutputMdspan&& output, InputMdspan&& input)
{
  ERROR_IF(!detail::copy_extents_match(output, input), "copy output shape does not match input shape");
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::copy_op{},
                          std::forward<OutputMdspan>(output), std::forward<InputMdspan>(input));
}

/// \brief Copy into a resizable or already-compatible tensor through an explicit selector.
/// \details Shape preparation occurs before either resolved mdspan is acquired.
/// \pre Input and output do not destructively overlap.
template <class BackendSelector, class OutputTensor, class InputTensor>
  requires detail::CopyTensors<OutputTensor, InputTensor>
void copy(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input)
{
  ensure_shape(output, input.extents());
  auto output_span = detail::tensor_device_mdspan(output);
  auto input_span = detail::tensor_device_mdspan(input);
  copy(std::forward<BackendSelector>(selector), output_span, input_span);
}

/// \brief Copy into a tensor using the operands' default backend selector.
template <class OutputTensor, class InputTensor>
  requires detail::CopyTensors<OutputTensor, InputTensor>
void copy(OutputTensor&& output, InputTensor const& input)
{
  ensure_shape(output, input.extents());
  auto output_span = detail::tensor_device_mdspan(output);
  auto input_span = detail::tensor_device_mdspan(input);
#if UNI20_BACKEND_CUDA
  if constexpr (detail::is_pageable_cuda_transfer<OutputTensor, InputTensor>)
  {
    copy(linalg::CudaReferenceBackend{}, output_span, input_span);
  }
  else
#endif
  {
    auto selector = linalg::select_backend(linalg::copy_op{}, output, input);
    copy(selector, output_span, input_span);
  }
}

/// \brief Assign tensor values through a mutable tensor alias descriptor.
/// \details Async alias assignment discovers this function through ADL. The
///          descriptor itself remains unchanged while `copy` writes its values.
template <MutableDeviceTensorView Output, DeviceTensorView Input>
  requires(device_tensor_mdspan_t<Output>::rank() == device_tensor_mdspan_t<Input>::rank())
void assign_through(Output& output, Input const& input)
{
  copy(output, input);
}

} // namespace uni20
