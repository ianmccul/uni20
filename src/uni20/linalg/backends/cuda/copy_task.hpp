#pragma once

/**
 * \file copy_task.hpp
 * \ingroup linalg
 * \brief Coroutine kernel-dispatch customization for CUDA Tensor copies.
 */

#include <uni20/async/cuda_task.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>
#include <uni20/linalg/async/kernel_task.hpp>
#include <uni20/linalg/backends/cuda/copy.hpp>

namespace uni20::linalg
{
namespace detail
{

template <class Scalar> async::CudaTask cuda_reference_copy_task(cuda_reference::CopyPlan<Scalar> plan)
{
  auto& resources = plan.output_buffer->resources();
  co_await uni20::cuda::set_device(resources.device());
  auto stream = co_await uni20::cuda::acquire_stream(resources.streams());
  cuda_reference::enqueue_device_copy(plan, stream);
  co_return;
}

} // namespace detail

/// \brief Return a deferred non-blocking CUDA-to-CUDA copy submission.
template <class OutputMdspan, class InputMdspan>
  requires(detail::cuda_reference::SupportedCopyMdspans<OutputMdspan, InputMdspan> &&
           detail::cuda_reference::is_raw_cuda_mdspan<OutputMdspan> &&
           detail::cuda_reference::is_raw_cuda_mdspan<InputMdspan>)
auto try_kernel_task(CudaReferenceBackend, copy_op const&, OutputMdspan& output,
                     InputMdspan& input) -> KernelTaskAttempt<async::CudaTask>
{
  auto preparation = detail::cuda_reference::prepare_copy(output, input);
  if (!kernel_attempt_succeeded(preparation.attempt) || !preparation.has_work)
  {
    return KernelTaskAttempt<async::CudaTask>{preparation.attempt};
  }

  int const device = preparation.output_buffer->device().ordinal();
  auto task = detail::cuda_reference_copy_task(std::move(preparation));
  async::cuda_promise(task.handle()).bind_device(device);
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}

/// \brief Normalize tensor metadata inside the CUDA task backend.
template <uni20::MutableDeviceTensorView OutputTensor, uni20::DeviceTensorView InputTensor>
auto try_kernel_task(CudaReferenceBackend backend, copy_op const& operation, OutputTensor& output,
                     InputTensor& input) -> KernelTaskAttempt<async::CudaTask>
{
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto input_span = uni20::detail::tensor_device_mdspan(input);
  return try_kernel_task(backend, operation, output_span, input_span);
}

} // namespace uni20::linalg
