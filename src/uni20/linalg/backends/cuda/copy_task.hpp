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
namespace detail::cuda_reference
{

template <class Scalar> async::CudaTask co_copy_submission(CopyPlan<Scalar> plan)
{
  auto& resources = plan.output_buffer->resources();
  co_await uni20::cuda::set_device(resources.device());
  auto stream = co_await uni20::cuda::acquire_stream(resources.streams());
  enqueue_device_copy(plan, stream);
  co_return;
}

/// \brief Lower CUDA mdspans into a deferred non-blocking copy submission.
template <class OutputMdspan, class InputMdspan>
  requires(SupportedCopyMdspans<OutputMdspan, InputMdspan> && is_raw_cuda_mdspec<OutputMdspan> &&
           is_supported_cuda_mdspan<InputMdspan>)
auto try_make_copy_task(OutputMdspan& output, InputMdspan& input) -> KernelTaskAttempt<async::CudaTask>
{
  auto preparation = prepare_copy(output, input);
  if (!kernel_attempt_succeeded(preparation.attempt) || !preparation.has_work)
  {
    return KernelTaskAttempt<async::CudaTask>{preparation.attempt};
  }

  int const device = preparation.output_buffer->device().ordinal();
  auto task = co_copy_submission(std::move(preparation));
  async::cuda_promise(task.handle()).bind_device(device);
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}
} // namespace detail::cuda_reference

/// \brief Create a CUDA copy task from normalized mdspecs.
template <uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan>
  requires(detail::cuda_reference::SupportedCopyMdspans<OutputMdspan, InputMdspan> &&
           detail::cuda_reference::is_raw_cuda_mdspec<OutputMdspan> &&
           detail::cuda_reference::is_supported_cuda_mdspan<InputMdspan>)
auto try_make_kernel_task(CudaReferenceBackend, copy_op const&, OutputMdspan& output,
                          InputMdspan& input) -> KernelTaskAttempt<async::CudaTask>
{
  return detail::cuda_reference::try_make_copy_task(output, input);
}

} // namespace uni20::linalg
