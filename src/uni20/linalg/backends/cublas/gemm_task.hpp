#pragma once

/**
 * \file gemm_task.hpp
 * \ingroup linalg
 * \brief Coroutine kernel-dispatch customization for CublasBackend GEMM.
 */

#include <uni20/async/cuda_task.hpp>
#include <uni20/backend/cublas/task_awaiters.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>
#include <uni20/linalg/async/kernel_task.hpp>
#include <uni20/linalg/backends/cublas/gemm.hpp>

#include <utility>

namespace uni20::linalg
{
namespace detail::cublas_backend
{

template <uni20::cublas::CublasScalar Scalar>
async::CudaTask gemm_submission_task(GemmPlan<Scalar> plan, Scalar alpha, Scalar beta)
{
  auto& resources = plan.resources();
  co_await uni20::cuda::set_device(resources.device());
  auto execution = co_await uni20::cublas::acquire_execution(uni20::cublas::execution_pool(resources));
  execute_gemm(execution, plan, alpha, beta);
  co_return;
}

/// \brief Prepare cuBLAS GEMM and return a deferred non-blocking submission task.
/// \details Runtime operand acceptance is decided before the task is created.
///          A decline is side-effect free. Once the returned task is awaited,
///          resource or provider failures are terminal and cannot fall through
///          to another backend.
template <uni20::cublas::CublasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires GemmMdspans<Scalar, OutputMdspan, LhsMdspan, RhsMdspan>
auto try_gemm_task(OutputMdspan& output, Scalar const& alpha, LhsMdspan& lhs, RhsMdspan& rhs,
                   Scalar const& beta) -> KernelTaskAttempt<async::CudaTask>
{
  auto preparation = prepare_gemm<Scalar>(output, lhs, rhs);
  if (!kernel_attempt_succeeded(preparation.attempt) || !preparation.plan.has_work)
  {
    return KernelTaskAttempt<async::CudaTask>{preparation.attempt};
  }

  int const device = preparation.plan.device;
  auto task = gemm_submission_task(std::move(preparation.plan), alpha, beta);
  async::cuda_promise(task.handle()).bind_device(device);
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}
} // namespace detail::cublas_backend

/// \brief Lower DeviceTensorView operands before preparing asynchronous cuBLAS work.
template <uni20::cublas::CublasScalar Scalar, class OutputTensor, class LhsTensor, class RhsTensor>
  requires uni20::MutableRankedDeviceTensorView<OutputTensor, 2> && uni20::RankedDeviceTensorView<LhsTensor, 2> &&
               uni20::RankedDeviceTensorView<RhsTensor, 2>
auto try_kernel_task(CublasBackend, gemm_op const&, OutputTensor& output, Scalar const& alpha, LhsTensor& lhs,
                     RhsTensor& rhs, Scalar const& beta) -> KernelTaskAttempt<async::CudaTask>
{
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  return detail::cublas_backend::try_gemm_task(output_span, alpha, lhs_span, rhs_span, beta);
}

} // namespace uni20::linalg
