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

namespace uni20::linalg
{
namespace detail
{

template <uni20::cublas::CublasScalar Scalar>
async::CudaTask cublas_gemm_task(cublas_backend::GemmPlan<Scalar> plan, Scalar alpha, Scalar beta)
{
  auto& resources = plan.resources();
  co_await uni20::cuda::set_device(resources.device());
  auto execution = co_await uni20::cublas::acquire_execution(uni20::cublas::execution_pool(resources));
  cublas_backend::execute_gemm(execution, plan, alpha, beta);
  co_return;
}

} // namespace detail

/// \brief Prepare cuBLAS GEMM and return a deferred non-blocking submission task.
/// \details Runtime operand acceptance is decided before the task is created.
///          A decline is side-effect free. Once the returned task is awaited,
///          resource or provider failures are terminal and cannot fall through
///          to another backend.
template <uni20::cublas::CublasScalar Scalar, class OutputMdspan, uni20::RankedStridedDeviceSpanLike<2> LhsMdspan,
          uni20::RankedStridedDeviceSpanLike<2> RhsMdspan>
  requires uni20::MutableDeviceSpanLike<OutputMdspan> && uni20::RankedStridedDeviceSpanLike<OutputMdspan, 2> &&
               requires(OutputMdspan& output, LhsMdspan& lhs, RhsMdspan& rhs) {
                 detail::cublas_backend::prepare_gemm<Scalar>(output, lhs, rhs);
               }
auto try_kernel_task(CublasBackend, gemm_op const&, OutputMdspan& output, Scalar const& alpha, LhsMdspan& lhs,
                     RhsMdspan& rhs, Scalar const& beta) -> KernelTaskAttempt<async::CudaTask>
{
  auto preparation = detail::cublas_backend::prepare_gemm<Scalar>(output, lhs, rhs);
  if (!kernel_attempt_succeeded(preparation.attempt) || !preparation.plan.has_work)
  {
    return KernelTaskAttempt<async::CudaTask>{preparation.attempt};
  }

  int const device = preparation.plan.device;
  auto task = detail::cublas_gemm_task(std::move(preparation.plan), alpha, beta);
  async::cuda_promise(task.handle()).bind_device(device);
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}

/// \brief Lower DeviceTensorView operands before preparing asynchronous cuBLAS work.
template <uni20::cublas::CublasScalar Scalar, class OutputTensor, class LhsTensor, class RhsTensor>
  requires uni20::MutableRankedDeviceTensorView<OutputTensor, 2> && uni20::RankedDeviceTensorView<LhsTensor, 2> &&
               uni20::RankedDeviceTensorView<RhsTensor, 2>
auto try_kernel_task(CublasBackend backend, gemm_op const& op, OutputTensor& output, Scalar const& alpha,
                     LhsTensor& lhs, RhsTensor& rhs, Scalar const& beta) -> KernelTaskAttempt<async::CudaTask>
{
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  return try_kernel_task(backend, op, output_span, alpha, lhs_span, rhs_span, beta);
}

} // namespace uni20::linalg
