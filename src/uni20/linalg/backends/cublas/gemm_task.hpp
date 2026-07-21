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
template <uni20::cublas::CublasScalar Scalar, uni20::MutableRankedStridedMdspan<2> OutputMdspan,
          uni20::RankedStridedMdspan<2> LhsMdspan, uni20::RankedStridedMdspan<2> RhsMdspan>
  requires requires(OutputMdspan& output, LhsMdspan& lhs, RhsMdspan& rhs) {
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

} // namespace uni20::linalg
