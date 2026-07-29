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
async::CudaTask co_gemm_submission(GemmPlan<Scalar> plan, Scalar alpha, Scalar beta)
{
  auto& resources = plan.resources();
  co_await uni20::cuda::set_device(resources.device());
  auto execution = co_await uni20::cublas::acquire_execution(uni20::cublas::execution_pool(resources));
  execute_gemm(execution, plan, alpha, beta);
  co_return;
}

/// \brief Prepare cuBLAS GEMM and return a deferred non-blocking submission task.
/// \details Runtime operand acceptance is decided before the task is created.
///          A decline submits no work. Once the returned task is awaited,
///          resource or provider failures are terminal and cannot fall through
///          to another backend.
template <uni20::cublas::CublasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires GemmMdspans<Scalar, OutputMdspan, LhsMdspan, RhsMdspan>
auto try_make_gemm_task(OutputMdspan& output, Scalar const& alpha, LhsMdspan& lhs, RhsMdspan& rhs,
                        Scalar const& beta) -> KernelTaskAttempt<async::CudaTask>
{
  auto preparation = prepare_gemm<Scalar>(output, lhs, rhs);
  if (!kernel_attempt_succeeded(preparation.attempt) || !preparation.plan.has_work)
  {
    return KernelTaskAttempt<async::CudaTask>{preparation.attempt};
  }

  int const device = preparation.plan.device;
  auto task = co_gemm_submission(std::move(preparation.plan), alpha, beta);
  async::cuda_promise(task.handle()).bind_device(device);
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}
} // namespace detail::cublas_backend

/// \brief Prepare asynchronous cuBLAS work from normalized device mdspans.
template <uni20::cublas::CublasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires uni20::MutableRankedDeviceMdspanLike<OutputMdspan, 2> && uni20::RankedDeviceMdspanLike<LhsMdspan, 2> &&
               uni20::RankedDeviceMdspanLike<RhsMdspan, 2>
auto try_make_kernel_task(CublasBackend, gemm_op const&, OutputMdspan& output, Scalar const& alpha, LhsMdspan& lhs,
                          RhsMdspan& rhs, Scalar const& beta) -> KernelTaskAttempt<async::CudaTask>
{
  return detail::cublas_backend::try_make_gemm_task(output, alpha, lhs, rhs, beta);
}

/// \brief Prepare a replaceable output and lower the operands once for asynchronous cuBLAS work.
template <uni20::cublas::CublasScalar Scalar, class OutputTensor, class LhsTensor, class RhsTensor>
  requires uni20::MutableRankedDeviceTensorView<OutputTensor, 2> && uni20::RankedDeviceTensorView<LhsTensor, 2> &&
               uni20::RankedDeviceTensorView<RhsTensor, 2>
auto try_make_kernel_task(CublasBackend, assign_product_op const&, OutputTensor& output, Scalar const& alpha,
                          LhsTensor& lhs, RhsTensor& rhs) -> KernelTaskAttempt<async::CudaTask>
{
  auto lhs_span = uni20::detail::tensor_device_mdspan(std::as_const(lhs));
  auto rhs_span = uni20::detail::tensor_device_mdspan(std::as_const(rhs));
  auto const shape = detail::matrix_product_shape(lhs_span, rhs_span);
  auto const lhs_device = detail::cublas_backend::span_device(lhs_span);
  if (detail::cublas_backend::span_device(rhs_span) != lhs_device)
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::incompatible_devices};

  if constexpr (requires { uni20::prepare_output(output, shape, lhs_device); })
  {
    uni20::prepare_output(output, shape, lhs_device);
  }
  else if (!uni20::detail::tensor_extents_equal(output.extents(), shape))
  {
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::unsupported_shape};
  }

  auto output_span = uni20::detail::tensor_device_mdspan(output);
  if (detail::cublas_backend::span_device(output_span) != lhs_device)
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::incompatible_devices};
  return detail::cublas_backend::try_make_gemm_task(output_span, alpha, lhs_span, rhs_span, Scalar{});
}

/// \brief Prepare asynchronous cuBLAS assignment into deferred Tensor storage.
template <uni20::cublas::CublasScalar Scalar, uni20::MutableRankedDeviceTensorView<2> OutputTensor, class LhsTensor,
          class RhsTensor>
  requires uni20::RankedDeviceTensorView<LhsTensor, 2> && uni20::RankedDeviceTensorView<RhsTensor, 2> &&
               requires(async::shared_storage<OutputTensor>& storage, detail::matrix_product_extents const& shape,
                        uni20::cuda::Device device) { uni20::prepare_output(storage, shape, device); }
auto try_make_kernel_task(CublasBackend, assign_product_op const&, async::shared_storage<OutputTensor>& output_storage,
                          Scalar const& alpha, LhsTensor& lhs, RhsTensor& rhs) -> KernelTaskAttempt<async::CudaTask>
{
  auto lhs_span = uni20::detail::tensor_device_mdspan(std::as_const(lhs));
  auto rhs_span = uni20::detail::tensor_device_mdspan(std::as_const(rhs));
  auto const shape = detail::matrix_product_shape(lhs_span, rhs_span);
  auto const lhs_device = detail::cublas_backend::span_device(lhs_span);
  if (detail::cublas_backend::span_device(rhs_span) != lhs_device)
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::incompatible_devices};

  auto& output = uni20::prepare_output(output_storage, shape, lhs_device);
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  return detail::cublas_backend::try_make_gemm_task(output_span, alpha, lhs_span, rhs_span, Scalar{});
}

} // namespace uni20::linalg
