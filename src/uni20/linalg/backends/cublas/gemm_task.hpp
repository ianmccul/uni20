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
///          A decline is side-effect free. Once the returned task is awaited,
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

/// \brief Lower DeviceTensorView operands before preparing asynchronous cuBLAS work.
template <uni20::cublas::CublasScalar Scalar, class OutputTensor, class LhsTensor, class RhsTensor>
  requires uni20::MutableRankedDeviceTensorView<OutputTensor, 2> && uni20::RankedDeviceTensorView<LhsTensor, 2> &&
               uni20::RankedDeviceTensorView<RhsTensor, 2>
auto try_make_kernel_task(CublasBackend, gemm_op const&, OutputTensor& output, Scalar const& alpha, LhsTensor& lhs,
                          RhsTensor& rhs, Scalar const& beta) -> KernelTaskAttempt<async::CudaTask>
{
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  return detail::cublas_backend::try_make_gemm_task(output_span, alpha, lhs_span, rhs_span, beta);
}

/// \brief Lower replaceable-output Tensor operands before preparing asynchronous cuBLAS work.
template <uni20::cublas::CublasScalar Scalar, class OutputTensor, class LhsTensor, class RhsTensor>
  requires uni20::MutableRankedDeviceTensorView<OutputTensor, 2> && uni20::RankedDeviceTensorView<LhsTensor, 2> &&
               uni20::RankedDeviceTensorView<RhsTensor, 2>
auto try_make_kernel_task(CublasBackend backend, assign_product_op const&, OutputTensor& output, Scalar const& alpha,
                          LhsTensor& lhs, RhsTensor& rhs) -> KernelTaskAttempt<async::CudaTask>
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  auto const lhs_device = detail::cublas_backend::tensor_device(lhs);
  if (detail::cublas_backend::tensor_device(rhs) != lhs_device)
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::unsupported_instance};

  bool const shape_matches = detail::cublas_backend::output_shape_matches(output, shape);
  bool const device_matches = detail::cublas_backend::tensor_device(output) == lhs_device;
  bool const replacement_required = !shape_matches || !device_matches;
  if (replacement_required)
  {
    if constexpr (requires { uni20::prepare_output(output, shape, lhs_device); })
    {
      uni20::prepare_output(output, shape, lhs_device);
    }
    else
    {
      return KernelTaskAttempt<async::CudaTask>{shape_matches ? KernelAttempt::unsupported_instance
                                                              : KernelAttempt::unsupported_shape};
    }
  }

  Scalar const beta{};
  auto attempt = try_make_kernel_task(backend, gemm_op{}, output, alpha, lhs, rhs, beta);
  if (replacement_required) CHECK(kernel_attempt_succeeded(attempt.attempt()));
  return attempt;
}

/// \brief Prepare asynchronous cuBLAS assignment into deferred Tensor storage.
template <uni20::cublas::CublasScalar Scalar, uni20::TensorOutputStorage OutputStorage, class LhsTensor,
          class RhsTensor>
  requires uni20::MutableRankedDeviceTensorView<uni20::tensor_output_storage_t<OutputStorage>, 2> &&
               uni20::RankedDeviceTensorView<LhsTensor, 2> && uni20::RankedDeviceTensorView<RhsTensor, 2> &&
               requires(OutputStorage& storage, detail::matrix_product_extents const& shape,
                        uni20::cuda::Device device) { uni20::prepare_output(storage, shape, device); }
auto try_make_kernel_task(CublasBackend backend, assign_product_op const&, OutputStorage& output_storage,
                          Scalar const& alpha, LhsTensor& lhs, RhsTensor& rhs) -> KernelTaskAttempt<async::CudaTask>
{
  if (output_storage.constructed())
    return try_make_kernel_task(backend, assign_product_op{}, *output_storage, alpha, lhs, rhs);

  auto const shape = detail::matrix_product_shape(lhs, rhs);
  auto const lhs_device = detail::cublas_backend::tensor_device(lhs);
  if (detail::cublas_backend::tensor_device(rhs) != lhs_device)
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::unsupported_instance};

  auto& output = uni20::prepare_output(output_storage, shape, lhs_device);
  Scalar const beta{};
  auto attempt = try_make_kernel_task(backend, gemm_op{}, output, alpha, lhs, rhs, beta);
  CHECK(kernel_attempt_succeeded(attempt.attempt()));
  return attempt;
}

} // namespace uni20::linalg
