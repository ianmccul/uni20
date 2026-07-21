#pragma once

/**
 * \file co_gemm.hpp
 * \ingroup linalg
 * \brief Coroutine submission interface for fixed-output GEMM.
 */

#include <uni20/async/async_task.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>

#if UNI20_BACKEND_CUBLAS
#include <uni20/async/cuda_task.hpp>
#include <uni20/backend/cublas/task_awaiters.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>
#include <uni20/linalg/cublas/gemm.hpp>
#endif

#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class Selector> inline constexpr bool is_cublas_only_selector = false;

#if UNI20_BACKEND_CUBLAS
template <> inline constexpr bool is_cublas_only_selector<CublasBackend> = true;
template <> inline constexpr bool is_cublas_only_selector<backend_list<CublasBackend>> = true;

template <uni20::cublas::CublasScalar Scalar>
async::CudaTask cublas_gemm_task(uni20::linalg::cublas::GemmPlan<Scalar> plan, Scalar alpha, Scalar beta)
{
  if (!plan.has_work) co_return;

  auto& resources = plan.resources();
  co_await uni20::cuda::set_device(resources.device());
  auto execution = co_await uni20::cublas::acquire_execution(uni20::cublas::execution_pool(resources));
  uni20::linalg::cublas::execute_gemm(execution, plan, alpha, beta);
  co_return;
}
#endif
} // namespace detail

/// \brief Submit host GEMM through an ordinary non-suspending backend dispatch.
/// \details The returned task completes after the synchronous backend call
///          finishes. This default path deliberately excludes the CUDA-only
///          selector whose resource admission has a suspending overload.
template <class BackendSelector, uni20::MutableRankedSpanLike<2> OutputMdspan, uni20::Scalar Scalar,
          uni20::RankedSpanLike<2> LhsMdspan, uni20::RankedSpanLike<2> RhsMdspan>
  requires(!detail::is_cublas_only_selector<std::remove_cvref_t<BackendSelector>>)
async::AsyncTask co_gemm(BackendSelector selector, OutputMdspan output, Scalar alpha, LhsMdspan lhs, RhsMdspan rhs,
                         Scalar beta)
{
  dispatch_kernel(std::move(selector), gemm_op{}, output, alpha, lhs, rhs, beta);
  co_return;
}

#if UNI20_BACKEND_CUBLAS
/// \brief Submit cuBLAS GEMM through awaitable CUDA resource admission.
/// \details The task completes after provider submission and publication of
///          CUDA buffer completion records, not after device execution.
template <uni20::cublas::CublasScalar Scalar, uni20::MutableRankedStridedMdspan<2> OutputMdspan,
          uni20::RankedStridedMdspan<2> LhsMdspan, uni20::RankedStridedMdspan<2> RhsMdspan>
  requires requires(OutputMdspan& output, LhsMdspan& lhs, RhsMdspan& rhs) {
    uni20::linalg::cublas::prepare_gemm<Scalar>(output, lhs, rhs);
  }
async::CudaTask co_gemm(CublasBackend, OutputMdspan output, Scalar alpha, LhsMdspan lhs, RhsMdspan rhs, Scalar beta)
{
  auto preparation = uni20::linalg::cublas::prepare_gemm<Scalar>(output, lhs, rhs);
  ERROR_IF(!kernel_attempt_succeeded(preparation.attempt), "cuBLAS co_gemm declined prepared operands",
           kernel_attempt_name(preparation.attempt));
  int const device = preparation.plan.device;
  auto task = detail::cublas_gemm_task(std::move(preparation.plan), alpha, beta);
  async::cuda_promise(task.handle()).bind_device(device);
  return task;
}

/// \brief Submit GEMM through the CUDA storage policy's current one-backend list.
template <uni20::cublas::CublasScalar Scalar, uni20::MutableRankedStridedMdspan<2> OutputMdspan,
          uni20::RankedStridedMdspan<2> LhsMdspan, uni20::RankedStridedMdspan<2> RhsMdspan>
auto co_gemm(backend_list<CublasBackend>, OutputMdspan output, Scalar alpha, LhsMdspan lhs, RhsMdspan rhs,
             Scalar beta) -> async::CudaTask
{
  return co_gemm(CublasBackend{}, std::move(output), alpha, std::move(lhs), std::move(rhs), beta);
}
#endif

} // namespace uni20::linalg
