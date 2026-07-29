#pragma once

/**
 * \file matrix_product.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for matrix-product overwrite and update operations.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/dispatch.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/tensor/output.hpp>

#if UNI20_BACKEND_CUBLAS
#include <uni20/linalg/backends/cublas/gemm_task.hpp>
#endif

#include <concepts>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class Scalar>
using matrix_product_scalar_awaiter_t = std::remove_cvref_t<decltype(async::read(std::declval<Scalar>()))>;

template <class Value, class Scalar>
concept MatrixProductScalar = async::TaskAwaitable<matrix_product_scalar_awaiter_t<Value>> &&
                              requires(matrix_product_scalar_awaiter_t<Value>& awaiter) {
                                { awaiter.await_resume() } -> std::convertible_to<Scalar>;
                              };

template <class OutputTensor, class LhsTensor, class RhsTensor>
void validate_async_matrix_product_aliasing(async::Async<OutputTensor> const& output,
                                            async::Async<LhsTensor> const& lhs, async::Async<RhsTensor> const& rhs)
{
  auto const* const output_queue = std::addressof(output.queue());
  ERROR_IF(output_queue == std::addressof(lhs.queue()) || output_queue == std::addressof(rhs.queue()),
           "async matrix product output must not share an epoch queue with an input");
}

template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor, class AlphaAwaiter>
async::AsyncTask co_assign_product(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                                   async::ReadBuffer<LhsTensor> lhs, async::ReadBuffer<RhsTensor> rhs,
                                   AlphaAwaiter alpha)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, lhs, rhs, alpha);
  auto& storage = std::get<0>(awaited);
  auto const& lhs_value = std::get<1>(awaited);
  auto const& rhs_value = std::get<2>(awaited);
  auto const& alpha_value = std::get<3>(awaited);

  using scalar_type = uni20::tensor_element_t<OutputTensor>;
  scalar_type const alpha_scalar = static_cast<scalar_type>(alpha_value);
  co_await co_dispatch_kernel(selector, assign_product_op{}, storage, alpha_scalar, lhs_value, rhs_value);
  co_return;
}

template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor, class AlphaAwaiter,
          class BetaAwaiter>
async::AsyncTask co_gemm(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                         async::ReadBuffer<LhsTensor> lhs, async::ReadBuffer<RhsTensor> rhs, AlphaAwaiter alpha,
                         BetaAwaiter beta)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, lhs, rhs, alpha, beta);
  auto& storage = std::get<0>(awaited);
  auto const& lhs_value = std::get<1>(awaited);
  auto const& rhs_value = std::get<2>(awaited);
  auto const& alpha_value = std::get<3>(awaited);
  auto const& beta_value = std::get<4>(awaited);

  if (!storage.constructed()) throw async::buffer_write_uninitialized{};
  auto const shape = matrix_product_shape(lhs_value, rhs_value);
  uni20::require_output(*storage, shape);
  using scalar_type = uni20::tensor_element_t<OutputTensor>;
  scalar_type const alpha_scalar = static_cast<scalar_type>(alpha_value);
  scalar_type const beta_scalar = static_cast<scalar_type>(beta_value);
  co_await co_dispatch_kernel(selector, gemm_op{}, *storage, alpha_scalar, lhs_value, rhs_value, beta_scalar);
  co_return;
}

template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor, class AlphaAwaiter>
void schedule_async_assign_product(BackendSelector selector, async::Async<OutputTensor>& output,
                                   async::Async<LhsTensor> const& lhs, async::Async<RhsTensor> const& rhs,
                                   AlphaAwaiter alpha)
{
  validate_async_matrix_product_aliasing(output, lhs, rhs);
  auto task = co_assign_product(std::move(selector), output.write(), lhs.read(), rhs.read(), std::move(alpha));
  task.debug_name("assign_product");
  async::schedule(std::move(task));
}

template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor, class AlphaAwaiter,
          class BetaAwaiter>
void schedule_async_gemm(BackendSelector selector, async::Async<OutputTensor>& output,
                         async::Async<LhsTensor> const& lhs, async::Async<RhsTensor> const& rhs, AlphaAwaiter alpha,
                         BetaAwaiter beta)
{
  validate_async_matrix_product_aliasing(output, lhs, rhs);
  auto task = co_gemm(std::move(selector), output.write(), lhs.read(), rhs.read(), std::move(alpha), std::move(beta));
  task.debug_name("gemm");
  async::schedule(std::move(task));
}
} // namespace detail

/// \brief Schedule fixed-output GEMM with an explicit backend selector.
/// \details Computes `output = alpha * lhs * rhs + beta * output`. Every
///          Tensor operand is asynchronous. `alpha` and `beta` may be immediate
///          values or async readers yielding compatible scalars.
/// \pre The output is constructed with the matrix-product shape and does not
///      share an epoch queue with either input.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor, class Alpha,
          class Beta>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductScalar<Alpha, uni20::tensor_element_t<OutputTensor>> &&
           detail::MatrixProductScalar<Beta, uni20::tensor_element_t<OutputTensor>>
void gemm(BackendSelector selector, async::Async<OutputTensor>& output, Alpha&& alpha,
          async::Async<LhsTensor> const& lhs, async::Async<RhsTensor> const& rhs, Beta&& beta)
{
  detail::schedule_async_gemm(std::move(selector), output, lhs, rhs, async::read(std::forward<Alpha>(alpha)),
                              async::read(std::forward<Beta>(beta)));
}

/// \brief Schedule fixed-output GEMM using the static Tensor selector.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::RankedDeviceTensorView<2> LhsTensor,
          uni20::RankedDeviceTensorView<2> RhsTensor, class Alpha, class Beta>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductScalar<Alpha, uni20::tensor_element_t<OutputTensor>> &&
           detail::MatrixProductScalar<Beta, uni20::tensor_element_t<OutputTensor>>
void gemm(async::Async<OutputTensor>& output, Alpha&& alpha, async::Async<LhsTensor> const& lhs,
          async::Async<RhsTensor> const& rhs, Beta&& beta)
{
  auto selector = select_backend_for<OutputTensor, LhsTensor, RhsTensor>(gemm_op{});
  gemm(std::move(selector), output, std::forward<Alpha>(alpha), lhs, rhs, std::forward<Beta>(beta));
}

/// \brief Schedule `output = alpha * lhs * rhs` with an explicit backend selector.
/// \details Every Tensor operand is asynchronous. `alpha` may be an immediate
///          value or an async reader yielding a compatible scalar; `async::read`
///          normalizes both forms into an awaiter retained by the coroutine.
///          An unconstructed output is initialized when its Tensor type can be
///          constructed from its extents.
/// \pre The output must not share an epoch queue with either input.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor,
          class Alpha = uni20::tensor_element_t<OutputTensor>>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductScalar<Alpha, uni20::tensor_element_t<OutputTensor>>
void assign_product(BackendSelector selector, async::Async<OutputTensor>& output, async::Async<LhsTensor> const& lhs,
                    async::Async<RhsTensor> const& rhs, Alpha&& alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  detail::schedule_async_assign_product(std::move(selector), output, lhs, rhs, async::read(std::forward<Alpha>(alpha)));
}

/// \brief Schedule `output = alpha * lhs * rhs` using the static Tensor selector.
/// \details The immutable selector is resolved from Tensor/storage types before
///          scheduling. `alpha` may be immediate or asynchronous. Unhandled task
///          failures propagate to the output epoch.
/// \pre The output must not share an epoch queue with either input.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::RankedDeviceTensorView<2> LhsTensor,
          uni20::RankedDeviceTensorView<2> RhsTensor, class Alpha = uni20::tensor_element_t<OutputTensor>>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductScalar<Alpha, uni20::tensor_element_t<OutputTensor>>
void assign_product(async::Async<OutputTensor>& output, async::Async<LhsTensor> const& lhs,
                    async::Async<RhsTensor> const& rhs, Alpha&& alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  auto selector = select_backend_for<OutputTensor, LhsTensor, RhsTensor>(assign_product_op{});
  detail::schedule_async_assign_product(std::move(selector), output, lhs, rhs, async::read(std::forward<Alpha>(alpha)));
}

/// \brief Schedule `output += alpha * lhs * rhs` with an explicit backend selector.
/// \details Every Tensor operand is asynchronous. The existing output value and
///          shape are required because they participate in the update. `alpha`
///          may be an immediate value or an async reader yielding a compatible
///          scalar.
/// \pre The output must not share an epoch queue with either input.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> OutputTensor,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor,
          class Alpha = uni20::tensor_element_t<OutputTensor>>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductScalar<Alpha, uni20::tensor_element_t<OutputTensor>>
void add_product(BackendSelector selector, async::Async<OutputTensor>& output, async::Async<LhsTensor> const& lhs,
                 async::Async<RhsTensor> const& rhs, Alpha&& alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  gemm(std::move(selector), output, std::forward<Alpha>(alpha), lhs, rhs, uni20::tensor_element_t<OutputTensor>{1});
}

/// \brief Schedule `output += alpha * lhs * rhs` using the static Tensor selector.
/// \details The immutable selector is resolved from Tensor/storage types before
///          scheduling. `alpha` may be immediate or asynchronous. Unhandled task
///          failures propagate to the output epoch.
/// \pre The output must not share an epoch queue with either input.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, uni20::RankedDeviceTensorView<2> LhsTensor,
          uni20::RankedDeviceTensorView<2> RhsTensor, class Alpha = uni20::tensor_element_t<OutputTensor>>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductScalar<Alpha, uni20::tensor_element_t<OutputTensor>>
void add_product(async::Async<OutputTensor>& output, async::Async<LhsTensor> const& lhs,
                 async::Async<RhsTensor> const& rhs, Alpha&& alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  gemm(output, std::forward<Alpha>(alpha), lhs, rhs, uni20::tensor_element_t<OutputTensor>{1});
}

} // namespace uni20::linalg
