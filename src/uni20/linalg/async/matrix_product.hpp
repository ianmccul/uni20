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
#include <uni20/linalg/async/co_gemm.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/tensor/output.hpp>

#if UNI20_BACKEND_CUBLAS
#include <uni20/storage/cuda_async_storage.hpp>
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
template <class Alpha>
using matrix_product_alpha_awaiter_t = std::remove_cvref_t<decltype(async::read(std::declval<Alpha>()))>;

template <class Alpha, class Scalar>
concept MatrixProductAlpha = async::TaskAwaitable<matrix_product_alpha_awaiter_t<Alpha>> &&
                             requires(matrix_product_alpha_awaiter_t<Alpha>& awaiter) {
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

#if UNI20_BACKEND_CUBLAS
template <class OutputTensor, class LhsTensor>
concept CudaAsyncMatrixProductOutput =
    std::same_as<uni20::detail::tensor_storage_policy_t<OutputTensor>, uni20::CudaAsyncStorage> &&
    std::same_as<uni20::detail::tensor_storage_policy_t<LhsTensor>, uni20::CudaAsyncStorage>;

template <uni20::TensorView Tensor>
[[nodiscard]] uni20::cuda::DeviceResources& cuda_tensor_resources(Tensor const& tensor)
{
  return tensor.mdspan().data_handle().buffer().resources();
}

#endif

template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor>
[[nodiscard]] OutputTensor& prepare_async_assign_product_output(async::shared_storage<OutputTensor>& storage,
                                                                matrix_product_extents const& shape,
                                                                LhsTensor const& lhs)
{
  if (storage.constructed()) return *storage;

  using extents_type = uni20::tensor_extents_t<OutputTensor>;
  auto const extents = uni20::detail::convert_tensor_extents<extents_type>(shape);
#if UNI20_BACKEND_CUBLAS
  if constexpr (CudaAsyncMatrixProductOutput<OutputTensor, LhsTensor> &&
                std::constructible_from<OutputTensor, uni20::cuda::DeviceResources&, extents_type const&>)
  {
    return storage.emplace(cuda_tensor_resources(lhs), extents);
  }
#endif
  if constexpr (std::constructible_from<OutputTensor, extents_type const&>)
  {
    return storage.emplace(extents);
  }
  else
  {
    throw async::buffer_write_uninitialized{};
  }
}

template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor, class AlphaAwaiter>
async::AsyncTask async_assign_product_task(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                                           async::ReadBuffer<LhsTensor> lhs, async::ReadBuffer<RhsTensor> rhs,
                                           AlphaAwaiter alpha)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, lhs, rhs, alpha);
  auto& storage = std::get<0>(awaited);
  auto const& lhs_value = std::get<1>(awaited);
  auto const& rhs_value = std::get<2>(awaited);
  auto const& alpha_value = std::get<3>(awaited);

  auto const shape = matrix_product_shape(lhs_value, rhs_value);
  auto& output_value = prepare_async_assign_product_output<OutputTensor>(storage, shape, lhs_value);
  uni20::ensure_shape(output_value, shape);
  using scalar_type = uni20::tensor_element_t<OutputTensor>;
  co_await co_gemm(selector, output_value.mdspan(), static_cast<scalar_type>(alpha_value), lhs_value.mdspan(),
                   rhs_value.mdspan(), scalar_type{});
  co_return;
}

template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor, class AlphaAwaiter>
async::AsyncTask async_add_product_task(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                                        async::ReadBuffer<LhsTensor> lhs, async::ReadBuffer<RhsTensor> rhs,
                                        AlphaAwaiter alpha)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, lhs, rhs, alpha);
  auto& storage = std::get<0>(awaited);
  auto const& lhs_value = std::get<1>(awaited);
  auto const& rhs_value = std::get<2>(awaited);
  auto const& alpha_value = std::get<3>(awaited);

  if (!storage.constructed()) throw async::buffer_write_uninitialized{};
  auto const shape = matrix_product_shape(lhs_value, rhs_value);
  uni20::require_shape(*storage, shape);
  using scalar_type = uni20::tensor_element_t<OutputTensor>;
  co_await co_gemm(selector, storage->mdspan(), static_cast<scalar_type>(alpha_value), lhs_value.mdspan(),
                   rhs_value.mdspan(), scalar_type{1});
  co_return;
}

template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor, class AlphaAwaiter>
void schedule_async_assign_product(BackendSelector selector, async::Async<OutputTensor>& output,
                                   async::Async<LhsTensor> const& lhs, async::Async<RhsTensor> const& rhs,
                                   AlphaAwaiter alpha)
{
  validate_async_matrix_product_aliasing(output, lhs, rhs);
  auto task = async_assign_product_task(std::move(selector), output.write(), lhs.read(), rhs.read(), std::move(alpha));
  task.debug_name("assign_product");
  async::schedule(std::move(task));
}

template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor, class AlphaAwaiter>
void schedule_async_add_product(BackendSelector selector, async::Async<OutputTensor>& output,
                                async::Async<LhsTensor> const& lhs, async::Async<RhsTensor> const& rhs,
                                AlphaAwaiter alpha)
{
  validate_async_matrix_product_aliasing(output, lhs, rhs);
  auto task = async_add_product_task(std::move(selector), output.write(), lhs.read(), rhs.read(), std::move(alpha));
  task.debug_name("add_product");
  async::schedule(std::move(task));
}
} // namespace detail

/// \brief Schedule `output = alpha * lhs * rhs` with an explicit backend selector.
/// \details Every Tensor operand is asynchronous. `alpha` may be an immediate
///          value or an async reader yielding a compatible scalar; `async::read`
///          normalizes both forms into an awaiter retained by the coroutine.
///          An unconstructed output is initialized when its Tensor type can be
///          constructed from its extents.
/// \pre The output must not share an epoch queue with either input.
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor, class Alpha = uni20::tensor_element_t<OutputTensor>>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductAlpha<Alpha, uni20::tensor_element_t<OutputTensor>>
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
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor, class Alpha = uni20::tensor_element_t<OutputTensor>>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductAlpha<Alpha, uni20::tensor_element_t<OutputTensor>>
void assign_product(async::Async<OutputTensor>& output, async::Async<LhsTensor> const& lhs,
                    async::Async<RhsTensor> const& rhs, Alpha&& alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  auto selector = select_backend_for<OutputTensor, LhsTensor, RhsTensor>(gemm_op{});
  detail::schedule_async_assign_product(std::move(selector), output, lhs, rhs, async::read(std::forward<Alpha>(alpha)));
}

/// \brief Schedule `output += alpha * lhs * rhs` with an explicit backend selector.
/// \details Every Tensor operand is asynchronous. The existing output value and
///          shape are required because they participate in the update. `alpha`
///          may be an immediate value or an async reader yielding a compatible
///          scalar.
/// \pre The output must not share an epoch queue with either input.
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor, class Alpha = uni20::tensor_element_t<OutputTensor>>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductAlpha<Alpha, uni20::tensor_element_t<OutputTensor>>
void add_product(BackendSelector selector, async::Async<OutputTensor>& output, async::Async<LhsTensor> const& lhs,
                 async::Async<RhsTensor> const& rhs, Alpha&& alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  detail::schedule_async_add_product(std::move(selector), output, lhs, rhs, async::read(std::forward<Alpha>(alpha)));
}

/// \brief Schedule `output += alpha * lhs * rhs` using the static Tensor selector.
/// \details The immutable selector is resolved from Tensor/storage types before
///          scheduling. `alpha` may be immediate or asynchronous. Unhandled task
///          failures propagate to the output epoch.
/// \pre The output must not share an epoch queue with either input.
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor, class Alpha = uni20::tensor_element_t<OutputTensor>>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor> &&
           detail::MatrixProductAlpha<Alpha, uni20::tensor_element_t<OutputTensor>>
void add_product(async::Async<OutputTensor>& output, async::Async<LhsTensor> const& lhs,
                 async::Async<RhsTensor> const& rhs, Alpha&& alpha = uni20::tensor_element_t<OutputTensor>{1})
{
  auto selector = select_backend_for<OutputTensor, LhsTensor, RhsTensor>(gemm_op{});
  detail::schedule_async_add_product(std::move(selector), output, lhs, rhs, async::read(std::forward<Alpha>(alpha)));
}

} // namespace uni20::linalg
