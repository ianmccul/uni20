#pragma once

/**
 * \file matrix_norm.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for dense matrix norms.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/linalg/ops/matrix_norm.hpp>

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class MatrixTensor>
using preserving_matrix_norm_result_t = std::remove_cvref_t<decltype(uni20::linalg::matrix_norm(
    std::declval<MatrixTensor const&>(), MatrixNorm::Frobenius))>;

template <class BackendSelector, class ResultTensor, uni20::RankedTensorView<2> MatrixTensor>
async::AsyncTask co_matrix_norm(BackendSelector const selector, async::WriteBuffer<ResultTensor> output,
                                async::ReadBuffer<MatrixTensor> matrix, MatrixNorm const kind)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, matrix);
  auto& storage = std::get<0>(awaited);
  auto const& matrix_value = std::get<1>(awaited);

  auto result = uni20::linalg::matrix_norm(selector, matrix_value, kind);
  storage.emplace(std::move(result));
  co_return;
}

template <class BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
async::AsyncTask
co_matrix_norm_host(BackendSelector const selector,
                    async::WriteBuffer<uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>> output,
                    async::ReadBuffer<MatrixTensor> matrix, MatrixNorm const kind)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, matrix);
  auto& storage = std::get<0>(awaited);
  auto const& matrix_value = std::get<1>(awaited);

  auto result = uni20::linalg::matrix_norm_host(selector, matrix_value, kind);
  storage.emplace(std::move(result));
  co_return;
}

template <class BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::RankedTensorView<2> MatrixTensor>
async::AsyncTask co_matrix_norm_output(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                                       async::ReadBuffer<MatrixTensor> matrix, MatrixNorm const kind)
{
  if constexpr (async::is_async_alias_v<OutputTensor>)
  {
    uni20::detail::AsyncAliasWriteDescriptorAwaiter output_descriptor_awaiter(output);
    auto awaited = co_await async::all(output_descriptor_awaiter, matrix);
    auto output_value = std::get<0>(awaited);
    uni20::linalg::matrix_norm(selector, output_value, std::get<1>(awaited), kind);
  }
  else
  {
    auto output_storage = output.storage();
    auto awaited = co_await async::all(output_storage, matrix);
    auto& output_value = uni20::detail::prepare_async_scalar_output<OutputTensor>(std::get<0>(awaited));
    uni20::linalg::matrix_norm(selector, output_value, std::get<1>(awaited), kind);
  }
  co_return;
}

template <class BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_matrix_norm(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                        MatrixNorm kind)
{
  using result_type = preserving_matrix_norm_result_t<MatrixTensor>;
  async::Async<result_type> output;
  output.debug_name("matrix_norm.result");
  auto task = co_matrix_norm(std::move(selector), output.write(), matrix.read(), kind);
  task.debug_name("matrix_norm");
  async::schedule(std::move(task));
  return output;
}

template <class BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_matrix_norm_host(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                             MatrixNorm kind)
{
  using result_type = uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>;
  async::Async<result_type> output;
  output.debug_name("matrix_norm_host.result");
  auto task = co_matrix_norm_host(std::move(selector), output.write(), matrix.read(), kind);
  task.debug_name("matrix_norm_host");
  async::schedule(std::move(task));
  return output;
}

template <class BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::RankedTensorView<2> MatrixTensor>
void schedule_matrix_norm_output(BackendSelector selector, async::Async<OutputTensor>& output,
                                 async::Async<MatrixTensor> const& matrix, MatrixNorm kind)
{
  ERROR_IF(std::addressof(output.queue()) == std::addressof(matrix.queue()),
           "async matrix-norm output must not share an epoch queue with its input");
  auto task = co_matrix_norm_output(std::move(selector), output.write(), matrix.read(), kind);
  task.debug_name("matrix_norm");
  async::schedule(std::move(task));
}
} // namespace detail

/// \brief Schedule a matrix norm into an explicit async scalar Tensor.
/// \pre The output must not share an epoch queue with the matrix.
template <KernelBackendSelector BackendSelector, uni20::AsyncTensorOutput OutputTensor,
          uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::MutableScalarTensorView<OutputTensor> &&
           uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>> &&
           std::same_as<uni20::tensor_element_t<OutputTensor>,
                        uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>>
void matrix_norm(BackendSelector selector, async::Async<OutputTensor>& output, async::Async<MatrixTensor> const& matrix,
                 MatrixNorm kind)
{
  detail::schedule_matrix_norm_output(std::move(selector), output, matrix, kind);
}

/// \brief Schedule a matrix norm into an explicit async scalar Tensor using storage policy.
/// \pre The output must not share an epoch queue with the matrix.
template <uni20::AsyncTensorOutput OutputTensor, uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::MutableScalarTensorView<OutputTensor> &&
           uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>> &&
           std::same_as<uni20::tensor_element_t<OutputTensor>,
                        uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>>
void matrix_norm(async::Async<OutputTensor>& output, async::Async<MatrixTensor> const& matrix, MatrixNorm kind)
{
  auto selector = select_backend_for<OutputTensor, MatrixTensor>(matrix_norm_op{.kind = kind});
  detail::schedule_matrix_norm_output(std::move(selector), output, matrix, kind);
}

/// \brief Schedule a storage-preserving dense matrix norm through an explicit selector.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>> &&
           requires(MatrixTensor const& matrix) { uni20::norm(matrix); }
[[nodiscard]] auto matrix_norm(BackendSelector selector, async::Async<MatrixTensor> const& matrix, MatrixNorm kind)
{
  return detail::schedule_matrix_norm(std::move(selector), matrix, kind);
}

/// \brief Schedule a storage-preserving dense matrix norm using static Tensor policy.
template <uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>> &&
           requires(MatrixTensor const& matrix) { uni20::norm(matrix); }
[[nodiscard]] auto matrix_norm(async::Async<MatrixTensor> const& matrix, MatrixNorm kind)
{
  auto selector = select_backend_for<MatrixTensor>(matrix_norm_op{.kind = kind});
  return detail::schedule_matrix_norm(std::move(selector), matrix, kind);
}

/// \brief Schedule a dense matrix norm returning a host C++ scalar through an explicit selector.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto matrix_norm_host(BackendSelector selector, async::Async<MatrixTensor> const& matrix, MatrixNorm kind)
{
  return detail::schedule_matrix_norm_host(std::move(selector), matrix, kind);
}

/// \brief Schedule a dense matrix norm returning a host C++ scalar using static Tensor policy.
template <uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto matrix_norm_host(async::Async<MatrixTensor> const& matrix, MatrixNorm kind)
{
  auto selector = select_backend_for<MatrixTensor>(matrix_norm_op{.kind = kind});
  return detail::schedule_matrix_norm_host(std::move(selector), matrix, kind);
}

} // namespace uni20::linalg
