#pragma once

/**
 * \file self_adjoint_eigh.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for complete self-adjoint eigensystems.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>

#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class MatrixTensor>
using preserving_eigh_result_t =
    std::remove_cvref_t<decltype(uni20::linalg::eigh(std::declval<MatrixTensor const&>(), MatrixTriangle::Upper))>;

template <class MatrixTensor>
using consuming_eigh_result_t =
    std::remove_cvref_t<decltype(uni20::linalg::eigh(std::declval<MatrixTensor&&>(), MatrixTriangle::Upper))>;

template <class Result>
using async_eigh_result_t = SelfAdjointEighResult<async::Async<typename Result::eigenvalue_tensor_type>,
                                                  async::Async<typename Result::eigenvector_tensor_type>>;

template <class Result> [[nodiscard]] constexpr auto select_async_eigh_backend(MatrixTriangle triangle)
{
  return select_backend_for<typename Result::eigenvalue_tensor_type, typename Result::eigenvector_tensor_type>(
      self_adjoint_eigh_op{.compute_vectors = true, .triangle = triangle});
}

template <class BackendSelector, class EigenvalueTensor, class EigenvectorTensor,
          uni20::RankedTensorView<2> MatrixTensor>
async::AsyncTask co_preserving_eigh(BackendSelector const selector, async::WriteBuffer<EigenvalueTensor> eigenvalues,
                                    async::WriteBuffer<EigenvectorTensor> eigenvectors,
                                    async::ReadBuffer<MatrixTensor> matrix, MatrixTriangle const triangle)
{
  auto eigenvalue_storage_awaiter = eigenvalues.storage();
  auto eigenvector_storage_awaiter = eigenvectors.storage();
  auto& eigenvalue_storage = co_await eigenvalue_storage_awaiter;
  auto& eigenvector_storage = co_await eigenvector_storage_awaiter;
  auto const& matrix_value = co_await matrix;

  auto result = uni20::linalg::eigh(selector, matrix_value, triangle);
  eigenvalue_storage.emplace(std::move(result.eigenvalues));
  eigenvector_storage.emplace(std::move(result.eigenvectors));
  co_return;
}

template <class BackendSelector, class EigenvalueTensor, class EigenvectorTensor,
          uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
async::AsyncTask co_consuming_eigh(BackendSelector const selector, async::WriteBuffer<EigenvalueTensor> eigenvalues,
                                   async::WriteBuffer<EigenvectorTensor> eigenvectors,
                                   async::WriteBuffer<MatrixTensor> matrix, MatrixTriangle const triangle)
{
  auto eigenvalue_storage_awaiter = eigenvalues.storage();
  auto eigenvector_storage_awaiter = eigenvectors.storage();
  auto& eigenvalue_storage = co_await eigenvalue_storage_awaiter;
  auto& eigenvector_storage = co_await eigenvector_storage_awaiter;
  // Retain the consumed writer gate until both outputs commit, so a failure is
  // published to all three epochs before any dependent reader can resume.
  auto matrix_value = co_await matrix.take();

  auto result = uni20::linalg::eigh(selector, std::move(matrix_value), triangle);
  eigenvalue_storage.emplace(std::move(result.eigenvalues));
  eigenvector_storage.emplace(std::move(result.eigenvectors));
  co_return;
}

template <class BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_preserving_eigh(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                            MatrixTriangle triangle)
{
  using result_type = preserving_eigh_result_t<MatrixTensor>;
  async_eigh_result_t<result_type> outputs;
  outputs.eigenvalues.debug_name("eigh.eigenvalues");
  outputs.eigenvectors.debug_name("eigh.eigenvectors");

  auto task = co_preserving_eigh(std::move(selector), outputs.eigenvalues.write(), outputs.eigenvectors.write(),
                                 matrix.read(), triangle);
  task.debug_name("eigh");
  async::schedule(std::move(task));
  return outputs;
}

template <class BackendSelector, uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
[[nodiscard]] auto schedule_consuming_eigh(BackendSelector selector, async::Async<MatrixTensor>& matrix,
                                           MatrixTriangle triangle)
{
  using result_type = consuming_eigh_result_t<MatrixTensor>;
  async_eigh_result_t<result_type> outputs;
  outputs.eigenvalues.debug_name("eigh.eigenvalues");
  outputs.eigenvectors.debug_name("eigh.eigenvectors");

  auto task = co_consuming_eigh(std::move(selector), outputs.eigenvalues.write(), outputs.eigenvectors.write(),
                                matrix.write(), triangle);
  task.debug_name("eigh");
  async::schedule(std::move(task));
  return outputs;
}
} // namespace detail

/// \brief Schedule a preserving self-adjoint eigensystem through an explicit selector.
/// \details The returned eigenvalue and eigenvector handles have independent
///          output epochs and may be passed directly to subsequent async
///          operations. Both receive any unhandled task failure.
template <class BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto eigh(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                        MatrixTriangle triangle = MatrixTriangle::Upper)
{
  return detail::schedule_preserving_eigh(std::move(selector), matrix, triangle);
}

/// \brief Schedule a preserving self-adjoint eigensystem using the static Tensor selector.
/// \details Structured binding yields two independent `Async<Tensor>` values:
///          `auto [eigenvalues, eigenvectors] = eigh(matrix);`.
template <uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto eigh(async::Async<MatrixTensor> const& matrix, MatrixTriangle triangle = MatrixTriangle::Upper)
{
  using result_type = detail::preserving_eigh_result_t<MatrixTensor>;
  auto selector = detail::select_async_eigh_backend<result_type>(triangle);
  return detail::schedule_preserving_eigh(std::move(selector), matrix, triangle);
}

/// \brief Schedule a consuming self-adjoint eigensystem through an explicit selector.
/// \details The stored owning matrix is taken from its epoch and may transfer
///          its allocation to the eigenvector output. The input writer and both
///          output writers receive any failure that occurs after enrollment.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto eigh(BackendSelector selector, async::Async<MatrixTensor>&& matrix,
                        MatrixTriangle triangle = MatrixTriangle::Upper)
{
  return detail::schedule_consuming_eigh(std::move(selector), matrix, triangle);
}

/// \brief Schedule a consuming self-adjoint eigensystem using the static Tensor selector.
/// \details Passing the async matrix as an rvalue grants permission to remove
///          its stored owning value. On success, that input timeline no longer
///          contains a readable matrix.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto eigh(async::Async<MatrixTensor>&& matrix, MatrixTriangle triangle = MatrixTriangle::Upper)
{
  using result_type = detail::consuming_eigh_result_t<MatrixTensor>;
  auto selector = detail::select_async_eigh_backend<result_type>(triangle);
  return detail::schedule_consuming_eigh(std::move(selector), matrix, triangle);
}

} // namespace uni20::linalg
