#pragma once

/**
 * \file matrix_set.hpp
 * \ingroup linalg
 * \brief Async Tensor wrapper for structured matrix initialization.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async/concepts.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/linalg/async/dispatch.hpp>
#include <uni20/linalg/ops/matrix_set.hpp>

#include <tuple>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class BackendSelector, uni20::AsyncTensorOutput MatrixTensor, class DiagonalAwaiter, class OffDiagonalAwaiter>
async::AsyncTask co_set_matrix(BackendSelector const selector, async::WriteBuffer<MatrixTensor> matrix,
                               DiagonalAwaiter diagonal, OffDiagonalAwaiter off_diagonal, MatrixRegion const region)
{
  auto matrix_awaiter = uni20::detail::mutable_async_tensor_awaiter(matrix);
  auto awaited = co_await async::all(matrix_awaiter, diagonal, off_diagonal);
  decltype(auto) matrix_value = uni20::detail::mutable_async_tensor_value<MatrixTensor>(std::get<0>(awaited));
  using scalar_type = uni20::tensor_element_t<MatrixTensor>;
  scalar_type const diagonal_value = static_cast<scalar_type>(std::get<1>(awaited));
  scalar_type const off_diagonal_value = static_cast<scalar_type>(std::get<2>(awaited));

  auto descriptor = uni20::mdspec_of(matrix_value);
  co_await co_dispatch_kernel(selector, matrix_set_op{.region = region}, descriptor, diagonal_value,
                              off_diagonal_value);
  co_return;
}

template <class BackendSelector, uni20::AsyncTensorOutput MatrixTensor, class DiagonalAwaiter, class OffDiagonalAwaiter>
void schedule_async_set_matrix(BackendSelector selector, async::Async<MatrixTensor>& matrix, DiagonalAwaiter diagonal,
                               OffDiagonalAwaiter off_diagonal, MatrixRegion region)
{
  auto task = co_set_matrix(std::move(selector), matrix.write(), std::move(diagonal), std::move(off_diagonal), region);
  task.debug_name("set_matrix");
  async::schedule(std::move(task));
}
} // namespace detail

/// \brief Schedule structured matrix initialization through an explicit selector.
/// \details The matrix is asynchronous; diagonal values may be immediate or Async scalars.
template <KernelBackendSelector BackendSelector, uni20::AsyncTensorOutput MatrixTensor, class Diagonal,
          class OffDiagonal>
  requires uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           AsyncOperationScalar<Diagonal, uni20::tensor_element_t<MatrixTensor>> &&
           AsyncOperationScalar<OffDiagonal, uni20::tensor_element_t<MatrixTensor>>
void set_matrix(BackendSelector selector, async::Async<MatrixTensor>& matrix, Diagonal&& diagonal,
                OffDiagonal&& off_diagonal, MatrixRegion region = MatrixRegion::All)
{
  detail::schedule_async_set_matrix(std::move(selector), matrix, async::read(std::forward<Diagonal>(diagonal)),
                                    async::read(std::forward<OffDiagonal>(off_diagonal)), region);
}

/// \brief Schedule structured matrix initialization using static Tensor policy.
template <uni20::AsyncTensorOutput MatrixTensor, class Diagonal, class OffDiagonal>
  requires uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           AsyncOperationScalar<Diagonal, uni20::tensor_element_t<MatrixTensor>> &&
           AsyncOperationScalar<OffDiagonal, uni20::tensor_element_t<MatrixTensor>>
void set_matrix(async::Async<MatrixTensor>& matrix, Diagonal&& diagonal, OffDiagonal&& off_diagonal,
                MatrixRegion region = MatrixRegion::All)
{
  auto selector = select_backend_for<MatrixTensor>(matrix_set_op{.region = region});
  set_matrix(std::move(selector), matrix, std::forward<Diagonal>(diagonal), std::forward<OffDiagonal>(off_diagonal),
             region);
}

} // namespace uni20::linalg
