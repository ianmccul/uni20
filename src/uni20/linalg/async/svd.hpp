#pragma once

/**
 * \file svd.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for exact dense singular value decompositions.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/ops/svd.hpp>

#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class MatrixTensor>
using preserving_singular_values_result_t =
    std::remove_cvref_t<decltype(uni20::linalg::singular_values(std::declval<MatrixTensor const&>()))>;

template <class MatrixTensor>
using consuming_singular_values_result_t =
    std::remove_cvref_t<decltype(uni20::linalg::singular_values(std::declval<MatrixTensor&&>()))>;

template <class MatrixTensor>
using preserving_svd_left_result_t = std::remove_cvref_t<decltype(uni20::linalg::svd_left(
    std::declval<MatrixTensor const&>(), SvdVectorExtent::Reduced))>;

template <class MatrixTensor>
using consuming_svd_left_result_t =
    std::remove_cvref_t<decltype(uni20::linalg::svd_left(std::declval<MatrixTensor&&>(), SvdVectorExtent::Reduced))>;

template <class MatrixTensor>
using preserving_svd_right_result_t = std::remove_cvref_t<decltype(uni20::linalg::svd_right(
    std::declval<MatrixTensor const&>(), SvdVectorExtent::Reduced))>;

template <class MatrixTensor>
using consuming_svd_right_result_t =
    std::remove_cvref_t<decltype(uni20::linalg::svd_right(std::declval<MatrixTensor&&>(), SvdVectorExtent::Reduced))>;

template <class MatrixTensor>
using preserving_svd_result_t =
    std::remove_cvref_t<decltype(uni20::linalg::svd(std::declval<MatrixTensor const&>(), SvdOptions{}))>;

template <class MatrixTensor>
using consuming_svd_result_t =
    std::remove_cvref_t<decltype(uni20::linalg::svd(std::declval<MatrixTensor&&>(), SvdOptions{}))>;

template <class Result>
using async_svd_left_result_t = SvdLeftResult<async::Async<typename Result::left_singular_vector_tensor_type>,
                                              async::Async<typename Result::singular_value_tensor_type>>;

template <class Result>
using async_svd_right_result_t =
    SvdRightResult<async::Async<typename Result::singular_value_tensor_type>,
                   async::Async<typename Result::right_singular_vector_adjoint_tensor_type>>;

template <class Result>
using async_svd_result_t = SvdResult<async::Async<typename Result::left_singular_vector_tensor_type>,
                                     async::Async<typename Result::singular_value_tensor_type>,
                                     async::Async<typename Result::right_singular_vector_adjoint_tensor_type>>;

template <class Result, class MatrixTensor>
[[nodiscard]] constexpr auto select_async_consuming_singular_values_backend()
{
  if constexpr (can_transfer_svd_storage<MatrixTensor>())
  {
    return select_backend_for<Result, std::remove_cvref_t<MatrixTensor>>(singular_values_op{});
  }
  else
  {
    using work_type = svd_matrix_tensor_t<MatrixTensor>;
    return select_singular_values_backend<work_type>();
  }
}

template <class Result, class MatrixTensor>
[[nodiscard]] constexpr auto select_async_consuming_svd_left_backend(SvdVectorExtent extent)
{
  if constexpr (can_transfer_svd_storage<MatrixTensor>())
  {
    return select_backend_for<typename Result::singular_value_tensor_type,
                              typename Result::left_singular_vector_tensor_type, std::remove_cvref_t<MatrixTensor>>(
        svd_left_op{.left = extent});
  }
  else
  {
    using work_type = svd_matrix_tensor_t<MatrixTensor>;
    return select_svd_left_backend<work_type>(extent);
  }
}

template <class Result, class MatrixTensor>
[[nodiscard]] constexpr auto select_async_consuming_svd_right_backend(SvdVectorExtent extent)
{
  if constexpr (can_transfer_svd_storage<MatrixTensor>())
  {
    return select_backend_for<typename Result::singular_value_tensor_type,
                              typename Result::right_singular_vector_adjoint_tensor_type,
                              std::remove_cvref_t<MatrixTensor>>(svd_right_op{.right = extent});
  }
  else
  {
    using work_type = svd_matrix_tensor_t<MatrixTensor>;
    return select_svd_right_backend<work_type>(extent);
  }
}

template <class Result, class MatrixTensor>
[[nodiscard]] constexpr auto select_async_consuming_svd_backend(SvdOptions options)
{
  if constexpr (can_transfer_svd_storage<MatrixTensor>())
  {
    return select_backend_for<
        typename Result::singular_value_tensor_type, typename Result::left_singular_vector_tensor_type,
        typename Result::right_singular_vector_adjoint_tensor_type, std::remove_cvref_t<MatrixTensor>>(
        svd_op{.left = options.left, .right = options.right});
  }
  else
  {
    using work_type = svd_matrix_tensor_t<MatrixTensor>;
    return select_svd_backend<work_type>(options);
  }
}

template <class BackendSelector, class SingularValueTensor, uni20::RankedImmediateTensorView<2> MatrixTensor>
async::AsyncTask co_preserving_singular_values(BackendSelector const selector,
                                               async::WriteBuffer<SingularValueTensor> singular_values,
                                               async::ReadBuffer<MatrixTensor> matrix)
{
  auto singular_value_storage_awaiter = singular_values.storage();
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto const& matrix_value = co_await matrix;
  singular_value_storage.emplace(uni20::linalg::singular_values(selector, matrix_value));
  co_return;
}

template <class BackendSelector, class SingularValueTensor, uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
async::AsyncTask co_consuming_singular_values(BackendSelector const selector,
                                              async::WriteBuffer<SingularValueTensor> singular_values,
                                              async::WriteBuffer<MatrixTensor> matrix)
{
  auto singular_value_storage_awaiter = singular_values.storage();
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto matrix_value = co_await matrix.take();
  singular_value_storage.emplace(uni20::linalg::singular_values(selector, std::move(matrix_value)));
  co_return;
}

template <class BackendSelector, class LeftTensor, class SingularValueTensor,
          uni20::RankedImmediateTensorView<2> MatrixTensor>
async::AsyncTask co_preserving_svd_left(BackendSelector const selector,
                                        async::WriteBuffer<LeftTensor> left_singular_vectors,
                                        async::WriteBuffer<SingularValueTensor> singular_values,
                                        async::ReadBuffer<MatrixTensor> matrix, SvdVectorExtent const extent)
{
  auto left_storage_awaiter = left_singular_vectors.storage();
  auto singular_value_storage_awaiter = singular_values.storage();
  auto& left_storage = co_await left_storage_awaiter;
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto const& matrix_value = co_await matrix;

  auto result = uni20::linalg::svd_left(selector, matrix_value, extent);
  left_storage.emplace(std::move(result.left_singular_vectors));
  singular_value_storage.emplace(std::move(result.singular_values));
  co_return;
}

template <class BackendSelector, class LeftTensor, class SingularValueTensor,
          uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
async::AsyncTask co_consuming_svd_left(BackendSelector const selector,
                                       async::WriteBuffer<LeftTensor> left_singular_vectors,
                                       async::WriteBuffer<SingularValueTensor> singular_values,
                                       async::WriteBuffer<MatrixTensor> matrix, SvdVectorExtent const extent)
{
  auto left_storage_awaiter = left_singular_vectors.storage();
  auto singular_value_storage_awaiter = singular_values.storage();
  auto& left_storage = co_await left_storage_awaiter;
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto matrix_value = co_await matrix.take();

  auto result = uni20::linalg::svd_left(selector, std::move(matrix_value), extent);
  left_storage.emplace(std::move(result.left_singular_vectors));
  singular_value_storage.emplace(std::move(result.singular_values));
  co_return;
}

template <class BackendSelector, class SingularValueTensor, class RightAdjointTensor,
          uni20::RankedImmediateTensorView<2> MatrixTensor>
async::AsyncTask co_preserving_svd_right(BackendSelector const selector,
                                         async::WriteBuffer<SingularValueTensor> singular_values,
                                         async::WriteBuffer<RightAdjointTensor> right_singular_vectors_adjoint,
                                         async::ReadBuffer<MatrixTensor> matrix, SvdVectorExtent const extent)
{
  auto singular_value_storage_awaiter = singular_values.storage();
  auto right_storage_awaiter = right_singular_vectors_adjoint.storage();
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto& right_storage = co_await right_storage_awaiter;
  auto const& matrix_value = co_await matrix;

  auto result = uni20::linalg::svd_right(selector, matrix_value, extent);
  singular_value_storage.emplace(std::move(result.singular_values));
  right_storage.emplace(std::move(result.right_singular_vectors_adjoint));
  co_return;
}

template <class BackendSelector, class SingularValueTensor, class RightAdjointTensor,
          uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
async::AsyncTask co_consuming_svd_right(BackendSelector const selector,
                                        async::WriteBuffer<SingularValueTensor> singular_values,
                                        async::WriteBuffer<RightAdjointTensor> right_singular_vectors_adjoint,
                                        async::WriteBuffer<MatrixTensor> matrix, SvdVectorExtent const extent)
{
  auto singular_value_storage_awaiter = singular_values.storage();
  auto right_storage_awaiter = right_singular_vectors_adjoint.storage();
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto& right_storage = co_await right_storage_awaiter;
  auto matrix_value = co_await matrix.take();

  auto result = uni20::linalg::svd_right(selector, std::move(matrix_value), extent);
  singular_value_storage.emplace(std::move(result.singular_values));
  right_storage.emplace(std::move(result.right_singular_vectors_adjoint));
  co_return;
}

template <class BackendSelector, class LeftTensor, class SingularValueTensor, class RightAdjointTensor,
          uni20::RankedImmediateTensorView<2> MatrixTensor>
async::AsyncTask co_preserving_svd(BackendSelector const selector, async::WriteBuffer<LeftTensor> left_singular_vectors,
                                   async::WriteBuffer<SingularValueTensor> singular_values,
                                   async::WriteBuffer<RightAdjointTensor> right_singular_vectors_adjoint,
                                   async::ReadBuffer<MatrixTensor> matrix, SvdOptions const options)
{
  auto left_storage_awaiter = left_singular_vectors.storage();
  auto singular_value_storage_awaiter = singular_values.storage();
  auto right_storage_awaiter = right_singular_vectors_adjoint.storage();
  auto& left_storage = co_await left_storage_awaiter;
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto& right_storage = co_await right_storage_awaiter;
  auto const& matrix_value = co_await matrix;

  auto result = uni20::linalg::svd(selector, matrix_value, options);
  left_storage.emplace(std::move(result.left_singular_vectors));
  singular_value_storage.emplace(std::move(result.singular_values));
  right_storage.emplace(std::move(result.right_singular_vectors_adjoint));
  co_return;
}

template <class BackendSelector, class LeftTensor, class SingularValueTensor, class RightAdjointTensor,
          uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
async::AsyncTask co_consuming_svd(BackendSelector const selector, async::WriteBuffer<LeftTensor> left_singular_vectors,
                                  async::WriteBuffer<SingularValueTensor> singular_values,
                                  async::WriteBuffer<RightAdjointTensor> right_singular_vectors_adjoint,
                                  async::WriteBuffer<MatrixTensor> matrix, SvdOptions const options)
{
  auto left_storage_awaiter = left_singular_vectors.storage();
  auto singular_value_storage_awaiter = singular_values.storage();
  auto right_storage_awaiter = right_singular_vectors_adjoint.storage();
  auto& left_storage = co_await left_storage_awaiter;
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto& right_storage = co_await right_storage_awaiter;
  // Retain the consumed writer gate until every output commits, so a failure
  // reaches the input and all independent result epochs.
  auto matrix_value = co_await matrix.take();

  auto result = uni20::linalg::svd(selector, std::move(matrix_value), options);
  left_storage.emplace(std::move(result.left_singular_vectors));
  singular_value_storage.emplace(std::move(result.singular_values));
  right_storage.emplace(std::move(result.right_singular_vectors_adjoint));
  co_return;
}

template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_preserving_singular_values(BackendSelector selector,
                                                       async::Async<MatrixTensor> const& matrix)
{
  async::Async<preserving_singular_values_result_t<MatrixTensor>> output;
  output.debug_name("singular_values.values");
  auto task = co_preserving_singular_values(std::move(selector), output.write(), matrix.read());
  task.debug_name("singular_values");
  async::schedule(std::move(task));
  return output;
}

template <class BackendSelector, uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
[[nodiscard]] auto schedule_consuming_singular_values(BackendSelector selector, async::Async<MatrixTensor>& matrix)
{
  async::Async<consuming_singular_values_result_t<MatrixTensor>> output;
  output.debug_name("singular_values.values");
  auto task = co_consuming_singular_values(std::move(selector), output.write(), matrix.write());
  task.debug_name("singular_values");
  async::schedule(std::move(task));
  return output;
}

template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_preserving_svd_left(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                                SvdVectorExtent extent)
{
  using result_type = preserving_svd_left_result_t<MatrixTensor>;
  async_svd_left_result_t<result_type> outputs;
  outputs.left_singular_vectors.debug_name("svd_left.left_singular_vectors");
  outputs.singular_values.debug_name("svd_left.singular_values");
  auto task = co_preserving_svd_left(std::move(selector), outputs.left_singular_vectors.write(),
                                     outputs.singular_values.write(), matrix.read(), extent);
  task.debug_name("svd_left");
  async::schedule(std::move(task));
  return outputs;
}

template <class BackendSelector, uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
[[nodiscard]] auto schedule_consuming_svd_left(BackendSelector selector, async::Async<MatrixTensor>& matrix,
                                               SvdVectorExtent extent)
{
  using result_type = consuming_svd_left_result_t<MatrixTensor>;
  async_svd_left_result_t<result_type> outputs;
  outputs.left_singular_vectors.debug_name("svd_left.left_singular_vectors");
  outputs.singular_values.debug_name("svd_left.singular_values");
  auto task = co_consuming_svd_left(std::move(selector), outputs.left_singular_vectors.write(),
                                    outputs.singular_values.write(), matrix.write(), extent);
  task.debug_name("svd_left");
  async::schedule(std::move(task));
  return outputs;
}

template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_preserving_svd_right(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                                 SvdVectorExtent extent)
{
  using result_type = preserving_svd_right_result_t<MatrixTensor>;
  async_svd_right_result_t<result_type> outputs;
  outputs.singular_values.debug_name("svd_right.singular_values");
  outputs.right_singular_vectors_adjoint.debug_name("svd_right.right_singular_vectors_adjoint");
  auto task = co_preserving_svd_right(std::move(selector), outputs.singular_values.write(),
                                      outputs.right_singular_vectors_adjoint.write(), matrix.read(), extent);
  task.debug_name("svd_right");
  async::schedule(std::move(task));
  return outputs;
}

template <class BackendSelector, uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
[[nodiscard]] auto schedule_consuming_svd_right(BackendSelector selector, async::Async<MatrixTensor>& matrix,
                                                SvdVectorExtent extent)
{
  using result_type = consuming_svd_right_result_t<MatrixTensor>;
  async_svd_right_result_t<result_type> outputs;
  outputs.singular_values.debug_name("svd_right.singular_values");
  outputs.right_singular_vectors_adjoint.debug_name("svd_right.right_singular_vectors_adjoint");
  auto task = co_consuming_svd_right(std::move(selector), outputs.singular_values.write(),
                                     outputs.right_singular_vectors_adjoint.write(), matrix.write(), extent);
  task.debug_name("svd_right");
  async::schedule(std::move(task));
  return outputs;
}

template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_preserving_svd(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                           SvdOptions options)
{
  using result_type = preserving_svd_result_t<MatrixTensor>;
  async_svd_result_t<result_type> outputs;
  outputs.left_singular_vectors.debug_name("svd.left_singular_vectors");
  outputs.singular_values.debug_name("svd.singular_values");
  outputs.right_singular_vectors_adjoint.debug_name("svd.right_singular_vectors_adjoint");

  auto task =
      co_preserving_svd(std::move(selector), outputs.left_singular_vectors.write(), outputs.singular_values.write(),
                        outputs.right_singular_vectors_adjoint.write(), matrix.read(), options);
  task.debug_name("svd");
  async::schedule(std::move(task));
  return outputs;
}

template <class BackendSelector, uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
[[nodiscard]] auto schedule_consuming_svd(BackendSelector selector, async::Async<MatrixTensor>& matrix,
                                          SvdOptions options)
{
  using result_type = consuming_svd_result_t<MatrixTensor>;
  async_svd_result_t<result_type> outputs;
  outputs.left_singular_vectors.debug_name("svd.left_singular_vectors");
  outputs.singular_values.debug_name("svd.singular_values");
  outputs.right_singular_vectors_adjoint.debug_name("svd.right_singular_vectors_adjoint");

  auto task =
      co_consuming_svd(std::move(selector), outputs.left_singular_vectors.write(), outputs.singular_values.write(),
                       outputs.right_singular_vectors_adjoint.write(), matrix.write(), options);
  task.debug_name("svd");
  async::schedule(std::move(task));
  return outputs;
}
} // namespace detail

/// \brief Schedule preserving exact singular values through an explicit selector.
template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto singular_values(BackendSelector selector, async::Async<MatrixTensor> const& matrix)
{
  return detail::schedule_preserving_singular_values(std::move(selector), matrix);
}

/// \brief Schedule preserving exact singular values using the static Tensor selector.
template <uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto singular_values(async::Async<MatrixTensor> const& matrix)
{
  using work_type = detail::svd_matrix_tensor_t<MatrixTensor>;
  auto selector = detail::select_singular_values_backend<work_type>();
  return detail::schedule_preserving_singular_values(std::move(selector), matrix);
}

/// \brief Schedule consuming exact singular values through an explicit selector.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto singular_values(BackendSelector selector, async::Async<MatrixTensor>&& matrix)
{
  return detail::schedule_consuming_singular_values(std::move(selector), matrix);
}

/// \brief Schedule consuming exact singular values using the static Tensor selector.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto singular_values(async::Async<MatrixTensor>&& matrix)
{
  using result_type = detail::consuming_singular_values_result_t<MatrixTensor>;
  auto selector = detail::select_async_consuming_singular_values_backend<result_type, MatrixTensor>();
  return detail::schedule_consuming_singular_values(std::move(selector), matrix);
}

/// \brief Schedule preserving left singular vectors through an explicit selector.
template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto svd_left(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                            SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  return detail::schedule_preserving_svd_left(std::move(selector), matrix, extent);
}

/// \brief Schedule preserving left singular vectors using the static Tensor selector.
template <uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto svd_left(async::Async<MatrixTensor> const& matrix, SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  using work_type = detail::svd_matrix_tensor_t<MatrixTensor>;
  auto selector = detail::select_svd_left_backend<work_type>(extent);
  return detail::schedule_preserving_svd_left(std::move(selector), matrix, extent);
}

/// \brief Schedule consuming left singular vectors through an explicit selector.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto svd_left(BackendSelector selector, async::Async<MatrixTensor>&& matrix,
                            SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  return detail::schedule_consuming_svd_left(std::move(selector), matrix, extent);
}

/// \brief Schedule consuming left singular vectors using the static Tensor selector.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto svd_left(async::Async<MatrixTensor>&& matrix, SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  using result_type = detail::consuming_svd_left_result_t<MatrixTensor>;
  auto selector = detail::select_async_consuming_svd_left_backend<result_type, MatrixTensor>(extent);
  return detail::schedule_consuming_svd_left(std::move(selector), matrix, extent);
}

/// \brief Schedule preserving right singular vectors through an explicit selector.
template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto svd_right(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                             SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  return detail::schedule_preserving_svd_right(std::move(selector), matrix, extent);
}

/// \brief Schedule preserving right singular vectors using the static Tensor selector.
template <uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto svd_right(async::Async<MatrixTensor> const& matrix,
                             SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  using work_type = detail::svd_matrix_tensor_t<MatrixTensor>;
  auto selector = detail::select_svd_right_backend<work_type>(extent);
  return detail::schedule_preserving_svd_right(std::move(selector), matrix, extent);
}

/// \brief Schedule consuming right singular vectors through an explicit selector.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto svd_right(BackendSelector selector, async::Async<MatrixTensor>&& matrix,
                             SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  return detail::schedule_consuming_svd_right(std::move(selector), matrix, extent);
}

/// \brief Schedule consuming right singular vectors using the static Tensor selector.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto svd_right(async::Async<MatrixTensor>&& matrix, SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  using result_type = detail::consuming_svd_right_result_t<MatrixTensor>;
  auto selector = detail::select_async_consuming_svd_right_backend<result_type, MatrixTensor>(extent);
  return detail::schedule_consuming_svd_right(std::move(selector), matrix, extent);
}

/// \brief Schedule a preserving exact SVD through an explicit selector.
/// \details The returned `U`, `s`, and `Vh` handles have independent output
///          epochs and may be passed directly to subsequent async operations.
///          All three receive any unhandled task failure.
template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto svd(BackendSelector selector, async::Async<MatrixTensor> const& matrix, SvdOptions options = {})
{
  return detail::schedule_preserving_svd(std::move(selector), matrix, options);
}

/// \brief Schedule a preserving exact SVD using the static Tensor selector.
/// \details Structured binding yields three independent `Async<Tensor>`
///          values: `auto [u, s, vh] = svd(matrix);`.
template <uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto svd(async::Async<MatrixTensor> const& matrix, SvdOptions options = {})
{
  using work_type = detail::svd_matrix_tensor_t<MatrixTensor>;
  auto selector = detail::select_svd_backend<work_type>(options);
  return detail::schedule_preserving_svd(std::move(selector), matrix, options);
}

/// \brief Schedule a consuming exact SVD through an explicit selector.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto svd(BackendSelector selector, async::Async<MatrixTensor>&& matrix, SvdOptions options = {})
{
  return detail::schedule_consuming_svd(std::move(selector), matrix, options);
}

/// \brief Schedule a consuming exact SVD using the static Tensor selector.
/// \details The stored owning matrix is taken from its epoch. Compatible
///          reduced factors may adopt that allocation, and the input timeline
///          no longer contains a readable matrix on success.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto svd(async::Async<MatrixTensor>&& matrix, SvdOptions options = {})
{
  using result_type = detail::consuming_svd_result_t<MatrixTensor>;
  auto selector = detail::select_async_consuming_svd_backend<result_type, MatrixTensor>(options);
  return detail::schedule_consuming_svd(std::move(selector), matrix, options);
}

} // namespace uni20::linalg
