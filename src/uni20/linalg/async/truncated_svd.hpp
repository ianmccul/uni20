#pragma once

/**
 * \file truncated_svd.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for truncated dense singular value decompositions.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async/svd.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>

#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class MatrixTensor>
using svd_truncation_policy_t = SvdTruncationPolicy<uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>>;

template <class MatrixTensor>
using preserving_truncated_svd_result_t = std::remove_cvref_t<decltype(uni20::linalg::truncated_svd(
    std::declval<MatrixTensor const&>(), std::declval<svd_truncation_policy_t<MatrixTensor>>()))>;

template <class MatrixTensor>
using consuming_truncated_svd_result_t = std::remove_cvref_t<decltype(uni20::linalg::truncated_svd(
    std::declval<MatrixTensor&&>(), std::declval<svd_truncation_policy_t<MatrixTensor>>()))>;

template <class Result>
using async_truncated_svd_result_t =
    TruncatedSvdResult<async::Async<typename Result::left_singular_vector_tensor_type>,
                       async::Async<typename Result::singular_value_tensor_type>,
                       async::Async<typename Result::right_singular_vector_adjoint_tensor_type>,
                       async::Async<typename Result::truncation_info_type>>;

template <class BackendSelector, class LeftTensor, class SingularValueTensor, class RightAdjointTensor,
          class TruncationInfo, uni20::RankedImmediateTensorView<2> MatrixTensor>
async::AsyncTask
co_preserving_truncated_svd(BackendSelector const selector, async::WriteBuffer<LeftTensor> left_singular_vectors,
                            async::WriteBuffer<SingularValueTensor> singular_values,
                            async::WriteBuffer<RightAdjointTensor> right_singular_vectors_adjoint,
                            async::WriteBuffer<TruncationInfo> truncation, async::ReadBuffer<MatrixTensor> matrix,
                            svd_truncation_policy_t<MatrixTensor> const policy)
{
  auto left_storage_awaiter = left_singular_vectors.storage();
  auto singular_value_storage_awaiter = singular_values.storage();
  auto right_storage_awaiter = right_singular_vectors_adjoint.storage();
  auto truncation_storage_awaiter = truncation.storage();
  auto& left_storage = co_await left_storage_awaiter;
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto& right_storage = co_await right_storage_awaiter;
  auto& truncation_storage = co_await truncation_storage_awaiter;
  auto const& matrix_value = co_await matrix;

  auto result = uni20::linalg::truncated_svd(selector, matrix_value, policy);
  left_storage.emplace(std::move(result.left_singular_vectors));
  singular_value_storage.emplace(std::move(result.singular_values));
  right_storage.emplace(std::move(result.right_singular_vectors_adjoint));
  truncation_storage.emplace(std::move(result.truncation));
  co_return;
}

template <class BackendSelector, class LeftTensor, class SingularValueTensor, class RightAdjointTensor,
          class TruncationInfo, uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
async::AsyncTask
co_consuming_truncated_svd(BackendSelector const selector, async::WriteBuffer<LeftTensor> left_singular_vectors,
                           async::WriteBuffer<SingularValueTensor> singular_values,
                           async::WriteBuffer<RightAdjointTensor> right_singular_vectors_adjoint,
                           async::WriteBuffer<TruncationInfo> truncation, async::WriteBuffer<MatrixTensor> matrix,
                           svd_truncation_policy_t<MatrixTensor> const policy)
{
  auto left_storage_awaiter = left_singular_vectors.storage();
  auto singular_value_storage_awaiter = singular_values.storage();
  auto right_storage_awaiter = right_singular_vectors_adjoint.storage();
  auto truncation_storage_awaiter = truncation.storage();
  auto& left_storage = co_await left_storage_awaiter;
  auto& singular_value_storage = co_await singular_value_storage_awaiter;
  auto& right_storage = co_await right_storage_awaiter;
  auto& truncation_storage = co_await truncation_storage_awaiter;
  auto matrix_value = co_await matrix.take();

  auto result = uni20::linalg::truncated_svd(selector, std::move(matrix_value), policy);
  left_storage.emplace(std::move(result.left_singular_vectors));
  singular_value_storage.emplace(std::move(result.singular_values));
  right_storage.emplace(std::move(result.right_singular_vectors_adjoint));
  truncation_storage.emplace(std::move(result.truncation));
  co_return;
}

template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto schedule_preserving_truncated_svd(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                                     svd_truncation_policy_t<MatrixTensor> policy)
{
  using result_type = preserving_truncated_svd_result_t<MatrixTensor>;
  async_truncated_svd_result_t<result_type> outputs;
  outputs.left_singular_vectors.debug_name("truncated_svd.left_singular_vectors");
  outputs.singular_values.debug_name("truncated_svd.singular_values");
  outputs.right_singular_vectors_adjoint.debug_name("truncated_svd.right_singular_vectors_adjoint");
  outputs.truncation.debug_name("truncated_svd.truncation");

  auto task = co_preserving_truncated_svd(
      std::move(selector), outputs.left_singular_vectors.write(), outputs.singular_values.write(),
      outputs.right_singular_vectors_adjoint.write(), outputs.truncation.write(), matrix.read(), std::move(policy));
  task.debug_name("truncated_svd");
  async::schedule(std::move(task));
  return outputs;
}

template <class BackendSelector, uni20::MutableRankedImmediateTensorView<2> MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor>
[[nodiscard]] auto schedule_consuming_truncated_svd(BackendSelector selector, async::Async<MatrixTensor>& matrix,
                                                    svd_truncation_policy_t<MatrixTensor> policy)
{
  using result_type = consuming_truncated_svd_result_t<MatrixTensor>;
  async_truncated_svd_result_t<result_type> outputs;
  outputs.left_singular_vectors.debug_name("truncated_svd.left_singular_vectors");
  outputs.singular_values.debug_name("truncated_svd.singular_values");
  outputs.right_singular_vectors_adjoint.debug_name("truncated_svd.right_singular_vectors_adjoint");
  outputs.truncation.debug_name("truncated_svd.truncation");

  auto task = co_consuming_truncated_svd(
      std::move(selector), outputs.left_singular_vectors.write(), outputs.singular_values.write(),
      outputs.right_singular_vectors_adjoint.write(), outputs.truncation.write(), matrix.write(), std::move(policy));
  task.debug_name("truncated_svd");
  async::schedule(std::move(task));
  return outputs;
}
} // namespace detail

/// \brief Schedule a preserving truncated SVD through an explicit selector.
template <class BackendSelector, uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto truncated_svd(BackendSelector selector, async::Async<MatrixTensor> const& matrix,
                                 detail::svd_truncation_policy_t<MatrixTensor> policy = {})
{
  return detail::schedule_preserving_truncated_svd(std::move(selector), matrix, std::move(policy));
}

/// \brief Schedule a preserving truncated SVD using the static Tensor selector.
template <uni20::RankedImmediateTensorView<2> MatrixTensor>
[[nodiscard]] auto truncated_svd(async::Async<MatrixTensor> const& matrix,
                                 detail::svd_truncation_policy_t<MatrixTensor> policy = {})
{
  using work_type = detail::svd_matrix_tensor_t<MatrixTensor>;
  auto selector = detail::select_svd_backend<work_type>(SvdOptions{});
  return detail::schedule_preserving_truncated_svd(std::move(selector), matrix, std::move(policy));
}

/// \brief Schedule a consuming truncated SVD through an explicit selector.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto truncated_svd(BackendSelector selector, async::Async<MatrixTensor>&& matrix,
                                 detail::svd_truncation_policy_t<MatrixTensor> policy = {})
{
  return detail::schedule_consuming_truncated_svd(std::move(selector), matrix, std::move(policy));
}

/// \brief Schedule a consuming truncated SVD using the static Tensor selector.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto truncated_svd(async::Async<MatrixTensor>&& matrix,
                                 detail::svd_truncation_policy_t<MatrixTensor> policy = {})
{
  using exact_result_type = detail::consuming_svd_result_t<MatrixTensor>;
  auto selector = detail::select_async_consuming_svd_backend<exact_result_type, MatrixTensor>(SvdOptions{});
  return detail::schedule_consuming_truncated_svd(std::move(selector), matrix, std::move(policy));
}

} // namespace uni20::linalg
