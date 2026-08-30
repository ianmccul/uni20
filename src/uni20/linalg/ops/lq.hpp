#pragma once

/**
 * \file lq.hpp
 * \ingroup linalg
 * \brief Destructive-workspace and value-producing reduced real LQ factorizations.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/backends/lapack/lq.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/tensor.hpp>

#include <algorithm>
#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
using lq_matrix_extents = stdex::dextents<uni20::index_type, 2>;

template <class LTensor, class QTensor, class MatrixTensor>
concept CompatibleLqTensors =
    uni20::MutableRankedTensorView<LTensor, 2> && uni20::MutableRankedTensorView<QTensor, 2> &&
    uni20::MutableRankedTensorView<MatrixTensor, 2> && uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>> &&
    std::same_as<uni20::tensor_element_t<LTensor>, uni20::tensor_element_t<MatrixTensor>> &&
    std::same_as<uni20::tensor_element_t<QTensor>, uni20::tensor_element_t<MatrixTensor>>;

template <uni20::MutableRankedTensorView<2> LTensor, uni20::MutableRankedTensorView<2> QTensor,
          uni20::MutableRankedTensorView<2> MatrixTensor>
void prepare_lq(LTensor& l, QTensor& q, MatrixTensor const& matrix_work)
{
  auto const rows = static_cast<uni20::index_type>(matrix_work.extent(0));
  auto const cols = static_cast<uni20::index_type>(matrix_work.extent(1));
  auto const rank = std::min(rows, cols);
  uni20::prepare_output(l, lq_matrix_extents{rows, rank});
  uni20::prepare_output(q, lq_matrix_extents{rank, cols});
}

template <class BackendSelector, uni20::MutableRankedTensorView<2> LTensor, uni20::MutableRankedTensorView<2> QTensor,
          uni20::MutableRankedTensorView<2> MatrixTensor>
void dispatch_lq(BackendSelector&& selector, LTensor& l, QTensor& q, MatrixTensor& matrix_work)
{
  auto l_descriptor = uni20::mdspec_of(l);
  auto q_descriptor = uni20::mdspec_of(q);
  auto matrix_descriptor = uni20::mdspec_of(matrix_work);
  dispatch_kernel(std::forward<BackendSelector>(selector), lq_op{}, l_descriptor, q_descriptor, matrix_descriptor);
}
} // namespace detail

/// \brief Compute reduced LQ in a destructive input workspace through an explicit selector.
/// \details `matrix_work` is overwritten. For an `m x n` input, `l` is
///          prepared with shape `m x k` and `q` with shape `k x n`, where
///          `k = min(m,n)`.
/// \pre The two outputs and input workspace do not overlap.
template <KernelBackendSelector BackendSelector, class LTensor, class QTensor, class MatrixTensor>
  requires detail::CompatibleLqTensors<LTensor, QTensor, MatrixTensor>
void lq_factorization(BackendSelector&& selector, LTensor&& l, QTensor&& q, MatrixTensor&& matrix_work)
{
  detail::prepare_lq(l, q, matrix_work);
  detail::dispatch_lq(std::forward<BackendSelector>(selector), l, q, matrix_work);
}

/// \brief Compute reduced LQ in a destructive workspace using tensor storage policy.
template <class LTensor, class QTensor, class MatrixTensor>
  requires detail::CompatibleLqTensors<LTensor, QTensor, MatrixTensor>
void lq_factorization(LTensor&& l, QTensor&& q, MatrixTensor&& matrix_work)
{
  detail::prepare_lq(l, q, matrix_work);
  auto selector = select_backend(lq_op{}, l, q, matrix_work);
  detail::dispatch_lq(selector, l, q, matrix_work);
}

/// \brief Owning reduced factors returned by `lq`.
/// \details Aggregate member order supports `auto [l, q] = lq(matrix)`.
template <class LTensor, class QTensor> struct LqResult
{
    using l_tensor_type = LTensor;
    using q_tensor_type = QTensor;

    LTensor l;
    QTensor q;
};

namespace detail
{
template <class MatrixTensor> using lq_matrix_tensor_t = uni20::Tensor<uni20::tensor_element_t<MatrixTensor>, 2>;

template <class MatrixTensor> consteval bool can_use_lq_storage_as_workspace()
{
  return std::same_as<std::remove_cvref_t<MatrixTensor>, lq_matrix_tensor_t<MatrixTensor>>;
}

template <class BackendSelector, uni20::OwningTensor MatrixTensor>
  requires uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto lq_from_work_matrix(BackendSelector&& selector, MatrixTensor matrix_work)
{
  lq_matrix_tensor_t<MatrixTensor> l;
  lq_matrix_tensor_t<MatrixTensor> q;
  lq_factorization(std::forward<BackendSelector>(selector), l, q, matrix_work);
  return LqResult<decltype(l), decltype(q)>{.l = std::move(l), .q = std::move(q)};
}

template <class MatrixTensor> [[nodiscard]] constexpr auto select_lq_backend()
{
  return select_backend_for<lq_matrix_tensor_t<MatrixTensor>, lq_matrix_tensor_t<MatrixTensor>,
                            std::remove_cvref_t<MatrixTensor>>(lq_op{});
}
} // namespace detail

/// \brief Preserve a real matrix and return its reduced LQ factorization through an explicit selector.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto lq(BackendSelector&& selector, MatrixTensor const& matrix)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  return detail::lq_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work));
}

/// \brief Preserve a real matrix and return its reduced LQ factorization.
template <uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto lq(MatrixTensor const& matrix)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  auto selector = detail::select_lq_backend<decltype(matrix_work)>();
  return detail::lq_from_work_matrix(std::move(selector), std::move(matrix_work));
}

/// \brief Consume an owning real matrix and return its reduced LQ factorization through an explicit selector.
/// \details An exact column-major host tensor is moved directly into the
///          destructive workspace. Other owning tensor views are materialized.
template <KernelBackendSelector BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>> && (!std::is_lvalue_reference_v<MatrixTensor>) &&
           (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto lq(BackendSelector&& selector, MatrixTensor&& matrix)
{
  if constexpr (detail::can_use_lq_storage_as_workspace<MatrixTensor>())
  {
    return detail::lq_from_work_matrix(std::forward<BackendSelector>(selector), std::forward<MatrixTensor>(matrix));
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    return detail::lq_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work));
  }
}

/// \brief Consume an owning real matrix and return its reduced LQ factorization.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>> && (!std::is_lvalue_reference_v<MatrixTensor>) &&
           (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto lq(MatrixTensor&& matrix)
{
  using work_type = std::conditional_t<detail::can_use_lq_storage_as_workspace<MatrixTensor>(),
                                       std::remove_cvref_t<MatrixTensor>, detail::lq_matrix_tensor_t<MatrixTensor>>;
  auto selector = detail::select_lq_backend<work_type>();
  return uni20::linalg::lq(std::move(selector), std::forward<MatrixTensor>(matrix));
}

} // namespace uni20::linalg
