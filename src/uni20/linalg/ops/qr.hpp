#pragma once

/**
 * \file qr.hpp
 * \ingroup linalg
 * \brief Destructive-workspace and value-producing reduced real QR factorizations.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/backends/lapack/qr.hpp>
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
using qr_matrix_extents = stdex::dextents<uni20::index_type, 2>;

template <class QTensor, class RTensor, class MatrixTensor>
concept CompatibleQrTensors =
    uni20::MutableRankedTensorView<QTensor, 2> && uni20::MutableRankedTensorView<RTensor, 2> &&
    uni20::MutableRankedTensorView<MatrixTensor, 2> && uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>> &&
    std::same_as<uni20::tensor_element_t<QTensor>, uni20::tensor_element_t<MatrixTensor>> &&
    std::same_as<uni20::tensor_element_t<RTensor>, uni20::tensor_element_t<MatrixTensor>>;

template <uni20::MutableRankedTensorView<2> QTensor, uni20::MutableRankedTensorView<2> RTensor,
          uni20::MutableRankedTensorView<2> MatrixTensor>
void prepare_qr(QTensor& q, RTensor& r, MatrixTensor const& matrix_work)
{
  auto const rows = static_cast<uni20::index_type>(matrix_work.extent(0));
  auto const cols = static_cast<uni20::index_type>(matrix_work.extent(1));
  auto const rank = std::min(rows, cols);
  uni20::prepare_output(q, qr_matrix_extents{rows, rank});
  uni20::prepare_output(r, qr_matrix_extents{rank, cols});
}

template <class BackendSelector, uni20::MutableRankedTensorView<2> QTensor, uni20::MutableRankedTensorView<2> RTensor,
          uni20::MutableRankedTensorView<2> MatrixTensor>
void dispatch_qr(BackendSelector&& selector, QTensor& q, RTensor& r, MatrixTensor& matrix_work)
{
  auto q_descriptor = uni20::mdspec_of(q);
  auto r_descriptor = uni20::mdspec_of(r);
  auto matrix_descriptor = uni20::mdspec_of(matrix_work);
  dispatch_kernel(std::forward<BackendSelector>(selector), qr_op{}, q_descriptor, r_descriptor, matrix_descriptor);
}
} // namespace detail

/// \brief Compute reduced QR in a destructive input workspace through an explicit selector.
/// \details `matrix_work` is overwritten. For an `m x n` input, `q` is
///          prepared with shape `m x k` and `r` with shape `k x n`, where
///          `k = min(m,n)`.
/// \pre The two outputs and input workspace do not overlap.
template <KernelBackendSelector BackendSelector, class QTensor, class RTensor, class MatrixTensor>
  requires detail::CompatibleQrTensors<QTensor, RTensor, MatrixTensor>
void qr_factorization(BackendSelector&& selector, QTensor&& q, RTensor&& r, MatrixTensor&& matrix_work)
{
  detail::prepare_qr(q, r, matrix_work);
  detail::dispatch_qr(std::forward<BackendSelector>(selector), q, r, matrix_work);
}

/// \brief Compute reduced QR in a destructive workspace using tensor storage policy.
template <class QTensor, class RTensor, class MatrixTensor>
  requires detail::CompatibleQrTensors<QTensor, RTensor, MatrixTensor>
void qr_factorization(QTensor&& q, RTensor&& r, MatrixTensor&& matrix_work)
{
  detail::prepare_qr(q, r, matrix_work);
  auto selector = select_backend(qr_op{}, q, r, matrix_work);
  detail::dispatch_qr(selector, q, r, matrix_work);
}

/// \brief Owning reduced factors returned by `qr`.
/// \details Aggregate member order supports `auto [q, r] = qr(matrix)`.
template <class QTensor, class RTensor> struct QrResult
{
    using q_tensor_type = QTensor;
    using r_tensor_type = RTensor;

    QTensor q;
    RTensor r;
};

namespace detail
{
template <class MatrixTensor> using qr_matrix_tensor_t = uni20::Tensor<uni20::tensor_element_t<MatrixTensor>, 2>;

template <class MatrixTensor> consteval bool can_use_qr_storage_as_workspace()
{
  return std::same_as<std::remove_cvref_t<MatrixTensor>, qr_matrix_tensor_t<MatrixTensor>>;
}

template <class BackendSelector, uni20::OwningTensor MatrixTensor>
  requires uni20::MutableRankedImmediateTensorView<MatrixTensor, 2>
[[nodiscard]] auto qr_from_work_matrix(BackendSelector&& selector, MatrixTensor matrix_work)
{
  qr_matrix_tensor_t<MatrixTensor> q;
  qr_matrix_tensor_t<MatrixTensor> r;
  qr_factorization(std::forward<BackendSelector>(selector), q, r, matrix_work);
  return QrResult<decltype(q), decltype(r)>{.q = std::move(q), .r = std::move(r)};
}

template <class MatrixTensor> [[nodiscard]] constexpr auto select_qr_backend()
{
  return select_backend_for<qr_matrix_tensor_t<MatrixTensor>, qr_matrix_tensor_t<MatrixTensor>,
                            std::remove_cvref_t<MatrixTensor>>(qr_op{});
}
} // namespace detail

/// \brief Preserve a real matrix and return its reduced QR factorization through an explicit selector.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto qr(BackendSelector&& selector, MatrixTensor const& matrix)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  return detail::qr_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work));
}

/// \brief Preserve a real matrix and return its reduced QR factorization.
template <uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto qr(MatrixTensor const& matrix)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  auto selector = detail::select_qr_backend<decltype(matrix_work)>();
  return detail::qr_from_work_matrix(std::move(selector), std::move(matrix_work));
}

/// \brief Consume an owning real matrix and return its reduced QR factorization through an explicit selector.
/// \details An exact column-major host tensor is moved directly into the
///          destructive workspace. Other owning tensor views are materialized.
template <KernelBackendSelector BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>> && (!std::is_lvalue_reference_v<MatrixTensor>) &&
           (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto qr(BackendSelector&& selector, MatrixTensor&& matrix)
{
  if constexpr (detail::can_use_qr_storage_as_workspace<MatrixTensor>())
  {
    return detail::qr_from_work_matrix(std::forward<BackendSelector>(selector), std::forward<MatrixTensor>(matrix));
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    return detail::qr_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work));
  }
}

/// \brief Consume an owning real matrix and return its reduced QR factorization.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           uni20::LapackReal<uni20::tensor_element_t<MatrixTensor>> && (!std::is_lvalue_reference_v<MatrixTensor>) &&
           (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto qr(MatrixTensor&& matrix)
{
  using work_type = std::conditional_t<detail::can_use_qr_storage_as_workspace<MatrixTensor>(),
                                       std::remove_cvref_t<MatrixTensor>, detail::qr_matrix_tensor_t<MatrixTensor>>;
  auto selector = detail::select_qr_backend<work_type>();
  return uni20::linalg::qr(std::move(selector), std::forward<MatrixTensor>(matrix));
}

} // namespace uni20::linalg
