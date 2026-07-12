#pragma once

/**
 * \file self_adjoint_eigh.hpp
 * \ingroup linalg
 * \brief In-place and value-producing dense self-adjoint eigensystems.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backends/lapack/self_adjoint_eigh.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/tensor.hpp>

#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
using self_adjoint_eigenvalue_extents = stdex::dextents<uni20::index_type, 1>;

template <uni20::MutableRankedTensorView<1> EigenvalueTensor, uni20::MutableRankedTensorView<2> MatrixTensor>
[[nodiscard]] self_adjoint_eigh_op prepare_self_adjoint_eigh(EigenvalueTensor& eigenvalues, MatrixTensor& matrix_work,
                                                             SelfAdjointEighOptions options)
{
  ERROR_IF(matrix_work.extent(0) != matrix_work.extent(1), "self-adjoint eigensystem requires a square matrix");
  uni20::ensure_shape(eigenvalues,
                      self_adjoint_eigenvalue_extents{static_cast<uni20::index_type>(matrix_work.extent(0))});
  return {.compute_vectors = options.compute_vectors, .triangle = options.triangle};
}

template <class BackendSelector, uni20::MutableRankedTensorView<1> EigenvalueTensor,
          uni20::MutableRankedTensorView<2> MatrixTensor>
void dispatch_self_adjoint_eigh(BackendSelector&& selector, self_adjoint_eigh_op operation,
                                EigenvalueTensor& eigenvalues, MatrixTensor& matrix_work)
{
  auto eigenvalue_span = eigenvalues.mdspan();
  auto matrix_span = matrix_work.mdspan();
  dispatch_kernel(std::forward<BackendSelector>(selector), operation, eigenvalue_span, matrix_span);
}
} // namespace detail

/// \brief Compute an in-place self-adjoint eigensystem through an explicit selector.
/// \details The selected triangle of `matrix_work` is read. LAPACK destroys the
///          matrix input; when vectors are requested, its columns contain the
///          normalized eigenvectors on return. `eigenvalues` is resized when
///          its output policy permits.
/// \pre Eigenvalue and matrix storage do not overlap.
template <class BackendSelector, uni20::MutableRankedTensorView<1> EigenvalueTensor,
          uni20::MutableRankedTensorView<2> MatrixTensor>
void self_adjoint_eigh(BackendSelector&& selector, EigenvalueTensor&& eigenvalues, MatrixTensor&& matrix_work,
                       SelfAdjointEighOptions options = {})
{
  auto operation = detail::prepare_self_adjoint_eigh(eigenvalues, matrix_work, options);
  detail::dispatch_self_adjoint_eigh(std::forward<BackendSelector>(selector), operation, eigenvalues, matrix_work);
}

/// \brief Compute an in-place self-adjoint eigensystem using tensor storage policy.
template <uni20::MutableRankedTensorView<1> EigenvalueTensor, uni20::MutableRankedTensorView<2> MatrixTensor>
void self_adjoint_eigh(EigenvalueTensor&& eigenvalues, MatrixTensor&& matrix_work, SelfAdjointEighOptions options = {})
{
  auto operation = detail::prepare_self_adjoint_eigh(eigenvalues, matrix_work, options);
  auto selector = select_backend(operation, eigenvalues, matrix_work);
  detail::dispatch_self_adjoint_eigh(selector, operation, eigenvalues, matrix_work);
}

/// \brief Owning eigenvalues and eigenvectors returned by `eigh`.
template <class EigenvalueTensor, class EigenvectorTensor> struct SelfAdjointEighResult
{
    EigenvalueTensor eigenvalues;
    EigenvectorTensor eigenvectors;
};

/// \brief Preserve a matrix and return its complete self-adjoint eigensystem.
/// \details This explicit value operation materializes a column-major work
///          matrix through `copy_op`, then calls the destructive in-place
///          eigensolver with eigenvectors enabled.
template <uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto eigh(MatrixTensor const& matrix, MatrixTriangle triangle = MatrixTriangle::Upper)
{
  ERROR_IF(matrix.extent(0) != matrix.extent(1), "self-adjoint eigensystem requires a square matrix");
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  using real_type = uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>;
  uni20::Tensor<real_type, 1> eigenvalues;
  self_adjoint_eigh(eigenvalues, matrix_work, SelfAdjointEighOptions{.compute_vectors = true, .triangle = triangle});
  return SelfAdjointEighResult<decltype(eigenvalues), decltype(matrix_work)>{.eigenvalues = std::move(eigenvalues),
                                                                             .eigenvectors = std::move(matrix_work)};
}

} // namespace uni20::linalg
