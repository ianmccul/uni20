#pragma once

/**
 * \file nonsymmetric_eigen.hpp
 * \ingroup linalg
 * \brief Fixed-output dense nonsymmetric eigensystem operation.
 */

#include <uni20/linalg/backends/lapack/nonsymmetric_eigen.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <span>
#include <utility>

namespace uni20::linalg
{

/// \brief Compute a nonsymmetric eigensystem through an explicit selector.
template <class BackendSelector, uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> RightEigenvectorTensor>
void nonsymmetric_eigen(BackendSelector&& selector, MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues,
                        RightEigenvectorTensor&& right_eigenvectors, bool compute_right_vectors)
{
  dispatch_kernel(std::forward<BackendSelector>(selector),
                  nonsymmetric_eigen_op{.compute_right_vectors = compute_right_vectors}, matrix_work.mdspan(),
                  eigenvalues, right_eigenvectors.mdspan());
}

/// \brief Compute a nonsymmetric eigensystem using tensor storage policy.
template <uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> RightEigenvectorTensor>
void nonsymmetric_eigen(MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues,
                        RightEigenvectorTensor&& right_eigenvectors, bool compute_right_vectors)
{
  auto operation = nonsymmetric_eigen_op{.compute_right_vectors = compute_right_vectors};
  auto selector = select_backend(operation, matrix_work, right_eigenvectors);
  nonsymmetric_eigen(selector, std::forward<MatrixTensor>(matrix_work), eigenvalues,
                     std::forward<RightEigenvectorTensor>(right_eigenvectors), compute_right_vectors);
}

} // namespace uni20::linalg
