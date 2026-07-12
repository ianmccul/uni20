#pragma once

/**
 * \file tridiagonal_eigen.hpp
 * \ingroup linalg
 * \brief Real symmetric tridiagonal eigensystem operation.
 */

#include <uni20/linalg/backends/lapack/tridiagonal_eigen.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <span>
#include <utility>

namespace uni20::linalg
{

/// \brief Compute a tridiagonal eigensystem through an explicit backend selector.
template <class BackendSelector, uni20::LapackReal Scalar, uni20::MutableRankedTensorView<2> EigenvectorTensor>
void symmetric_tridiagonal_eigen(BackendSelector&& selector, std::span<Scalar> diagonal, std::span<Scalar> subdiagonal,
                                 EigenvectorTensor&& eigenvectors, bool compute_vectors)
{
  dispatch_kernel(std::forward<BackendSelector>(selector),
                  symmetric_tridiagonal_eigen_op{.compute_vectors = compute_vectors}, diagonal, subdiagonal,
                  eigenvectors.mdspan());
}

/// \brief Compute a tridiagonal eigensystem using host tensor backend policy.
template <uni20::LapackReal Scalar, uni20::MutableRankedTensorView<2> EigenvectorTensor>
void symmetric_tridiagonal_eigen(std::span<Scalar> diagonal, std::span<Scalar> subdiagonal,
                                 EigenvectorTensor&& eigenvectors, bool compute_vectors)
{
  auto selector = select_backend(symmetric_tridiagonal_eigen_op{.compute_vectors = compute_vectors}, eigenvectors);
  symmetric_tridiagonal_eigen(selector, diagonal, subdiagonal, std::forward<EigenvectorTensor>(eigenvectors),
                              compute_vectors);
}

} // namespace uni20::linalg
