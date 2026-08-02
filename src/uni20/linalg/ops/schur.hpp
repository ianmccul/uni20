#pragma once

/**
 * \file schur.hpp
 * \ingroup linalg
 * \brief Fixed-output dense Schur decomposition and reordering operations.
 */

#include <uni20/linalg/backends/lapack/schur.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <span>
#include <utility>

namespace uni20::linalg
{

/// \brief Compute a Schur decomposition through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void schur(BackendSelector&& selector, MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues,
           SchurVectorTensor&& schur_vectors, bool compute_vectors)
{
  auto matrix_descriptor = uni20::mdspec_of(matrix_work);
  auto vector_descriptor = uni20::mdspec_of(schur_vectors);
  dispatch_kernel(std::forward<BackendSelector>(selector), schur_op{.compute_vectors = compute_vectors},
                  matrix_descriptor, eigenvalues, vector_descriptor);
}

/// \brief Compute a Schur decomposition using tensor storage policy.
template <uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void schur(MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues, SchurVectorTensor&& schur_vectors,
           bool compute_vectors)
{
  auto operation = schur_op{.compute_vectors = compute_vectors};
  auto selector = select_backend(operation, matrix_work, schur_vectors);
  schur(selector, std::forward<MatrixTensor>(matrix_work), eigenvalues, std::forward<SchurVectorTensor>(schur_vectors),
        compute_vectors);
}

/// \brief Compute a real Hessenberg Schur decomposition through an explicit selector.
template <class BackendSelector, uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void hessenberg_schur(BackendSelector&& selector, MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues,
                      SchurVectorTensor&& schur_vectors, bool compute_vectors)
{
  auto matrix_descriptor = uni20::mdspec_of(matrix_work);
  auto vector_descriptor = uni20::mdspec_of(schur_vectors);
  dispatch_kernel(std::forward<BackendSelector>(selector), hessenberg_schur_op{.compute_vectors = compute_vectors},
                  matrix_descriptor, eigenvalues, vector_descriptor);
}

/// \brief Compute a real Hessenberg Schur decomposition using tensor storage policy.
template <uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void hessenberg_schur(MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues, SchurVectorTensor&& schur_vectors,
                      bool compute_vectors)
{
  auto operation = hessenberg_schur_op{.compute_vectors = compute_vectors};
  auto selector = select_backend(operation, matrix_work, schur_vectors);
  hessenberg_schur(selector, std::forward<MatrixTensor>(matrix_work), eigenvalues,
                   std::forward<SchurVectorTensor>(schur_vectors), compute_vectors);
}

/// \brief Move one Schur block or entry through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedTensorView<2> SchurFormTensor,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void reorder_schur(BackendSelector&& selector, SchurFormTensor&& schur_form, SchurVectorTensor&& schur_vectors,
                   std::size_t from, std::size_t to, bool update_vectors)
{
  auto form_descriptor = uni20::mdspec_of(schur_form);
  auto vector_descriptor = uni20::mdspec_of(schur_vectors);
  dispatch_kernel(std::forward<BackendSelector>(selector),
                  schur_reorder_op{.from = from, .to = to, .update_vectors = update_vectors}, form_descriptor,
                  vector_descriptor);
}

/// \brief Move one Schur block or entry using tensor storage policy.
template <uni20::MutableRankedTensorView<2> SchurFormTensor, uni20::MutableRankedTensorView<2> SchurVectorTensor>
void reorder_schur(SchurFormTensor&& schur_form, SchurVectorTensor&& schur_vectors, std::size_t from, std::size_t to,
                   bool update_vectors)
{
  auto operation = schur_reorder_op{.from = from, .to = to, .update_vectors = update_vectors};
  auto selector = select_backend(operation, schur_form, schur_vectors);
  reorder_schur(selector, std::forward<SchurFormTensor>(schur_form), std::forward<SchurVectorTensor>(schur_vectors),
                from, to, update_vectors);
}

} // namespace uni20::linalg
