#pragma once

/**
 * \file matrix_set.hpp
 * \ingroup linalg
 * \brief Structured dense matrix initialization operation.
 */

#include <uni20/linalg/backends/cpu/matrix_set.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <utility>

namespace uni20::linalg
{

/// \brief Initialize a matrix region through an explicit backend selector.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<2> MatrixTensor, class Scalar>
void set_matrix(BackendSelector&& selector, MatrixTensor&& matrix, Scalar diagonal, Scalar off_diagonal,
                MatrixRegion region = MatrixRegion::All)
{
  dispatch_kernel(std::forward<BackendSelector>(selector), matrix_set_op{.region = region}, matrix, diagonal,
                  off_diagonal);
}

/// \brief Initialize a matrix region using the matrix storage's backend selector.
template <uni20::MutableRankedDeviceTensorView<2> MatrixTensor, class Scalar>
void set_matrix(MatrixTensor&& matrix, Scalar diagonal, Scalar off_diagonal, MatrixRegion region = MatrixRegion::All)
{
  auto selector = select_backend(matrix_set_op{.region = region}, matrix);
  set_matrix(selector, std::forward<MatrixTensor>(matrix), diagonal, off_diagonal, region);
}

} // namespace uni20::linalg
