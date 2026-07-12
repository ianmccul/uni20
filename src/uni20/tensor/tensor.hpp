#pragma once

/**
 * \file tensor.hpp
 * \ingroup tensor
 * \brief General-purpose owning tensors with runtime extents.
 */

#include "basic_tensor.hpp"
#include "conjugate.hpp"
#include "output.hpp"

#include <cstddef>

namespace uni20
{

/// \brief General-purpose owning tensor with runtime extents and compile-time rank.
/// \ingroup tensor
/// \tparam ElementType Value type stored by the tensor.
/// \tparam Rank Static rank of the tensor.
/// \tparam StoragePolicy Policy controlling ownership and allocation of the buffer.
/// \tparam LayoutPolicy Layout policy that determines index ordering and stride computation.
/// \tparam AccessorFactory Factory that produces accessors for the storage handle.
template <typename ElementType, std::size_t Rank, typename StoragePolicy = VectorStorage,
          typename LayoutPolicy = stdex::layout_stride, typename AccessorFactory = DefaultAccessorFactory>
using Tensor =
    BasicTensor<ElementType, stdex::dextents<index_type, Rank>, StoragePolicy, LayoutPolicy, AccessorFactory>;

/// \brief Conventional row-major contiguous mdspan layout.
using RowMajor = stdex::layout_right;

/// \brief Conventional column-major contiguous mdspan layout.
using ColumnMajor = stdex::layout_left;

/// \brief Owning dense host matrix with a compile-time contiguous layout.
/// \ingroup tensor
/// \tparam ElementType Value type stored by the matrix.
/// \tparam LayoutPolicy Contiguous matrix layout; column-major by default for
///                      direct LAPACK interoperability.
template <typename ElementType, typename LayoutPolicy = ColumnMajor>
using DenseMatrix = Tensor<ElementType, 2, VectorStorage, LayoutPolicy>;

} // namespace uni20
