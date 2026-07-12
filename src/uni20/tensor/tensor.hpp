#pragma once

#include "basic_tensor.hpp"

namespace uni20
{

/// \brief Owning tensor alias with compile-time rank convenience.
/// \ingroup tensor
/// \tparam ElementType Value type stored by the tensor.
/// \tparam Rank Static rank of the tensor extents.
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
