#pragma once

#include "basic_tensor.hpp"

/// \brief Owning tensor alias with compile-time rank convenience.
/// \ingroup tensor
/// \tparam ElementType Value type stored by the tensor.
/// \tparam Rank Static rank of the tensor extents.
/// \tparam StoragePolicy Policy controlling ownership and allocation of the buffer.
/// \tparam LayoutPolicy Layout policy that determines index ordering and stride computation.
/// \tparam AccessorFactory Factory that produces accessors for the storage handle.
template <typename ElementType, std::size_t Rank, typename StoragePolicy = VectorStorage,
          typename LayoutPolicy = stdex::layout_stride,
          typename AccessorFactory = DefaultAccessorFactory>
using Tensor = BasicTensor<ElementType, stdex::dextents<index_type, Rank>, StoragePolicy, LayoutPolicy,
                          AccessorFactory>;
