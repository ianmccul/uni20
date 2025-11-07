#pragma once

#include "basic_tensor.hpp"

template <typename ElementType, std::size_t Rank, typename StoragePolicy = VectorStorage,
          typename LayoutPolicy = stdex::layout_stride,
          typename AccessorFactory = DefaultAccessorFactory>
using Tensor = BasicTensor<ElementType, stdex::dextents<index_type, Rank>, StoragePolicy, LayoutPolicy,
                          AccessorFactory>;
