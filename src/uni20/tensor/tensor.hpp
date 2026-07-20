#pragma once

/**
 * \file tensor.hpp
 * \ingroup tensor
 * \brief General-purpose owning tensors with runtime extents.
 */

#include "basic_tensor.hpp"
#include "conjugate.hpp"
#include "generated.hpp"
#include "output.hpp"
#include "reshape.hpp"

#if UNI20_BACKEND_CUDA
#include <uni20/storage/cuda_async_storage.hpp>
#endif

#include <cstddef>

namespace uni20
{

/// \brief Conventional row-major contiguous mdspan layout.
using RowMajor = stdex::layout_right;

/// \brief Conventional column-major contiguous mdspan layout.
using ColumnMajor = stdex::layout_left;

/// \brief Owning runtime-extents tensor with canonical column-major storage.
template <typename ElementType, std::size_t Rank, typename StoragePolicy = VectorStorage,
          typename AccessorFactory = detail::tensor_accessor_factory_t<StoragePolicy>>
using ColumnMajorTensor = Tensor<ElementType, Rank, StoragePolicy, ColumnMajor, AccessorFactory>;

/// \brief Owning runtime-extents tensor with canonical row-major storage.
template <typename ElementType, std::size_t Rank, typename StoragePolicy = VectorStorage,
          typename AccessorFactory = detail::tensor_accessor_factory_t<StoragePolicy>>
using RowMajorTensor = Tensor<ElementType, Rank, StoragePolicy, RowMajor, AccessorFactory>;

/// \brief Owning runtime-extents tensor with an explicitly strided mapping.
template <typename ElementType, std::size_t Rank, typename StoragePolicy = VectorStorage,
          typename AccessorFactory = detail::tensor_accessor_factory_t<StoragePolicy>>
using StridedTensor = Tensor<ElementType, Rank, StoragePolicy, stdex::layout_stride, AccessorFactory>;

/// \brief Owning rank-zero tensor that retains storage and backend semantics.
template <typename ElementType, typename StoragePolicy = VectorStorage,
          typename AccessorFactory = detail::tensor_accessor_factory_t<StoragePolicy>>
using ScalarTensor = Tensor<ElementType, 0, StoragePolicy, ColumnMajor, AccessorFactory>;

/// \brief Owning dense host matrix with a compile-time contiguous layout.
/// \ingroup tensor
/// \tparam ElementType Value type stored by the matrix.
/// \tparam LayoutPolicy Contiguous matrix layout; column-major by default for
///                      direct LAPACK interoperability.
template <typename ElementType, typename LayoutPolicy = ColumnMajor>
using DenseMatrix = Tensor<ElementType, 2, VectorStorage, LayoutPolicy>;

#if UNI20_BACKEND_CUDA
/// \brief Owning runtime-extents Tensor in non-blocking CUDA storage.
/// \details Construction requires an explicit `cuda::DeviceContext`. Resolved
///          mdspans expose opaque `cuda::CudaBufferView` handles and never
///          perform host/device transfer or resource acquisition.
template <typename ElementType, std::size_t Rank, typename LayoutPolicy = ColumnMajor>
using CudaAsyncTensor = Tensor<ElementType, Rank, CudaAsyncStorage, LayoutPolicy>;

/// \brief Owning column-major matrix in non-blocking CUDA storage.
template <typename ElementType, typename LayoutPolicy = ColumnMajor>
using CudaAsyncMatrix = CudaAsyncTensor<ElementType, 2, LayoutPolicy>;
#endif

} // namespace uni20
