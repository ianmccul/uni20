#pragma once

/**
 * \file tensor.hpp
 * \ingroup tensor
 * \brief General-purpose owning tensors with runtime extents.
 */

#include "access.hpp"
#include "basic_tensor.hpp"
#include "conjugate.hpp"
#include "generated.hpp"
#include "mdspec_tensor_view.hpp"
#include "output.hpp"
#include "reshape.hpp"

#if UNI20_BACKEND_CUDA
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/tensor/cuda_access.hpp>
#endif

#include <cstddef>

namespace uni20
{

/// \brief Conventional row-major contiguous mdspan layout.
using RowMajor = stdex::layout_right;

/// \brief Conventional column-major contiguous mdspan layout.
using ColumnMajor = stdex::layout_left;

/// \brief Owning runtime-extents tensor with canonical column-major storage.
template <typename ElementType, std::size_t Rank, typename StoragePolicy = HostStorage,
          typename AccessorFactory = detail::tensor_accessor_factory_t<StoragePolicy>>
using ColumnMajorTensor = Tensor<ElementType, Rank, StoragePolicy, ColumnMajor, AccessorFactory>;

/// \brief Owning runtime-extents tensor with canonical row-major storage.
template <typename ElementType, std::size_t Rank, typename StoragePolicy = HostStorage,
          typename AccessorFactory = detail::tensor_accessor_factory_t<StoragePolicy>>
using RowMajorTensor = Tensor<ElementType, Rank, StoragePolicy, RowMajor, AccessorFactory>;

/// \brief Owning runtime-extents tensor with an explicitly strided mapping.
template <typename ElementType, std::size_t Rank, typename StoragePolicy = HostStorage,
          typename AccessorFactory = detail::tensor_accessor_factory_t<StoragePolicy>>
using StridedTensor = Tensor<ElementType, Rank, StoragePolicy, stdex::layout_stride, AccessorFactory>;

/// \brief Owning rank-zero tensor that retains storage and backend semantics.
template <typename ElementType, typename StoragePolicy = HostStorage,
          typename AccessorFactory = detail::tensor_accessor_factory_t<StoragePolicy>>
using ScalarTensor = Tensor<ElementType, 0, StoragePolicy, ColumnMajor, AccessorFactory>;

/// \brief Owning dense host matrix with a compile-time contiguous layout.
/// \ingroup tensor
/// \tparam ElementType Value type stored by the matrix.
/// \tparam LayoutPolicy Contiguous matrix layout; column-major by default for
///                      direct LAPACK interoperability.
template <typename ElementType, typename LayoutPolicy = ColumnMajor>
using DenseMatrix = Tensor<ElementType, 2, HostStorage, LayoutPolicy>;

#if UNI20_BACKEND_CUDA
/// \brief Owning runtime-extents Tensor in CUDA device storage.
/// \details Ordinary construction uses the installed CUDA runtime's default
///          device. Passing an explicit `cuda::DeviceResources` selects a
///          particular resource set. `mdspec()` exposes a
///          `cuda::CudaBufferView` descriptor, mapping, and eventual pointer
///          accessor. It does not resolve a usable data handle or perform
///          host/device transfer.
template <typename ElementType, std::size_t Rank, typename LayoutPolicy = ColumnMajor>
using CudaTensor = Tensor<ElementType, Rank, CudaStorage, LayoutPolicy>;

/// \brief Owning column-major matrix in CUDA device storage.
template <typename ElementType, typename LayoutPolicy = ColumnMajor>
using CudaMatrix = CudaTensor<ElementType, 2, LayoutPolicy>;
#endif

} // namespace uni20
