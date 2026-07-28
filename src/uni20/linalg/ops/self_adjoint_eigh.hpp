#pragma once

/**
 * \file self_adjoint_eigh.hpp
 * \ingroup linalg
 * \brief In-place and value-producing dense self-adjoint eigensystems.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backends/lapack/self_adjoint_eigh.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/conjugate_inplace.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/tensor.hpp>

#include <array>
#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
using self_adjoint_eigenvalue_extents = stdex::dextents<uni20::index_type, 1>;

template <uni20::MutableRankedDeviceTensorView<1> EigenvalueTensor,
          uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] self_adjoint_eigh_op prepare_self_adjoint_eigh(EigenvalueTensor& eigenvalues, MatrixTensor& matrix_work,
                                                             SelfAdjointEighOptions options)
{
  ERROR_IF(matrix_work.extent(0) != matrix_work.extent(1), "self-adjoint eigensystem requires a square matrix");
  uni20::ensure_shape(eigenvalues,
                      self_adjoint_eigenvalue_extents{static_cast<uni20::index_type>(matrix_work.extent(0))});
  return {.compute_vectors = options.compute_vectors, .triangle = options.triangle};
}

template <class BackendSelector, uni20::MutableRankedDeviceTensorView<1> EigenvalueTensor,
          uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
void dispatch_self_adjoint_eigh(BackendSelector&& selector, self_adjoint_eigh_op operation,
                                EigenvalueTensor& eigenvalues, MatrixTensor& matrix_work)
{
  dispatch_kernel(std::forward<BackendSelector>(selector), operation, eigenvalues, matrix_work);
}
} // namespace detail

/// \brief Compute an in-place self-adjoint eigensystem through an explicit selector.
/// \details The selected triangle of `matrix_work` is read. LAPACK destroys the
///          matrix input; when vectors are requested, its columns contain the
///          normalized eigenvectors on return. `eigenvalues` is resized when
///          its output policy permits.
/// \pre Eigenvalue and matrix storage do not overlap.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<1> EigenvalueTensor,
          uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
void self_adjoint_eigh(BackendSelector&& selector, EigenvalueTensor&& eigenvalues, MatrixTensor&& matrix_work,
                       SelfAdjointEighOptions options = {})
{
  auto operation = detail::prepare_self_adjoint_eigh(eigenvalues, matrix_work, options);
  detail::dispatch_self_adjoint_eigh(std::forward<BackendSelector>(selector), operation, eigenvalues, matrix_work);
}

/// \brief Compute an in-place self-adjoint eigensystem using tensor storage policy.
template <uni20::MutableRankedDeviceTensorView<1> EigenvalueTensor,
          uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
void self_adjoint_eigh(EigenvalueTensor&& eigenvalues, MatrixTensor&& matrix_work, SelfAdjointEighOptions options = {})
{
  auto operation = detail::prepare_self_adjoint_eigh(eigenvalues, matrix_work, options);
  auto selector = select_backend(operation, eigenvalues, matrix_work);
  detail::dispatch_self_adjoint_eigh(selector, operation, eigenvalues, matrix_work);
}

/// \brief Owning eigenvalues and eigenvectors returned by `eigh`.
template <class EigenvalueTensor, class EigenvectorTensor> struct SelfAdjointEighResult
{
    using eigenvalue_tensor_type = EigenvalueTensor;
    using eigenvector_tensor_type = EigenvectorTensor;

    EigenvalueTensor eigenvalues;
    EigenvectorTensor eigenvectors;
};

namespace detail
{
template <class MatrixTensor>
using self_adjoint_eigh_reuse_matrix_t =
    std::conditional_t<std::same_as<typename std::remove_cvref_t<MatrixTensor>::layout_type, uni20::ColumnMajor>,
                       std::remove_cvref_t<MatrixTensor>,
                       typename std::remove_cvref_t<MatrixTensor>::template rebind_layout_type<stdex::layout_stride>>;

template <class MatrixTensor> consteval bool can_transfer_self_adjoint_eigh_storage()
{
  using matrix_type = std::remove_cvref_t<MatrixTensor>;
  if constexpr (requires {
                  typename matrix_type::storage_policy;
                  typename matrix_type::layout_type;
                  typename matrix_type::accessor_factory_type;
                  typename matrix_type::storage_type;
                  typename matrix_type::template rebind_layout_type<stdex::layout_stride>;
                })
  {
    using work_type = self_adjoint_eigh_reuse_matrix_t<matrix_type>;
    return std::same_as<typename matrix_type::storage_policy, uni20::VectorStorage> &&
           std::same_as<typename matrix_type::accessor_factory_type, uni20::DefaultAccessorFactory> &&
           uni20::DefaultAccessorMdspanLike<uni20::mutable_tensor_mdspan_t<matrix_type>> &&
           requires(matrix_type& matrix, typename work_type::mapping_type mapping,
                    typename work_type::storage_type storage) {
             { std::move(matrix).release_storage() } -> std::same_as<typename work_type::storage_type>;
             { work_type::adopt_storage(std::move(mapping), std::move(storage)) } -> std::same_as<work_type>;
           };
  }
  else
  {
    return false;
  }
}

constexpr MatrixTriangle transposed_triangle(MatrixTriangle triangle)
{
  switch (triangle)
  {
    case MatrixTriangle::Upper:
      return MatrixTriangle::Lower;
    case MatrixTriangle::Lower:
      return MatrixTriangle::Upper;
  }
  PANIC("invalid MatrixTriangle", std::to_underlying(triangle));
}

template <class WorkMatrix, uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] WorkMatrix materialize_self_adjoint_eigh_work_matrix(MatrixTensor const& matrix)
{
  WorkMatrix result = [&] {
    if constexpr (std::same_as<typename WorkMatrix::layout_type, uni20::ColumnMajor>)
    {
      return WorkMatrix(matrix.extents());
    }
    else
    {
      return WorkMatrix(matrix.extents(), uni20::layout::LayoutLeft{});
    }
  }();
  uni20::copy(result, matrix);
  return result;
}

template <class WorkMatrix, uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto column_major_reuse_mapping(MatrixTensor const& matrix, uni20::blas_int leading_dimension) ->
    typename WorkMatrix::mapping_type
{
  if constexpr (std::same_as<typename WorkMatrix::layout_type, uni20::ColumnMajor>)
  {
    return typename WorkMatrix::mapping_type(matrix.extents());
  }
  else
  {
    using index_type = typename WorkMatrix::index_type;
    return typename WorkMatrix::mapping_type(
        matrix.extents(), std::array<index_type, 2>{index_type{1}, static_cast<index_type>(leading_dimension)});
  }
}

template <class MatrixTensor>
  requires uni20::RankedDeviceTensorView<MatrixTensor, 2>
[[nodiscard]] constexpr auto select_self_adjoint_eigh_backend(MatrixTriangle triangle)
{
  using real_type = uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>;
  using eigenvalue_tensor = uni20::Tensor<real_type, 1>;
  return select_backend_for<eigenvalue_tensor, std::remove_cvref_t<MatrixTensor>>(
      self_adjoint_eigh_op{.compute_vectors = true, .triangle = triangle});
}

template <class BackendSelector, uni20::OwningTensor MatrixTensor>
  requires uni20::MutableRankedTensorView<MatrixTensor, 2>
[[nodiscard]] auto eigh_from_work_matrix(BackendSelector&& selector, MatrixTensor matrix_work, MatrixTriangle triangle)
{
  using real_type = uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>;
  uni20::Tensor<real_type, 1> eigenvalues;
  self_adjoint_eigh(std::forward<BackendSelector>(selector), eigenvalues, matrix_work,
                    SelfAdjointEighOptions{.compute_vectors = true, .triangle = triangle});
  return SelfAdjointEighResult<decltype(eigenvalues), MatrixTensor>{.eigenvalues = std::move(eigenvalues),
                                                                    .eigenvectors = std::move(matrix_work)};
}

template <class BackendSelector, uni20::OwningTensor MatrixTensor>
  requires uni20::MutableRankedTensorView<MatrixTensor, 2>
[[nodiscard]] auto eigh_from_consumed_matrix(BackendSelector&& selector, MatrixTensor&& matrix, MatrixTriangle triangle)
{
  using matrix_type = std::remove_cvref_t<MatrixTensor>;
  using work_type = self_adjoint_eigh_reuse_matrix_t<matrix_type>;

  auto const matrix_span = matrix.mdspan();
  auto const stage = uni20::linalg::blas::try_mdspan_matrix_stage(matrix_span);
  if (stage && !stage->needs_conjugation)
  {
    bool const transpose_storage = stage->unit_stride_axis == 1;
    MatrixTriangle const work_triangle = transpose_storage ? transposed_triangle(triangle) : triangle;
    auto work_mapping = column_major_reuse_mapping<work_type>(matrix, stage->nonunit_stride);
    auto storage = std::move(matrix).release_storage();
    auto work_matrix = work_type::adopt_storage(std::move(work_mapping), std::move(storage));
    auto result = eigh_from_work_matrix(std::forward<BackendSelector>(selector), std::move(work_matrix), work_triangle);
    if constexpr (uni20::Complex<uni20::tensor_element_t<matrix_type>>)
    {
      if (transpose_storage) uni20::conjugate_inplace(result.eigenvectors);
    }
    return result;
  }

  auto work_matrix = materialize_self_adjoint_eigh_work_matrix<work_type>(matrix);
  return eigh_from_work_matrix(std::forward<BackendSelector>(selector), std::move(work_matrix), triangle);
}
} // namespace detail

/// \brief Preserve a matrix and return its complete self-adjoint eigensystem through an explicit selector.
/// \details Materialization uses the tensors' copy selector; the supplied
///          selector controls the destructive eigensolver dispatch.
template <class BackendSelector, uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] auto eigh(BackendSelector&& selector, MatrixTensor const& matrix,
                        MatrixTriangle triangle = MatrixTriangle::Upper)
{
  ERROR_IF(matrix.extent(0) != matrix.extent(1), "self-adjoint eigensystem requires a square matrix");
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  return detail::eigh_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work), triangle);
}

/// \brief Preserve a matrix and return its complete self-adjoint eigensystem.
/// \details This explicit value operation materializes a column-major work
///          matrix through `copy_op`, then calls the destructive in-place
///          eigensolver with eigenvectors enabled.
template <uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] auto eigh(MatrixTensor const& matrix, MatrixTriangle triangle = MatrixTriangle::Upper)
{
  ERROR_IF(matrix.extent(0) != matrix.extent(1), "self-adjoint eigensystem requires a square matrix");
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  auto selector = detail::select_self_adjoint_eigh_backend<decltype(matrix_work)>(triangle);
  return detail::eigh_from_work_matrix(std::move(selector), std::move(matrix_work), triangle);
}

/// \brief Consume an owning matrix and return its complete self-adjoint eigensystem through an explicit selector.
/// \details This overload has the same storage-transfer semantics as the
///          default consuming overload; the supplied selector controls the
///          eigensolver dispatch.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto eigh(BackendSelector&& selector, MatrixTensor&& matrix,
                        MatrixTriangle triangle = MatrixTriangle::Upper)
{
  ERROR_IF(matrix.extent(0) != matrix.extent(1), "self-adjoint eigensystem requires a square matrix");
  if constexpr (detail::can_transfer_self_adjoint_eigh_storage<MatrixTensor>())
  {
    return detail::eigh_from_consumed_matrix(std::forward<BackendSelector>(selector),
                                             std::forward<MatrixTensor>(matrix), triangle);
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    return detail::eigh_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work), triangle);
  }
}

/// \brief Consume an owning matrix and return its complete self-adjoint eigensystem.
/// \details A host tensor with a directly addressable BLAS matrix mapping
///          transfers its allocation to the eigenvector result. Row-major
///          storage is reinterpreted as its column-major transpose, with the
///          selected triangle adjusted before LAPACK and complex eigenvectors
///          conjugated afterward. The result preserves a compatible leading
///          dimension and any storage tail. Other owning tensor types are
///          materialized into a compatible work matrix. Passing an rvalue
///          grants permission to consume storage but does not guarantee reuse.
/// \warning Any non-owning views into storage transferred by this operation
///          are invalidated under the ordinary C++ moved-from owner rules.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto eigh(MatrixTensor&& matrix, MatrixTriangle triangle = MatrixTriangle::Upper)
{
  ERROR_IF(matrix.extent(0) != matrix.extent(1), "self-adjoint eigensystem requires a square matrix");
  if constexpr (detail::can_transfer_self_adjoint_eigh_storage<MatrixTensor>())
  {
    using work_type = detail::self_adjoint_eigh_reuse_matrix_t<std::remove_cvref_t<MatrixTensor>>;
    auto selector = detail::select_self_adjoint_eigh_backend<work_type>(triangle);
    return detail::eigh_from_consumed_matrix(std::move(selector), std::forward<MatrixTensor>(matrix), triangle);
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    auto selector = detail::select_self_adjoint_eigh_backend<decltype(matrix_work)>(triangle);
    return detail::eigh_from_work_matrix(std::move(selector), std::move(matrix_work), triangle);
  }
}

} // namespace uni20::linalg
