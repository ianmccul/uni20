#pragma once

/**
 * \file svd.hpp
 * \ingroup linalg
 * \brief Destructive and value-producing exact dense singular value decompositions.
 */

#include <uni20/linalg/backends/lapack/svd.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/tensor.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
using svd_value_extents = stdex::dextents<uni20::index_type, 1>;
using svd_matrix_extents = stdex::dextents<uni20::index_type, 2>;

template <
    uni20::MutableRankedDeviceTensorView<1> SingularValueTensor, uni20::MutableRankedDeviceTensorView<2> LeftTensor,
    uni20::MutableRankedDeviceTensorView<2> RightAdjointTensor, uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] svd_op prepare_svd(SingularValueTensor& singular_values, LeftTensor& left_singular_vectors,
                                 RightAdjointTensor& right_singular_vectors_adjoint, MatrixTensor& matrix_work,
                                 SvdOptions options)
{
  std::size_t const rows = static_cast<std::size_t>(matrix_work.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix_work.extent(1));
  std::size_t const rank = std::min(rows, cols);
  std::size_t const left_cols = options.left == SvdVectorExtent::Full ? rows : rank;
  std::size_t const right_rows = options.right == SvdVectorExtent::Full ? cols : rank;

  uni20::prepare_output(singular_values, svd_value_extents{static_cast<uni20::index_type>(rank)});
  uni20::prepare_output(left_singular_vectors, svd_matrix_extents{static_cast<uni20::index_type>(rows),
                                                                  static_cast<uni20::index_type>(left_cols)});
  uni20::prepare_output(right_singular_vectors_adjoint, svd_matrix_extents{static_cast<uni20::index_type>(right_rows),
                                                                           static_cast<uni20::index_type>(cols)});
  return {.left = options.left, .right = options.right};
}

template <class BackendSelector, uni20::MutableRankedDeviceTensorView<1> SingularValueTensor,
          uni20::MutableRankedDeviceTensorView<2> LeftTensor,
          uni20::MutableRankedDeviceTensorView<2> RightAdjointTensor,
          uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
void dispatch_svd(BackendSelector&& selector, svd_op operation, SingularValueTensor& singular_values,
                  LeftTensor& left_singular_vectors, RightAdjointTensor& right_singular_vectors_adjoint,
                  MatrixTensor& matrix_work)
{
  auto value_descriptor = uni20::device_mdspan_of(singular_values);
  auto left_descriptor = uni20::device_mdspan_of(left_singular_vectors);
  auto right_descriptor = uni20::device_mdspan_of(right_singular_vectors_adjoint);
  auto matrix_descriptor = uni20::device_mdspan_of(matrix_work);
  dispatch_kernel(std::forward<BackendSelector>(selector), operation, value_descriptor, left_descriptor,
                  right_descriptor, matrix_descriptor);
}

template <class BackendSelector, uni20::MutableRankedDeviceTensorView<1> SingularValueTensor,
          uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
void dispatch_singular_values(BackendSelector&& selector, SingularValueTensor& singular_values,
                              MatrixTensor& matrix_work)
{
  auto value_descriptor = uni20::device_mdspan_of(singular_values);
  auto matrix_descriptor = uni20::device_mdspan_of(matrix_work);
  dispatch_kernel(std::forward<BackendSelector>(selector), singular_values_op{}, value_descriptor, matrix_descriptor);
}

template <class BackendSelector, uni20::MutableRankedDeviceTensorView<1> SingularValueTensor,
          uni20::MutableRankedDeviceTensorView<2> LeftTensor, uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
void dispatch_svd_left(BackendSelector&& selector, svd_left_op operation, SingularValueTensor& singular_values,
                       LeftTensor& left_singular_vectors, MatrixTensor& matrix_work)
{
  auto value_descriptor = uni20::device_mdspan_of(singular_values);
  auto left_descriptor = uni20::device_mdspan_of(left_singular_vectors);
  auto matrix_descriptor = uni20::device_mdspan_of(matrix_work);
  dispatch_kernel(std::forward<BackendSelector>(selector), operation, value_descriptor, left_descriptor,
                  matrix_descriptor);
}

template <class BackendSelector, uni20::MutableRankedDeviceTensorView<1> SingularValueTensor,
          uni20::MutableRankedDeviceTensorView<2> RightAdjointTensor,
          uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
void dispatch_svd_right(BackendSelector&& selector, svd_right_op operation, SingularValueTensor& singular_values,
                        RightAdjointTensor& right_singular_vectors_adjoint, MatrixTensor& matrix_work)
{
  auto value_descriptor = uni20::device_mdspan_of(singular_values);
  auto right_descriptor = uni20::device_mdspan_of(right_singular_vectors_adjoint);
  auto matrix_descriptor = uni20::device_mdspan_of(matrix_work);
  dispatch_kernel(std::forward<BackendSelector>(selector), operation, value_descriptor, right_descriptor,
                  matrix_descriptor);
}
} // namespace detail

/// \brief Compute a destructive exact SVD through an explicit selector.
/// \details `matrix_work` is overwritten. Singular values are returned in
///          descending order. The right factor is `Vh`, the transpose for real
///          inputs and conjugate transpose for complex inputs. The workspace
///          and outputs must lower to writable LAPACK column-major operands.
/// \pre Output storage and matrix workspace do not overlap.
template <class BackendSelector, uni20::MutableRankedDeviceTensorView<1> SingularValueTensor,
          uni20::MutableRankedDeviceTensorView<2> LeftTensor,
          uni20::MutableRankedDeviceTensorView<2> RightAdjointTensor,
          uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
void singular_value_decomposition(BackendSelector&& selector, SingularValueTensor&& singular_values,
                                  LeftTensor&& left_singular_vectors,
                                  RightAdjointTensor&& right_singular_vectors_adjoint, MatrixTensor&& matrix_work,
                                  SvdOptions options = {})
{
  auto operation =
      detail::prepare_svd(singular_values, left_singular_vectors, right_singular_vectors_adjoint, matrix_work, options);
  detail::dispatch_svd(std::forward<BackendSelector>(selector), operation, singular_values, left_singular_vectors,
                       right_singular_vectors_adjoint, matrix_work);
}

/// \brief Compute a destructive exact SVD using tensor storage policy.
/// \details The workspace and outputs must lower to writable LAPACK
///          column-major operands.
template <
    uni20::MutableRankedDeviceTensorView<1> SingularValueTensor, uni20::MutableRankedDeviceTensorView<2> LeftTensor,
    uni20::MutableRankedDeviceTensorView<2> RightAdjointTensor, uni20::MutableRankedDeviceTensorView<2> MatrixTensor>
void singular_value_decomposition(SingularValueTensor&& singular_values, LeftTensor&& left_singular_vectors,
                                  RightAdjointTensor&& right_singular_vectors_adjoint, MatrixTensor&& matrix_work,
                                  SvdOptions options = {})
{
  auto operation =
      detail::prepare_svd(singular_values, left_singular_vectors, right_singular_vectors_adjoint, matrix_work, options);
  auto selector =
      select_backend(operation, singular_values, left_singular_vectors, right_singular_vectors_adjoint, matrix_work);
  detail::dispatch_svd(selector, operation, singular_values, left_singular_vectors, right_singular_vectors_adjoint,
                       matrix_work);
}

/// \brief Owning left singular vectors and singular values returned by `svd_left`.
/// \details Aggregate member order supports `auto [u, s] = svd_left(matrix)`.
template <class LeftTensor, class SingularValueTensor> struct SvdLeftResult
{
    using left_singular_vector_tensor_type = LeftTensor;
    using singular_value_tensor_type = SingularValueTensor;

    LeftTensor left_singular_vectors;
    SingularValueTensor singular_values;
};

/// \brief Owning singular values and right singular vectors returned by `svd_right`.
/// \details Aggregate member order supports `auto [s, vh] = svd_right(matrix)`.
template <class SingularValueTensor, class RightAdjointTensor> struct SvdRightResult
{
    using singular_value_tensor_type = SingularValueTensor;
    using right_singular_vector_adjoint_tensor_type = RightAdjointTensor;

    SingularValueTensor singular_values;
    RightAdjointTensor right_singular_vectors_adjoint;
};

/// \brief Owning factors returned by `svd`.
/// \details Aggregate member order supports `auto [u, s, vh] = svd(matrix)`.
template <class LeftTensor, class SingularValueTensor, class RightAdjointTensor> struct SvdResult
{
    using left_singular_vector_tensor_type = LeftTensor;
    using singular_value_tensor_type = SingularValueTensor;
    using right_singular_vector_adjoint_tensor_type = RightAdjointTensor;

    LeftTensor left_singular_vectors;
    SingularValueTensor singular_values;
    RightAdjointTensor right_singular_vectors_adjoint;
};

namespace detail
{
template <class MatrixTensor> using svd_scalar_t = uni20::tensor_element_t<MatrixTensor>;
template <class MatrixTensor> using svd_real_t = uni20::make_real_t<svd_scalar_t<MatrixTensor>>;
template <class MatrixTensor> using svd_value_tensor_t = uni20::Tensor<svd_real_t<MatrixTensor>, 1>;
template <class MatrixTensor> using svd_matrix_tensor_t = uni20::Tensor<svd_scalar_t<MatrixTensor>, 2>;

template <class MatrixTensor>
using svd_reuse_factor_t =
    typename std::remove_cvref_t<MatrixTensor>::template rebind_layout_type<stdex::layout_stride>;

template <class MatrixTensor> consteval bool can_transfer_svd_storage()
{
  using matrix_type = std::remove_cvref_t<MatrixTensor>;
  if constexpr (requires {
                  typename matrix_type::storage_policy;
                  typename matrix_type::layout_type;
                  typename matrix_type::accessor_factory_type;
                  typename matrix_type::storage_type;
                  typename matrix_type::extents_type;
                  typename matrix_type::template rebind_layout_type<stdex::layout_stride>;
                })
  {
    using factor_type = svd_reuse_factor_t<matrix_type>;
    return matrix_type::extents_type::rank_dynamic() == 2 &&
           std::same_as<typename matrix_type::storage_policy, uni20::VectorStorage> &&
           std::same_as<typename matrix_type::accessor_factory_type, uni20::DefaultAccessorFactory> &&
           uni20::DefaultAccessorMdspanLike<uni20::mutable_tensor_mdspan_t<matrix_type>> &&
           requires(matrix_type& matrix, typename factor_type::mapping_type mapping,
                    typename factor_type::storage_type storage,
                    typename factor_type::accessor_factory_type accessor_factory) {
             { std::move(matrix).release_storage() } -> std::same_as<typename factor_type::storage_type>;
             {
               std::move(matrix).release_accessor_factory()
             } -> std::same_as<typename factor_type::accessor_factory_type>;
             {
               factor_type::adopt_storage(std::move(mapping), std::move(storage), std::move(accessor_factory))
             } -> std::same_as<factor_type>;
           };
  }
  else
  {
    return false;
  }
}

template <class FactorTensor> [[nodiscard]] FactorTensor make_svd_factor(std::size_t rows, std::size_t cols)
{
  using index_type = typename FactorTensor::index_type;
  using extents_type = typename FactorTensor::extents_type;
  return FactorTensor(extents_type{static_cast<index_type>(rows), static_cast<index_type>(cols)},
                      uni20::layout::LayoutLeft{});
}

template <class FactorTensor>
[[nodiscard]] auto svd_reuse_mapping(std::size_t rows, std::size_t cols, uni20::blas_int leading_dimension) ->
    typename FactorTensor::mapping_type
{
  using index_type = typename FactorTensor::index_type;
  using extents_type = typename FactorTensor::extents_type;
  extents_type extents{static_cast<index_type>(rows), static_cast<index_type>(cols)};
  return typename FactorTensor::mapping_type(
      extents, std::array<index_type, 2>{index_type{1}, static_cast<index_type>(leading_dimension)});
}

template <class FactorTensor, class MatrixTensor>
[[nodiscard]] FactorTensor adopt_svd_factor(MatrixTensor&& matrix, std::size_t rows, std::size_t cols,
                                            uni20::blas_int leading_dimension)
{
  auto mapping = svd_reuse_mapping<FactorTensor>(rows, cols, leading_dimension);
  auto accessor_factory = std::move(matrix).release_accessor_factory();
  auto storage = std::move(matrix).release_storage();
  return FactorTensor::adopt_storage(std::move(mapping), std::move(storage), std::move(accessor_factory));
}

template <class MatrixTensor> [[nodiscard]] auto direct_column_major_svd_stage(MatrixTensor& matrix)
{
  auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(matrix.mdspan());
  if (!stage || stage->unit_stride_axis != 0 || stage->needs_conjugation)
  {
    return decltype(stage){};
  }
  return stage;
}

template <class BackendSelector, uni20::OwningTensor MatrixTensor>
  requires uni20::MutableRankedTensorView<MatrixTensor, 2>
[[nodiscard]] auto singular_values_from_work_matrix(BackendSelector&& selector, MatrixTensor matrix_work)
{
  svd_value_tensor_t<MatrixTensor> values(
      svd_value_extents{static_cast<uni20::index_type>(std::min(matrix_work.extent(0), matrix_work.extent(1)))});
  dispatch_singular_values(std::forward<BackendSelector>(selector), values, matrix_work);
  return values;
}

template <class BackendSelector, uni20::OwningTensor MatrixTensor>
  requires uni20::MutableRankedTensorView<MatrixTensor, 2>
[[nodiscard]] auto svd_left_from_work_matrix(BackendSelector&& selector, MatrixTensor matrix_work,
                                             SvdVectorExtent extent)
{
  std::size_t const rows = static_cast<std::size_t>(matrix_work.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix_work.extent(1));
  std::size_t const rank = std::min(rows, cols);
  std::size_t const left_cols = extent == SvdVectorExtent::Full ? rows : rank;
  svd_matrix_tensor_t<MatrixTensor> left(rows, left_cols);
  svd_value_tensor_t<MatrixTensor> values(rank);
  dispatch_svd_left(std::forward<BackendSelector>(selector), svd_left_op{.left = extent}, values, left, matrix_work);
  return SvdLeftResult<decltype(left), decltype(values)>{.left_singular_vectors = std::move(left),
                                                         .singular_values = std::move(values)};
}

template <class BackendSelector, uni20::OwningTensor MatrixTensor>
  requires uni20::MutableRankedTensorView<MatrixTensor, 2>
[[nodiscard]] auto svd_right_from_work_matrix(BackendSelector&& selector, MatrixTensor matrix_work,
                                              SvdVectorExtent extent)
{
  std::size_t const rows = static_cast<std::size_t>(matrix_work.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix_work.extent(1));
  std::size_t const rank = std::min(rows, cols);
  std::size_t const right_rows = extent == SvdVectorExtent::Full ? cols : rank;
  svd_value_tensor_t<MatrixTensor> values(rank);
  svd_matrix_tensor_t<MatrixTensor> right(right_rows, cols);
  dispatch_svd_right(std::forward<BackendSelector>(selector), svd_right_op{.right = extent}, values, right,
                     matrix_work);
  return SvdRightResult<decltype(values), decltype(right)>{.singular_values = std::move(values),
                                                           .right_singular_vectors_adjoint = std::move(right)};
}

template <class BackendSelector, uni20::OwningTensor MatrixTensor>
  requires uni20::MutableRankedTensorView<MatrixTensor, 2>
[[nodiscard]] auto svd_from_work_matrix(BackendSelector&& selector, MatrixTensor matrix_work, SvdOptions options)
{
  using scalar_type = uni20::tensor_element_t<MatrixTensor>;
  using real_type = uni20::make_real_t<scalar_type>;
  uni20::Tensor<scalar_type, 2> left_singular_vectors;
  uni20::Tensor<real_type, 1> singular_values;
  uni20::Tensor<scalar_type, 2> right_singular_vectors_adjoint;
  singular_value_decomposition(std::forward<BackendSelector>(selector), singular_values, left_singular_vectors,
                               right_singular_vectors_adjoint, matrix_work, options);
  return SvdResult<decltype(left_singular_vectors), decltype(singular_values),
                   decltype(right_singular_vectors_adjoint)>{.left_singular_vectors = std::move(left_singular_vectors),
                                                             .singular_values = std::move(singular_values),
                                                             .right_singular_vectors_adjoint =
                                                                 std::move(right_singular_vectors_adjoint)};
}

template <class MatrixTensor> [[nodiscard]] constexpr auto select_singular_values_backend()
{
  return select_backend_for<svd_value_tensor_t<MatrixTensor>, std::remove_cvref_t<MatrixTensor>>(singular_values_op{});
}

template <class MatrixTensor> [[nodiscard]] constexpr auto select_svd_left_backend(SvdVectorExtent extent)
{
  return select_backend_for<svd_value_tensor_t<MatrixTensor>, svd_matrix_tensor_t<MatrixTensor>,
                            std::remove_cvref_t<MatrixTensor>>(svd_left_op{.left = extent});
}

template <class MatrixTensor> [[nodiscard]] constexpr auto select_svd_right_backend(SvdVectorExtent extent)
{
  return select_backend_for<svd_value_tensor_t<MatrixTensor>, svd_matrix_tensor_t<MatrixTensor>,
                            std::remove_cvref_t<MatrixTensor>>(svd_right_op{.right = extent});
}

template <class MatrixTensor> [[nodiscard]] constexpr auto select_svd_backend(SvdOptions options)
{
  return select_backend_for<svd_value_tensor_t<MatrixTensor>, svd_matrix_tensor_t<MatrixTensor>,
                            svd_matrix_tensor_t<MatrixTensor>, std::remove_cvref_t<MatrixTensor>>(
      svd_op{.left = options.left, .right = options.right});
}

template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2>
[[nodiscard]] auto singular_values_from_consumed_matrix(BackendSelector&& selector, MatrixTensor&& matrix)
{
  if constexpr (can_transfer_svd_storage<MatrixTensor>())
  {
    if (direct_column_major_svd_stage(matrix))
    {
      svd_value_tensor_t<MatrixTensor> values(std::min(matrix.extent(0), matrix.extent(1)));
      dispatch_singular_values(std::forward<BackendSelector>(selector), values, matrix);
      return values;
    }
  }
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  return singular_values_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work));
}

template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (can_transfer_svd_storage<MatrixTensor>())
[[nodiscard]] auto svd_left_from_transferable_matrix(BackendSelector&& selector, MatrixTensor&& matrix,
                                                     SvdVectorExtent extent)
{
  using factor_type = svd_reuse_factor_t<MatrixTensor>;
  using value_type = svd_value_tensor_t<MatrixTensor>;
  std::size_t const rows = static_cast<std::size_t>(matrix.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix.extent(1));
  std::size_t const rank = std::min(rows, cols);
  std::size_t const left_cols = extent == SvdVectorExtent::Full ? rows : rank;
  value_type values(rank);

  auto stage = direct_column_major_svd_stage(matrix);
  if (stage)
  {
    if (extent == SvdVectorExtent::Reduced)
    {
      auto values_span = values.mdspan();
      auto matrix_span = matrix.mdspan();
      auto operation = svd_left_op{.left = extent, .overwrite_input = true};
      if constexpr (requires { dispatch_kernel(selector, operation, values_span, matrix_span); })
      {
        dispatch_kernel(std::forward<BackendSelector>(selector), operation, values_span, matrix_span);
        auto left =
            adopt_svd_factor<factor_type>(std::forward<MatrixTensor>(matrix), rows, rank, stage->nonunit_stride);
        return SvdLeftResult<factor_type, value_type>{.left_singular_vectors = std::move(left),
                                                      .singular_values = std::move(values)};
      }
    }

    auto left = make_svd_factor<factor_type>(rows, left_cols);
    dispatch_svd_left(std::forward<BackendSelector>(selector), svd_left_op{.left = extent}, values, left, matrix);
    return SvdLeftResult<factor_type, value_type>{.left_singular_vectors = std::move(left),
                                                  .singular_values = std::move(values)};
  }

  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  auto left = make_svd_factor<factor_type>(rows, left_cols);
  dispatch_svd_left(std::forward<BackendSelector>(selector), svd_left_op{.left = extent}, values, left, matrix_work);
  return SvdLeftResult<factor_type, value_type>{.left_singular_vectors = std::move(left),
                                                .singular_values = std::move(values)};
}

template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (can_transfer_svd_storage<MatrixTensor>())
[[nodiscard]] auto svd_right_from_transferable_matrix(BackendSelector&& selector, MatrixTensor&& matrix,
                                                      SvdVectorExtent extent)
{
  using factor_type = svd_reuse_factor_t<MatrixTensor>;
  using value_type = svd_value_tensor_t<MatrixTensor>;
  std::size_t const rows = static_cast<std::size_t>(matrix.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix.extent(1));
  std::size_t const rank = std::min(rows, cols);
  std::size_t const right_rows = extent == SvdVectorExtent::Full ? cols : rank;
  value_type values(rank);

  auto stage = direct_column_major_svd_stage(matrix);
  if (stage)
  {
    if (extent == SvdVectorExtent::Reduced)
    {
      auto values_span = values.mdspan();
      auto matrix_span = matrix.mdspan();
      auto operation = svd_right_op{.right = extent, .overwrite_input = true};
      if constexpr (requires { dispatch_kernel(selector, operation, values_span, matrix_span); })
      {
        dispatch_kernel(std::forward<BackendSelector>(selector), operation, values_span, matrix_span);
        auto right =
            adopt_svd_factor<factor_type>(std::forward<MatrixTensor>(matrix), rank, cols, stage->nonunit_stride);
        return SvdRightResult<value_type, factor_type>{.singular_values = std::move(values),
                                                       .right_singular_vectors_adjoint = std::move(right)};
      }
    }

    auto right = make_svd_factor<factor_type>(right_rows, cols);
    dispatch_svd_right(std::forward<BackendSelector>(selector), svd_right_op{.right = extent}, values, right, matrix);
    return SvdRightResult<value_type, factor_type>{.singular_values = std::move(values),
                                                   .right_singular_vectors_adjoint = std::move(right)};
  }

  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  auto right = make_svd_factor<factor_type>(right_rows, cols);
  dispatch_svd_right(std::forward<BackendSelector>(selector), svd_right_op{.right = extent}, values, right,
                     matrix_work);
  return SvdRightResult<value_type, factor_type>{.singular_values = std::move(values),
                                                 .right_singular_vectors_adjoint = std::move(right)};
}

template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (can_transfer_svd_storage<MatrixTensor>())
[[nodiscard]] auto svd_from_transferable_matrix(BackendSelector&& selector, MatrixTensor&& matrix, SvdOptions options)
{
  using factor_type = svd_reuse_factor_t<MatrixTensor>;
  using value_type = svd_value_tensor_t<MatrixTensor>;
  std::size_t const rows = static_cast<std::size_t>(matrix.extent(0));
  std::size_t const cols = static_cast<std::size_t>(matrix.extent(1));
  std::size_t const rank = std::min(rows, cols);
  std::size_t const left_cols = options.left == SvdVectorExtent::Full ? rows : rank;
  std::size_t const right_rows = options.right == SvdVectorExtent::Full ? cols : rank;
  value_type values(rank);
  factor_type left;
  factor_type right;

  auto stage = direct_column_major_svd_stage(matrix);
  if (!stage)
  {
    left = make_svd_factor<factor_type>(rows, left_cols);
    right = make_svd_factor<factor_type>(right_rows, cols);
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    auto operation = svd_op{.left = options.left, .right = options.right};
    auto values_span = values.mdspan();
    auto left_span = left.mdspan();
    auto right_span = right.mdspan();
    auto matrix_span = matrix_work.mdspan();
    dispatch_kernel(std::forward<BackendSelector>(selector), operation, values_span, left_span, right_span,
                    matrix_span);
  }
  else if (options.left == SvdVectorExtent::Reduced && (options.right == SvdVectorExtent::Full || rows >= cols))
  {
    right = make_svd_factor<factor_type>(right_rows, cols);
    auto operation = svd_op{.left = options.left, .right = options.right, .overwrite = SvdOverwrite::Left};
    auto values_span = values.mdspan();
    auto right_span = right.mdspan();
    auto matrix_span = matrix.mdspan();
    if constexpr (requires { dispatch_kernel(selector, operation, values_span, right_span, matrix_span); })
    {
      dispatch_kernel(std::forward<BackendSelector>(selector), operation, values_span, right_span, matrix_span);
      left = adopt_svd_factor<factor_type>(std::forward<MatrixTensor>(matrix), rows, rank, stage->nonunit_stride);
    }
    else
    {
      left = make_svd_factor<factor_type>(rows, left_cols);
      auto normal_operation = svd_op{.left = options.left, .right = options.right};
      auto left_span = left.mdspan();
      dispatch_kernel(std::forward<BackendSelector>(selector), normal_operation, values_span, left_span, right_span,
                      matrix_span);
    }
  }
  else if (options.right == SvdVectorExtent::Reduced)
  {
    left = make_svd_factor<factor_type>(rows, left_cols);
    auto operation = svd_op{.left = options.left, .right = options.right, .overwrite = SvdOverwrite::Right};
    auto values_span = values.mdspan();
    auto left_span = left.mdspan();
    auto matrix_span = matrix.mdspan();
    if constexpr (requires { dispatch_kernel(selector, operation, values_span, left_span, matrix_span); })
    {
      dispatch_kernel(std::forward<BackendSelector>(selector), operation, values_span, left_span, matrix_span);
      right = adopt_svd_factor<factor_type>(std::forward<MatrixTensor>(matrix), rank, cols, stage->nonunit_stride);
    }
    else
    {
      right = make_svd_factor<factor_type>(right_rows, cols);
      auto normal_operation = svd_op{.left = options.left, .right = options.right};
      auto right_span = right.mdspan();
      dispatch_kernel(std::forward<BackendSelector>(selector), normal_operation, values_span, left_span, right_span,
                      matrix_span);
    }
  }
  else
  {
    left = make_svd_factor<factor_type>(rows, left_cols);
    right = make_svd_factor<factor_type>(right_rows, cols);
    auto operation = svd_op{.left = options.left, .right = options.right};
    auto values_span = values.mdspan();
    auto left_span = left.mdspan();
    auto right_span = right.mdspan();
    auto matrix_span = matrix.mdspan();
    dispatch_kernel(std::forward<BackendSelector>(selector), operation, values_span, left_span, right_span,
                    matrix_span);
  }

  return SvdResult<factor_type, value_type, factor_type>{.left_singular_vectors = std::move(left),
                                                         .singular_values = std::move(values),
                                                         .right_singular_vectors_adjoint = std::move(right)};
}
} // namespace detail

/// \brief Preserve a matrix and return its exact singular values through an explicit selector.
template <class BackendSelector, uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] auto singular_values(BackendSelector&& selector, MatrixTensor const& matrix)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  return detail::singular_values_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work));
}

/// \brief Preserve a matrix and return its exact singular values.
template <uni20::RankedDeviceTensorView<2> MatrixTensor> [[nodiscard]] auto singular_values(MatrixTensor const& matrix)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  auto selector = detail::select_singular_values_backend<decltype(matrix_work)>();
  return detail::singular_values_from_work_matrix(std::move(selector), std::move(matrix_work));
}

/// \brief Consume an owning matrix and return its exact singular values through an explicit selector.
/// \details The input allocation is used directly as destructive workspace
///          when its layout is LAPACK-compatible.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto singular_values(BackendSelector&& selector, MatrixTensor&& matrix)
{
  return detail::singular_values_from_consumed_matrix(std::forward<BackendSelector>(selector),
                                                      std::forward<MatrixTensor>(matrix));
}

/// \brief Consume an owning matrix and return its exact singular values.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto singular_values(MatrixTensor&& matrix)
{
  using work_type = std::conditional_t<detail::can_transfer_svd_storage<MatrixTensor>(),
                                       std::remove_cvref_t<MatrixTensor>, detail::svd_matrix_tensor_t<MatrixTensor>>;
  auto selector = detail::select_singular_values_backend<work_type>();
  return detail::singular_values_from_consumed_matrix(std::move(selector), std::forward<MatrixTensor>(matrix));
}

/// \brief Preserve a matrix and return exact left singular vectors and singular values.
template <class BackendSelector, uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] auto svd_left(BackendSelector&& selector, MatrixTensor const& matrix,
                            SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  return detail::svd_left_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work), extent);
}

/// \brief Preserve a matrix and return exact left singular vectors and singular values.
template <uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] auto svd_left(MatrixTensor const& matrix, SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  auto selector = detail::select_svd_left_backend<decltype(matrix_work)>(extent);
  return detail::svd_left_from_work_matrix(std::move(selector), std::move(matrix_work), extent);
}

/// \brief Consume an owning matrix and return exact left singular vectors and singular values.
/// \details A reduced left factor may adopt the input allocation through
///          LAPACK `JOBU='O'`. A full factor is allocated separately.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto svd_left(BackendSelector&& selector, MatrixTensor&& matrix,
                            SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  if constexpr (detail::can_transfer_svd_storage<MatrixTensor>())
  {
    return detail::svd_left_from_transferable_matrix(std::forward<BackendSelector>(selector),
                                                     std::forward<MatrixTensor>(matrix), extent);
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    return detail::svd_left_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work), extent);
  }
}

/// \brief Consume an owning matrix and return exact left singular vectors and singular values.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto svd_left(MatrixTensor&& matrix, SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  if constexpr (detail::can_transfer_svd_storage<MatrixTensor>())
  {
    using factor_type = detail::svd_reuse_factor_t<MatrixTensor>;
    using value_type = detail::svd_value_tensor_t<MatrixTensor>;
    auto operation = svd_left_op{.left = extent};
    auto selector = select_backend_for<value_type, factor_type, std::remove_cvref_t<MatrixTensor>>(operation);
    return detail::svd_left_from_transferable_matrix(std::move(selector), std::forward<MatrixTensor>(matrix), extent);
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    auto selector = detail::select_svd_left_backend<decltype(matrix_work)>(extent);
    return detail::svd_left_from_work_matrix(std::move(selector), std::move(matrix_work), extent);
  }
}

/// \brief Preserve a matrix and return exact singular values and right singular vectors.
template <class BackendSelector, uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] auto svd_right(BackendSelector&& selector, MatrixTensor const& matrix,
                             SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  return detail::svd_right_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work), extent);
}

/// \brief Preserve a matrix and return exact singular values and right singular vectors.
template <uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] auto svd_right(MatrixTensor const& matrix, SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  auto selector = detail::select_svd_right_backend<decltype(matrix_work)>(extent);
  return detail::svd_right_from_work_matrix(std::move(selector), std::move(matrix_work), extent);
}

/// \brief Consume an owning matrix and return exact singular values and right singular vectors.
/// \details A reduced right factor may adopt the input allocation through
///          LAPACK `JOBVT='O'`. A full factor is allocated separately.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto svd_right(BackendSelector&& selector, MatrixTensor&& matrix,
                             SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  if constexpr (detail::can_transfer_svd_storage<MatrixTensor>())
  {
    return detail::svd_right_from_transferable_matrix(std::forward<BackendSelector>(selector),
                                                      std::forward<MatrixTensor>(matrix), extent);
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    return detail::svd_right_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work), extent);
  }
}

/// \brief Consume an owning matrix and return exact singular values and right singular vectors.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto svd_right(MatrixTensor&& matrix, SvdVectorExtent extent = SvdVectorExtent::Reduced)
{
  if constexpr (detail::can_transfer_svd_storage<MatrixTensor>())
  {
    using factor_type = detail::svd_reuse_factor_t<MatrixTensor>;
    using value_type = detail::svd_value_tensor_t<MatrixTensor>;
    auto operation = svd_right_op{.right = extent};
    auto selector = select_backend_for<value_type, factor_type, std::remove_cvref_t<MatrixTensor>>(operation);
    return detail::svd_right_from_transferable_matrix(std::move(selector), std::forward<MatrixTensor>(matrix), extent);
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    auto selector = detail::select_svd_right_backend<decltype(matrix_work)>(extent);
    return detail::svd_right_from_work_matrix(std::move(selector), std::move(matrix_work), extent);
  }
}

/// \brief Preserve a matrix and return its exact singular value decomposition through an explicit selector.
/// \details The default reduced decomposition returns `U` with shape `m x k`,
///          `s` with length `k`, and `Vh` with shape `k x n`, where
///          `k = min(m,n)`. Full left and right extents are independently
///          selectable.
template <class BackendSelector, uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] auto svd(BackendSelector&& selector, MatrixTensor const& matrix, SvdOptions options = {})
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  return detail::svd_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work), options);
}

/// \brief Preserve a matrix and return its exact singular value decomposition.
/// \details Zero inner extent returns empty reduced factors; a requested
///          unconstrained full left or right factor is the identity matrix.
template <uni20::RankedDeviceTensorView<2> MatrixTensor>
[[nodiscard]] auto svd(MatrixTensor const& matrix, SvdOptions options = {})
{
  auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  auto selector = detail::select_svd_backend<decltype(matrix_work)>(options);
  return detail::svd_from_work_matrix(std::move(selector), std::move(matrix_work), options);
}

/// \brief Consume an owning matrix and return its exact singular value decomposition.
/// \details The supplied selector controls the destructive SVD
///          dispatch. Compatible reduced factors may adopt the input
///          allocation through a LAPACK overwrite job.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto svd(BackendSelector&& selector, MatrixTensor&& matrix, SvdOptions options = {})
{
  if constexpr (detail::can_transfer_svd_storage<MatrixTensor>())
  {
    return detail::svd_from_transferable_matrix(std::forward<BackendSelector>(selector),
                                                std::forward<MatrixTensor>(matrix), options);
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    return detail::svd_from_work_matrix(std::forward<BackendSelector>(selector), std::move(matrix_work), options);
  }
}

/// \brief Consume an owning matrix and return its exact singular value decomposition.
/// \details A directly addressable column-major input becomes LAPACK work
///          storage. When either requested factor is reduced, LAPACK may write
///          that factor into the input allocation and Uni20 adopts it under a
///          strided owning descriptor that preserves the leading dimension and
///          unused storage tail. Other inputs are materialized. Passing an
///          rvalue grants permission to consume storage but does not guarantee
///          allocation reuse.
/// \warning Existing views into transferred input storage are invalidated by
///          the ordinary C++ moved-from owner rules.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto svd(MatrixTensor&& matrix, SvdOptions options = {})
{
  if constexpr (detail::can_transfer_svd_storage<MatrixTensor>())
  {
    using factor_type = detail::svd_reuse_factor_t<MatrixTensor>;
    using value_type = detail::svd_value_tensor_t<MatrixTensor>;
    auto operation = svd_op{.left = options.left, .right = options.right};
    auto selector =
        select_backend_for<value_type, factor_type, factor_type, std::remove_cvref_t<MatrixTensor>>(operation);
    return detail::svd_from_transferable_matrix(std::move(selector), std::forward<MatrixTensor>(matrix), options);
  }
  else
  {
    auto matrix_work = uni20::make_tensor<uni20::ColumnMajor>(matrix);
    auto selector = detail::select_svd_backend<decltype(matrix_work)>(options);
    return detail::svd_from_work_matrix(std::move(selector), std::move(matrix_work), options);
  }
}

} // namespace uni20::linalg
