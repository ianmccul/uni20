#pragma once

/**
 * \file mdspan_matrix_operand.hpp
 * \ingroup linalg
 * \brief Lower mdspan matrix stages to provider-ready BLAS operands.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/matrix_operand.hpp>
#include <uni20/linalg/blas/mdspan_matrix_stage.hpp>

#include <optional>

namespace uni20::linalg::blas
{

/// \brief Transform from the provider matrix shape to the logical mdspan shape.
template <class Scalar, class Handle>
constexpr MatrixTransform storage_transform(MdspanMatrixStage<Scalar, Handle> const& stage)
{
  return stage.unit_stride_axis == 0 ? MatrixTransform::normal : MatrixTransform::transpose;
}

/// \brief Return the provider-ready writable operand for a staged mdspan matrix.
template <class Scalar, class Handle>
constexpr auto
blas_writable_matrix(MdspanMatrixStage<Scalar, Handle> const& stage) -> BlasWritableMatrix<Scalar, Handle>
{
  if (stage.unit_stride_axis == 0)
  {
    return {
        .data = stage.data, .rows = stage.extent0, .cols = stage.extent1, .leading_dimension = stage.nonunit_stride};
  }
  return {.data = stage.data, .rows = stage.extent1, .cols = stage.extent0, .leading_dimension = stage.nonunit_stride};
}

/// \brief Return the provider-ready readable operand for a staged mdspan matrix.
template <class Scalar, class Handle>
constexpr auto
blas_readable_matrix(MdspanMatrixStage<Scalar, Handle> const& stage) -> BlasReadableMatrix<Scalar, Handle>
{
  MatrixTransform transform = storage_transform(stage);
  if (stage.needs_conjugation)
  {
    transform = compose(MatrixTransform::conjugate, transform);
  }

  auto writable = blas_writable_matrix(stage);
  return {.data = writable.data,
          .rows = writable.rows,
          .cols = writable.cols,
          .leading_dimension = writable.leading_dimension,
          .transform = transform};
}

/// \brief Try to lower a mutable mdspan directly to a writable provider operand.
template <uni20::MutableStridedMdspan Mdspan>
  requires uni20::BlasScalar<std::remove_cv_t<typename Mdspan::element_type>>
auto try_blas_writable_matrix(Mdspan const& span)
    -> std::optional<
        BlasWritableMatrix<std::remove_cv_t<typename Mdspan::element_type>, typename Mdspan::data_handle_type>>
{
  auto stage = try_mdspan_matrix_stage(span);
  if (!stage)
  {
    return std::nullopt;
  }
  return blas_writable_matrix(*stage);
}

/// \brief Try to lower an mdspan directly to a readable provider operand.
template <uni20::StridedMdspan Mdspan>
  requires uni20::BlasScalar<std::remove_cv_t<typename Mdspan::element_type>>
auto try_blas_readable_matrix(Mdspan const& span)
    -> std::optional<
        BlasReadableMatrix<std::remove_cv_t<typename Mdspan::element_type>, typename Mdspan::data_handle_type>>
{
  auto stage = try_mdspan_matrix_stage(span);
  if (!stage)
  {
    return std::nullopt;
  }
  return blas_readable_matrix(*stage);
}

/// \brief Try to lower an mdspan to the strict column-major shape expected by LAPACK.
template <uni20::MutableStridedMdspan Mdspan>
  requires uni20::LapackScalar<std::remove_cv_t<typename Mdspan::element_type>>
auto try_lapack_writable_matrix(Mdspan const& span)
    -> std::optional<
        BlasWritableMatrix<std::remove_cv_t<typename Mdspan::element_type>, typename Mdspan::data_handle_type>>
{
  auto stage = try_mdspan_matrix_stage(span);
  if (!stage || stage->unit_stride_axis != 0 || stage->needs_conjugation)
  {
    return std::nullopt;
  }
  return blas_writable_matrix(*stage);
}

} // namespace uni20::linalg::blas
