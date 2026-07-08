#pragma once

/**
 * \file mdspan_matrix.hpp
 * \ingroup linalg
 * \brief Lower mdspan matrix views to provider-ready BLAS operands.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/blas_matrix.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>

#include <optional>
#include <type_traits>
#include <utility>

namespace uni20::linalg::blas
{

static_assert(std::is_signed_v<blas_int>, "BLAS/LAPACK integer ABI type must be signed");

/// \brief Logical mdspan matrix layout before provider-specific BLAS lowering.
template <class Scalar, class Handle> struct MdspanMatrixStage
{
    Handle data{};
    blas_int extent0 = 0;
    blas_int extent1 = 0;
    /// \brief Mdspan stride orthogonal to the unit-stride axis.
    blas_int nonunit_stride = 0;
    int unit_stride_axis = 0;
    bool needs_conjugation = false;
};

namespace detail
{
inline constexpr blas_int invalid_blas_int = blas_int{-1};

constexpr bool is_valid_blas_int(blas_int value) noexcept { return value >= 0; }

template <typename Value> constexpr blas_int try_blas_int(Value value) noexcept
{
  if (!std::in_range<blas_int>(value))
  {
    return invalid_blas_int;
  }

  auto const converted = static_cast<blas_int>(value);
  if (converted < 0)
  {
    return invalid_blas_int;
  }
  return converted;
}

inline blas_int minimum_leading_dimension(blas_int rows) { return rows > 1 ? rows : 1; }

inline std::optional<blas_int> normalized_nonunit_stride(blas_int nonunit_stride, blas_int provider_rows,
                                                         blas_int provider_cols)
{
  blas_int const minimum = minimum_leading_dimension(provider_rows);
  if (nonunit_stride >= minimum)
  {
    return nonunit_stride;
  }

  // If there is only one provider column, the stride between columns is
  // unspecified by the logical view. Choose the BLAS-compatible representative.
  if (provider_cols <= 1)
  {
    return minimum;
  }

  return std::nullopt;
}

template <class Accessor>
struct is_blas_direct_read_accessor : std::bool_constant<uni20::is_default_accessor_v<Accessor>>
{};

template <uni20::AccessorPolicy Accessor>
struct is_blas_direct_read_accessor<uni20::conjugated_accessor<Accessor>>
    : std::bool_constant<uni20::is_default_accessor_v<Accessor>>
{};

template <class Accessor>
inline constexpr bool is_blas_direct_read_accessor_v =
    is_blas_direct_read_accessor<std::remove_cvref_t<Accessor>>::value;

template <class Mdspan>
concept blas_readable_mdspan =
    uni20::RankedStridedMdspan<Mdspan, 2> &&
    uni20::BlasScalar<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>> &&
    is_blas_direct_read_accessor_v<typename std::remove_cvref_t<Mdspan>::accessor_type>;

template <class Mdspan>
concept blas_writable_mdspan =
    uni20::MutableRankedStridedMdspan<Mdspan, 2> &&
    uni20::BlasScalar<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>> &&
    uni20::DefaultAccessorMdspan<Mdspan>;

template <class Mdspan>
concept lapack_writable_mdspan =
    uni20::MutableRankedStridedMdspan<Mdspan, 2> &&
    uni20::LapackScalar<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>> &&
    uni20::DefaultAccessorMdspan<Mdspan>;
} // namespace detail

/// \brief Build an mdspan matrix staging descriptor when direct BLAS lowering is possible.
template <uni20::RankedStridedMdspan<2> Mdspan>
auto try_mdspan_matrix_stage(Mdspan const& span)
    -> std::optional<
        MdspanMatrixStage<std::remove_cv_t<typename Mdspan::element_type>, typename Mdspan::data_handle_type>>
{
  using scalar_type = std::remove_cv_t<typename Mdspan::element_type>;
  using handle_type = typename Mdspan::data_handle_type;

  auto const& mapping = span.mapping();
  if (!mapping.is_unique())
  {
    return std::nullopt;
  }

  auto const extent0 = detail::try_blas_int(span.extent(0));
  auto const extent1 = detail::try_blas_int(span.extent(1));
  auto const stride0 = detail::try_blas_int(mapping.stride(0));
  auto const stride1 = detail::try_blas_int(mapping.stride(1));
  if (!detail::is_valid_blas_int(extent0) || !detail::is_valid_blas_int(extent1) ||
      !detail::is_valid_blas_int(stride0) || !detail::is_valid_blas_int(stride1))
  {
    return std::nullopt;
  }

  int unit_stride_axis = -1;
  blas_int nonunit_stride = 0;
  blas_int provider_rows = 0;
  blas_int provider_cols = 0;

  if (stride0 == 1)
  {
    unit_stride_axis = 0;
    nonunit_stride = stride1;
    provider_rows = extent0;
    provider_cols = extent1;
  }
  else if (stride1 == 1)
  {
    unit_stride_axis = 1;
    nonunit_stride = stride0;
    provider_rows = extent1;
    provider_cols = extent0;
  }
  else
  {
    return std::nullopt;
  }

  auto const normalized_stride = detail::normalized_nonunit_stride(nonunit_stride, provider_rows, provider_cols);
  if (!normalized_stride)
  {
    return std::nullopt;
  }

  return MdspanMatrixStage<scalar_type, handle_type>{.data = span.data_handle(),
                                                     .extent0 = extent0,
                                                     .extent1 = extent1,
                                                     .nonunit_stride = *normalized_stride,
                                                     .unit_stride_axis = unit_stride_axis,
                                                     .needs_conjugation = mdspan_needs_conjugation_v<Mdspan>};
}

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
template <detail::blas_writable_mdspan Mdspan>
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
template <detail::blas_readable_mdspan Mdspan>
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
template <detail::lapack_writable_mdspan Mdspan>
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
