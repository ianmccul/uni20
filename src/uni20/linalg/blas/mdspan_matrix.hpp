#pragma once

/**
 * \file mdspan_matrix.hpp
 * \ingroup linalg
 * \brief Lower mdspan matrix views to provider-ready BLAS operands.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/backend/blas/blas_int.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/blas_matrix.hpp>
#include <uni20/linalg/blas/mdspan_access.hpp>

#include <optional>
#include <type_traits>
#include <utility>

namespace uni20::linalg::blas
{

/// \brief Handle-independent matrix metadata used for BLAS eligibility checks.
struct MdspanMatrixMetadata
{
    blas_int extent0 = 0;
    blas_int extent1 = 0;
    /// \brief Mdspan stride orthogonal to the unit-stride axis.
    blas_int nonunit_stride = 0;
    int unit_stride_axis = 0;
    bool needs_conjugation = false;
};

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
template <class Span, class = void> struct SpanData
{
    using type = typename std::remove_cvref_t<Span>::data_handle_type;

    [[nodiscard]] static constexpr auto get(Span const& span) -> type { return span.data_handle(); }
};

template <class Span> struct SpanData<Span, std::void_t<typename std::remove_cvref_t<Span>::data_descriptor_type>>
{
    using type = typename std::remove_cvref_t<Span>::data_descriptor_type;

    [[nodiscard]] static constexpr auto get(Span const& span) -> type { return span.data_descriptor(); }
};

template <class Span> using span_data_t = typename SpanData<Span>::type;

template <class Span> [[nodiscard]] constexpr auto span_data(Span const& span) -> span_data_t<Span>
{
  return SpanData<Span>::get(span);
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

  // If there are no provider rows or at most one provider column, the stride
  // between columns does not address two logical elements. Choose the
  // BLAS-compatible representative.
  if (provider_rows == 0 || provider_cols <= 1)
  {
    return minimum;
  }

  return std::nullopt;
}

template <class Mdspan>
concept lapack_writable_mdspan =
    uni20::MutableRankedStridedMdspanLike<Mdspan, 2> &&
    uni20::LapackScalar<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>> &&
    uni20::DefaultAccessorMdspanLike<Mdspan>;
} // namespace detail

/// \brief Inspect matrix layout and accessor semantics without resolving a data handle.
template <uni20::RankedStridedMdspecLike<2> Mdspan>
auto try_mdspan_matrix_metadata(Mdspan const& span) -> std::optional<MdspanMatrixMetadata>
{
  auto const& mapping = span.mapping();
  if (!mapping.is_unique())
  {
    return std::nullopt;
  }

  auto const extent0 = uni20::blas::try_blas_int(span.extent(0));
  auto const extent1 = uni20::blas::try_blas_int(span.extent(1));
  if (!uni20::blas::is_valid_blas_int(extent0) || !uni20::blas::is_valid_blas_int(extent1))
  {
    return std::nullopt;
  }

  // Empty and singleton axes do not observe their stride. Canonicalize them
  // before checking the provider integer range or selecting a unit-stride axis.
  blas_int stride0 = 1;
  if (extent0 > 1)
  {
    stride0 = uni20::blas::try_blas_int(mapping.stride(0));
    if (!uni20::blas::is_valid_blas_int(stride0)) return std::nullopt;
  }
  blas_int stride1 = 1;
  if (extent1 > 1)
  {
    stride1 = uni20::blas::try_blas_int(mapping.stride(1));
    if (!uni20::blas::is_valid_blas_int(stride1)) return std::nullopt;
  }

  int unit_stride_axis = -1;
  blas_int nonunit_stride = 0;
  blas_int provider_rows = 0;
  blas_int provider_cols = 0;

  // Prefer a genuinely unit-stride non-singleton axis. This preserves the
  // natural provider orientation before using a singleton as the unit axis.
  if (extent0 > 1 && stride0 == 1)
  {
    unit_stride_axis = 0;
    nonunit_stride = stride1;
    provider_rows = extent0;
    provider_cols = extent1;
  }
  else if (extent1 > 1 && stride1 == 1)
  {
    unit_stride_axis = 1;
    nonunit_stride = stride0;
    provider_rows = extent1;
    provider_cols = extent0;
  }
  else if (extent0 <= 1)
  {
    unit_stride_axis = 0;
    nonunit_stride = stride1;
    provider_rows = extent0;
    provider_cols = extent1;
  }
  else if (extent1 <= 1)
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

  return MdspanMatrixMetadata{.extent0 = extent0,
                              .extent1 = extent1,
                              .nonunit_stride = *normalized_stride,
                              .unit_stride_axis = unit_stride_axis,
                              .needs_conjugation = mdspan_needs_conjugation_v<Mdspan>};
}

namespace detail
{
/// \brief Attach a matrix data handle or descriptor after metadata validation.
template <uni20::RankedStridedMdspecLike<2> Mdspan>
auto make_mdspan_matrix_stage(Mdspan const& span, MdspanMatrixMetadata const& metadata)
    -> MdspanMatrixStage<std::remove_cv_t<typename Mdspan::element_type>, detail::span_data_t<Mdspan>>
{
  using scalar_type = std::remove_cv_t<typename Mdspan::element_type>;
  using handle_type = detail::span_data_t<Mdspan>;
  return MdspanMatrixStage<scalar_type, handle_type>{.data = detail::span_data(span),
                                                     .extent0 = metadata.extent0,
                                                     .extent1 = metadata.extent1,
                                                     .nonunit_stride = metadata.nonunit_stride,
                                                     .unit_stride_axis = metadata.unit_stride_axis,
                                                     .needs_conjugation = metadata.needs_conjugation};
}
} // namespace detail

/// \brief Build a matrix staging descriptor when direct provider lowering is possible.
template <uni20::RankedStridedMdspecLike<2> Mdspan>
auto try_mdspan_matrix_stage(Mdspan const& span)
    -> std::optional<MdspanMatrixStage<std::remove_cv_t<typename Mdspan::element_type>, detail::span_data_t<Mdspan>>>
{
  auto metadata = try_mdspan_matrix_metadata(span);
  if (!metadata) return std::nullopt;
  return detail::make_mdspan_matrix_stage(span, *metadata);
}

/// \brief Transform from the provider matrix shape to the logical mdspan shape.
constexpr MatrixTransform storage_transform(MdspanMatrixMetadata const& metadata)
{
  return metadata.unit_stride_axis == 0 ? MatrixTransform::normal : MatrixTransform::transpose;
}

/// \brief Transform from the provider matrix shape to the logical mdspan shape.
template <class Scalar, class Handle>
constexpr MatrixTransform storage_transform(MdspanMatrixStage<Scalar, Handle> const& stage)
{
  return stage.unit_stride_axis == 0 ? MatrixTransform::normal : MatrixTransform::transpose;
}

/// \brief Return the provider-ready writable operand for a staged mdspan matrix.
template <class Scalar, class Handle>
constexpr auto blas_writable_matrix(MdspanMatrixStage<Scalar, Handle> const& stage)
    -> BlasWritableMatrix<Scalar, Handle>
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
constexpr auto blas_readable_matrix(MdspanMatrixStage<Scalar, Handle> const& stage)
    -> BlasReadableMatrix<Scalar, Handle>
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
template <class Mdspan>
  requires detail::blas_writable_mdspan<Mdspan, 2>
auto try_blas_writable_matrix(Mdspan const& span) -> std::optional<
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
template <class Mdspan>
  requires detail::blas_readable_mdspan<Mdspan, 2>
auto try_blas_readable_matrix(Mdspan const& span) -> std::optional<
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
auto try_lapack_writable_matrix(Mdspan const& span) -> std::optional<
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
