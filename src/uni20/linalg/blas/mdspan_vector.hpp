#pragma once

/**
 * \file mdspan_vector.hpp
 * \ingroup linalg
 * \brief Lower rank-one mdspan views to provider-ready BLAS operands.
 */

#include <uni20/backend/blas/blas_int.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/blas/blas_vector.hpp>
#include <uni20/linalg/blas/mdspan_access.hpp>

#include <optional>
#include <type_traits>

namespace uni20::linalg::blas
{

/// \brief Logical rank-one mdspan layout before provider-specific BLAS lowering.
template <class Scalar, class Handle> struct MdspanVectorStage
{
    Handle data{};
    blas_int extent = 0;
    blas_int increment = 1;
    bool needs_conjugation = false;
};

/// \brief Build an mdspan vector staging descriptor when direct BLAS lowering is possible.
template <uni20::RankedStridedMdspan<1> Mdspan>
auto try_mdspan_vector_stage(Mdspan const& span)
    -> std::optional<
        MdspanVectorStage<std::remove_cv_t<typename Mdspan::element_type>, typename Mdspan::data_handle_type>>
{
  using scalar_type = std::remove_cv_t<typename Mdspan::element_type>;
  using handle_type = typename Mdspan::data_handle_type;

  auto const& mapping = span.mapping();
  if (!mapping.is_unique())
  {
    return std::nullopt;
  }

  blas_int const extent = uni20::blas::try_blas_int(span.extent(0));
  if (!uni20::blas::is_valid_blas_int(extent))
  {
    return std::nullopt;
  }

  // Empty and singleton mappings do not observe a step between elements, so
  // choose the canonical BLAS increment regardless of the mapping's stride.
  blas_int increment = 1;
  if (extent > 1)
  {
    increment = uni20::blas::try_blas_int(mapping.stride(0));
    if (!uni20::blas::is_valid_blas_int(increment) || increment == 0)
    {
      return std::nullopt;
    }
  }

  return MdspanVectorStage<scalar_type, handle_type>{.data = span.data_handle(),
                                                     .extent = extent,
                                                     .increment = increment,
                                                     .needs_conjugation = mdspan_needs_conjugation_v<Mdspan>};
}

/// \brief Return the provider-ready writable operand for a staged mdspan vector.
/// \pre The staged accessor does not apply conjugation.
template <class Scalar, class Handle>
constexpr auto
blas_writable_vector(MdspanVectorStage<Scalar, Handle> const& stage) -> BlasWritableVector<Scalar, Handle>
{
  CHECK(!stage.needs_conjugation, "writable BLAS vectors cannot discard accessor conjugation");
  return {.data = stage.data, .size = stage.extent, .increment = stage.increment};
}

/// \brief Return the provider-ready readable storage operand for a staged mdspan vector.
/// \pre The staged accessor does not apply conjugation, which portable BLAS
///      vector operands cannot represent.
template <class Scalar, class Handle>
constexpr auto
blas_readable_vector(MdspanVectorStage<Scalar, Handle> const& stage) -> BlasReadableVector<Scalar, Handle>
{
  CHECK(!stage.needs_conjugation, "readable BLAS vectors cannot discard accessor conjugation");
  return {.data = stage.data, .size = stage.extent, .increment = stage.increment};
}

/// \brief Try to lower a mutable rank-one mdspan directly to a writable provider operand.
template <class Mdspan>
  requires detail::blas_writable_mdspan<Mdspan, 1>
auto try_blas_writable_vector(Mdspan const& span)
    -> std::optional<
        BlasWritableVector<std::remove_cv_t<typename Mdspan::element_type>, typename Mdspan::data_handle_type>>
{
  auto stage = try_mdspan_vector_stage(span);
  if (!stage || stage->needs_conjugation)
  {
    return std::nullopt;
  }
  return blas_writable_vector(*stage);
}

/// \brief Try to lower a readable rank-one mdspan directly to a provider operand.
template <class Mdspan>
  requires detail::blas_readable_mdspan<Mdspan, 1>
auto try_blas_readable_vector(Mdspan const& span)
    -> std::optional<
        BlasReadableVector<std::remove_cv_t<typename Mdspan::element_type>, typename Mdspan::data_handle_type>>
{
  auto stage = try_mdspan_vector_stage(span);
  if (!stage || stage->needs_conjugation)
  {
    return std::nullopt;
  }
  return blas_readable_vector(*stage);
}

} // namespace uni20::linalg::blas
