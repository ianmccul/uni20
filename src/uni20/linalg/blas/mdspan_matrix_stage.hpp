#pragma once

/**
 * \file mdspan_matrix_stage.hpp
 * \ingroup linalg
 * \brief Mdspan-axis staging descriptors for BLAS-compatible matrix lowering.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <limits>
#include <optional>
#include <type_traits>

namespace uni20::linalg::blas
{

/// \brief Trait used by mdspan accessors that conjugate values on read.
template <class Accessor> struct accessor_applies_conjugation : std::false_type
{};

/// \brief True when an mdspan accessor presents conjugated values.
template <class Accessor>
inline constexpr bool accessor_applies_conjugation_v =
    accessor_applies_conjugation<std::remove_cvref_t<Accessor>>::value;

/// \brief True when the mdspan's accessor presents conjugated values.
template <class Mdspan>
inline constexpr bool mdspan_needs_conjugation_v =
    accessor_applies_conjugation_v<typename std::remove_cvref_t<Mdspan>::accessor_type>;

/// \brief Logical mdspan matrix layout before provider-specific BLAS lowering.
template <class Scalar, class Handle> struct MdspanMatrixStage
{
    Handle data{};
    blas_int extent0 = 0;
    blas_int extent1 = 0;
    blas_int nonunit_stride = 0;
    int unit_stride_axis = 0;
    bool needs_conjugation = false;
};

namespace detail
{
template <typename Value> std::optional<blas_int> try_blas_int(Value value)
{
  if constexpr (std::is_signed_v<Value>)
  {
    if (value < Value{})
    {
      return std::nullopt;
    }
  }

  auto constexpr max_value = static_cast<unsigned long long>(std::numeric_limits<blas_int>::max());
  auto const unsigned_value = static_cast<unsigned long long>(value);
  if (unsigned_value > max_value)
  {
    return std::nullopt;
  }
  return static_cast<blas_int>(value);
}

inline bool leading_dimension_is_valid(blas_int leading_dimension, blas_int rows)
{
  return leading_dimension >= (rows > 1 ? rows : 1);
}
} // namespace detail

/// \brief Build an mdspan matrix staging descriptor when direct BLAS lowering is possible.
template <uni20::StridedMdspan Mdspan>
auto try_mdspan_matrix_stage(Mdspan const& span)
    -> std::optional<
        MdspanMatrixStage<std::remove_cv_t<typename Mdspan::element_type>, typename Mdspan::data_handle_type>>
{
  static_assert(Mdspan::rank() == 2, "BLAS matrix staging requires a rank-2 mdspan");

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
  if (!extent0 || !extent1 || !stride0 || !stride1)
  {
    return std::nullopt;
  }

  int unit_stride_axis = -1;
  blas_int nonunit_stride = 0;
  blas_int provider_rows = 0;

  if (*stride0 == 1)
  {
    unit_stride_axis = 0;
    nonunit_stride = *stride1;
    provider_rows = *extent0;
  }
  else if (*stride1 == 1)
  {
    unit_stride_axis = 1;
    nonunit_stride = *stride0;
    provider_rows = *extent1;
  }
  else
  {
    return std::nullopt;
  }

  if (!detail::leading_dimension_is_valid(nonunit_stride, provider_rows))
  {
    return std::nullopt;
  }

  return MdspanMatrixStage<scalar_type, handle_type>{.data = span.data_handle(),
                                                     .extent0 = *extent0,
                                                     .extent1 = *extent1,
                                                     .nonunit_stride = nonunit_stride,
                                                     .unit_stride_axis = unit_stride_axis,
                                                     .needs_conjugation = mdspan_needs_conjugation_v<Mdspan>};
}

} // namespace uni20::linalg::blas
