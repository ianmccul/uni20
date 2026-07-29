#pragma once

/**
 * \file matrix_product_shape.hpp
 * \ingroup linalg
 * \brief Backend-independent dense matrix-product shape validation.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/types.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/tensor/concepts.hpp>

namespace uni20::linalg::detail
{

using matrix_product_extents = stdex::dextents<uni20::index_type, 2>;

/// \brief Validate matrix-product input extents and return the output shape.
template <uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
[[nodiscard]] matrix_product_extents matrix_product_shape(LhsTensor const& lhs, RhsTensor const& rhs)
{
  ERROR_IF(lhs.extent(1) != rhs.extent(0), "matrix product inner extents do not agree", lhs.extent(1), rhs.extent(0));
  return matrix_product_extents{static_cast<uni20::index_type>(lhs.extent(0)),
                                static_cast<uni20::index_type>(rhs.extent(1))};
}

} // namespace uni20::linalg::detail
