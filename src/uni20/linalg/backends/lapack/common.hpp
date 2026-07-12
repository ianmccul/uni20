#pragma once

/**
 * \file common.hpp
 * \ingroup linalg
 * \brief Shared helpers for LAPACK operation-tag backend adapters.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/math.hpp>

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace uni20::linalg::lapack_detail
{

/// \brief Convert a provider workspace query result to the configured LAPACK integer type.
template <class Scalar> blas_int workspace_size(Scalar const& query)
{
  auto const recommended = uni20::real(query);
  using real_type = std::remove_cv_t<decltype(recommended)>;
  CHECK(uni20::isfinite(recommended), recommended);
  CHECK(recommended >= real_type{1}, recommended);

  auto const maximum = static_cast<real_type>(uni20::numeric_limits<blas_int>::max());
  if constexpr (uni20::numeric_limits<real_type>::digits < uni20::numeric_limits<blas_int>::digits)
  {
    // A low-precision real may round max(blas_int) up to the first
    // unrepresentable integer, so equality is not a safe conversion bound.
    CHECK(recommended < maximum, recommended);
  }
  else
  {
    CHECK(recommended <= maximum, recommended);
  }

  blas_int const result = static_cast<blas_int>(recommended);
  CHECK(result >= 1, result);
  CHECK(std::in_range<std::size_t>(result), result);
  return result;
}

/// \brief Multiply allocation extents without wrapping `std::size_t`.
constexpr std::optional<std::size_t> try_size_product(std::size_t left, std::size_t right) noexcept
{
  if (left != 0 && right > uni20::numeric_limits<std::size_t>::max() / left)
  {
    return std::nullopt;
  }
  return left * right;
}

} // namespace uni20::linalg::lapack_detail
