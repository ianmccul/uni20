#pragma once

/**
 * \file blas_int.hpp
 * \ingroup backend_blas
 * \brief Safe conversion helpers for the configured signed BLAS integer ABI.
 */

#include <uni20/config.hpp>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20::blas
{

static_assert(std::is_signed_v<blas_int>, "BLAS/LAPACK integer ABI type must be signed");

inline constexpr blas_int invalid_blas_int = blas_int{-1};

/// \brief Return whether a non-throwing BLAS integer conversion succeeded.
constexpr bool is_valid_blas_int(blas_int value) noexcept { return value >= 0; }

/// \brief Convert a non-negative integer to `blas_int`, or return the negative invalid sentinel.
template <std::integral Value> constexpr blas_int try_blas_int(Value value) noexcept
{
  if (!std::in_range<blas_int>(value))
  {
    return invalid_blas_int;
  }

  auto const converted = static_cast<blas_int>(value);
  return converted >= 0 ? converted : invalid_blas_int;
}

/// \brief Convert a dense dimension to `blas_int` or throw on overflow.
inline blas_int checked_blas_int(std::size_t value)
{
  blas_int const converted = try_blas_int(value);
  if (!is_valid_blas_int(converted))
  {
    throw std::overflow_error("dense dimension does not fit BLAS integer type");
  }
  return converted;
}

} // namespace uni20::blas
