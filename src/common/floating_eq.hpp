#pragma once

#include "core/scalar_traits.hpp"
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace uni20::check
{

/// \brief Return the signed distance in ULPs between two IEEE-754 values.
/// \details
///  * Positive if `b > a`, negative if `a > b`.
///  * Returns 0 if `a == b` (including +0 vs -0).
///  * Returns `max<long long>` if either value is NaN or if infinities differ.
/// \pre Requires `std::numeric_limits<T>::is_iec559 == true`.
template <std::floating_point T> inline long long float_distance(T a, T b)
{
  static_assert(std::numeric_limits<T>::is_iec559, "float_distance requires IEEE-754 floating point");

  using UInt =
      std::conditional_t<sizeof(T) == 4, std::uint32_t, std::conditional_t<sizeof(T) == 8, std::uint64_t, __uint128_t>>;

  if (std::isnan(a) || std::isnan(b))
  {
    return std::numeric_limits<long long>::max();
  }
  if (std::isinf(a) || std::isinf(b))
  {
    return (a == b) ? 0 : std::numeric_limits<long long>::max();
  }
  if (a == b)
  {
    return 0; // handles +0 == -0
  }

  auto ai = std::bit_cast<UInt>(a);
  auto bi = std::bit_cast<UInt>(b);

  // Map negative floats into lexicographically ordered space
  if (ai >> (sizeof(UInt) * 8 - 1)) ai = ~ai + 1;
  if (bi >> (sizeof(UInt) * 8 - 1)) bi = ~bi + 1;

  return static_cast<long long>(bi) - static_cast<long long>(ai);
}

/// \brief Compare floating point or complex values within a given ULP tolerance.
///
/// \details
/// This template is specialized for `float`, `double`, and `std::complex<T>`
/// where `T` is one of those types.  It can be extended by specializing
/// `FloatingULP<T>` for other scalar-like types such as bfloat16.
///
/// \note
/// Default tolerance is 4 ULPs, chosen to match GoogleTest’s
/// `ASSERT_FLOAT_EQ` semantics.
///
/// \tparam T The type to compare. Must be floating point or supported specialization.
template <typename T> struct FloatingULP;

template <typename T>
  requires std::floating_point<T>
struct FloatingULP<T>
{
    static bool eq(T a, T b, unsigned max_ulps = 4)
    {
      auto dist = float_distance(a, b);
      return dist != std::numeric_limits<long long>::max() && std::llabs(dist) <= static_cast<long long>(max_ulps);
    }
};

/// \brief ULP comparator for complex numbers over floating point.
template <typename T>
  requires uni20::is_complex<T>
struct FloatingULP<T>
{
    static bool eq(T const& a, T const& b, unsigned max_ulps = 4)
    {
      using S = uni20::make_real_t<T>;
      return FloatingULP<S>::eq(a.real(), b.real(), max_ulps) && FloatingULP<S>::eq(a.imag(), b.imag(), max_ulps);
    }
};

/// \brief Concept satisfied if T can be compared in ULPs.
template <typename T>
concept is_ulp_comparable = requires(T a, T b) {
                              {
                                FloatingULP<T>::eq(a, b)
                                } -> std::same_as<bool>;
                            };

} // namespace uni20::check
