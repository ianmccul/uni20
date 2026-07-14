#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>

namespace uni20::check
{

namespace detail
{

template <typename T>
inline constexpr bool has_ieee_binary_interchange_layout_v =
    (sizeof(T) == 4 && uni20::numeric_limits<T>::digits == 24 && uni20::numeric_limits<T>::max_exponent == 128) ||
    (sizeof(T) == 8 && uni20::numeric_limits<T>::digits == 53 && uni20::numeric_limits<T>::max_exponent == 1024) ||
    (sizeof(T) == 16 && uni20::numeric_limits<T>::digits == 113 && uni20::numeric_limits<T>::max_exponent == 16384);

} // namespace detail

/// \brief Real scalar stored in an IEEE binary32, binary64, or binary128 interchange representation.
/// \details This excludes padded extended-precision representations such as
///          x87 80-bit `long double`, whose object representation contains
///          non-value bits that cannot participate in a portable ULP ordering.
/// \tparam T Real scalar type to inspect.
template <typename T>
concept IeeeBinaryReal =
    uni20::Real<std::remove_cvref_t<T>> && uni20::numeric_limits<std::remove_cvref_t<T>>::is_iec559 &&
    uni20::numeric_limits<std::remove_cvref_t<T>>::radix == 2 && std::is_trivially_copyable_v<std::remove_cvref_t<T>> &&
    detail::has_ieee_binary_interchange_layout_v<std::remove_cvref_t<T>>;

namespace detail
{

template <IeeeBinaryReal T>
using ulp_uint_t =
    std::conditional_t<sizeof(T) == 4, std::uint32_t, std::conditional_t<sizeof(T) == 8, std::uint64_t, __uint128_t>>;

template <IeeeBinaryReal T> constexpr ulp_uint_t<T> ordered_ulp_key(T value)
{
  using UInt = ulp_uint_t<T>;
  constexpr UInt SignBit = UInt{1} << (sizeof(UInt) * 8 - 1);
  UInt const bits = std::bit_cast<UInt>(value);

  // Collapse the two signed-zero encodings and order negative values before
  // positive values while preserving adjacency between finite values.
  return (bits & SignBit) != 0 ? ~bits + UInt{1} : bits | SignBit;
}

template <IeeeBinaryReal T> constexpr ulp_uint_t<T> ulp_distance_magnitude(T a, T b)
{
  auto const ai = ordered_ulp_key(a);
  auto const bi = ordered_ulp_key(b);
  return ai < bi ? bi - ai : ai - bi;
}

} // namespace detail

/// \brief Return the signed distance in ULPs between two IEEE-754 values.
/// \details
///  * Positive if `b > a`, negative if `a > b`.
///  * Returns 0 if `a == b` (including +0 vs -0).
///  * Returns `max<long long>` if either value is NaN or if infinities differ.
///  * Saturates when the finite distance exceeds the signed diagnostic range.
/// \tparam T IEEE binary interchange scalar type.
template <IeeeBinaryReal T> inline long long float_distance(T a, T b)
{
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

  using UInt = detail::ulp_uint_t<T>;
  auto const ai = detail::ordered_ulp_key(a);
  auto const bi = detail::ordered_ulp_key(b);
  UInt const magnitude = ai < bi ? bi - ai : ai - bi;
  constexpr auto MaxDistance = std::numeric_limits<long long>::max();
  if (magnitude >= static_cast<UInt>(MaxDistance))
  {
    return bi > ai ? MaxDistance : -MaxDistance;
  }

  auto const signed_magnitude = static_cast<long long>(magnitude);
  return bi > ai ? signed_magnitude : -signed_magnitude;
}

/// \brief Compare floating point or complex values within a given ULP tolerance.
///
/// \details
/// This template is specialized for IEEE binary32, binary64, and binary128
/// Uni20 real scalars and their `uni20::complex<T>` counterparts. It can be
/// extended by specializing `FloatingULP<T>` for other scalar-like types.
///
/// \note
/// Default tolerance is 4 ULPs, chosen to match GoogleTest’s
/// `ASSERT_FLOAT_EQ` semantics.
///
/// \tparam T The type to compare. Must be floating point or supported specialization.
template <typename T> struct FloatingULP;

template <typename T>
  requires IeeeBinaryReal<T>
struct FloatingULP<T>
{
    static bool eq(T a, T b, std::int64_t max_ulps = 4)
    {
      if (max_ulps < 0 || std::isnan(a) || std::isnan(b))
      {
        return false;
      }
      if (a == b)
      {
        return true;
      }
      if (std::isinf(a) || std::isinf(b))
      {
        return false;
      }

      using UInt = detail::ulp_uint_t<T>;
      return detail::ulp_distance_magnitude(a, b) <= static_cast<UInt>(max_ulps);
    }
};

/// \brief ULP comparator for complex numbers over floating point.
template <typename T>
  requires uni20::Complex<T> && IeeeBinaryReal<uni20::make_real_t<T>>
struct FloatingULP<T>
{
    static bool eq(T const& a, T const& b, std::int64_t max_ulps = 4)
    {
      using S = uni20::make_real_t<T>;
      return FloatingULP<S>::eq(a.real(), b.real(), max_ulps) && FloatingULP<S>::eq(a.imag(), b.imag(), max_ulps);
    }
};

/// \brief Concept satisfied if `T` can be compared in ULPs.
template <typename T>
concept UlpComparable = requires(T a, T b) {
  { FloatingULP<T>::eq(a, b) } -> std::same_as<bool>;
};

/// \brief Return the absolute diagnostic ULP distance between two real scalars.
template <IeeeBinaryReal T> inline long long float_abs_distance(T a, T b)
{
  auto dist = float_distance(a, b);
  return (dist == std::numeric_limits<long long>::max()) ? std::numeric_limits<long long>::max() : std::llabs(dist);
}

template <uni20::Complex T>
  requires IeeeBinaryReal<uni20::make_real_t<T>>
inline long long float_abs_distance(T a, T b)
{
  auto dr = float_abs_distance(a.real(), b.real());
  auto di = float_abs_distance(a.imag(), b.imag());
  return std::max(dr, di);
}

} // namespace uni20::check
