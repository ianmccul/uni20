#pragma once

#include <limits>
#include <type_traits>

namespace uni20
{

/// \brief Project-level numeric limits customization point.
/// \details The primary template inherits `std::numeric_limits<T>` so ordinary arithmetic types use the standard
///          library implementation. Uni20 scalar backends should specialize this template, not `std::numeric_limits`,
///          for extension or library scalar types whose standard-library limits are missing or incomplete.
/// \tparam T Scalar type to inspect.
/// \ingroup core_math
template <typename T> struct numeric_limits : std::numeric_limits<T>
{};

template <typename T> struct numeric_limits<T const> : numeric_limits<T>
{};

template <typename T> struct numeric_limits<T volatile> : numeric_limits<T>
{};

template <typename T> struct numeric_limits<T const volatile> : numeric_limits<T>
{};

/// \brief True when Uni20 has numeric limits for `T`.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool has_numeric_limits_v = numeric_limits<std::remove_cvref_t<T>>::is_specialized;

} // namespace uni20
