// string_util.hpp
//
// This header provides several string utility functions and type traits to aid in
// modern C++ string manipulation and conversion without relying on Boost.
//
// It includes:
//   - Operator+ overloads for concatenating std::string and std::string_view,
//     enabled only if the standard library lacks this support (__cpp_lib_string_view < 202403L).
//   - A case-insensitive string comparison function, iequals, for std::string_view.
//   - A trait (has_stream_extractor) to detect if a type T supports stream extraction (operator>>).
//   - A helper constant (dependent_false_v) for use in SFINAE and static_asserts in templates.
//   - A generic from_string template function that converts a std::string_view to a value of type T.
//       - For arithmetic types, it uses std::from_chars.
//       - For types constructible from std::string, it uses that constructor.
//       - Otherwise, if T supports operator>>, it falls back to istringstream extraction.
//       - If none of these apply, a static_assert is triggered.

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

// Only define these overloads if the standard library doesn't provide them.
#if !defined(__cpp_lib_string_view) || (__cpp_lib_string_view < 202403L)

namespace hack_string_view
{

// Overload for std::string + std::string_view.
inline std::string operator+(const std::string& lhs, std::string_view rhs) { return lhs + std::string(rhs); }

// Overload for std::string_view + std::string.
inline std::string operator+(std::string_view lhs, const std::string& rhs) { return std::string(lhs) + rhs; }

} // namespace hack_string_view

// Bring the overloads into the global namespace
using hack_string_view::operator+;

#endif

// A simple case-insensitive comparison using std::string_view.
inline bool iequals(std::string_view a, std::string_view b)
{
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](unsigned char ca, unsigned char cb) {
           return std::tolower(ca) == std::tolower(cb);
         });
}

// Trait to check if operator>> is defined for T from an std::istream.
template <typename T, typename = void> struct has_stream_extractor : std::false_type
{};

/// \cond
template <typename T>
struct has_stream_extractor<T, std::void_t<decltype(std::declval<std::istream&>() >> std::declval<T&>())>>
    : std::true_type
{};
/// \endcond

template <typename T> inline constexpr bool has_stream_extractor_v = has_stream_extractor<T>::value;

// A helper for static_assert in an uninstantiated branch.
template <typename> inline constexpr bool dependent_false_v = false;

// from_string: converts a std::string_view to a value of type T.
// 1. If T is arithmetic, use std::from_chars.
// 2. Else if T is constructible from std::string, use that constructor.
// 3. Else if T supports operator>> extraction, use an istringstream.
// 4. Otherwise, trigger a static_assert.
template <typename T> T from_string(std::string_view s)
{
  if constexpr (std::is_arithmetic_v<T>)
  {
    T result{};
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), result);
    if (ec != std::errc()) throw std::runtime_error("from_string: conversion failed for arithmetic type");
    return result;
  }
  else if constexpr (std::is_constructible_v<T, std::string>)
  {
    return T{std::string(s)};
  }
  else if constexpr (has_stream_extractor_v<T>)
  {
    std::istringstream iss{std::string(s)};
    T result{};
    if (!(iss >> result))
      throw std::runtime_error("from_string: conversion failed for non-arithmetic type using stream extraction");
    return result;
  }
  else
  {
    static_assert(dependent_false_v<T>, "from_string: No conversion available for type T");
  }
}
