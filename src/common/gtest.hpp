#pramga once

/// \file gtest_floating_eq.hpp
/// \brief GoogleTest integration for ULP-based floating-point comparisons.
///
/// This header defines two macros, `EXPECT_FLOATING_EQ` and `ASSERT_FLOATING_EQ`,
/// for use inside GoogleTest unit tests. They extend the standard GTest
/// floating-point comparison macros (`EXPECT_FLOAT_EQ`, `EXPECT_DOUBLE_EQ`) to:
///
/// - Work with any IEEE-754 floating point type (`float`, `double`, `long double`).
/// - Work with `std::complex<T>` where `T` is floating point.
/// - Allow explicit specification of ULP tolerance.
/// - Default to a tolerance of 4 ULPs if none is provided, matching GoogleTest.
/// - Accept arbitrary additional parameters, which are forwarded to the
///   diagnostic output on failure.
///
/// \details
/// Usage patterns:
/// \code
/// float a = 1.0f;
/// float b = std::nextafter(a, 2.0f);
///
/// // Default tolerance of 4 ULPs
/// EXPECT_FLOATING_EQ(a, b);
///
/// // Explicit tolerance of 1 ULP
/// EXPECT_FLOATING_EQ(a, b, 1);
///
/// // Explicit tolerance plus extra context
/// EXPECT_FLOATING_EQ(a, b, 2, "during normalization");
///
/// // ASSERT_ variant aborts the current test case on failure
/// ASSERT_FLOATING_EQ(a, b, 1);
/// \endcode
///
/// Failure output shows:
/// - The source file and line number
/// - The compared expressions and their evaluated values
/// - The allowed tolerance in ULPs
/// - The actual ULP distance, computed via `uni20::check::float_distance`
///
/// \note These macros are intended for unit tests only.
/// For assertions in library code, use `CHECK_FLOATING_EQ` / `PRECONDITION_FLOATING_EQ`
/// from `trace.hpp`.
///
/// \pre The compared type must satisfy `uni20::check::is_ulp_comparable`.
/// \post On failure, the macros report via GoogleTest (`ADD_FAILURE` or `FAIL`).
///
/// \see CHECK_FLOATING_EQ, PRECONDITION_FLOATING_EQ, uni20::check::float_distance

#include "floating_eq.hpp"
#include "trace.h"
#include <gtest/gtest.h>

#define EXPECT_FLOATING_EQ(a, b, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    auto va = (a);                                                                                                     \
    auto vb = (b);                                                                                                     \
    using T = std::decay_t<decltype(va)>;                                                                              \
    static_assert(::uni20::check::is_ulp_comparable<T>, "EXPECT_FLOATING_EQ requires a ULP-comparable scalar type");   \
    unsigned ulps = ::trace::detail::get_ulps(va, vb, __VA_ARGS__);                                                    \
    if (!::uni20::check::FloatingULP<T>::eq(va, vb, ulps))                                                             \
    {                                                                                                                  \
      ::testing::Message msg;                                                                                          \
      msg << "EXPECT_FLOATING_EQ failed at " << __FILE__ << ":" << __LINE__ << "\n  " #a " = " << va                   \
          << "\n  " #b " = " << vb << "\n  allowed tolerance: " << ulps << " ULPs"                                     \
          << "\n  actual distance: " << ::uni20::check::float_distance(va, vb);                                        \
      ADD_FAILURE() << msg;                                                                                            \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define ASSERT_FLOATING_EQ(a, b, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    auto va = (a);                                                                                                     \
    auto vb = (b);                                                                                                     \
    using T = std::decay_t<decltype(va)>;                                                                              \
    static_assert(::uni20::check::is_ulp_comparable<T>, "ASSERT_FLOATING_EQ requires a ULP-comparable scalar type");   \
    unsigned ulps = ::trace::detail::default_ulps(va, vb, __VA_ARGS__);                                                \
    if (!::uni20::check::FloatingULP<T>::eq(va, vb, ulps))                                                             \
    {                                                                                                                  \
      ::testing::Message msg;                                                                                          \
      msg << "ASSERT_FLOATING_EQ failed at " << __FILE__ << ":" << __LINE__ << "\n  " #a " = " << va                   \
          << "\n  " #b " = " << vb << "\n  allowed tolerance: " << ulps << " ULPs"                                     \
          << "\n  actual distance: " << ::uni20::check::float_distance(va, vb);                                        \
      FAIL() << msg;                                                                                                   \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)
