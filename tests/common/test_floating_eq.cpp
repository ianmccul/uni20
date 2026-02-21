#include "common/gtest.hpp"

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

#include <gtest/gtest-spi.h>

// --- EXPECT_FLOATING_EQ ---

TEST(FloatingEqGTest, ExpectPassesWithinTolerance)
{
  float a = 1.0f;
  float b = std::nextafter(a, 2.0f); // 1 ULP away
  EXPECT_FLOATING_EQ(a, b, 1);       // should pass
}

TEST(FloatingEqGTest, ExpectFailsOutsideTolerance)
{
  float a = 1.0f;
  float b = std::nextafter(a, 0.0f); // 1 ULP away in the opposite direction
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), "EXPECT_FLOATING_EQ failed");
}

TEST(FloatingEqGTest, ExpectDefaultToleranceIsFour)
{
  float a = 1.0f;
  float b = std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) + 4);
  EXPECT_FLOATING_EQ(a, b); // default tolerance is 4 ULP
}

TEST(FloatingEqGTest, ExpectReportsToleranceAndDistance)
{
  float a = 1.0f;
  float b = std::nextafter(a, 0.0f); // 1 ULP away
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), "allowed tolerance: 0 ULP");
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), "actual distance: 1");
}

TEST(FloatingEqGTest, ExpectRejectsNegativeTolerance)
{
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(1.0f, 1.0f, -1), "non-negative ULP tolerance");
}

// --- ASSERT_FLOATING_EQ ---

TEST(FloatingEqGTest, AssertPassesWithinTolerance)
{
  double a = 1.0;
  double b = std::nextafter(a, 2.0);
  ASSERT_FLOATING_EQ(a, b, 1); // should pass
  SUCCEED();
}

TEST(FloatingEqGTest, AssertFailsOutsideTolerance)
{
  static double const a = 1.0; // these need to be static because EXPECT_FATAL_FAILURE uses a lambda with no capture
  static double const b = std::nextafter(a, 0.0);
  EXPECT_FATAL_FAILURE(ASSERT_FLOATING_EQ(a, b, 0), "ASSERT_FLOATING_EQ failed");
}

TEST(FloatingEqGTest, AssertDefaultToleranceIsFour)
{
  double a = 1.0;
  double b = std::bit_cast<double>(std::bit_cast<std::uint64_t>(a) + 4);
  ASSERT_FLOATING_EQ(a, b); // default tolerance is 4 ULP
  SUCCEED();
}

TEST(FloatingEqGTest, AssertRejectsNegativeTolerance)
{
  EXPECT_FATAL_FAILURE(ASSERT_FLOATING_EQ(1.0, 1.0, -1), "non-negative ULP tolerance");
}

// --- Complex numbers ---

TEST(FloatingEqGTest, ComplexPasses)
{
  std::complex<float> a{1.0f, 2.0f};
  std::complex<float> b{std::nextafter(1.0f, 2.0f), 2.0f};
  EXPECT_FLOATING_EQ(a, b, 1); // real differs by 1 ULP
}

TEST(FloatingEqGTest, ComplexFails)
{
  std::complex<double> a{1.0, 2.0};
  std::complex<double> b{1.0, 2.1};
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 1), "EXPECT_FLOATING_EQ failed");
}

// --- NaN and infinity behavior ---

TEST(FloatingEqGTest, NaNFails)
{
  float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(nan, nan), "unrepresentable");
}

TEST(FloatingEqGTest, SameInfinityPasses)
{
  double inf = std::numeric_limits<double>::infinity();
  EXPECT_FLOATING_EQ(inf, inf);
}

TEST(FloatingEqGTest, OppositeInfinityFails)
{
  static double const pos_inf = std::numeric_limits<double>::infinity();
  static double const neg_inf = -std::numeric_limits<double>::infinity();
  EXPECT_FATAL_FAILURE(ASSERT_FLOATING_EQ(pos_inf, neg_inf), "unrepresentable");
}
