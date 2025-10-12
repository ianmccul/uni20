#include "common/gtest.hpp"

#include <cmath>
#include <complex>

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

// --- Complex numbers ---

TEST(FloatingEqGTest, ComplexPasses)
{
  std::complex<float> a{1.0f, 2.0f};
  std::complex<float> b{std::nextafter(1.0f, 2.0f), 2.0f};
  EXPECT_FLOATING_EQ(a, b, 1); // real differs by 1 ULP
}
