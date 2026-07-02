#include <uni20/core/math.hpp>

#include <gtest/gtest.h>

#include <limits>

TEST(MathTest, IsFiniteHandlesIntegerScalars)
{
  EXPECT_TRUE(uni20::isfinite(0));
  EXPECT_TRUE(uni20::isfinite(-42));
}

TEST(MathTest, IsFiniteHandlesBuiltinRealScalars)
{
  auto const inf = std::numeric_limits<double>::infinity();
  auto const nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_TRUE(uni20::isfinite(1.25));
  EXPECT_TRUE(uni20::isfinite(-0.0));
  EXPECT_FALSE(uni20::isfinite(inf));
  EXPECT_FALSE(uni20::isfinite(-inf));
  EXPECT_FALSE(uni20::isfinite(nan));
}

TEST(MathTest, IsFiniteHandlesComplexScalars)
{
  auto const inf = std::numeric_limits<double>::infinity();
  auto const nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_TRUE(uni20::isfinite(uni20::complex<double>{1.0, -2.0}));
  EXPECT_FALSE(uni20::isfinite(uni20::complex<double>{inf, 1.0}));
  EXPECT_FALSE(uni20::isfinite(uni20::complex<double>{1.0, -inf}));
  EXPECT_FALSE(uni20::isfinite(uni20::complex<double>{nan, 1.0}));
}
