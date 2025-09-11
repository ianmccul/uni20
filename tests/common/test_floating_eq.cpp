#include "common/gtest.hpp"

// --- EXPECT_FLOATING_EQ ---

TEST(FloatingEqGTest, ExpectPassesWithinTolerance)
{
  float a = 1.0f;
  float b = std::nextafter(a, 2.0f); // 1 ULP away
  EXPECT_FLOATING_EQ(a, b, 1);       // should pass
}

// --- ASSERT_FLOATING_EQ ---

TEST(FloatingEqGTest, AssertPassesWithinTolerance)
{
  double a = 1.0;
  double b = std::nextafter(a, 2.0);
  ASSERT_FLOATING_EQ(a, b, 1); // should pass
  SUCCEED();
}

// --- Complex numbers ---

TEST(FloatingEqGTest, ComplexPasses)
{
  std::complex<float> a{1.0f, 2.0f};
  std::complex<float> b{std::nextafter(1.0f, 2.0f), 2.0f};
  EXPECT_FLOATING_EQ(a, b, 1); // real differs by 1 ULP
}

// It doesn't seem possible to write a test where EXPECT_FLOATING_EQ will fail, but still have the test succeed
