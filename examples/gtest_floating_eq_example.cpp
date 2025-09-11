#include "common/gtest.hpp"
#include <gtest/gtest.h>

// Demonstrates EXPECT_FLOATING_EQ
TEST(FloatingEqExample, ExpectPass)
{
  float a = 1.0f;
  float b = std::nextafter(a, 2.0f); // 1 ULP away
  EXPECT_FLOATING_EQ(a, b, 1);       // passes
  EXPECT_FLOATING_EQ(a, b);          // also passes with default tolerance of 4 ULPs
}

TEST(FloatingEqExample, ExpectFailButContinue)
{
  float a = 1.0f;
  float b = std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) + 100);
  EXPECT_FLOATING_EQ(a, b, 1); // will fail, but test continues
  EXPECT_TRUE(true);           // still executes
}

// Demonstrates ASSERT_FLOATING_EQ
TEST(FloatingEqExample, AssertPass)
{
  double a = 1.0;
  double b = std::nextafter(a, 2.0);
  ASSERT_FLOATING_EQ(a, b, 1); // passes
  EXPECT_TRUE(true);           // still executes
}

TEST(FloatingEqExample, AssertFailStopsTest)
{
  double a = 1.0;
  double b = std::bit_cast<double>(std::bit_cast<std::uint64_t>(a) + 1000);

  // ASSERT_ will stop execution of this test immediately
  ASSERT_FLOATING_EQ(a, b, 1);
  // This line will never run
  EXPECT_TRUE(false);
}

// Demonstrates complex numbers
TEST(FloatingEqExample, ComplexComparison)
{
  std::complex<float> a{1.0f, 2.0f};
  std::complex<float> b{std::nextafter(1.0f, 2.0f), 2.0f};

  EXPECT_FLOATING_EQ(a, b, 1); // compares component-wise
}

// Standard GTest main
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
