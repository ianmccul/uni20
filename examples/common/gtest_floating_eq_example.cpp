#include <uni20/common/gtest.hpp>
#include <gtest/gtest.h>
#include <iostream>

// Demonstrates EXPECT_FLOATING_EQ
TEST(FloatingEqExample, ExpectPass)
{
  float a = 1.0f;
  float b = std::nextafter(a, 2.0f); // 1 ULP away
  EXPECT_FLOATING_EQ(a, b, 1);       // passes
  EXPECT_FLOATING_EQ(a, b);          // also passes with default tolerance of 4 ULPs
}

// This test intentionally fails to demonstrate that EXPECT_* reports a failure
// but allows the test body to continue executing.
TEST(FloatingEqExample, ExpectFailButContinue)
{
  float a = 1.0f;
  float b = std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) + 100);
  EXPECT_FLOATING_EQ(a, b, 1); // intentionally fails
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

// This test intentionally fails to demonstrate that ASSERT_* aborts the
// current test body immediately.
TEST(FloatingEqExample, AssertFailStopsTest)
{
  double a = 1.0;
  double b = std::bit_cast<double>(std::bit_cast<std::uint64_t>(a) + 1000);
  ASSERT_FLOATING_EQ(a, b, 1); // intentionally fails
  EXPECT_TRUE(false);          // should never run
}

// Demonstrates complex numbers
TEST(FloatingEqExample, ComplexComparison)
{
  std::complex<float> a{1.0f, 2.0f};
  std::complex<float> b{std::nextafter(std::nextafter(1.0f, 2.0f), 2.0f), 2.0f};

  EXPECT_FLOATING_EQ(a, b, 2); // this should succeed
  EXPECT_FLOATING_EQ(a, b, 1); // intentionally fails
}

// Standard GTest main
int main(int argc, char** argv)
{
  std::cout << "[gtest_floating_eq_example] This executable intentionally includes failing tests.\n"
               "[gtest_floating_eq_example] Expected failing tests:\n"
               "  - FloatingEqExample.ExpectFailButContinue\n"
               "  - FloatingEqExample.AssertFailStopsTest\n"
               "  - FloatingEqExample.ComplexComparison\n"
               "[gtest_floating_eq_example] A non-zero exit code is expected.\n";

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
