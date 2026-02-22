#include "core/dummy.hpp"
#include "gtest/gtest.h"

TEST(DummyTest, Addition)
{
  EXPECT_EQ(uni20::add(2, 3), 5);
  EXPECT_EQ(uni20::add(-1, 1), 0);
}

TEST(DummyTest, Multiplication) { EXPECT_DOUBLE_EQ(uni20::multiply(2.0, 3.0), 6.0); }

TEST(DummyTest, HeavyOperation)
{
  // This test checks that the heavy operation produces a positive result.
  double result = uni20::compute_heavy_operation(2.0, 3.0);
  EXPECT_GT(result, 0.0);
}
