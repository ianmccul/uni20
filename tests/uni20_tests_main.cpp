// main() function for the combined tests
// Set the death_test_style to play nicely with TBB

#include <gtest/gtest.h>

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  return RUN_ALL_TESTS();
}
