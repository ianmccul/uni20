// Shared main() for Uni20 GoogleTest executables.

#include <cstdio>
#include <gtest/gtest.h>

#ifdef __linux__
#include <sys/prctl.h>
#endif

int main(int argc, char** argv)
{
#ifdef __linux__
  // Death tests intentionally abort child processes. Prevent those expected
  // crashes from invoking system-wide core and crash-reporting services.
  if (prctl(PR_SET_DUMPABLE, 0) == -1) std::perror("prctl(PR_SET_DUMPABLE)");
#endif

  ::testing::InitGoogleTest(&argc, argv);

#ifdef UNI20_TEST_DEATH_TEST_STYLE_THREADSAFE
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#endif

  return RUN_ALL_TESTS();
}
