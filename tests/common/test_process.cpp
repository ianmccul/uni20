#include <gtest/gtest.h>

#ifdef __linux__
#include <sys/prctl.h>

TEST(TestProcess, CoreDumpingIsDisabled) { EXPECT_EQ(prctl(PR_GET_DUMPABLE), 0); }
#endif
