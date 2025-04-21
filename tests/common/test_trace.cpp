#include "common/trace.hpp"
#include <gtest/gtest.h>

using namespace trace;

// Disable ANSI colors so death‑tests see plain text
namespace
{
struct DisableColor
{
    DisableColor()
    {
      using CO = trace::FormattingOptions::ColorOptions;
      trace::FormattingOptions::set_color_output(CO::no);
    }
} _disableColor;
} // namespace

// CHECK
TEST(CheckMacro, FailingCheckAborts)
{
  EXPECT_DEATH({ CHECK(false); }, "false is false!");
}
TEST(CheckMacro, PassingCheckDoesNotAbort) { CHECK(true); }

// CHECK_EQUAL
TEST(CheckEqualMacro, FailingCheckEqualAborts)
{
  EXPECT_DEATH({ CHECK_EQUAL(1, 2); }, "1 is not equal to 2!");
}
TEST(CheckEqualMacro, PassingCheckEqualDoesNotAbort) { CHECK_EQUAL(42, 42); }

// PRECONDITION
TEST(PreconditionMacro, FailingPreconditionAborts)
{
  EXPECT_DEATH({ PRECONDITION(false); }, "false is false!");
}
TEST(PreconditionMacro, PassingPreconditionDoesNotAbort) { PRECONDITION(true); }

// PRECONDITION_EQUAL
TEST(PreconditionEqualMacro, FailingPreconditionEqualAborts)
{
  EXPECT_DEATH({ PRECONDITION_EQUAL(3, 4); }, "3 is not equal to 4!");
}
TEST(PreconditionEqualMacro, PassingPreconditionEqualDoesNotAbort) { PRECONDITION_EQUAL(5, 5); }

// PANIC
TEST(PanicMacro, PanicAlwaysAborts)
{
  EXPECT_DEATH({ PANIC("unconditional panic"); }, "unconditional panic");
}

// ERROR / ERROR_IF in abort mode
TEST(ErrorMacro, ErrorAlwaysAbortsWhenConfigured)
{
  trace::FormattingOptions::set_errors_abort(true);
  EXPECT_DEATH({ ERROR("fatal error"); }, "fatal error");
}
TEST(ErrorIfMacro, ErrorIfTrueAbortsWhenConfigured)
{
  trace::FormattingOptions::set_errors_abort(true);
  EXPECT_DEATH({ ERROR_IF(true, "conditional error"); }, "conditional error");
}
TEST(ErrorIfMacro, ErrorIfFalseDoesNotAbort)
{
  trace::FormattingOptions::set_errors_abort(true);
  ERROR_IF(false, "should not abort");
}

// ERROR / ERROR_IF in throw mode
TEST(ErrorMacro, ErrorThrowsWhenAbortDisabled)
{
  trace::FormattingOptions::set_errors_abort(false);
  EXPECT_THROW(ERROR("must throw"), std::runtime_error);
}

TEST(ErrorIfMacro, ErrorIfTrueThrowsWhenAbortDisabled)
{
  trace::FormattingOptions::set_errors_abort(false);
  EXPECT_THROW(ERROR_IF(true, "must throw"), std::runtime_error);
}

TEST(ErrorIfMacro, ErrorIfFalseDoesNotThrowWhenAbortDisabled)
{
  trace::FormattingOptions::set_errors_abort(false);
  ERROR_IF(false, "no throw");
}
