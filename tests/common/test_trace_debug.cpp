// test trace macros without NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <gtest/gtest.h>

// Enable our test module
#define ENABLE_TRACE_TESTMODULE 1

#include "common/trace.hpp"

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

// DEBUG_CHECK / DEBUG_CHECK_EQUAL
TEST(DebugCheckMacro, FailingDebugCheckAborts)
{
  EXPECT_DEATH({ DEBUG_CHECK(false); }, "false is false!");
}

TEST(DebugCheckMacro, PassingDebugCheckDoesNotAbort) { DEBUG_CHECK(true); }

TEST(DebugCheckEqualMacro, FailingDebugCheckEqualAborts)
{
  EXPECT_DEATH({ DEBUG_CHECK_EQUAL(1, 2); }, "1 is not equal to 2!");
}

TEST(DebugCheckEqualMacro, PassingDebugCheckEqualDoesNotAbort) { DEBUG_CHECK_EQUAL(42, 42); }

// DEBUG_PRECONDITION / DEBUG_PRECONDITION_EQUAL
TEST(DebugPreconditionMacro, FailingDebugPreconditionAborts)
{
  EXPECT_DEATH({ DEBUG_PRECONDITION(false); }, "false is false!");
}

TEST(DebugPreconditionMacro, PassingDebugPreconditionDoesNotAbort) { DEBUG_PRECONDITION(true); }

TEST(DebugPreconditionEqualMacro, FailingDebugPreconditionEqualAborts)
{
  EXPECT_DEATH({ DEBUG_PRECONDITION_EQUAL(3, 4); }, "3 is not equal to 4!");
}

TEST(DebugPreconditionEqualMacro, PassingDebugPreconditionEqualDoesNotAbort) { DEBUG_PRECONDITION_EQUAL(5, 5); }

// TRACE_MODULE / DEBUG_TRACE_MODULE / DEBUG_TRACE_MODULE_IF
TEST(TraceModuleMacro, TRACE_MODULE_AlwaysAvailable)
{
  TRACE_MODULE(TESTMODULE, "foo", 123);
  SUCCEED();
}

TEST(DebugTraceModuleMacro, DEBUG_TRACE_MODULE_EmitsWhenEnabled)
{
  DEBUG_TRACE_MODULE(TESTMODULE, "bar", 456);
  SUCCEED();
}

TEST(DebugTraceModuleIfMacro, DEBUG_TRACE_MODULE_IF_EmitsWhenTrue)
{
  DEBUG_TRACE_MODULE_IF(TESTMODULE, true, "baz");
  SUCCEED();
}
