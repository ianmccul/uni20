// test trace macros without NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <gtest/gtest.h>

#include "common/trace.hpp"

#if ENABLE_TRACE_TESTMODULE == 0
COMPILER_NOTE("ENABLE_TRACE_TESTMODULE is 0 — trace tests will likely fail.")
#endif

// Disable ANSI colors so death‑tests see plain text
namespace
{
struct DisableColor
{
    DisableColor() { trace::get_formatting_options().set_color_output(trace::FormattingOptions::ColorOptions::no); }
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
  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 123;
  TRACE_MODULE(TESTMODULE, "foo", n);
  EXPECT_NE(oss.str().find("foo, n = 123"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::get_formatting_options().set_output_stream(stderr);
}

TEST(DebugTraceModuleMacro, DEBUG_TRACE_MODULE_EmitsWhenEnabled)
{
  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 456;
  DEBUG_TRACE_MODULE(TESTMODULE, "bar", n);
  EXPECT_NE(oss.str().find("bar, n = 456"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::get_formatting_options().set_output_stream(stderr);
}

TEST(DebugTraceModuleIfMacro, DEBUG_TRACE_MODULE_IF_EmitsWhenTrue)
{
  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });
  bool x = true;
  auto n = 123;
  DEBUG_TRACE_MODULE_IF(TESTMODULE, x, "baz", n);
  EXPECT_NE(oss.str().find("baz, n = 123"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::get_formatting_options().set_output_stream(stderr);
}
