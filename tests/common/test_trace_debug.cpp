// test trace macros without NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif
#ifdef TRACE_DISABLE
#undef TRACE_DISABLE
#define TRACE_DISABLE 0
#endif

#include <gtest/gtest.h>

#include "common/trace.hpp"

#if ENABLE_TRACE_TESTMODULE == 0
COMPILER_NOTE("ENABLE_TRACE_TESTMODULE is 0 - trace tests will likely fail.")
#endif

// Disable ANSI colors so death‑tests see plain text
namespace
{
struct DisableColor
{
    DisableColor() { trace::get_formatting_options().set_color_output(trace::FormattingOptions::ColorOptions::no); }
} _disableColor;

#if UNI20_HAS_STACKTRACE
char const* kStacktraceDiagnosticRegex = "Stacktrace:";
#else
char const* kStacktraceDiagnosticRegex = "WARNING: std::stacktrace is unavailable";
#endif
} // namespace

// DEBUG_CHECK / DEBUG_CHECK_EQUAL
TEST(DebugCheckMacro, FailingDebugCheckAborts)
{
  EXPECT_DEATH({ DEBUG_CHECK(false); }, "false is false!");
}

TEST(DebugCheckMacro, FailingDebugCheckIncludesStacktraceDiagnostic)
{
  EXPECT_DEATH({ DEBUG_CHECK(false); }, kStacktraceDiagnosticRegex);
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

TEST(DebugPreconditionMacro, FailingDebugPreconditionIncludesStacktraceDiagnostic)
{
  EXPECT_DEATH({ DEBUG_PRECONDITION(false); }, kStacktraceDiagnosticRegex);
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
  trace::get_formatting_options("TESTMODULE").set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 123;
  TRACE_MODULE(TESTMODULE, "foo", n);
  EXPECT_NE(oss.str().find("foo, n = 123"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::get_formatting_options("TESTMODULE").set_output_stream(stderr);
}

TEST(DebugTraceModuleMacro, DEBUG_TRACE_MODULE_EmitsWhenEnabled)
{
  std::ostringstream oss;
  trace::get_formatting_options("TESTMODULE").set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 456;
  DEBUG_TRACE_MODULE(TESTMODULE, "bar", n);
  EXPECT_NE(oss.str().find("bar, n = 456"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::get_formatting_options("TESTMODULE").set_output_stream(stderr);
}

TEST(DebugTraceModuleIfMacro, DEBUG_TRACE_MODULE_IF_EmitsWhenTrue)
{
  std::ostringstream oss;
  trace::get_formatting_options("TESTMODULE").set_sink([&oss](std::string msg) { oss << msg; });
  bool x = true;
  auto n = 123;
  DEBUG_TRACE_MODULE_IF(TESTMODULE, x, "baz", n);
  EXPECT_NE(oss.str().find("baz, n = 123"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::get_formatting_options("TESTMODULE").set_output_stream(stderr);
}

TEST(DebugTraceStackMacro, DebugTraceStackIncludesStacktraceDiagnostic)
{
  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 789;
  DEBUG_TRACE_STACK("debug-trace-stack", n);
  auto const output = oss.str();
  EXPECT_NE(output.find("debug-trace-stack, n = 789"), std::string::npos) << "Trace output was:\n" << output;
#if UNI20_HAS_STACKTRACE
  EXPECT_NE(output.find("Stacktrace:"), std::string::npos) << "Trace output was:\n" << output;
#else
  EXPECT_NE(output.find("WARNING: std::stacktrace is unavailable"), std::string::npos) << "Trace output was:\n"
                                                                                        << output;
#endif
  trace::get_formatting_options().set_output_stream(stderr);
}

TEST(DebugCheckFloatingEq, PassesWithinTolerance)
{
  float a = 1.0f;
  float b = std::nextafter(a, 2.0f); // 1 ULP away
  CHECK_FLOATING_EQ(a, b, 1);        // should not abort
  SUCCEED();
}

TEST(DebugCheckFloatingEq, FailsOutsideTolerance)
{
  float a = 1.0f;
  float b = std::bit_cast<float>(std::bit_cast<uint32_t>(a) + 10);
  EXPECT_DEATH({ CHECK_FLOATING_EQ(a, b, 1); }, "CHECK_FLOATING_EQ");
}

TEST(DebugCheckFloatingEq, DefaultToleranceIsFour)
{
  float a = 1.0f;
  float b = std::bit_cast<float>(std::bit_cast<uint32_t>(a) + 4);
  CHECK_FLOATING_EQ(a, b);                                            // 4 ULPs away, should pass
  EXPECT_DEATH({ CHECK_FLOATING_EQ(a, b, 3); }, "CHECK_FLOATING_EQ"); // but not within 3
}

TEST(DebugPreconditionFloatingEq, PassesWithinTolerance)
{
  float a = 1.0f;
  float b = std::nextafter(a, 2.0f); // 1 ULP away
  PRECONDITION_FLOATING_EQ(a, b, 1); // should not abort
  SUCCEED();
}

TEST(DebugPreconditionFloatingEq, FailsOutsideTolerance)
{
  float a = 1.0f;
  float b = std::bit_cast<float>(std::bit_cast<uint32_t>(a) + 10);
  EXPECT_DEATH({ PRECONDITION_FLOATING_EQ(a, b, 1); }, "PRECONDITION_FLOATING_EQ");
}

TEST(DebugPreconditionFloatingEq, DefaultToleranceIsFour)
{
  float a = 1.0f;
  float b = std::bit_cast<float>(std::bit_cast<uint32_t>(a) + 4);
  PRECONDITION_FLOATING_EQ(a, b);                                                   // 4 ULPs away, should pass
  EXPECT_DEATH({ PRECONDITION_FLOATING_EQ(a, b, 3); }, "PRECONDITION_FLOATING_EQ"); // but not within 3
}
