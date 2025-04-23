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

// TRACE
TEST(TraceMacro, TraceVariable)
{
  std::ostringstream oss;
  trace::formatting_options.set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 123;
  TRACE("foo", n);
  EXPECT_NE(oss.str().find("foo, n = 123"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::formatting_options.set_output_stream(stderr);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"

TEST(TraceMacro, TraceBrackets)
{
  std::ostringstream oss;
  trace::formatting_options.set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 123;
  TRACE(("foo", n));
  EXPECT_NE(oss.str().find("(\"foo\", n) = 123"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::formatting_options.set_output_stream(stderr);
}

#pragma GCC diagnostic pop

TEST(TraceMacro, TraceSquareBrackets)
{
  struct Dummy2D
  {
      std::string operator[](int i, int j) const { return "result of [i,j]"; }
  };

  std::ostringstream oss;
  trace::formatting_options.set_sink([&oss](std::string msg) { oss << msg; });
  Dummy2D n;
  TRACE(n[2, 3]);
  EXPECT_NE(oss.str().find("n[2, 3] = result of [i,j]"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::formatting_options.set_output_stream(stderr);
}

// in consteval context, we cannot write to the screen but instead the TRACE() macros are a no-op
template <typename T> consteval void TraceConsteval(T const& x) { TRACE(x); }

TEST(TraceMacro, TraceConsteval)
{
  auto n = 123;
  TraceConsteval(n);
  SUCCEED();
}

// CHECK
TEST(CheckMacro, FailingCheckAborts)
{
  EXPECT_DEATH({ CHECK(false); }, "false is false!");
}

TEST(CheckMacro, PassingCheckDoesNotAbort) { CHECK(true); }

// CHECK in consteval context
consteval bool CheckConsteval(bool b)
{
  CHECK(b);
  return b;
}

TEST(CheckMacro, PassingCheckConsteval) { CHECK(CheckConsteval(true)); }

// CheckConsteval(false) should produce a compile-time error.

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

// It is possible to call PRECONDITION in constexpr context, where it is equivalent to static_assert
constexpr bool test_precondition()
{
  PRECONDITION(true);
  return true;
}

static_assert(test_precondition(), "PRECONDITION(true) should not fire");

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
