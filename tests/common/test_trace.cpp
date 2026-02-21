#include "common/trace.hpp"
#include <gtest/gtest.h>

using namespace trace;

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

// TRACE
TEST(TraceMacro, TraceVariable)
{
  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 123;
  TRACE("foo", n);
  EXPECT_NE(oss.str().find("foo, n = 123"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::get_formatting_options().set_output_stream(stderr);
}

TEST(TraceStackMacro, TraceStackIncludesStacktraceDiagnostic)
{
  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 123;
  TRACE_STACK("trace-stack", n);
  auto const output = oss.str();
  EXPECT_NE(output.find("trace-stack, n = 123"), std::string::npos) << "Trace output was:\n" << output;
#if UNI20_HAS_STACKTRACE
  EXPECT_NE(output.find("Stacktrace:"), std::string::npos) << "Trace output was:\n" << output;
#else
  EXPECT_NE(output.find("WARNING: std::stacktrace is unavailable"), std::string::npos) << "Trace output was:\n"
                                                                                        << output;
#endif
  trace::get_formatting_options().set_output_stream(stderr);
}

// Disable warning that ("foo", n) discards "foo"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"

TEST(TraceMacro, TraceBrackets)
{
  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });
  auto n = 123;
  TRACE(("foo", n));
  EXPECT_NE(oss.str().find("(\"foo\", n) = 123"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::get_formatting_options().set_output_stream(stderr);
}

#pragma GCC diagnostic pop

TEST(TraceMacro, TraceSquareBrackets)
{
  struct Dummy2D
  {
      std::string operator[](int i, int j) const { return "result of [i,j]"; }
  };

  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });
  Dummy2D n;
  TRACE(n[2, 3]);
  EXPECT_NE(oss.str().find("n[2, 3] = result of [i,j]"), std::string::npos) << "Trace output was:\n" << oss.str();
  trace::get_formatting_options().set_output_stream(stderr);
}

// in consteval context, we cannot write to the screen but instead the TRACE() macros are a no-op
template <typename T> consteval void TraceConsteval(T const& x) { TRACE(x); }

TEST(TraceMacro, TraceConsteval)
{
  auto n = 123;
  TraceConsteval(n);
  SUCCEED();
}

TEST(TraceMacro, TraceOnceFiresOnlyOnce)
{
  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });

  for (int i = 0; i < 5; ++i)
  {
    TRACE_ONCE("hello", i);
  }

  auto output = oss.str();
  // Should contain exactly one occurrence of "hello"
  EXPECT_NE(output.find("hello"), std::string::npos);
  EXPECT_EQ(output.find("hello", output.find("hello") + 1), std::string::npos) << "TRACE_ONCE emitted more than once:\n"
                                                                               << output;

  trace::get_formatting_options().set_output_stream(stderr);
}

TEST(TraceMacro, TraceOnceDifferentSitesAreIndependent)
{
  std::ostringstream oss;
  trace::get_formatting_options().set_sink([&oss](std::string msg) { oss << msg; });

  for (int i = 0; i < 3; ++i)
  {
    TRACE_ONCE("siteA", i);
    TRACE_ONCE("siteB", i);
  }

  auto output = oss.str();
  // Each call site should fire exactly once
  EXPECT_NE(output.find("siteA"), std::string::npos);
  EXPECT_EQ(output.find("siteA", output.find("siteA") + 1), std::string::npos);
  EXPECT_NE(output.find("siteB"), std::string::npos);
  EXPECT_EQ(output.find("siteB", output.find("siteB") + 1), std::string::npos);

  trace::get_formatting_options().set_output_stream(stderr);
}

// CHECK
TEST(CheckMacro, FailingCheckAborts)
{
  EXPECT_DEATH({ CHECK(false); }, "false is false!");
}

TEST(CheckMacro, FailingCheckIncludesStacktraceDiagnostic)
{
  EXPECT_DEATH({ CHECK(false); }, kStacktraceDiagnosticRegex);
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

TEST(PreconditionMacro, FailingPreconditionIncludesStacktraceDiagnostic)
{
  EXPECT_DEATH({ PRECONDITION(false); }, kStacktraceDiagnosticRegex);
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
  trace::get_formatting_options().set_errors_abort(true);
  EXPECT_DEATH({ ERROR("fatal error"); }, "fatal error");
}
TEST(ErrorIfMacro, ErrorIfTrueAbortsWhenConfigured)
{
  trace::get_formatting_options().set_errors_abort(true);
  EXPECT_DEATH({ ERROR_IF(true, "conditional error"); }, "conditional error");
}
TEST(ErrorIfMacro, ErrorIfFalseDoesNotAbort)
{
  trace::get_formatting_options().set_errors_abort(true);
  ERROR_IF(false, "should not abort");
}

// ERROR / ERROR_IF in throw mode
TEST(ErrorMacro, ErrorThrowsWhenAbortDisabled)
{
  trace::get_formatting_options().set_errors_abort(false);
  EXPECT_THROW(ERROR("must throw"), std::runtime_error);
}

TEST(ErrorIfMacro, ErrorIfTrueThrowsWhenAbortDisabled)
{
  trace::get_formatting_options().set_errors_abort(false);
  EXPECT_THROW(ERROR_IF(true, "must throw"), std::runtime_error);
}

TEST(ErrorIfMacro, ErrorIfFalseDoesNotThrowWhenAbortDisabled)
{
  trace::get_formatting_options().set_errors_abort(false);
  ERROR_IF(false, "no throw");
}

// CHECK_FLOATING_EQ
TEST(CheckFloatingEq, EqualScalarsPass)
{
  float x = 1.0f;
  float y = std::nextafter(x, 2.0f); // within 1 ULP
  CHECK_FLOATING_EQ(x, y);           // should not abort
  SUCCEED();
}

TEST(CheckFloatingEq, UnequalScalarsAbort)
{
  double x = 1.0;
  double y = 1.1; // many ULPs apart
  EXPECT_DEATH({ CHECK_FLOATING_EQ(x, y); }, "CHECK_FLOATING_EQ");
}

// --- Complex numbers ---
TEST(CheckFloatingEq, ComplexEqualPass)
{
  std::complex<double> a{1.0, 2.0};
  std::complex<double> b{std::nextafter(1.0, 2.0), 2.0};
  CHECK_FLOATING_EQ(a, b); // real differs by 1 ULP, imag equal
  SUCCEED();
}

TEST(CheckFloatingEq, ComplexUnequalAbort)
{
  std::complex<float> a{1.0f, 2.0f};
  std::complex<float> b{1.0f, 2.1f}; // imag off by many ULPs
  EXPECT_DEATH({ CHECK_FLOATING_EQ(a, b); }, "CHECK_FLOATING_EQ");
}

// --- PRECONDITION_FLOATING_EQ ---
TEST(PreconditionFloatingEq, EqualPass)
{
  PRECONDITION_FLOATING_EQ(1.0f, std::nextafter(1.0f, 2.0f));
  SUCCEED();
}

TEST(PreconditionFloatingEq, UnequalAbort)
{
  EXPECT_DEATH({ PRECONDITION_FLOATING_EQ(1.0f, 2.0f); }, "PRECONDITION_FLOATING_EQ");
}

// --- CHECK_FLOATING_EQ with explicit ULPs ---
TEST(CheckFloatingEq, ThreeParamExplicitUlpsPass)
{
  float x = 1.0f;
  float y = std::nextafter(x, 2.0f); // 1 ULP away
  CHECK_FLOATING_EQ(x, y, 1);        // should pass with ulps = 1
  SUCCEED();
}

TEST(CheckFloatingEq, ThreeParamExplicitUlpsAbort)
{
  double x = 1.0;
  double y = 1.0000000000001; // many ULPs away
  EXPECT_DEATH({ CHECK_FLOATING_EQ(x, y, 1); }, "CHECK_FLOATING_EQ");
}

// --- CHECK_FLOATING_EQ with extra context parameters ---
TEST(CheckFloatingEq, FourParamWithMessagePass)
{
  double x = 1.0;
  double y = std::nextafter(1.0, 2.0); // within 1 ULP
  CHECK_FLOATING_EQ(x, y, 2, "values should be close");
  SUCCEED();
}

TEST(CheckFloatingEq, FourParamWithMessageAbort)
{
  float x = 1.0f;
  float y = 1.1f;
  EXPECT_DEATH({ CHECK_FLOATING_EQ(x, y, 2, "extra context", 42); }, "CHECK_FLOATING_EQ");
}

// Helper: move 'n' ULPs away from a
template <typename T> T offset_by_ulps(T value, int n)
{
  T x = value;
  if (n > 0)
  {
    for (int i = 0; i < n; i++)
      x = std::nextafter(x, std::numeric_limits<T>::infinity());
  }
  else if (n < 0)
  {
    for (int i = 0; i < -n; i++)
      x = std::nextafter(x, -std::numeric_limits<T>::infinity());
  }
  return x;
}

// --- PRECONDITION_FLOATING_EQ with explicit ULPs ---
TEST(PreconditionFloatingEq, ThreeParamExplicitUlpsPass)
{
  PRECONDITION_FLOATING_EQ(1.0, std::nextafter(1.0, 2.0), 1);
  SUCCEED();
}

TEST(PreconditionFloatingEq, FourParamWithMessageAbort)
{
  EXPECT_DEATH({ PRECONDITION_FLOATING_EQ(1.0, 1.5, 1, "bad precondition"); }, "PRECONDITION_FLOATING_EQ");
}

TEST(CheckFloatingEq, UlpToleranceOnePassesOneAway)
{
  float a = 1.0f;
  float b = offset_by_ulps(a, 1);
  CHECK_FLOATING_EQ(a, b, 1); // within 1 ULP
  SUCCEED();
}

TEST(CheckFloatingEq, UlpToleranceOneFailsTwoAway)
{
  float a = 1.0f;
  float b = offset_by_ulps(a, 2);
  EXPECT_DEATH({ CHECK_FLOATING_EQ(a, b, 1); }, "CHECK_FLOATING_EQ");
}

TEST(CheckFloatingEq, UlpToleranceTwoPassesTwoAway)
{
  double a = 1.0;
  double b = offset_by_ulps(a, 2);
  CHECK_FLOATING_EQ(a, b, 2); // should pass
  SUCCEED();
}

TEST(CheckFloatingEq, UlpToleranceTwoFailsThreeAway)
{
  double a = 1.0;
  double b = offset_by_ulps(a, 3);
  EXPECT_DEATH({ CHECK_FLOATING_EQ(a, b, 2); }, "CHECK_FLOATING_EQ");
}

TEST(CheckFloatingEq, ComplexWithinTolerancePass)
{
  std::complex<float> a{1.0f, 2.0f};
  // real part differs by 1 ULP, imag identical
  std::complex<float> b{std::nextafter(1.0f, 2.0f), 2.0f};
  CHECK_FLOATING_EQ(a, b, 1); // should pass
  SUCCEED();
}

TEST(CheckFloatingEq, ComplexOutsideToleranceFail)
{
  std::complex<double> a{1.0, 2.0};
  // imag part shifted by 10 ULPs
  double imag_shift = std::bit_cast<double>(std::bit_cast<std::uint64_t>(2.0) + 10);
  std::complex<double> b{1.0, imag_shift};
  EXPECT_DEATH({ CHECK_FLOATING_EQ(a, b, 1); }, "CHECK_FLOATING_EQ");
}

TEST(CheckFloatingEq, ComplexDefaultTolerance)
{
  std::complex<float> a{1.0f, 2.0f};
  // real part is 4 ULPs away, imag identical
  float shifted = std::bit_cast<float>(std::bit_cast<std::uint32_t>(1.0f) + 4);
  std::complex<float> b{shifted, 2.0f};
  CHECK_FLOATING_EQ(a, b); // should pass with default 4 ULPs
  EXPECT_DEATH({ CHECK_FLOATING_EQ(a, b, 3); }, "CHECK_FLOATING_EQ");
}
