// test trace macros with NDEBUG

#ifndef NDEBUG
#define NDEBUG
#endif

#include "common/trace.hpp"
#include <gtest/gtest.h>

// Disable ANSI colors so death‑tests see plain text
namespace
{
struct DisableColor
{
    DisableColor() { trace::get_formatting_options().set_color_output(trace::FormattingOptions::ColorOptions::no); }
} _disableColor;
} // namespace

// In NDEBUG builds all DEBUG_… macros should be no‐ops:

TEST(DebugMacrosNoOp, DebugCheckDoesNothing)
{
  DEBUG_CHECK(false);
  DEBUG_CHECK(true);
  SUCCEED();
}

TEST(DebugMacrosNoOp, DebugCheckEqualDoesNothing)
{
  DEBUG_CHECK_EQUAL(1, 2);
  DEBUG_CHECK_EQUAL(3, 3);
  SUCCEED();
}

TEST(DebugMacrosNoOp, DebugCheckFloatingEqDoesNothing)
{
  DEBUG_CHECK_FLOATING_EQ(1, 2);
  DEBUG_CHECK_FLOATING_EQ(3, 3);
  SUCCEED();
}

TEST(DebugMacrosNoOp, DebugPreconditionDoesNothing)
{
  DEBUG_PRECONDITION(false);
  DEBUG_PRECONDITION(true);
  SUCCEED();
}

TEST(DebugMacrosNoOp, DebugPreconditionEqualDoesNothing)
{
  DEBUG_PRECONDITION_EQUAL(2, 3);
  DEBUG_PRECONDITION_EQUAL(4, 4);
  SUCCEED();
}

TEST(DebugMacrosNoOp, DebugPreconditionFloatingEqDoesNothing)
{
  DEBUG_PRECONDITION_FLOATING_EQ(1, 2);
  DEBUG_PRECONDITION_FLOATING_EQ(3, 3);
  SUCCEED();
}

TEST(DebugMacrosNoOp, DebugTraceModuleDoesNothing)
{
  // Even if module is enabled in code, with NDEBUG this macro is empty
  DEBUG_TRACE_MODULE(TESTMODULE, "hello", 999);
  DEBUG_TRACE_MODULE_IF(TESTMODULE, true, "world");
  SUCCEED();
}
