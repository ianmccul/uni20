#pragma once

#include "config.hpp"

#include "demangle.hpp"
#include "floating_eq.hpp"
#include "namedenum.hpp"
#include "terminal.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <complex>
#include <coroutine>
#include <cstdio>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// Check for stacktrace support (C++23 and GCC 13+ or Clang+libc++)
#if UNI20_HAS_STACKTRACE
#define TRACE_HAS_STACKTRACE 1
#include <stacktrace>
#else
#define TRACE_HAS_STACKTRACE 0
#endif

#ifndef TRACE_DISABLE
#define TRACE_DISABLE 0
#endif

// stringize helper (two levels needed so that macro args expand first)
#define TRACE_STRINGIZE_IMPL(x) #x
#define TRACE_STRINGIZE(x) TRACE_STRINGIZE_IMPL(x)

// COMPILER_NOTE emits an information message during compilation. msg must be a string literal.
#if defined(_MSC_VER)
// On MSVC, use __pragma without needing quotes around the entire pragma.
#define COMPILER_NOTE(msg) __pragma(message(msg))
#else
// On GCC/Clang, use the standard _Pragma operator with a string.
#define COMPILER_NOTE(msg) _Pragma(TRACE_STRINGIZE(message msg))
#endif

// COMPILER_WARNING_NOTE(msg):
//   – On GCC/Clang: emits a compile‐time warning via an unnamed lambda with a deprecation attribute.
//   – On MSVC: no effect.
#if defined(__clang__) || defined(__GNUC__)
#define COMPILER_WARNING_NOTE(msg)                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    ([]() __attribute__((deprecated(msg))){})();                                                                       \
  }                                                                                                                    \
  while (0)
#else
#define COMPILER_WARNING_NOTE(msg)                                                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#endif

// TRACE MACROS
// These macros forward both the stringified expression list and the evaluated
// arguments, along with file and line info, to the corresponding trace functions.
// In constexpr context there doesn't seem to be any way to make these work - the only
// option is that in constexpr context they expand to nothing
#define TRACE(...)                                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (TRACE_DISABLE)                                                                                     \
      {}                                                                                                               \
      else                                                                                                             \
      {                                                                                                                \
        ::trace::TraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                                \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_IF(cond, ...)                                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (TRACE_DISABLE)                                                                                     \
      {}                                                                                                               \
      else if (cond)                                                                                                   \
      {                                                                                                                \
        if                                                                                                             \
          consteval {}                                                                                                 \
        else                                                                                                           \
        {                                                                                                              \
          ::trace::TraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_ONCE(...)                                                                                                \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (TRACE_DISABLE)                                                                                     \
      {}                                                                                                               \
      else                                                                                                             \
      {                                                                                                                \
        static std::atomic_flag _trace_once_flag = ATOMIC_FLAG_INIT;                                                   \
        if (!_trace_once_flag.test_and_set(std::memory_order_relaxed))                                                 \
        {                                                                                                              \
          ::trace::TraceOnceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                          \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_MODULE(m, ...)                                                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (ENABLE_TRACE_##m)                                                                                  \
      {                                                                                                                \
        if constexpr (TRACE_DISABLE)                                                                                   \
        {}                                                                                                             \
        else                                                                                                           \
        {                                                                                                              \
          ::trace::TraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                    \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_MODULE_IF(m, cond, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (ENABLE_TRACE_##m)                                                                                  \
      {                                                                                                                \
        if (cond)                                                                                                      \
        {                                                                                                              \
          if constexpr (TRACE_DISABLE)                                                                                 \
          {}                                                                                                           \
          else                                                                                                         \
          {                                                                                                            \
            ::trace::TraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                  \
          }                                                                                                            \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_STACK(...)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (TRACE_DISABLE)                                                                                     \
      {}                                                                                                               \
      else                                                                                                             \
      {                                                                                                                \
        ::trace::TraceStackCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                           \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_IF_STACK(cond, ...)                                                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (TRACE_DISABLE)                                                                                     \
      {}                                                                                                               \
      else if (cond)                                                                                                   \
      {                                                                                                                \
        if                                                                                                             \
          consteval {}                                                                                                 \
        else                                                                                                           \
        {                                                                                                              \
          ::trace::TraceStackCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                         \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_ONCE_STACK(...)                                                                                          \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (TRACE_DISABLE)                                                                                     \
      {}                                                                                                               \
      else                                                                                                             \
      {                                                                                                                \
        static std::atomic_flag _trace_once_stack_flag = ATOMIC_FLAG_INIT;                                             \
        if (!_trace_once_stack_flag.test_and_set(std::memory_order_relaxed))                                           \
        {                                                                                                              \
          ::trace::TraceStackOnceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                     \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_MODULE_STACK(m, ...)                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (ENABLE_TRACE_##m)                                                                                  \
      {                                                                                                                \
        if constexpr (TRACE_DISABLE)                                                                                   \
        {}                                                                                                             \
        else                                                                                                           \
        {                                                                                                              \
          ::trace::TraceModuleStackCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));               \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_MODULE_IF_STACK(m, cond, ...)                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (ENABLE_TRACE_##m)                                                                                  \
      {                                                                                                                \
        if (cond)                                                                                                      \
        {                                                                                                              \
          if constexpr (TRACE_DISABLE)                                                                                 \
          {}                                                                                                           \
          else                                                                                                         \
          {                                                                                                            \
            ::trace::TraceModuleStackCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));             \
          }                                                                                                            \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

// CHECK and PRECONDITION MACROS
// These macros check a condition and, if false, print diagnostic information and abort.
// They forward additional debug information similarly to the TRACE macros.

#define CHECK(cond, ...)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
      ::trace::CheckCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                           \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define CHECK_EQUAL(a, b, ...)                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!((a) == (b)))                                                                                                 \
    {                                                                                                                  \
      ::trace::CheckEqualCall(#a, #b, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a,                 \
                              b __VA_OPT__(, __VA_ARGS__));                                                            \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define CHECK_FLOATING_EQ(a, b, ...)                                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    auto va = (a);                                                                                                     \
    auto vb = (b);                                                                                                     \
    using T = std::decay_t<decltype(va)>;                                                                              \
    static_assert(::uni20::check::is_ulp_comparable<T>, "CHECK_FLOATING_EQ requires a scalar type");                   \
    ::std::int64_t ulps = ::trace::detail::get_ulps(a, b __VA_OPT__(, __VA_ARGS__));                                   \
    if (!::uni20::check::FloatingULP<T>::eq(va, vb, ulps))                                                             \
    {                                                                                                                  \
      ::trace::CheckFloatingEqCall(#a, #b, ulps, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a,      \
                                   b __VA_OPT__(, __VA_ARGS__));                                                       \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define PRECONDITION(cond, ...)                                                                                        \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
      ::trace::PreconditionCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                    \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define PRECONDITION_EQUAL(a, b, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!((a) == (b)))                                                                                                 \
    {                                                                                                                  \
      ::trace::PreconditionEqualCall(#a, #b, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a,          \
                                     b __VA_OPT__(, __VA_ARGS__));                                                     \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define PRECONDITION_FLOATING_EQ(a, b, ...)                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    auto va = (a);                                                                                                     \
    auto vb = (b);                                                                                                     \
    using T = std::decay_t<decltype(va)>;                                                                              \
    static_assert(::uni20::check::is_ulp_comparable<T>, "PRECONDITION_FLOATING_EQ requires a scalar type");            \
    ::std::int64_t ulps = ::trace::detail::get_ulps(a, b __VA_OPT__(, __VA_ARGS__));                                   \
    if (!::uni20::check::FloatingULP<T>::eq(va, vb, ulps))                                                             \
    {                                                                                                                  \
      ::trace::PreconditionFloatingEqCall(#a, #b, ulps, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__,  \
                                          a, b __VA_OPT__(, __VA_ARGS__));                                             \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

// PANIC is used to unconditionally abort
#define PANIC(...) trace::PanicCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));

// ERROR MACROS
// These macros report an error, printing debug information and then either abort
// or throw an exception based on a global configuration flag.
#define ERROR(...) trace::ErrorCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));

#define ERROR_IF(cond, ...)                                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if (cond)                                                                                                          \
    {                                                                                                                  \
      ::trace::ErrorIfCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                         \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

// ---------------------------------------------------------------------------
// DEBUG MACROS (compile to nothing if NDEBUG is defined)
#if defined(NDEBUG)
#define DEBUG_TRACE(...)                                                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_IF(...)                                                                                            \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_ONCE(...)                                                                                          \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_MODULE(...)                                                                                        \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_MODULE_IF(...)                                                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_STACK(...)                                                                                         \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_IF_STACK(...)                                                                                      \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_ONCE_STACK(...)                                                                                    \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_MODULE_STACK(...)                                                                                  \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_MODULE_IF_STACK(...)                                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_CHECK(...)                                                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_CHECK_EQUAL(...)                                                                                         \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_CHECK_FLOATING_EQ(...)                                                                                   \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_PRECONDITION(...)                                                                                        \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_PRECONDITION_EQUAL(...)                                                                                  \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_PRECONDITION_FLOATING_EQ(...)                                                                            \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#else

#define DEBUG_TRACE(...)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (TRACE_DISABLE)                                                                                       \
    {}                                                                                                                 \
    else                                                                                                               \
      ::trace::DebugTraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                             \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_IF(cond, ...)                                                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (TRACE_DISABLE)                                                                                       \
    {}                                                                                                                 \
    else if (cond)                                                                                                     \
    {                                                                                                                  \
      ::trace::DebugTraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                             \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_ONCE(...)                                                                                          \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (TRACE_DISABLE)                                                                                     \
      {}                                                                                                               \
      else                                                                                                             \
      {                                                                                                                \
        static std::atomic_flag _trace_once_flag = ATOMIC_FLAG_INIT;                                                   \
        if (!_trace_once_flag.test_and_set(std::memory_order_relaxed))                                                 \
        {                                                                                                              \
          ::trace::DebugTraceOnceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                     \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_MODULE(m, ...)                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (TRACE_DISABLE)                                                                                       \
    {}                                                                                                                 \
    else if constexpr (ENABLE_TRACE_##m)                                                                               \
    {                                                                                                                  \
      ::trace::DebugTraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                   \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_MODULE_IF(m, cond, ...)                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (TRACE_DISABLE)                                                                                       \
    {}                                                                                                                 \
    else if constexpr (ENABLE_TRACE_##m)                                                                               \
    {                                                                                                                  \
      if (cond)                                                                                                        \
      {                                                                                                                \
        ::trace::DebugTraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                 \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_STACK(...)                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (TRACE_DISABLE)                                                                                       \
    {}                                                                                                                 \
    else                                                                                                               \
      ::trace::DebugTraceStackCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                        \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_IF_STACK(cond, ...)                                                                                \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (TRACE_DISABLE)                                                                                       \
    {}                                                                                                                 \
    else if (cond)                                                                                                     \
    {                                                                                                                  \
      ::trace::DebugTraceStackCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                        \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_ONCE_STACK(...)                                                                                    \
  do                                                                                                                   \
  {                                                                                                                    \
    if                                                                                                                 \
      consteval {}                                                                                                     \
    else                                                                                                               \
    {                                                                                                                  \
      if constexpr (TRACE_DISABLE)                                                                                     \
      {}                                                                                                               \
      else                                                                                                             \
      {                                                                                                                \
        static std::atomic_flag _debug_trace_once_stack_flag = ATOMIC_FLAG_INIT;                                       \
        if (!_debug_trace_once_stack_flag.test_and_set(std::memory_order_relaxed))                                     \
        {                                                                                                              \
          ::trace::DebugTraceStackOnceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_MODULE_STACK(m, ...)                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (TRACE_DISABLE)                                                                                       \
    {}                                                                                                                 \
    else if constexpr (ENABLE_TRACE_##m)                                                                               \
    {                                                                                                                  \
      ::trace::DebugTraceModuleStackCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));              \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_MODULE_IF_STACK(m, cond, ...)                                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (TRACE_DISABLE)                                                                                       \
    {}                                                                                                                 \
    else if constexpr (ENABLE_TRACE_##m)                                                                               \
    {                                                                                                                  \
      if (cond)                                                                                                        \
      {                                                                                                                \
        ::trace::DebugTraceModuleStackCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));            \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_CHECK(cond, ...)                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
      ::trace::DebugCheckCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                      \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_CHECK_EQUAL(a, b, ...)                                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!((a) == (b)))                                                                                                 \
    {                                                                                                                  \
      ::trace::DebugCheckEqualCall(#a, #b, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a,            \
                                   b __VA_OPT__(, __VA_ARGS__));                                                       \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_CHECK_FLOATING_EQ(a, b, ...)                                                                             \
  do                                                                                                                   \
  {                                                                                                                    \
    auto va = (a);                                                                                                     \
    auto vb = (b);                                                                                                     \
    using T = std::decay_t<decltype(va)>;                                                                              \
    static_assert(::uni20::check::is_ulp_comparable<T>, "DEBUG_CHECK_FLOATING_EQ requires a scalar type");             \
    ::std::int64_t ulps = ::trace::detail::get_ulps(a, b __VA_OPT__(, __VA_ARGS__));                                   \
    if (!::uni20::check::FloatingULP<T>::eq(va, vb, ulps))                                                             \
    {                                                                                                                  \
      ::trace::DebugCheckFloatingEqCall(#a, #b, ulps, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a, \
                                        b __VA_OPT__(, __VA_ARGS__));                                                  \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_PRECONDITION(cond, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
      ::trace::DebugPreconditionCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));               \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_PRECONDITION_EQUAL(a, b, ...)                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!((a) == (b)))                                                                                                 \
    {                                                                                                                  \
      ::trace::DebugPreconditionEqualCall(#a, #b, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a,     \
                                          b __VA_OPT__(, __VA_ARGS__));                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_PRECONDITION_FLOATING_EQ(a, b, ...)                                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    auto va = (a);                                                                                                     \
    auto vb = (b);                                                                                                     \
    using T = std::decay_t<decltype(va)>;                                                                              \
    static_assert(::uni20::check::is_ulp_comparable<T>, "DEBUG_PRECONDITION_FLOATING_EQ requires a scalar type");      \
    ::std::int64_t ulps = ::trace::detail::get_ulps(a, b __VA_OPT__(, __VA_ARGS__));                                   \
    if (!::uni20::check::FloatingULP<T>::eq(va, vb, ulps))                                                             \
    {                                                                                                                  \
      ::trace::DebugPreconditionFloatingEqCall(#a, #b, ulps, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__,       \
                                               __LINE__, a, b __VA_OPT__(, __VA_ARGS__));                              \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#endif

#include "trace_impl.hpp"

namespace trace
{

//
// TracingBaseClass
//

/// \brief Compile-time string wrapper for use as a non-type template parameter.
///
/// Usage: Use as `CTStr{"ClassName"}` to pass a string literal as a template parameter.
/// @tparam N The length of the string literal, including the null terminator.
template <size_t N> struct BaseName
{
    char value[N]; ///< The wrapped string literal.
    /// \brief Construct from a string literal.
    /// \param str The string literal (e.g., "AsyncTask").
    constexpr BaseName(const char (&str)[N])
    {
      for (size_t i = 0; i < N; ++i)
        value[i] = str[i];
    }
    /// \brief Convert to std::string_view (excluding null terminator).
    constexpr operator std::string_view() const { return {value, N - 1}; }
};

/// \brief CRTP tracing base class that emits messages with demangled type name.
/// \ingroup trace
///
/// @tparam Tag Compile-time string, e.g., CTStr{"AsyncTask"}.
///
/// Inherit from this class to automatically trace construction, destruction,
/// and assignment of derived objects. All trace messages are tagged with the class name.
template <typename Derived> struct TracingBaseClass
{
    /// \brief Get this pointer as Derived* (for CRTP usage).
    Derived const* This() const noexcept { return static_cast<Derived const*>(this); }
    Derived* This() noexcept { return static_cast<Derived*>(this); }

    /// \brief Get demangled class name of the CRTP derived class.
    std::string DerivedName() const { return uni20::demangle::demangle(typeid(Derived).name()); }

    template <typename T> detail::TraceNameValue OtherPointer(T* x) const
    {
      return detail::TraceNameValue("other", fmt::format("{:p}", fmt::ptr(x)));
    }
    detail::TraceNameValue ThisPointer() const
    {
      return detail::TraceNameValue("this", fmt::format("{:p}", fmt::ptr(this)));
    }

    /// \brief Default constructor. Emits a trace message.
    TracingBaseClass() { TRACE(DerivedName() + " default constructor", ThisPointer()); }

    template <typename... Args> TracingBaseClass(Args&&... args)
    {
      TRACE(DerivedName() + " forwarding constructor", ThisPointer(), args...);
    }

    /// \brief Copy constructor. Emits a trace message.
    TracingBaseClass(const TracingBaseClass& other)
    {
      TRACE(DerivedName() + " copy constructor", ThisPointer(), OtherPointer(&other));
    }

    /// \brief Move constructor. Emits a trace message.
    TracingBaseClass(TracingBaseClass&& other) noexcept
    {
      TRACE(DerivedName() + " move constructor", ThisPointer(), OtherPointer(&other));
    }

    /// \brief Move constructor. Emits a trace message.
    TracingBaseClass(Derived&& other) noexcept
    {
      TRACE(DerivedName() + " move constructor", ThisPointer(), OtherPointer(&other));
    }

    /// \brief Copy assignment. Emits a trace message.
    TracingBaseClass& operator=(const TracingBaseClass& other)
    {
      if (this != &other)
      {
        TRACE(DerivedName() + " copy assignment", ThisPointer(), OtherPointer(&other));
      }
      return *this;
    }

    /// \brief Move assignment. Emits a trace message.
    TracingBaseClass& operator=(TracingBaseClass&& other) noexcept
    {
      if (this != &other)
      {
        TRACE(DerivedName() + " move assignment", ThisPointer(), OtherPointer(&other));
      }
      return *this;
    }

    /// \brief Destructor. Emits a trace message.
    ~TracingBaseClass() { TRACE(DerivedName() + " destructor", ThisPointer()); }
};

#define TRACE_BASE_CLASS(Name) trace::TracingBaseClass<trace::BaseName(TRACE_STRINGIZE(Name)), Name>

// #define TRACE_BASE_CLASS(Identifier) trace::TracingBaseClass<trace::BaseName{#Identifier}, Identifier>
} // namespace trace
