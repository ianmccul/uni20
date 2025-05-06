#pragma once

#include "common/trace.hpp"
#include "config.hpp"

// intended as a hook to add tracing/logging
/// Step 1: ordinary stringize
#define UNI20_STR(x) #x
/// Step 2: force expansion, then stringize
#define UNI20_API_CALL_STRINGIZE(func) UNI20_STR(func)
#define UNI20_API_CALL(module, func, ...)                                                                              \
  TRACE_MODULE(module, "Calling API function " UNI20_API_CALL_STRINGIZE(func) __VA_OPT__(, __VA_ARGS__))

namespace uni20
{

struct cpu_tag
{};

} // namespace uni20
