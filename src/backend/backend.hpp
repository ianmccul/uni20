#pragma once

#include "common/trace.hpp"
#include "config.hpp"

/**
 * \defgroup backend Backend subsystem
 * \brief Facilities that coordinate concrete compute backends with instrumentation hooks.
 *
 * \details The backend subsystem centralizes all compile- and run-time glue that selects
 * different accelerator or BLAS providers.  Shared helpers here also implement the
 * tracing/logging hooks documented in docs/Doxygen.md so public entry points consistently
 * record side effects.
 */

/**
 * \defgroup backend_blas BLAS backend integration
 * \ingroup backend
 * \brief Glue that binds Uni20 abstractions to BLAS vendor implementations.
 */

///
/// \brief Converts the supplied token into a string literal without macro expansion.
/// \param x Token to stringify.
/// \return A string literal containing the textual representation of \p x.
/// \ingroup internal
#define UNI20_STR(x) #x

#define UNI20_API_CALL_STRINGIZE(func) UNI20_STR(func)

/// \brief Emits a trace log entry for an outgoing backend API call.
/// \details This macro wraps \ref TRACE_MODULE so every backend API call records a side
///          effect visible to the tracing subsystem.
/// \param module Name of the backend module emitting the trace entry.
///        This must be one of the defined modules in the top-level CMakeLists.txt
/// \param func function name.
/// \param ... function parameters.
/// \ingroup backend
#define UNI20_API_CALL(module, func, ...)                                                                              \
  TRACE_MODULE(module, "Calling API function " UNI20_API_CALL_STRINGIZE(func) __VA_OPT__(, __VA_ARGS__))
