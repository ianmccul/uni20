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
 *
 * \par Submodules
 * - \ref backend_blas — BLAS integration and vendor selection utilities.
 * - \ref backend_cuda — CUDA runtime orchestration helpers.
 * - \ref backend_cusolver — cuSOLVER-specific linear algebra adapters.
 */

/**
 * \defgroup backend_cuda CUDA backend integration
 * \ingroup backend
 * \brief Glue that binds Uni20 abstractions to CUDA runtime facilities.
 */

/**
 * \defgroup backend_cusolver cuSOLVER backend integration
 * \ingroup backend
 * \brief Helpers that adapt cuSOLVER handles and status codes to Uni20 abstractions.
 */

/**
 * \defgroup backend_blas BLAS backend integration
 * \ingroup backend
 * \brief Glue that binds Uni20 abstractions to BLAS vendor implementations.
 *
 * \par Submodules
 * - \ref backend_blas_mkl — Intel MKL-backed BLAS shims.
 * - \ref backend_blas_reference — Reference BLAS wrappers used for testing and fallbacks.
 */

/**
 * \defgroup internal Backend implementation details
 * \ingroup backend
 * \brief Internal macros and helpers that support backend integrations without being part of the
 *        public API surface.
 */

/// \brief Converts the supplied token into a string literal without macro expansion.
/// \param x Token to stringify.
/// \return A string literal containing the textual representation of \p x.
/// \ingroup internal
#define UNI20_INTERNAL_STRINGIFY_TOKEN(x) #x

/// \brief Produces the log-friendly spelling of a backend API symbol.
/// \param func Macro argument that names the symbol being traced.
/// \return String literal suitable for concatenating into trace output.
/// \ingroup internal
#define UNI20_INTERNAL_API_CALL_STRINGIZE(func) UNI20_INTERNAL_STRINGIFY_TOKEN(func)

/// \brief Emits a trace log entry for an outgoing backend API call.
/// \details This macro wraps \ref TRACE_MODULE so every backend API call records a side
///          effect visible to the tracing subsystem.
/// \param module Name of the backend module emitting the trace entry. This must match one of
///        the trace channels registered in the top-level CMake configuration.
/// \param func Symbol name for the backend function being invoked. The value is stringized for
///        inclusion in the trace message but is otherwise unmodified.
/// \param ... Optional comma-separated arguments that mirror the runtime parameters forwarded to
///        the backend function. Each argument is appended to the trace entry when provided.
/// \ingroup backend
#define UNI20_API_CALL(module, func, ...)                                                                              \
  TRACE_MODULE(module,                                                                                                \
               "Calling API function " UNI20_INTERNAL_API_CALL_STRINGIZE(func) __VA_OPT__(, __VA_ARGS__))
