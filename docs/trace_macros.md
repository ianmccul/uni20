# Trace Macros Developer Guide

This guide explains the purpose, usage, and customization of the various trace macros provided in the system. These macros allow you to output detailed trace messages during development and debugging. They include file/line information, support conditional and module‐specific tracing, and come with both always‐on and debug-only versions.

## Overview of Macros

The system provides the following macros:

- **General Tracing**
  - `TRACE(...)`
    Outputs a trace message along with the file name and line number.
  - `TRACE_IF(condition, ...)`
    Outputs a trace message **only if** the specified condition is true.

- **Module-Specific Tracing**
  - `TRACE_MODULE(module, ...)`
    Outputs a trace message for a given module. The module’s tracing flag is controlled by a compile‑time boolean (e.g. `ENABLE_TRACE_BLAS3`), generated automatically via CMake.
  - `TRACE_MODULE_IF(module, condition, ...)`
    Outputs a trace message for a module **only if** both the module is enabled and the condition is true.

- **Debug-Only Variants** (emit no code when `NDEBUG` is defined)
  - `DEBUG_TRACE(...)`
  - `DEBUG_TRACE_IF(condition, ...)`
  - `DEBUG_TRACE_MODULE(module, ...)`
  - `DEBUG_TRACE_MODULE_IF(module, condition, ...)`

Each macro automatically inserts the source file and line number into the trace output, so that you can quickly locate where the trace message was generated.

## How to Use the Macros

### Basic Usage

- **General Trace**

  ```cpp
  TRACE("Starting computation", x, y);
  ```

  This expands to a call to the trace function (e.g., `OutputTraceCall`) with the stringified expression list and the current file and line. The output might look like:

  ```
  TRACE at /path/to/source.cpp:123 : Starting computation, x = 42, y = 17
  ```

  If the parameter is a string literal (such as `"Starting computation"`), or contains a string literal (such as `"Result:" + R`) then the value of the string is displayed alone.  If the parameter is any other type of variable or expression, then the output is of the form `expression = value`.  There is no need to enclose expressions in brackets if they contain commas, so for example `TRACE(std::vector<int, std::allocator<int>>{0,1,2});` works fine. Even though the preprocessor treats this as a macro with two parameters, the trace function recognises that they are a single C++ parameter.

- **Conditional Trace**

  ```cpp
  TRACE_IF(x > 0, x);
  ```

  The trace message is printed only if the condition (`x > 0`) is true.

- **Module-Specific Trace**

  ```cpp
  TRACE_MODULE(BLAS3, "BLAS3: performing matrix multiplication with", A, B, C);
  ```

  This macro checks the compile‑time flag (e.g. `ENABLE_TRACE_BLAS3`) and, if enabled, outputs the trace message. If the flag is false, the entire expression is discarded - no code is emitted.

  There is a module named `TESTMODULE` that defaults to `enabled`, which is intended for testing the `TRACE_MODULE` macro itself.

- **Debug-Only Trace**

  ```cpp
  DEBUG_TRACE("Debug: current state =", currentState);
  ```

  When building a release version (with `NDEBUG` defined), these macros compile to no code.

### How the File and Line Number are Included

Each macro automatically passes `__FILE__` and `__LINE__` to the underlying trace function. For example, the basic `TRACE` macro is defined as follows:

```cpp
#define TRACE(...) OutputTraceCall(#__VA_ARGS__, __FILE__, __LINE__, __VA_ARGS__)
```

The trace output is then prefixed with something like:

```
TRACE at /path/to/source.cpp:123 : ...
```

## Customizing the Output Formatting

The trace macros ultimately rely on a helper function called `formatValue` to convert values into strings. The default implementation uses `fmt::formatter<T>`, so if you define `fmt` style I/O for your custom types then it will work with the `TRACE` macros. But if you want to customize the output, or you want to `TRACE` objects that do not have a `fmt::formatter<T>` specialization, you can customize the `trace::formatValue` function.

### Example: Custom Formatting for a User-Defined Type

Suppose you have a custom type `MyType` and want to display it in a special way:

```cpp
struct MyType {
    int id;
    std::string name;
};

// Default version for most types is already provided:
template<typename T>
std::string formatValue(const T& value, const FormattingOptions& opts) {
    return fmt::format("{}", value);
}

// Provide a custom overload for MyType:
std::string formatValue(const MyType& value, const FormattingOptions& opts) {
    // Customize the output format for MyType
    return fmt::format("MyType(id: {}, name: {})", value.id, value.name);
}
```

### Adjusting Floating-Point Precision

If you want to adjust the precision used for floating-point values, you can adjust the parameters, eg
```cpp
trace::formatting_options.fp_precision<double> = 10;
```

### Output stream

By default trace messages are sent to `stderr`.  You can redirect them to any other `FILE*` object by calling `trace::formatting_options.set_output_stream(FILE* stream)` to some other stream, for example `stdout`:
```cpp
trace::formatting_options.set_output_stream(stdout);
```

### Color output

The `TRACE` macro has color support. You can control this with `trace::formatting_options::set_color_output(c)`, where `c` is one of `trace::FormattingOptions::yes`, `trace::FormattingOptions::no`, or `trace::FormattingOptions::terminal`. If the color mode is set to `terminal`, then color output will be enabled if the output stream is a tty terminal, otherwise it will be disabled.

## Using `trace.hpp` in Different Projects

The module-specific macros (e.g. `TRACE_MODULE(BLAS3, ...)`) rely on compile‑time constants generated from a configuration header. In the top-level CMake, it defines the list of available `SUBMODULES`, which can easily be added to. To use the TRACE macros in another project, you can incorporate it into `CMakeLists.txt` as follows:

1. **Define Modules in CMake**

   In your `CMakeLists.txt`:

   ```cmake
   set(SUBMODULES MODULE1 MODULE2 MODULE3)

   foreach(module IN LISTS SUBMODULES)
     string(TOUPPER "${module}" MODULE_UPPER)
     option(ENABLE_TRACE_${MODULE_UPPER} "Enable tracing for ${module}" OFF)  # or default to ON if you prefer
   endforeach()

   # Build the definitions string for config.h
   set(TRACE_MODULE_DEFINITIONS "")
   foreach(module IN LISTS TRACE_MODULES)
     string(TOUPPER "${module}" MODULE_UPPER)
     if(${ENABLE_TRACE_${MODULE_UPPER}})
       set(TRACE_MODULE_DEFINITIONS "${TRACE_MODULE_DEFINITIONS}#define ENABLE_TRACE_${MODULE_UPPER} 1\n")
     else()
       set(TRACE_MODULE_DEFINITIONS "${TRACE_MODULE_DEFINITIONS}#define ENABLE_TRACE_${MODULE_UPPER} 0\n")
     endif()
   endforeach()

   configure_file(
       ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
       ${CMAKE_CURRENT_BINARY_DIR}/config.h
       @ONLY
   )

   include_directories(${CMAKE_CURRENT_BINARY_DIR})
   ```

2. **Template Header (`config.h.in`)**

   ```c
   #pragma once
   // Auto-generated trace module enable flags:
   @TRACE_MODULE_DEFINITIONS@
   ```

   Now you can write:

   ```cpp
   #include "trace.hpp"

   [...]

   TRACE_MODULE(MODULE1, "called with", a, b, c);
   ```
