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

  If the output of a variable spans multiple lines, then it will display that variable on a separate line. The trace library has built-in support for displaying containers (and nested containers up to 2 levels; this could be extended in the future). For example
  ```cpp

  std::vector<std::vector<int>> vec2{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  foo = 42;
  TRACE(vec2, foo);
  ```
  produces output similar to
  ```bash
  TRACE at file:line :
  vec2 = [ [ 1, 2, 3 ],
           [ 4, 5, 6 ],
           [ 7, 8, 9 ] ], foo = 42
  ```
  Note that the following parameters are not split onto separate lines (unless they also span multiple lines) in order to keep the output as compact as possible.

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

### Color output

The `TRACE` macro has color support. These are controlled by environment variables, and can be overridden under program control.
It supports both the traditional 16-color palette and 24-bit truecolor (RGB) values, along with text attributes. These options can be combined using semicolons in style strings and can be controlled via environment variables. Note that not all terminals support color output, and some terminals might support 16-color palettes but not 24-bit RGB values.

### Environment Variables for Color Customization

The trace library supports rich color customization via a set of environment variables. These variables control the color settings used for various parts of the trace output. You can override the defaults without changing any code—simply set the appropriate environment variable in your shell.

Below is a table summarizing the basic environment variables, their purposes, and their default values:

| Environment Variable             | Purpose                                         | Default Value          |
| -------------------------------- | ------------------------------------------------ | ---------------------- |
| **UNI20_COLOR**                  | Controls overall color usage. Acceptable values are:       | `auto`     |
|                                  | - `yes` – Always use color                           |
|                                  | - `no` – Never use color                 |
|                                  | - `auto` – Use color only if the output stream is a terminal     |   
| **UNI20_COLOR_TRACE**            | Color for general `TRACE` messages.             |`Cyan`                 |
| **UNI20_COLOR_DEBUG_TRACE**      | Color for `DEBUG_TRACE` messages.                       | `Green`                |
| **UNI20_COLOR_TRACE_EXPR**       | Color for expression names in trace output (e.g. the left-hand side of `name = value`).   | `Blue`        |
| **UNI20_COLOR_TRACE_VALUE**      | Color for expression values in trace output (e.g. the evaluated result).            | *(Empty – no override)* |
| **UNI20_COLOR_TRACE_STRING**      | Color for string-literal-like values in trace output                    | `LightBlue` |
| **UNI20_COLOR_TRACE_MODULE**     | Default color for module-specific trace messages (used by `TRACE_MODULE`).     | `LightCyan`  
| **UNI20_COLOR_TRACE_FILENAME**   | Color for displaying filenames in trace output.         | `Red`    |
| **UNI20_COLOR_TRACE_LINE**       | Color for displaying line numbers in trace output.         | `Bold`        |

In addition, you can customize the color for specific trace modules using environment variables of the form:

- **UNI20_COLOR_MODULE_XXXX**  
  Where `XXXX` is the module name. This variable overrides the default module color (`UNI20_COLOR_TRACE`) for that particular module.

### Usage Example

For example, in a Unix-like shell you might configure your trace output like this:

```bash
export UNI20_COLOR=yes
export UNI20_COLOR_TRACE=Magenta
export UNI20_COLOR_DEBUG_TRACE=Yellow
export UNI20_COLOR_TRACE_EXPR="rgb(50,25,250);bold;underline"
export UNI20_COLOR_TRACE_VALUE="#00FF00"
export UNI20_COLOR_TRACE_MODULE="LightGreen"
export UNI20_COLOR_TRACE_FILENAME="Red;Bold"
export UNI20_COLOR_TRACE_LINE=Bold
export UNI20_COLOR_TRACE_String=LightMagenta
export UNI20_COLOR_MODULE_TESTMODULE="fg:Black;bg:DarkGray"

examples/trace_example
```

You can also control these options in C++ code using `trace::formatting_options` member functions.

### Basic Named Colors

The library supports the 16 basic colors for both foreground and background. For foreground colors, the standard ANSI codes are used (30–37 for standard colors and 90–97 for bright colors). For background colors, the corresponding codes are 40–47 (standard) and 100–107 (bright).

| **Named Color**  | **Foreground Code** | **Background Code** |
|------------------|---------------------|---------------------|
| Black            | 30                  | 40                  |
| Red              | 31                  | 41                  |
| Green            | 32                  | 42                  |
| Yellow           | 33                  | 43                  |
| Blue             | 34                  | 44                  |
| Magenta          | 35                  | 45                  |
| Cyan             | 36                  | 46                  |
| LightGray        | 37                  | 47                  |
| DarkGray         | 90                  | 100                 |
| LightRed         | 91                  | 101                 |
| LightGreen       | 92                  | 102                 |
| LightYellow      | 93                  | 103                 |
| LightBlue        | 94                  | 104                 |
| LightMagenta     | 95                  | 105                 |
| LightCyan        | 96                  | 106                 |
| White            | 97                  | 107                 |

### Text Attributes

These modifiers affect the text appearance and can be combined with any color:

- **Bold** (ANSI code 1): Makes the text appear bolder.
- **Dim** (ANSI code 2): Reduces the brightness of the text.
- **Underline** (ANSI code 4): Underlines the text.

*Note:* Bold and Dim are intended to be mutually exclusive in terms of intensity, but either can be combined with Underline.

### RGB / Truecolor Options

For more precise color control, you can specify truecolor values:

- **RGB Function Notation:**  
  e.g., `rgb(255,0,0)` sets the color to red.  
  - Foreground: Generates the escape sequence fragment `38;2;255;0;0`  
  - Background: Generates `48;2;255;0;0`

- **Hexadecimal Notation:**  
  e.g., `#FF0000` (or shorthand `#F00`) is also supported and parsed as an RGB value.

### Combining Options

Options within a single style component are separated by semicolons:

- `"Red;Bold"` sets the foreground to red with bold text.
- `"fg:rgb(0,128,255);Underline, bg:#FFFFFF"` sets the foreground to a specific RGB blue with underline and the background to white.

If no target is specified, the default is to apply the style to the foreground.

### Output stream

By default trace messages are sent to `stderr`.  You can redirect them to any other `FILE*` object by calling `trace::formatting_options.set_output_stream(FILE* stream)` to some other stream, for example `stdout`:
```cpp
trace::formatting_options.set_output_stream(stdout);
```

### Adjusting Floating-Point Precision

If you want to adjust the precision used for floating-point values, you can adjust the parameters, eg
```cpp
trace::formatting_options.fp_precision<double> = 10;
```
It is not currently possible to set these precisions as environment variables.

### Supporting new types

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
