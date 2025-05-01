# Trace Macros Developer Guide

This guide explains the purpose, usage, and customization of the various trace macros provided in the header `src/common/trace.hpp`. These macros allow you to output detailed trace messages during development and debugging. They include file/line information, support conditional and module‐specific tracing, and come with both always‐on and debug-only versions.

## Overview of Macros

The system provides the following macros for logging, diagnostics, and error handling:

### General Tracing

- **`TRACE(...)`**  
  Outputs a trace message along with the source file name and line number.  

- **`TRACE_IF(condition, ...)`**  
  Outputs a trace message **only if** the specified condition is true.

  These macros are intended for debugging; generally speaking they should not appear in a stable release.

  By default, the `TRACE(...)` and `TRACE_IF(...)` output includes a **timestamp** (in microseconds) and the **current thread ID**. This helps disambiguate interleaved trace output in multi-threaded programs or analyze timing between events. These can be controlled by environment variables. To control the timestamp display set the environment variable `UNI20_TRACE_TIMESTAMP=x`, where `x` is `1` or `yes` to show a timestamp (the default), or `0` or `no` to disable the timestamp. Similarly, to control the thread ID display, set `UNI20_TRACE_THREAD_ID=x`.

  **Color output** can also be customized via environment variables; see section **Color output** below.

### Module-Specific Tracing

- **`TRACE_MODULE(module, ...)`**  
  Outputs a trace message for a given module. The module’s tracing flag is controlled by a configure option (e.g. `ENABLE_TRACE_BLAS`), which is generated automatically via `cmake`. To enable tracing of a given module, use `cmake -DENABLE_TRACE_BLAS ....`.

- **`TRACE_MODULE_IF(module, condition, ...)`**  
  Outputs a trace message for a module **only if** both the module is enabled and the specified condition is true.

Because the module-specific macros are only enabled if configured via `cmake`, there is no problem to use these macros freely. A typical use for a module-specific `TRACE` macro would be to trace all calls to an external API, for example.

Like general trace macros, module-specific trace output supports optional **timestamps** and **thread IDs**, controlled via environment variables. You can customize these globally or per-module.

### Debug-Only Variants  
These macros are compiled away (i.e. expand to no code) when `NDEBUG` is defined:

- **`DEBUG_TRACE(...)`**
- **`DEBUG_TRACE_IF(condition, ...)`**
- **`DEBUG_TRACE_MODULE(module, ...)`**
- **`DEBUG_TRACE_MODULE_IF(module, condition, ...)`**

Each of these macros automatically inserts the source file name and line number into the trace output, so that you can quickly locate where the message was generated. Although the `DEBUG` variants of the macros will not have an effect on a *Release* build, they should
still be removed as soon as possible, since spurious `DEBUG_TRACE` calls will cause unnecessary spam when debugging.

### Precondition / Check Macros

These macros serve as general assertion mechanisms for logic errors that indicate that the program is in an ill-defined state and it does not make sense to continue execution. They are intended for use in enforcing preconditions and checking invariants. These macros display a message and then call `std::abort()`.

- **`PRECONDITION(condition, ...)`**  
  Similar to `CHECK`, but prints "PRECONDITION" in its output to indicate that the condition is a precondition.  
  *Intended for validating function inputs and preconditions.*

- **`PRECONDITION_EQUAL(a, b, ...)`**  
  Checks that `a` equals `b` as a precondition. If not, it prints a precondition-specific failure message and aborts.

- **`CHECK(condition, ...)`**  
  Checks a condition and, if it evaluates to false, prints a diagnostic message and aborts execution.  
  *Intended for general assert checks.*

- **`CHECK_EQUAL(a, b, ...)`**  
  Checks that `a` equals `b`. If not, it prints a message showing the stringified expressions (including `a` and `b`) along with any additional debug information, then aborts.

The `PRECONDITION` macros behave identically to their `CHECK` counterparts except for the label printed in the diagnostic output (e.g. "PRECONDITION" vs. "CHECK"). This distinction helps differentiate between general runtime assertions and conditions that must be met before a function is executed.

- **`PANIC(...)`**
  Displays a message and immediately calls `std::abort()`.

### Error Macros

These macros are designed for handling logic errors where something unexpected has occurred but it is not necessarily fatal (e.g. invalid user input). While such errors typically result in aborting the program, the behavior is configurable via a global flag. In a typical C++ program, the default is to abort execution; however, when integrated with Python bindings (or similar), you can adjust the behavior so that an exception is thrown instead of aborting.

- **`ERROR(...)`**  
  Unconditionally reports an error by printing diagnostic information (including file, line, and any additional context) and then either aborts the program or throws an exception based on the global error configuration.

- **`ERROR_IF(condition, ...)`**  
  Reports an error (as described above) only if the specified condition is true.

By default, these macros display a message and call `std::abort()`, however they can be configured to instead throw an exception by calling `trace::formatting_options.set_errors_abort(false)`.

## How to Use the Macros

### Basic Usage

- **General Trace**

  ```cpp
  TRACE("Starting computation", x, y);
  ```

  This expands to a call to the trace function (e.g., `TraceCall`) with the stringified expression list and the current file and line number. The output might look like:

  ```
  [2025-02-02 10:59:29.011873] [TID 3c298ab44ebd69cd] TRACE at /path/to/source.cpp:123 : Starting computation, x = 42, y = 17
  ```

  *Behavior:*  
  - If the parameter is a string literal (e.g. `"Starting computation"`), or contains a string literal (e.g. `"Result:" + R`), then the literal value is displayed alone.  
  - Otherwise, the output shows the expression and its evaluated value (e.g. `x = 42`).

  Expressions containing commas need some care. If the comma is enclosed in round or square or curly brackets, then there is no problem. So `TRACE(std::vector<int>{0,1,2});` or `TRACE(mdarray[0,1,2])` work fine. But a comma that is not enclosed in brackets will confuse the output. In particular, template angle brackets, eg `TRACE(vector<int, alloc>(5).size())` will work, but the output will be garbled. The first parameter of `TRACE_IF(cond, ...)`, `CHECK(cond, ...)` etc need special care, since they are required to be a single *preprocessor token*, meaning that it is often necessary to enclose the expression in brackets. This is harmless, and is generally always possible, eg `CHECK((a[i,j] > 2), i, j)`.


  If the output of a variable spans multiple lines, then it will display that variable on a separate line. The trace library has built-in support for displaying containers (and nested containers up to 2 levels; this could be extended in the future). For example
  ```cpp

  std::vector<std::vector<int>> vec2d{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  int foo = 42;
  TRACE(vec2d, foo);
  ```
  produces output similar to
  ```
  [Timestamp] [ThreadID] TRACE at file:line :
  vec2d = [ [ 1, 2, 3 ],
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

  This macro checks the compile‑time flag (e.g. `ENABLE_TRACE_BLAS3`) and, if enabled, outputs the trace message along with the module name. If the flag is false, no code is emitted.

  *Note:* There is a module named `TESTMODULE` that defaults to enabled, intended for testing the `TRACE_MODULE` macro itself.

- **Debug-Only Trace**

  ```cpp
  DEBUG_TRACE("Debug: current state =", currentState);
  ```

  When building a release version (with `NDEBUG` defined), these macros compile to no code.

- **Precondition / Check Macros**

  These macros are used for asserting that certain conditions (or preconditions) hold true. They print diagnostic information (including file and line) and then abort execution if the condition is false.

  - **`CHECK(condition, ...)`**  
    Used for general assertions. For example:

    ```cpp
    CHECK(x != 0, "x must not be zero");
    ```

    If `x` is zero, this macro prints a message (including the stringified condition and any additional context) and aborts.

  - **`CHECK_EQUAL(a, b, ...)`**  
    Checks that `a` equals `b`. If not, it prints both the expressions and their evaluated values along with any extra debug information, then aborts.

    ```cpp
    CHECK_EQUAL(foo, 42, "foo must equal 42");
    ```

  - **`PRECONDITION(condition, ...)`**  
    Similar to `CHECK`, but intended for precondition checks on function inputs. The printed label is "PRECONDITION" rather than "CHECK".

    ```cpp
    PRECONDITION(ptr != nullptr, "ptr cannot be null");
    ```

  - **`PRECONDITION_EQUAL(a, b, ...)`**  
    Similar to `CHECK_EQUAL`, but labels the message as a precondition check.

    ```cpp
    PRECONDITION_EQUAL(x, y, "x and y must be equal as preconditions");
    ```

- **Error Macros**

  These macros are used when an error has occurred that is recoverable (for example, invalid user input). In a typical C++ program, the default behavior is to abort execution, but this can be configured (for example, by the Python bindings) to throw an exception instead.

  - **`ERROR(...)`**  
    Unconditionally reports an error. It prints a message with diagnostic information (file, line, and any additional context) and then either aborts or throws an exception based on a global configuration flag.

    ```cpp
    ERROR("Invalid parameter value for mode");
    ```

  - **`ERROR_IF(condition, ...)`**  
    Reports an error only if the specified condition is true.

    ```cpp
    ERROR_IF(user_input < 0, "User input must be non-negative");
    ```

Each of these macros automatically inserts the source file and line number into the output, ensuring that debugging and trace information is precise and helps quickly locate the origin of the message.

### Enabling Module-Specific Tracing with CMake Options

The trace library controls whether tracing is enabled for specific modules through compile‑time flags. For example, the macro `TRACE_MODULE(BLAS3, ...)` checks the flag `ENABLE_TRACE_BLAS3`. You can control this flag by passing a CMake option when configuring your build.

#### Enabling Tracing for a Module

To enable tracing for a module, for example `BLAS3`, you can run:

```bash
cmake -DENABLE_TRACE_BLAS3=ON path/to/your/source
```

This sets the CMake variable `ENABLE_TRACE_BLAS3` to `ON` (the default is usually OFF if not specified). CMake will then generate a configuration header (for example, `config.h`) in which `ENABLE_TRACE_BLAS3` is defined, and your source code will include that header.

## Customizing the Output Formatting

The library allows a lot of customization via **environment variables**.  These are generally of the form `UNI20_xxxx`, for example `UNI20_TRACE_COLOR` controls whether the macros should use color output. Each variable can be customized per-module, by using the form `UNI20_xxxx_MODULE_modulename`. For example, `UNI20_TRACE_COLOR_MODULE_BLAS` controls whether color output should be used in `TRACE_MODULE(BLAS,...)` calls. If no module-specific override exists, then the global override will be used. If there is no global override then it will use a sensible default value.

All of the customization described in this section can be controlled by the program as well; see **Advanced Usage** below for details.

### Output stream

By default trace messages are sent to `stderr`. This can be redirected to a different stream or a file by setting the environment variable `UNI20_TRACEFILE=name`, where `name` is:

| **`name`** | **Meaning** |
|------------|-------------|
| `stderr`   | standard error output (default) |
|  `-` or `stdout` | standard output |
| `filename` | direct output to a file, overwrite any existing file of that name |
| `+filename` | direct output to a file, append to the end  |

For example,
```bash
export UNI20_TRACEFILE=+trace.out
```
will redirect `TRACE` output to the file `trace.out`. If that file already exists, then it will appending new text to the end.  If we had specified instead `UNI20_TRACEFILE=trace.out` (without the `+`) then it would overwrite any existing `trace.out` file.


You can control these individually by module using the environment variable `UNI20_TRACEFILE_MODULE_xxxx=...` for the module name `xxxx`, or under program control using `trace::formatting_options("ModuleName").set_output_stream(...)`. For example,

```cpp
trace::formatting_options("BLAS").set_output_stream(stdout);
```

### Adjusting Floating-Point Precision

If you want to adjust the precision used for floating-point values, you can adjust the number of digits displayed by using the environment variables `UNI20_FP_PRECISION_FLOAT32=n` and `UNI20_FP_PRECISION_FLOAT64=n`. Or under program control using the `formatting_options()` variables `fp_precision_float32` and `fp_precision_float64`. For example,
```cpp
trace::formatting_options().fp_precision_float64 = 10;
```

The module-specific overrides also work, eg `UNI20_FP_PRECISION_FLOAT32_MODULE_xxxx` etc.

### Color output

The `TRACE` macro has color support. These are controlled by environment variables, and can be overridden under program control.
It supports both the traditional 16-color palette and 24-bit truecolor (RGB) values, along with text attributes. These options can be combined using semicolons in style strings and can be controlled via environment variables. Note that not all terminals support color output, and some terminals might support 16-color palettes but not 24-bit RGB values.

Below is a table summarizing the basic environment variables for color output, and their default values:

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
| **UNI20_COLOR_TRACE_STRING**     | Color for string-literal-like values in trace output                    | `LightBlue` |
| **UNI20_COLOR_TRACE_MODULE**     | Default color for module-specific trace messages (used by `TRACE_MODULE`).     | `Cyan;Bold`  
| **UNI20_COLOR_TRACE_FILENAME**   | Color for displaying filenames in trace output.         | `Red`    |
| **UNI20_COLOR_TRACE_LINE**       | Color for displaying line numbers in trace output.         | `Bold`        |
| **UNI20_COLOR_TIMESTAMP**        | Color for the timestamp field in `TRACE` output.     | `LightBlack`           |
| **UNI20_COLOR_THREAD_ID**        | Color for the thread ID field in `TRACE` output.     | `LightMagenta`     |
| **UNI20_COLOR_CHECK**            | Color for `CHECK` messages (general assertions).                                                                                                                   | `Red`                  |
| **UNI20_COLOR_DEBUG_CHECK**      | Color for `DEBUG_CHECK` messages (assertions only enabled in debug builds).                                                                                          | `Red`                  |
| **UNI20_COLOR_PRECONDITION**     | Color for `PRECONDITION` messages (assertions on function preconditions).                                                                                            | `Red`                  |
| **UNI20_COLOR_DEBUG_PRECONDITION**| Color for `DEBUG_PRECONDITION` messages (debug-only precondition checks).                                                                                            | `Red`                  |
| **UNI20_COLOR_PANIC**            | Color for `PANIC` messages (indicating unrecoverable errors).                                                                                                      | `Red`                  |
| **UNI20_COLOR_ERROR**            | Color for `ERROR` messages (unexpected errors that may result in an exception).                                                          | `Red`                  |

In addition, you can customize the color for specific trace modules using environment variables of the form:

- **UNI20_COLOR_TRACE_xxxxx_MODULE_\<MODULE>**  
  corresponding to each **UNI20_COLOR_TRACE_xxxxx** variables above. This variable overrides the default module color (**UNI20_COLOR_TRACE_xxxxx**) for that particular module, if you want to customize the output per-module.

### Environment Variables for Trace Metadata

The following variables control whether `TRACE`, `TRACE_IF`, and their module-specific variants include **timestamps** and **thread IDs**. These do not affect `CHECK`, `PANIC`, or `ERROR` macros.

| Environment Variable                         | Description                                                                 | Default |
|---------------------------------------------|-----------------------------------------------------------------------------|---------|
| **UNI20_TRACE_TIMESTAMP**                     | If `yes`, include a timestamp (in microseconds since epoch) in each trace. | `no`    |
| **UNI20_TRACE_THREAD_ID**                     | If `yes`, include a thread ID hash in each trace line.                     | `no`    |

For example:

```bash
export UNI20_TRACE_TIMESTAMP=yes
export UNI20_TRACE_THREAD_ID_MODULE_BLAS=yes
```

Sample output with both enabled:

```text
[ 1714543150294213] [TID  7f3e2a9c] TRACE at myfile.cpp:123 : x = 42
```

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

### Supporting new types

The trace macros ultimately rely on a helper function called `formatValue` to convert values into strings. The default implementation uses `fmt::formatter<T>`, so if you define `fmt` style I/O for your custom types then it will work with the `TRACE` macros. But if you want to customize the output, or you want to `TRACE` objects that do not have a `fmt::formatter<T>` specialization, you can customize the `trace::formatValue` function.

### Example: Custom Formatting for a User-Defined Type

Suppose you have a custom type `MyType` and want to display it in a special way:

```cpp
struct MyType {
    int id;
    std::string name;
};

namespace trace {

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

} // namespace trace
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
       ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
       ${CMAKE_CURRENT_BINARY_DIR}/config.hpp
       @ONLY
   )

   include_directories(${CMAKE_CURRENT_BINARY_DIR})
   ```

2. **Template Header (`config.hpp.in`)**

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
## Advanced Usage

You can access the formatting object for trace output using the function `trace::formatting_options()` for the global options, and `trace::formatting_options(module)` for a module. This gives access to all of the variables used for controlling the output.

While it is recommended to avoid this interface and prefer customization via environment variables, there are some potential uses. For example, you can redirect the output of the `TRACE` macros to a `FILE*` stream by calling `trace::formatting_options().set_output_stream(FILE* stream)`, or you can pass a function that accepts a `std::string` to `trace::formatting_options().set_output_stream(std::function<void>(std::string) Sink)`. The `Sink` function can do anything you want, for example it could log the output via some external logging library.

For other variables that can be set in the `FormattingOptions` class, see the `trace.hpp` source code.
