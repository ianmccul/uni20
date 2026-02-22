# Trace Macros Developer Guide

This guide documents the trace/assert macros in `src/common/trace.hpp`, runtime formatting controls, and stacktrace configuration.

## Quick Summary

If you are new to the trace/assert system, this is the shortest useful model:

- Use `TRACE...` macros to observe execution and values while debugging.
- Use `CHECK...` macros for invariants that must always hold inside correct code.
- Use `PRECONDITION...` macros for caller/input contract checks.
- Use `ERROR...` when you want to report an error and then abort or throw (configurable).
- Use `..._STACK` variants when you also want an immediate stacktrace.
- Use `DEBUG_...` variants for diagnostics/asserts that should compile out when `NDEBUG` is set.

Generally, use `CHECK...` and `PRECONDITION...` to test logical conditions that would indicate coding bugs, and use `ERROR...` where user input is involved. In Python bindings, `CHECK` and `PRECONDITION` will immediately halt the interpreter, whereas `ERROR` can be configured to propagate an exception into Python.

## Macro Families

### General Trace

| Macro | Behavior |
|---|---|
| `TRACE(...)` | Emit a trace line with file and line. |
| `TRACE_IF(cond, ...)` | Emit only when `cond` is true. |
| `TRACE_ONCE(...)` | Emit once per call site. |
| `TRACE_MODULE(MODULE, ...)` | Emit only when `ENABLE_TRACE_<MODULE>` is enabled at configure time. |
| `TRACE_MODULE_IF(MODULE, cond, ...)` | Module-gated + conditional emit. |

### Stack-Trace Variants

These emit the normal trace line and then a stacktrace block.

| Macro | Behavior |
|---|---|
| `TRACE_STACK(...)` | `TRACE` + stacktrace. |
| `TRACE_IF_STACK(cond, ...)` | `TRACE_IF` + stacktrace. |
| `TRACE_ONCE_STACK(...)` | `TRACE_ONCE` + stacktrace. |
| `TRACE_MODULE_STACK(MODULE, ...)` | `TRACE_MODULE` + stacktrace. |
| `TRACE_MODULE_IF_STACK(MODULE, cond, ...)` | `TRACE_MODULE_IF` + stacktrace. |

Naming convention: `_STACK` is always a suffix.

### Debug-Only Trace

When `NDEBUG` is defined, these expand to no-ops.

| Macro | Behavior |
|---|---|
| `DEBUG_TRACE(...)` | Debug-only `TRACE`. |
| `DEBUG_TRACE_IF(cond, ...)` | Debug-only conditional trace. |
| `DEBUG_TRACE_ONCE(...)` | Debug-only once-per-site trace. |
| `DEBUG_TRACE_MODULE(MODULE, ...)` | Debug-only module trace. |
| `DEBUG_TRACE_MODULE_IF(MODULE, cond, ...)` | Debug-only conditional module trace. |
| `DEBUG_TRACE_STACK(...)` | Debug-only stacktrace trace. |
| `DEBUG_TRACE_IF_STACK(cond, ...)` | Debug-only conditional stacktrace trace. |
| `DEBUG_TRACE_ONCE_STACK(...)` | Debug-only once stacktrace trace. |
| `DEBUG_TRACE_MODULE_STACK(MODULE, ...)` | Debug-only module stacktrace trace. |
| `DEBUG_TRACE_MODULE_IF_STACK(MODULE, cond, ...)` | Debug-only conditional module stacktrace trace. |

### Assertions / Fail-Fast

These print diagnostics and abort:

- `CHECK(cond, ...)`
- `CHECK_EQUAL(a, b, ...)`
- `CHECK_FLOATING_EQ(a, b, [ulps], [extra...])`
- `PRECONDITION(cond, ...)`
- `PRECONDITION_EQUAL(a, b, ...)`
- `PRECONDITION_FLOATING_EQ(a, b, [ulps], [extra...])`
- `PANIC(...)`

`CHECK*`, `PRECONDITION*`, and `PANIC` also print a stacktrace block before abort.

Debug-only assertion forms:

- `DEBUG_CHECK(...)`
- `DEBUG_CHECK_EQUAL(...)`
- `DEBUG_CHECK_FLOATING_EQ(...)`
- `DEBUG_PRECONDITION(...)`
- `DEBUG_PRECONDITION_EQUAL(...)`
- `DEBUG_PRECONDITION_FLOATING_EQ(...)`

### `CHECK_FLOATING_EQ` Examples

`CHECK_FLOATING_EQ` is for floating-point values where exact bitwise equality is usually too strict.

### What "ULP" Means

ULP means "Unit in the Last Place":

- Floating-point numbers are discrete representable points, not a continuous line.
- A difference of `1` ULP means the values are adjacent representable numbers.
- ULP distance scales with magnitude, so it is often more stable than a fixed absolute epsilon.

Practical guidance:

- `CHECK_EQUAL(a, b)` for integers, enums, pointers, and exact-match logic.
- `CHECK_FLOATING_EQ(a, b)` for float/double/complex comparisons.
- Start with default tolerance (`4` ULP), then tighten only if needed.

Near `1.0`, the step size is:

| Type | `1` ULP near `1.0` | `1.0 + 1 ULP` | `1.0 + 2 ULP` |
|---|---|---|---|
| `float` | `1.1920929e-07` | `1.00000012f` | `1.00000024f` |
| `double` | `2.2204460492503131e-16` | `1.0000000000000002` | `1.0000000000000004` |

Equivalent exact hex-float literals:

- `float`: `0x1p+0f`, `0x1.000002p+0f`, `0x1.000004p+0f`
- `double`: `0x1p+0`, `0x1.0000000000001p+0`, `0x1.0000000000002p+0`

Default tolerance is `4` ULP:

```cpp
CHECK_FLOATING_EQ(1.0f, 1.00000024f); // ~2 ULP away, passes (default 4 ULP)
CHECK_FLOATING_EQ(1.0, 1.0000000000000004); // ~2 ULP away, also passes
```

Specify explicit ULP tolerance:

```cpp
CHECK_FLOATING_EQ(1.0f, 1.00000024f, 2); // passes (2 ULP)
CHECK_FLOATING_EQ(1.0f, 1.00000024f, 1); // fails (needs > 1 ULP)
```

Add extra diagnostics (printed on failure):

```cpp
CHECK_FLOATING_EQ(ref, got, 2, iter, timestep, "solver drift");
```

Complex values compare both real and imaginary parts:

```cpp
std::complex<double> expected{1.0, -2.0};
std::complex<double> actual{1.0, -1.9999999999999998}; // imag differs by ~1 ULP
CHECK_FLOATING_EQ(expected, actual, 1);
```

`PRECONDITION_FLOATING_EQ(...)` and debug variants follow the same calling forms.

### GoogleTest Integration

For unit tests, there are GTest-oriented helpers in `src/common/gtest.hpp`:

- `EXPECT_FLOATING_EQ(a, b[, ulps])`
- `ASSERT_FLOATING_EQ(a, b[, ulps])`

These use the same ULP comparison engine as `CHECK_FLOATING_EQ`, but report through
GoogleTest (`ADD_FAILURE`/`FAIL`) instead of aborting the process.

```cpp
#include "common/gtest.hpp"

EXPECT_FLOATING_EQ(value, reference);    // default 4 ULP
ASSERT_FLOATING_EQ(value, reference, 2); // explicit tolerance
```

### Error Macros

- `ERROR(...)`
- `ERROR_IF(cond, ...)`

By default these abort. You can switch to throw mode with:

```cpp
trace::get_formatting_options().set_errors_abort(false);
```

## Stacktrace Configuration (`<stacktrace>`)

C++23 introduced the `<stacktrace>` library, which was supported by GCC in version 13 but only well supported in version 14, and in Clang LLVM 16+. Support in Linux varies by distro. The Ubuntu `gcc-13` does not include `<stacktrace>` support. Ubuntu 26.04 LTS should include good support for `<stacktrace>` with the GCC 15 compiler.

Stacktrace support is controlled in two layers:

1. CMake option:

```bash
cmake -DUNI20_ENABLE_STACKTRACE=ON ...
```

2. Standard library feature availability at compile time:
   - `__cpp_lib_stacktrace >= 202011L`

If either is missing, trace code still compiles. `_STACK` macros and abort diagnostics print:

`WARNING: std::stacktrace is unavailable in this build; stacktrace omitted.`

## Basic Usage

```cpp
TRACE("begin", n);
TRACE_IF(n > 0, n);
TRACE_MODULE(ASYNC, "scheduler tick", id);
TRACE_STACK("creating epoch", epoch_ptr, generation);
```

Example line format:

```text
2026-02-21 20:35:55.374552123 [TID ...] TRACE_STACK at /path/file.cpp:123 : creating epoch, ...
Stacktrace:
  ...
```

The timestamp is local time and uses nanosecond precision (`.NNNNNNNNN`).

### Additional Examples

Simple trace with expression/value expansion:

```cpp
int i = 4;
double x = 3.5;
TRACE(i, x * i); // prints: i = 4, x * i = 14
```

One-shot trace at a noisy call site:

```cpp
for (int iter = 0; iter < 1000; ++iter) {
  TRACE_ONCE("first iteration only", iter);
}
```

Debug-only stacktrace trace:

```cpp
DEBUG_TRACE_STACK("suspending task", task_id, state);
```

Assertion with contextual diagnostics:

```cpp
CHECK_EQUAL(expected_epoch, actual_epoch, task_id, writer_count, reader_count);
```

Route one module to a separate file:

```bash
export UNI20_TRACEFILE=stderr
export UNI20_TRACEFILE_MODULE_ASYNC=+async.trace.log
```

## Module Enable Flags

`TRACE_MODULE(...)` and variants are compile-time selected by CMAKE flags:

- `ENABLE_TRACE_<MODULE>`

Example:

```bash
cmake -DENABLE_TRACE_ASYNC=ON ...
```
These are also set as defined symbols in the generated `config.hpp` header as `ENABLE_TRACE_<MODULE>`, for example
```c++
#if ENABLE_TRACE_ASYNC
/// code
#endif
```

There is a module `TESTMODULE` that is always enabled, which can be used for testing the module system.

## Output and Formatting Controls

Formatting is controlled by `trace::get_formatting_options()` and environment variables.

### Output Selection

The output sink for TRACE messages is controlled by the variable `UNI20_TRACEFILE`:

- `UNI20_TRACEFILE=stderr` (default)
- `UNI20_TRACEFILE=stdout` or `UNI20_TRACEFILE=-`
- `UNI20_TRACEFILE=trace.log` (overwrite)
- `UNI20_TRACEFILE=+trace.log` (append)

Module-specific TRACE messages can be managed separately, so for example it is possible to set messages from different modules to different output files.

- `UNI20_TRACEFILE_MODULE_<MODULE>=...`

### Timestamp and Thread ID

`UNI20_TRACE_TIMESTAMP` accepts `yes`/`no` (aliases: `true`/`false`, `1`/`0`).
`UNI20_TRACE_THREAD_ID` additionally accepts `auto`.


| Variable | Default | Effect |
|---|---|---|
| `UNI20_TRACE_TIMESTAMP` | `true` | Show local-time timestamp prefix `YYYY-MM-DD HH:MM:SS.NNNNNNNNN` |
| `UNI20_TRACE_THREAD_ID` | `auto` | Show thread-id prefix (`yes`), disable it (`no`), or auto-detect (`auto`). |

Module-specific overrides:

- `UNI20_TRACE_TIMESTAMP_MODULE_<MODULE>`
- `UNI20_TRACE_THREAD_ID_MODULE_<MODULE>`

`UNI20_TRACE_THREAD_ID=auto` shows the thread-id only for non-main threads.

### Color Enable/Disable

| Variable | Default | Values |
|---|---|---|
| `UNI20_TRACE_COLOR` | `auto` | `yes`, `no`, `auto` |

When set to `auto`, color output is used if writing to a terminal that supports color, otherwise no color is used.

Module-specific override:

- `UNI20_TRACE_COLOR_MODULE_<MODULE>`

### Color Style Keys

Use `UNI20_COLOR_<KEY>=<style>`.

| Key | Default |
|---|---|
| `TRACE` | `Cyan` |
| `DEBUG_TRACE` | `Green` |
| `TRACE_EXPR` | `Blue` |
| `TRACE_VALUE` | *(empty)* |
| `TRACE_MODULE` | `Cyan;Bold` |
| `TRACE_FILENAME` | `Red` |
| `TRACE_LINE` | `Bold` |
| `TRACE_STRING` | `Cyan` |
| `CHECK` | `Red` |
| `DEBUG_CHECK` | `Red` |
| `PRECONDITION` | `Red` |
| `DEBUG_PRECONDITION` | `Red` |
| `PANIC` | `Red` |
| `ERROR` | `Red` |
| `TIMESTAMP` | `LightGray` |
| `THREAD_ID` | `LightMagenta` |

Module-specific style overrides:

- `UNI20_COLOR_<KEY>_MODULE_<MODULE>`

### Style String Syntax

Style strings support foreground/background colors plus attributes:

- Separate style components with `,`.
- Separate tokens within a component with `;`.
- Use `fg:` or `bg:` to target foreground/background.
- Unrecognized tokens are ignored.
- In shell exports, quote style strings because `;` is a shell command separator.

Examples:

- `LightCyan;Bold`
- `fg:Yellow;Underline, bg:DarkGray`
- `fg:#7FDBFF;Bold`
- `bg:rgb(40,40,40);LightGreen`

### Named Colors

Named colors are case-insensitive:

| Name | Notes |
|---|---|
| `Default` | Reset to terminal default for fg/bg. |
| `Black` |  |
| `Red` |  |
| `Green` |  |
| `Yellow` |  |
| `Blue` |  |
| `Magenta` |  |
| `Cyan` |  |
| `LightGray` |  |
| `DarkGray` | Bright black in many terminals. |
| `LightRed` |  |
| `LightGreen` |  |
| `LightYellow` |  |
| `LightBlue` |  |
| `LightMagenta` |  |
| `LightCyan` |  |
| `White` |  |

### Text Attributes

Attributes are case-insensitive and can be combined:

| Attribute | Effect |
|---|---|
| `Bold` | ANSI bold/intense text. |
| `Dim` | ANSI dim/faint text. |
| `Underline` | ANSI underline. |

### RGB and Hex Colors

In addition to named colors:

- `rgb(r,g,b)` for 24-bit color (`r/g/b` in `0..255`, `rgb` must be lowercase).
- `#RRGGBB` or `#RGB` hex color forms.

Both can be used with `fg:` and `bg:`:

- `UNI20_COLOR_TRACE='fg:rgb(255,200,0);Bold'`
- `UNI20_COLOR_TRACE_FILENAME='fg:#FF6B6B'`
- `UNI20_COLOR_TRACE_LINE='bg:#202020;LightCyan;Bold'`

### Common Color Recipes

| Goal | Example |
|---|---|
| Strong trace label | `UNI20_COLOR_TRACE='LightCyan;Bold'` |
| Subtle timestamp | `UNI20_COLOR_TIMESTAMP='DarkGray'` |
| Highlight file/line | `UNI20_COLOR_TRACE_FILENAME='Yellow'` and `UNI20_COLOR_TRACE_LINE='Bold'` |
| High-contrast checks | `UNI20_COLOR_CHECK='White;Bold,bg:Red'` |
| Module-specific ASYNC palette | `UNI20_COLOR_TRACE_MODULE_ASYNC='LightBlue;Bold'` |

### Floating-Point Precision

Global:

- `UNI20_FP_PRECISION_FLOAT32`
- `UNI20_FP_PRECISION_FLOAT64`

Module-specific:

- `UNI20_FP_PRECISION_FLOAT32_MODULE_<MODULE>`
- `UNI20_FP_PRECISION_FLOAT64_MODULE_<MODULE>`

## Programmatic API

Global options:

```cpp
auto& opts = trace::get_formatting_options();
```

Module options:

```cpp
auto& async_opts = trace::get_formatting_options("ASYNC");
```

Useful methods:

- `set_output_stream(FILE*)`
- `set_sink(std::function<void(std::string)>)`
- `set_color_output(trace::FormattingOptions::ColorOptions::yes/no/autocolor)`
- `set_errors_abort(bool)` (static setting; callable through the object)

Useful field for thread-id mode:

- `threadId = trace::FormattingOptions::ThreadIdOptions::yes|no|auto_detect`

## Expression Parsing Notes

Macro arguments are parsed by the preprocessor first. Commas that are not grouped by parentheses/brackets/braces split arguments. If needed, wrap expressions:

```cpp
CHECK((a[i, j] > 2), i, j);
TRACE((vector<int, Alloc>(5).size()));
```

## Build-Time Disable

Define `TRACE_DISABLE` to `1` before including `trace.hpp` to compile out trace emission paths while keeping macro call sites in place.
