# Trace Macros Developer Guide

This guide documents the trace/assert macros in `src/common/trace.hpp`, runtime formatting controls, and stacktrace configuration.

## Quick Summary

If you are new to the trace/assert system, this is the shortest useful model:

- Use `TRACE...` macros to observe execution and values while debugging.
- Use `CHECK...` macros for invariants that must always hold inside correct code.
- Use `PRECONDITION...` macros for caller/input contract checks.
- Use `ERROR...` when you want to report an error and then abort or throw (configurable).
- Use `trace::raise(exception)` for a structured exception that must preserve
  its concrete type and presentation metadata.
- Use `..._STACK` variants when you also want an immediate stacktrace.
- Use `DEBUG_...` variants for diagnostics/asserts that should compile out when `NDEBUG` is set.

Generally, use `CHECK...` and `PRECONDITION...` to test logical conditions that would indicate coding bugs, and use `ERROR...` where user input is involved. Importing the Python extension configures `ERROR` to throw, allowing nanobind to translate the failure into a Python exception. `CHECK`, `PRECONDITION`, and `PANIC` remain invariant failures that immediately halt the interpreter.

## Quick Start

Minimal usage in code:

```cpp
#include "common/trace.hpp"

void f(int n)
{
  PRECONDITION(n > 2, "n must be larger than 2", n);
  TRACE("begin", n);
  TRACE_IF(n > 0, n);
  TRACE_STACK("debug point", n); // same line + stacktrace block
}
```

Typical output:

```text
2026-02-22 08:37:10.692231374 TRACE at /path/file.cpp:6 → begin, n = 3
2026-02-22 08:37:10.692235001 TRACE at /path/file.cpp:7 → n = 3
2026-02-22 08:37:10.692238442 TRACE_STACK at /path/file.cpp:8 → debug point, n = 3
Stacktrace:
  ├─ #0 ...
  └─ #1 ...
```

By default, if the call is from a non-main thread, a `[TID ...]` prefix is inserted before `TRACE`, showing the thread ID of the caller. This can be customized - see section Output and Formatting Controls below.

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
Diagnostic headers use the same semantic glyph policy as trace output: emoji mode renders headers such as `❌ CHECK` and `🚨 PANIC`, Unicode mode renders `✗ CHECK` and `▲ PANIC`, and ASCII mode renders `[FAIL] CHECK` and `[WARN] PANIC`.

Debug-only assertion forms:

- `DEBUG_CHECK(...)`
- `DEBUG_CHECK_EQUAL(...)`
- `DEBUG_CHECK_FLOATING_EQ(...)`
- `DEBUG_PRECONDITION(...)`
- `DEBUG_PRECONDITION_EQUAL(...)`
- `DEBUG_PRECONDITION_FLOATING_EQ(...)`

### CHECK_FLOATING_EQ Examples

`CHECK_FLOATING_EQ` is for floating-point values where exact bitwise equality is usually too strict.

### What "ULP" Means

ULP means "Unit in the Last Place":

- Floating-point numbers are discrete representable points, not a continuous line.
- A difference of `1` ULP means the values are adjacent representable numbers.
- ULP distance scales with magnitude, so it is often more stable than a fixed absolute epsilon.

Practical guidance:

- `CHECK_EQUAL(a, b)` for integers, enums, pointers, and exact-match logic.
- `CHECK_FLOATING_EQ(a, b)` for IEEE binary32, binary64, configured binary128,
  and corresponding complex comparisons.
- Start with default tolerance (`4` ULP), then tighten only if needed.

Near `1.0`, the step size is:

| Type | `1` ULP near `1.0` | `1.0 + 1 ULP` | `1.0 + 2 ULP` |
|---|---|---|---|
| `float` | `1.1920929e-07` | `1.00000012f` | `1.00000024f` |
| `double` | `2.2204460492503131e-16` | `1.0000000000000002` | `1.0000000000000004` |
| `uni20::float128` | `1.92592994438723585305597794258492732e-34` | see exact values below | see exact values below |

`uni20::float128` support is present when Uni20 is configured with an IEEE
binary128 provider. Padded x87 extended-precision `long double` is deliberately
excluded because its object representation is not an IEEE interchange format.

Equivalent exact hex-float literals:

- `float`: `0x1p+0f`, `0x1.000002p+0f`, `0x1.000004p+0f`
- `double`: `0x1p+0`, `0x1.0000000000001p+0`, `0x1.0000000000002p+0`

Binary128 needs 28 hexadecimal fractional digits, so its adjacent literals are
clearer outside the table. C++23's standard `F128` suffix is available in the
supported GNU MPLAPACK configuration:

```cpp
uni20::float128 const one = 0x1p+0F128;                                      // 1.0
uni20::float128 const next = 0x1.0000000000000000000000000001p+0F128;       // 1.0 + 1 ULP
uni20::float128 const next_next = 0x1.0000000000000000000000000002p+0F128;  // 1.0 + 2 ULP
```

The hexadecimal values are exact. For provider-independent runtime conversion
from decimal or hexadecimal text, use `uni20::parse_real<uni20::float128>()`.

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
uni20::complex<double> expected{1.0, -2.0};
uni20::complex<double> actual{1.0, -1.9999999999999998}; // imag differs by ~1 ULP
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

By default these emit a stacktrace and abort. You can switch to throw mode with:

```cpp
trace::get_formatting_options().set_errors_abort(false);
```

The Python extension selects throw mode process-wide when the module is initialized. This also covers C++ calls made on behalf of Python after the immediate binding function has returned. Worker-thread and asynchronous task boundaries must catch such exceptions and propagate them to the Python-facing result; an exception escaping a C++ thread still invokes `std::terminate`.

### Structured Exceptions

`trace::raise(exception)` is the ordinary-function counterpart for structured
exceptions. Its default `std::source_location` captures the raise site, so no
macro is required:

```cpp
trace::raise(KernelDispatchError{operation, failure, backend_attempts});
```

Throw mode preserves the concrete exception type. Exceptions derived from
`uni20::diagnostic_error` also receive the source location and, when available,
the captured `std::stacktrace`. Abort mode asks ADL for
`diagnostic_report(exception)` returning a `presentation::report_builder`; when
that customization is absent, it renders `exception.what()`. Both abort paths
then emit the usual stacktrace before terminating. `trace::format_diagnostic(...)`
uses the same report and tree-formatted stacktrace path without emitting it,
which is useful when a binding, example, or other recoverable boundary catches
the exception. An overload accepts an explicit presentation policy for
non-terminal render targets. Stacktrace frames use the policy width, wrapping
continuation lines beneath the frame description. Structured abort mode does
not emit a second stacktrace when one is already attached.

## Stacktrace Configuration (<stacktrace>)

C++23 introduced the `<stacktrace>` library, but availability depends on the
compiler, standard-library headers, and packaged support library rather than
the compiler version alone. For example, libstdc++ may provide the implementation
through a separately linked `stdc++exp` library. Uni20 therefore tests the
complete compile-and-link capability instead of maintaining a compiler-version
allowlist.

During a fresh configure, CMake compiles and links a small `std::stacktrace`
program. `UNI20_ENABLE_STACKTRACE` defaults to `ON` when that probe succeeds and
to `OFF` otherwise. Some libstdc++ installations require the separately linked
`stdc++exp` library; the same probe detects and records that dependency.

Disable otherwise available stacktraces explicitly with:

```bash
cmake -DUNI20_ENABLE_STACKTRACE=OFF ...
```

Forcing `UNI20_ENABLE_STACKTRACE=ON` with an unsupported compiler or standard
library is a configuration error. The generated configuration also verifies
`__cpp_lib_stacktrace >= 202011L` before compiling stacktrace-dependent code.

When stacktraces are disabled, `_STACK` macros and abort diagnostics print:

`🚨 WARNING: std::stacktrace is unavailable in this build; stacktrace omitted.`

## Basic Usage

```cpp
TRACE("begin", n);
TRACE_IF(n > 0, n);
TRACE_MODULE(ASYNC, "scheduler tick", id);
TRACE_STACK("creating epoch", epoch_ptr, generation);
```

Example line format:

```text
2026-02-21 20:35:55.374552123 [TID ...] TRACE_STACK at /path/file.cpp:123 → creating epoch, ...
Stacktrace:
  ├─ #0 ...
  └─ #1 ...
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

### Presentation Controls

Trace diagnostics are rendered through the common [presentation formatting](presentation.md) layer. Plain/file output suppresses ANSI escapes, semantic glyph fallback follows the shared output policy, strict ASCII modes apply to whole trace lines, and container-style trace output aligns by display cells. Mdspan-like values and tensor/view-like objects render as bounded presentation tensor-art previews: vectors use a row form, matrices use aligned bracket art, higher-rank tensors use labeled matrix slices, and large values elide edge rows/columns/slices to respect the terminal width.

Normal trace output never writes side files. Abort diagnostics (`CHECK*`, `PRECONDITION*`, `PANIC`, and `ERROR*` when abort mode is enabled) write a full mdspan/tensor dump when the visible preview had to elide data.

| Variable | Default | Values | Effect |
|---|---|---|---|
| `UNI20_GLYPHS` | `emoji` | `unicode`, `emoji`, `ascii` | Select semantic glyph spelling for all presentation output, including trace. |
| `UNI20_CHARSET` | `utf8` | `utf8`, `escape`, `replace` | Select fallback for non-ASCII text in all presentation output, including trace. `utf-8`, `ascii_escape`, and `ascii_replace` aliases are accepted. |
| `UNI20_COLOR` | `auto` | `auto`, `yes`, `always`, `no`, `never`, plus boolean aliases | Control ANSI style emission globally. |
| `COLUMNS` | terminal columns | positive integer | Override detected terminal width for trace layout and wrapping. |
| `UNI20_TRACE_DUMP` | enabled | `never`, `no`, `off`, `false`, `0` disable | Control full mdspan/tensor dump files for abort diagnostics that elide preview output. |
| `UNI20_TRACE_DUMP_DIR` | system temp dir under `uni20-trace` | directory path | Directory for full mdspan/tensor dump files written by abort diagnostics. |

When `UNI20_COLOR=auto`, color output is used if `NO_COLOR` is unset or empty and the output stream is a terminal.
Set `NO_COLOR` to any non-empty value to disable automatic color output by default. Explicit `UNI20_COLOR=yes`
or `UNI20_COLOR=no` overrides `NO_COLOR`.

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
- `UNI20_FP_PRECISION_FLOAT128`

These control the number of digits after the decimal point for trace real values. Complex values use the same precision for real and imaginary components, and tensor/mdspan trace output applies the same scalar precision inside the presentation tensor-art renderer.

Module-specific:

- `UNI20_FP_PRECISION_FLOAT32_MODULE_<MODULE>`
- `UNI20_FP_PRECISION_FLOAT64_MODULE_<MODULE>`
- `UNI20_FP_PRECISION_FLOAT128_MODULE_<MODULE>`

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
- `set_color_output(uni20::presentation::color_mode::always/never/automatic)`
- `set_errors_abort(bool)` (static setting; callable through the object)

Useful field for thread-id mode:

- `threadId = trace::FormattingOptions::ThreadIdOptions::yes|no|auto_detect`

Useful presentation policies:

- `presentation_policy()` controls glyph, charset, width, and color rendering.
- `mdspan_format_policy()` controls tensor-art shape labels, slice labels, and matrix axes for mdspan/tensor trace values.
- `mdspan_preview_policy()` controls bounded tensor preview limits such as full-output element limit, edge item count, and maximum displayed slices.

## Expression Parsing Notes

Macro arguments are parsed by the preprocessor first. Commas that are not grouped by parentheses/brackets/braces split arguments. If needed, wrap expressions:

```cpp
CHECK((a[i, j] > 2), i, j);
TRACE((vector<int, Alloc>(5).size()));
```

## Build-Time Disable

Define `TRACE_DISABLE` to `1` before including `trace.hpp` to compile out trace emission paths while keeping macro call sites in place.
