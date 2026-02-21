# Trace Macros Developer Guide

This guide documents the trace/assert macros in `src/common/trace.hpp`, runtime formatting controls, and stacktrace configuration.

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

Alias spellings are also available:

| Alias | Equivalent |
|---|---|
| `TRACE_STACK_IF` | `TRACE_IF_STACK` |
| `TRACE_STACK_ONCE` | `TRACE_ONCE_STACK` |
| `TRACE_STACK_MODULE` | `TRACE_MODULE_STACK` |
| `TRACE_STACK_MODULE_IF` | `TRACE_MODULE_IF_STACK` |

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

Debug alias spellings:

| Alias | Equivalent |
|---|---|
| `DEBUG_TRACE_STACK_IF` | `DEBUG_TRACE_IF_STACK` |
| `DEBUG_TRACE_STACK_ONCE` | `DEBUG_TRACE_ONCE_STACK` |
| `DEBUG_TRACE_STACK_MODULE` | `DEBUG_TRACE_MODULE_STACK` |
| `DEBUG_TRACE_STACK_MODULE_IF` | `DEBUG_TRACE_MODULE_IF_STACK` |

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

### Error Macros

- `ERROR(...)`
- `ERROR_IF(cond, ...)`

By default these abort. You can switch to throw mode with:

```cpp
trace::get_formatting_options().set_errors_abort(false);
```

## Stacktrace Configuration (`<stacktrace>`)

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
2026-02-21 20:35:55.374552 [TID ...] TRACE_STACK at /path/file.cpp:123 : creating epoch, ...
Stacktrace:
  ...
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

These can be turned on or off by setting variables to `true`/`on`/`1` or `false`/`off`/`0`.


| Variable | Default | Effect |
|---|---|---|
| `UNI20_TRACE_TIMESTAMP` | `true` | Show timestamp prefix `YYYY-MM-DD HH:MM:SS.NNNNNNNNN` |
| `UNI20_TRACE_THREAD_ID` | `true` | Show thread-id prefix. |

Module-specific overrides:

- `UNI20_TRACE_TIMESTAMP_MODULE_<MODULE>`
- `UNI20_TRACE_THREAD_ID_MODULE_<MODULE>`

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

## Expression Parsing Notes

Macro arguments are parsed by the preprocessor first. Commas that are not grouped by parentheses/brackets/braces split arguments. If needed, wrap expressions:

```cpp
CHECK((a[i, j] > 2), i, j);
TRACE((vector<int, Alloc>(5).size()));
```

## Build-Time Disable

Define `TRACE_DISABLE` to `1` before including `trace.hpp` to compile out trace emission paths while keeping macro call sites in place.
