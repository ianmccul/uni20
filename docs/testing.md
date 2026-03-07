# Testing uni20

This document describes how uni20 tests are configured, built, and run.

## Overview

uni20 testing uses:

- [Google Test](https://github.com/google/googletest) for C++ unit tests
- [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) for test discovery and execution
- CMake [`gtest_discover_tests`](https://cmake.org/cmake/help/latest/module/GoogleTest.html) for automatic registration

Tests live in `tests/` and are organized by module.

## Test Layout

Current test modules include:

- `tests/async`
- `tests/backend`
- `tests/common`
- `tests/core`
- `tests/expokit`
- `tests/kernel`
- `tests/level1`
- `tests/linalg`
- `tests/mdspan`
- `tests/tensor`
- `tests/python` (only when `UNI20_BUILD_PYTHON=ON`)

## Configuration Options

Primary CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `UNI20_BUILD_TESTS` | `ON` | Build and register C++ tests |
| `UNI20_BUILD_COMBINED_TESTS` | `ON` | Build `tests/uni20_tests` combined test executable |
| `UNI20_BUILD_PYTHON` | `ON` | Also enable `tests/python` CTest entries |

Example configure:

```bash
cmake -S . -B build -DUNI20_BUILD_TESTS=ON -DUNI20_BUILD_COMBINED_TESTS=ON
```

To disable tests:

```bash
cmake -S . -B build -DUNI20_BUILD_TESTS=OFF
```

## Building Tests

After configuration, build with:

```bash
cmake --build build -j
```

## Running Tests

List discovered tests:

```bash
ctest --test-dir build -N
```

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

Run only matching tests:

```bash
ctest --test-dir build --output-on-failure -R IterationPlan
```

You can also run binaries directly:

```bash
./build/tests/common/uni20_common_tests
./build/tests/uni20_tests --gtest_filter=TraitsTest.*
```

## Separate vs Combined Test Executables

uni20 supports both modes:

- Per-module executables like `uni20_common_tests`, `uni20_async_tests`
- Combined executable `uni20_tests`

When both modes are enabled, CTest intentionally registers both.

## Adding New Tests

1. Add the source file in the appropriate `tests/<module>/` directory.
2. Add it to that module’s `tests/<module>/CMakeLists.txt` via `add_test_module(...)`.

Example:

```cmake
add_test_module(level1
  SOURCES
    test_apply_unary.cpp
    test_assign.cpp
    test_sum.cpp
    test_new_case.cpp
  LIBS
    uni20_level1
    mdspan
)
```

## Internals

`cmake/Uni20TestHelpers.cmake` defines `add_test_module(...)`, creates per-module test executables, and accumulates sources/libs for the optional combined executable.

## Linux crash handler slowdown (Apport)

uni20 includes many GoogleTest death tests (`EXPECT_DEATH`/`ASSERT_DEATH`) that intentionally call `std::abort()` to verify contract failures.

The configuration of some Linux systems may interpret these as system crashes which are directed to Apport, via the `kernel.core_pattern` sysctl knob. When this is enabled, each death test crash can launch an Apport Python helper. With many death tests, this can dramatically slow down the total test time.

Check current setting:

```bash
cat /proc/sys/kernel/core_pattern
```

For development and CI environments focused on fast death-test runs, prefer:

```bash
sudo sysctl -w kernel.core_pattern=core
```

To persist this setting you could use

```bash
echo "kernel.core_pattern=core" | sudo tee /etc/sysctl.d/99-local-core-pattern.conf
sudo sysctl --system
```
or disable the `apport` service.
