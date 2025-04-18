# Testing uni20

This document describes the testing infrastructure in the `uni20` library. It covers the test configuration system, the layout of the test sources, how to build and run tests using `CTest`, and how the test harness integrates with Google Test (`gtest`).

The goal is to provide both a solid verification suite for the library and a convenient development environment for implementing new functionality.

## Overview

The `uni20` testing system is based on:

- [Google Test](https://github.com/google/googletest): A C++ unit testing framework
- [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html): CMake’s test runner and reporting tool
- CMake’s [`gtest_discover_tests`](https://cmake.org/cmake/help/latest/module/GoogleTest.html) integration for automatic test registration

Tests are written using the standard `TEST(...)` and `EXPECT_...` macros provided by Google Test. All tests can be executed via CTest, and are grouped by module.

---

## Test Layout

Tests are located in the `tests/` directory and organized by subdirectory:

```
tests/
├── common/
│   ├── test_types.cpp
│   └── test_terminal_color.cpp
├── level1/
│   └── test_iteration_plan.cpp
...
```

Each subdirectory corresponds to a logical component or layer of the library (e.g., `common`, `level1`, etc.).


## Configuration Options

The following CMake options control the test system:

| Option | Default | Description |
|--------|---------|-------------|
| `UNI20_BUILD_TESTS` | `ON` | Enables test compilation and registration |
| `UNI20_BUILD_COMBINED_TESTS` | `ON` | Builds an additional combined test executable aggregating all test sources |

To disable all testing:

```bash
cmake -DUNI20_BUILD_TESTS=OFF ..
```

To build per-module test executables but skip the combined test:

```bash
cmake -DUNI20_BUILD_COMBINED_TESTS=OFF ..
```


## Building Tests

After configuring the project with CMake (with testing enabled), the test targets are automatically added to the build.

To build all tests:

```bash
make
```

The following test executables will be created (paths relative to the build directory):

- `tests/common/uni20_common_tests`
- `tests/level1/uni20_level1_tests`
- ...
- `tests/uni20_tests` (if `UNI20_BUILD_COMBINED_TESTS=ON`)


## Running Tests

### With CTest

CTest is the preferred way to execute tests. It provides filtering, output control, and integration with CI systems.

List all available tests:

```bash
ctest -N
```

Run all tests (quiet mode):

```bash
ctest
```

Run all tests with verbose output:

```bash
ctest -V
```

Filter tests by name (regex match):

```bash
ctest -R IterationPlan
```

### Running Test Executables Directly

Each test executable can also be invoked directly. This may be more convenient during development.

Examples:

```bash
./tests/common/uni20_common_tests
./tests/uni20_tests --gtest_filter=TraitsTest.*
```

The Google Test command-line interface allows fine-grained control over which tests are run, output formatting, and more. See the [Google Test Advanced Guide](https://github.com/google/googletest/blob/main/docs/advanced.md) for details.


## Separate vs Combined Test Executables

The `uni20` testing framework supports both **per-module** and **combined** test execution.

- **Per-module tests** are built individually and compiled with only the sources relevant to that module. This is useful during development and debugging of a specific component.

- The **combined test executable** aggregates all test sources into a single binary. This is convenient for CI pipelines and global test runs.

Both systems are registered with CTest and can be run independently.

Tests in the combined executable will appear **twice** in `ctest -N` output if both modes are enabled. This is intentional and allows independent test control.


## Adding New Tests

To add a new test file:

1. Place the test source in the appropriate subdirectory under `tests/`.
2. Add the filename to the `SOURCES` list in that subdirectory’s `CMakeLists.txt`, using the `add_test_module(...)` macro.

For example:

```cmake
# tests/level1/CMakeLists.txt

add_test_module(level1
  SOURCES
    test_iteration_plan.cpp
    test_tensor_assign.cpp   # <-- New test
  LIBS
    uni20_level1
    mdspan
)
```

The build system will automatically compile and register the new test with both the per-module and combined executables (if enabled).


## Internals

Test registration is centralized through a CMake macro defined in `cmake/Uni20TestHelpers.cmake`. This macro accumulates test sources and libraries for both separate and combined builds.

The combined executable is only constructed **after** all per-module sources have been discovered, ensuring the source list is complete.


## Summary

- Tests are organized by module and built using CMake and Google Test
- Use `ctest` to run tests, either globally or filtered
- Combined and per-module test modes can be enabled independently
- The system requires minimal effort to extend and add new tests
