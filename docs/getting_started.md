# uni20 - A C++20 Tensor-Network Library

Welcome to **uni20**! This repository contains a high-performance tensor-network library written in C++20. The library is designed with multiple backends, GPU/MPI support, and Python bindings (via pybind11) to ease user interaction. This guide is intended to help new developers quickly set up the project, run tests, and execute benchmarks.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Building the Project](#building-the-project)
- [Running Tests](#running-tests)
- [Running Benchmarks](#running-benchmarks)
- [Python Bindings](#python-bindings)
- [Coding Style](#code-style)
- [Additional Notes](#additional-notes)
- [Contributing](#contributing)

## Project Structure

A brief overview of the directory layout:

```
uni20/                              # Project root
├── CMakeLists.txt                  # Top-level CMake configuration
├── README.md                       # Project overview and instructions
├── LICENSE                         # License information
├── docs/                           # Documentation (this guide, API docs, etc.)
│   ├── getting_started.md
│   ├── testing.md
│   └── trace_macros.md
├── cmake/                          # Custom CMake modules and helpers
│   ├── ClangFormat.cmake
│   ├── DetectBlasVendor.cmake
│   └── Uni20TestHelpers.cmake
├── src/                            # C++ source code for the uni20 library
│   ├── common/                     # Shared types, traits, and utilities
│   ├── core/                       # (currently empty)
│   ├── level1/                     # Implementation of Level 1 tensor operations (linear computation / memory use)
│   ├── backend/                    # Backend API wrappers (BLAS, MKL, CUDA, etc.)
│   │   ├── blas/                   # BLAS API headers, vendor detection
│   │   ├── mkl/                    # Intel MKL-specific wrappers
│   │   ├── cuda/                   # CUDA backend stubs (WIP)
│   │   └── cusolver/               # cuSOLVER backend stubs (WIP)
│   ├── kernel/                     # High-level kernel dispatch (e.g. tensor contraction)
│   │   ├── cpu/                    # CPU-specific kernel implementations
│   │   ├── blas/                   # BLAS-backed kernel implementations
│   │   ├── mkl/                    # MKL-backed kernel implementations
│   │   ├── cuda/                   # CUDA kernel stubs (WIP)
│   │   ├── operations.hpp          # Kernel entry point for general operations
│   │   └── contract.hpp            # Kernel entry point for `contract` function
│   ├── async/                      # Coroutine-based async execution and scheduling
│   ├── mdspan/                     # Extensions and traits for stdex::mdspan
│   ├── storage/                    # Storage implementations (e.g. vector-backed)
│   └── tensor/                     # Tensor and TensorView abstraction layer
├── tests/                          # Unit tests using GoogleTest
│   ├── common/                     # Tests for src/common utilities
│   ├── level1/                     # Tests for Level 1 operations
│   ├── kernel/                     # Kernel-level tests (e.g. contraction)
│   └── helpers.hpp                 # Shared testing helpers and mocks
├── benchmarks/                     # Performance benchmarks using Google Benchmark
├── examples/                       # Demonstration programs
│   ├── mdspan_example.cpp
│   ├── async_example.cpp
│   └── trace_example.cpp
├── bindings/                       # Language bindings (e.g. Python via pybind11)
│   └── python/
└── asm/                            # Contains sample code for testing generated assembly output
```

## Prerequisites

Before building the project, ensure you have the following installed:

- **CMake 3.18+** (or newer; see the [CMake documentation](https://cmake.org))
- A C++23-compliant compiler (e.g., GCC 13+, Clang 16+, MSVC 2022)
- Git (for cloning the repository and fetching dependencies)
- BLAS and LAPACK libraries are essential; any library that implements the standard Fortran interface will work.
- (Optional) Python 3.x and pybind11 dependencies for building the Python bindings

> **Note:** The project uses CMake’s FetchContent module to automatically download external libraries (like fmt, mdspan, GoogleTest, and Google Benchmark). If you prefer to manage these dependencies differently, adjust the CMake configurations accordingly.

## Dependencies

```bash
apt-get install libopenblas-dev liblapack-dev
```

## Building the Project

It is recommended to use an out-of-source build. From the project root, run:

```bash
# Create and configure the build directory:
cmake -S . -B build

# Build all targets (library, tests, benchmarks, and Python module):
cmake --build build
```

If you need to disable CUDA (default is OFF) or adjust other build options, you can pass flags:

```bash
cmake -S . -B build -DUNI20_ENABLE_CUDA=OFF -DUNI20_BUILD_TESTS=ON -DUNI20_BUILD_BENCH=ON
```

### BLAS/LAPACK Detection and Vendor-Specific Features

(this is a work in progress!)

The project leverages CMake’s built-in modules (`FindBLAS.cmake` and `FindLAPACK.cmake`) to automatically detect available BLAS and LAPACK libraries. This means:
- If a standard BLAS/LAPACK implementation is present, the project will link against it.
- If vendor-specific libraries such as Intel MKL or OpenBLAS are installed, the detection modules will set variables (e.g., `BLAS_VENDOR`) accordingly.  
- In the case of MKL, if the environment variable `MKLROOT` is defined, the project will add MKL’s include directory (e.g., `$MKLROOT/include`) to the target’s include paths, allowing vendor-specific extensions to be used.

## Running Tests

The `uni20` library includes a comprehensive suite of unit tests, written using [Google Test](https://github.com/google/googletest) and managed via [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html).

You can run tests using either **CTest** or by executing test binaries directly. The test system supports both **separate** (per-module) test executables and an optional **combined** test binary.

See [docs/testing.md](testing.md) for detailed configuration options, test architecture, and best practices.

### Build and Run Tests

Tests are enabled by default. After building:

```bash
cd build
ctest --output-on-failure
```

This will run all tests registered with CTest, including both per-module and combined executables (if enabled).

To filter tests by name or suite:

```bash
ctest -R IterationPlan        # Run all tests matching the regex "IterationPlan"
ctest -V                      # Verbose output for debugging
```

### Run Tests Directly

You may also run test executables manually. For example:

```bash
./tests/uni20_tests --gtest_filter=TraitsTest.*
./tests/common/uni20_common_tests
```

The Google Test interface supports additional flags (e.g., `--gtest_list_tests`) for exploring and selecting tests interactively.

### Disable or Reconfigure Tests

To disable all tests or change test modes, pass options to CMake during configuration:

```bash
cmake -S . -B build -DUNI20_BUILD_TESTS=OFF
cmake -S . -B build -DUNI20_BUILD_COMBINED_TESTS=OFF
```

See [docs/testing.md](testing.md) for a full explanation of these options and how they affect the build.

## Running Benchmarks

The project uses Google Benchmark for performance measurement.

### Option 1: Using a Custom Target

If your `benchmarks/CMakeLists.txt` defines a custom target (e.g., `run_benchmarks`), run:

```bash
cmake --build build --target run_benchmarks
```

### Option 2: Running Directly

Or run the benchmark executable directly:

```bash
cd build/benchmarks
./uni20_benchmarks
```

Benchmarks will output performance metrics (execution time, iterations, etc.) to the console.

## Python Bindings

A simple Python binding is provided via pybind11. After building:

1. Locate the Python module in the build directory (typically under `build/bindings/python`).
2. Add that directory to your `PYTHONPATH`:

   ```bash
   export PYTHONPATH=~/path/to/uni20/build/bindings/python:$PYTHONPATH
   ```

3. Test the module in a Python shell:

   ```bash
   python3 -c "import uni20_python; print(uni20_python.greet())"
   ```

   You should see the output:
   ```
   Hello from uni20!
   ```

## Coding Style and Formatting

To help maintain a consistent code style across the project, we've integrated clang-format and several other configuration tools. This section outlines the code formatting preferences, how to use clang-format from the command line (and in your editor), and highlights other recent enhancements.

### Clang-Format Integration

The project uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to enforce a consistent code style, although the formatting isn't compulsory. A sample `.clang-format` file is provided at the root of the repository with the following key style settings:

- **Indentation:** 2 spaces per indent
- **Brace Style:** Allman style (opening braces on their own line)
- **Short Functions:** Allowed on a single line for brevity (both inline and, if very simple, non-member functions)
- **Access Specifiers & Case Labels:** Indented for clarity

#### CMake Integration

To streamline formatting, we have created a separate CMake module in `cmake/ClangFormat.cmake`. This module defines a custom target that automatically runs clang-format on all source files. The module looks for all C++ source and header files (in `src/`, `tests/`, and `bindings/python/`) and creates a target named `clang_format`.

You can run the formatting target with:

```bash
cmake --build build --target clang_format
```

This will invoke clang-format in-place on all matching files. Many editors have some form of `clang-format` integration, which may be helpful.

## Additional Notes

- **CMake Options:**  
  Use project-specific CMake options (prefixed with `UNI20_`) to enable/disable features like CUDA, MPI, testing, and benchmarking.

- **Dependency Management:**  
  External dependencies are managed via CMake’s FetchContent. If you have network issues or prefer local copies, adjust the `FETCHCONTENT_BASE_DIR` variable or use submodules.

- **Directory Structure & CTest:**  
  If you run CTest from the top-level build directory and no tests are found, use the `--recursive` flag, or run tests from the appropriate subdirectory (e.g., `build/tests`).

## Contributing

Contributions to uni20 are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (e.g., `feature/new-backend`).
3. Make your changes and add tests/benchmarks as needed.
4. Submit a pull request with a clear description of your changes.

For further details, please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) file.
