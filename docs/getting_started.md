# uni20 - A C++23 Tensor-Network Library

Welcome to **uni20**. This repository contains an early-stage tensor-network library written in C++23 with optional BLAS, CUDA, benchmark, documentation, and Python binding support. This guide covers the current CMake-based workflow for configuring, building, testing, benchmarking, and generating documentation.

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
├── bindings/                       # Language bindings (currently Python via nanobind)
│   └── python/
└── asm/                            # Contains sample code for testing generated assembly output
```

## Prerequisites

Before building the project, ensure you have the following installed:

- **CMake 3.24+**.
- A C++23-compliant compiler (e.g., GCC 13+, Clang 16+, MSVC 2022)
- Git (for cloning the repository and fetching dependencies)
- BLAS and LAPACK libraries are essential; any library that implements the standard Fortran interface will work.
- Python 3 with development headers if you want to build the Python bindings.

> **Note:** Uni20 prefers system installations of `fmt`, `TBB`, Google Benchmark, and other optional dependencies when compatible versions are available. Otherwise CMake can fetch missing dependencies from source during configuration.

## Dependencies

```bash
apt-get install libopenblas-dev liblapack-dev
```

Optional developer packages such as `libtbb-dev`, `libbenchmark-dev`, `libfmt-dev`, and `libgtest-dev` can also be installed from the system, but Uni20 can fetch them automatically when needed.

## Building the Project

Use an out-of-source build. From the project root, run:

```bash
cmake -S . -B build

# Build the default target graph:
cmake --build build
```

If you prefer Ninja and have it installed:

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

Common configuration toggles include:

```bash
cmake -S . -B build \
  -DUNI20_BUILD_TESTS=ON \
  -DUNI20_BUILD_BENCH=ON \
  -DUNI20_BUILD_PYTHON=ON \
  -DUNI20_ENABLE_CUDA=OFF
```

### BLAS/LAPACK Detection

Uni20 uses CMake's `FindBLAS` and `FindLAPACK` modules to locate a compatible BLAS/LAPACK implementation. Use `-DUNI20_BLAS_VENDOR=<vendor>` to forward a vendor hint to `BLA_VENDOR` when you want to prefer a specific implementation such as `OpenBLAS` or an Intel MKL variant.

Vendor extension APIs are opt-in:

```bash
cmake -S . -B build -DUNI20_BLAS_VENDOR=OpenBLAS -DUNI20_BACKEND_OPENBLAS=ON
cmake -S . -B build -DUNI20_BLAS_VENDOR=Intel10_64lp_seq -DUNI20_BACKEND_MKL_SEQUENTIAL=ON
cmake -S . -B build -DUNI20_BLAS_VENDOR=Intel10_64_dyn -DUNI20_BACKEND_MKL_THREADED=ON
```

The generic BLAS wrappers remain available through `uni20::blas`. When enabled, vendor utility functions are exposed under `uni20::blas::openblas` and `uni20::blas::mkl`.
For Uni20's async runtime, prefer `UNI20_BACKEND_MKL_SEQUENTIAL=ON` unless you explicitly want MKL to run its own internal worker threads.
`UNI20_BACKEND_MKL=ON` remains as a compatibility option and selects the sequential MKL backend unless a specific MKL variant is enabled.

## Running Tests

Uni20 ships a large GoogleTest-based suite and registers the per-module tests with [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html).

You can run tests using either **CTest** or by executing test binaries directly. The test system supports both **separate** (per-module) test executables and an optional **combined** test binary.

See [docs/testing.md](testing.md) for detailed configuration options, test architecture, and best practices.

### Build and Run Tests

Tests are enabled by default. After building:

```bash
ctest --test-dir build --output-on-failure
```

This runs all tests registered with CTest. The optional combined `uni20_tests` executable is built for manual runs, but it is not registered with CTest by default.

To filter tests by name or suite:

```bash
ctest --test-dir build --output-on-failure -R IterationPlan
ctest --test-dir build -N
```

### Run Tests Directly

You may also run test executables manually. For example:

```bash
./build/tests/uni20_tests --gtest_filter=TraitsTest.*
./build/tests/common/uni20_common_tests
```

The Google Test interface supports additional flags (e.g., `--gtest_list_tests`) for exploring and selecting tests interactively.

### Disable or Reconfigure Tests

To disable all tests or change test modes, pass options to CMake during configuration:

```bash
cmake -S . -B build -DUNI20_BUILD_TESTS=OFF
cmake -S . -B build -DUNI20_BUILD_COMBINED_TESTS=OFF
```

See [testing.md](testing.md) for a full explanation of these options and how they affect the build.

## Running Benchmarks

The project uses Google Benchmark for performance measurement.

### Option 1: Using a Custom Target

Uni20 provides a `run_benchmarks` helper target:

```bash
cmake --build build --target run_benchmarks
```

### Option 2: Running Directly

Or run the benchmark executable directly:

```bash
./build/benchmarks/uni20_benchmarks
```

Benchmarks will output performance metrics (execution time, iterations, etc.) to the console.

## Python Bindings

The Python extension is built with [nanobind](https://github.com/wjakob/nanobind). Configure with `-DUNI20_BUILD_PYTHON=ON` and either build the default target graph or the extension target directly:

```bash
cmake --build build --target uni20_python
```

The compiled extension is written under `build/bindings/python/`. To try the sample binding:

```bash
export PYTHONPATH="$(pwd)/build/bindings/python:${PYTHONPATH}"
python3 -c "import uni20; print(uni20.greet())"
```

For more detail see [Python.md](Python.md).

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
  External dependencies are managed via CMake `FetchContent`. `UNI20_FETCHCONTENT_BASE_DIR` controls build and stamp files. `UNI20_FETCHCONTENT_SOURCE` controls whether fetched sources live under the build directory or a shared cache rooted at `UNI20_FETCHCONTENT_SOURCE_BASE_DIR`.

- **Directory Structure & CTest:**  
  Prefer `ctest --test-dir build`. The top-level build tree already includes the discovered tests; no extra `--recursive` flag is required.

## Contributing

Contributions to uni20 are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (e.g., `feature/new-backend`).
3. Make your changes and add tests/benchmarks as needed.
4. Submit a pull request with a clear description of your changes.

For further details, please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) file.
