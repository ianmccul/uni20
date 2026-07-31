# uni20 - A C++23 Tensor-Network Library

Welcome to **uni20**. This repository contains an early-stage tensor-network library written in C++23 with optional BLAS, CUDA, benchmark, documentation, and Python binding support. This guide covers the current CMake-based workflow for configuring, building, testing, benchmarking, and generating documentation.

## Project Structure

A brief overview of the directory layout:

```
uni20/                         # Project root
├── CMakeLists.txt             # Top-level CMake configuration
├── README.md                  # Project overview
├── AGENTS.md                  # Contributor and coding policy
├── docs/                      # Topic-oriented developer documentation
│   ├── README.md              # Documentation index
│   ├── architecture/          # Dispatch, ordering, and execution design
│   ├── async/                 # Async runtime, schedulers, AD, and diagnostics
│   ├── tensor/                # Tensor values, views, operations, and scalars
│   ├── linalg/                # Dense linalg and provider lowering
│   ├── krylov/                # Krylov algorithms and validation
│   ├── symmetry/              # QNum and block-sparse design
│   └── backends/              # CUDA and MPI design
├── cmake/                     # CMake modules and dependency helpers
├── src/uni20/                 # C++ library source
│   ├── common/                # Shared traits, diagnostics, and presentation
│   ├── async/                 # Coroutine runtime and schedulers
│   ├── backend/               # Provider and ABI facades
│   ├── mdspan/                # Mdspan concepts, views, and accessors
│   ├── storage/               # Owning storage policies
│   ├── tensor/                # Tensor values, aliases, and front-end operations
│   ├── linalg/                # Operation tags, dispatch, and dense kernels
│   ├── krylov/                # Matrix-free Krylov algorithms
│   ├── symmetry/              # Quantum-number foundations
│   └── kernel/                # Lower-level/reference kernel infrastructure
├── tests/                     # GoogleTest modules organized by subsystem
├── benchmarks/                # Google Benchmark targets
├── examples/                  # Runnable examples organized by subsystem
└── bindings/python/           # Nanobind Python module and smoke tests
```

## Prerequisites

Before building the project, ensure you have the following installed:

- **CMake 3.24+**.
- A supported C++23 compiler: GCC 13 or newer, or upstream Clang 19 or newer.
  Clang 18 is not supported because its alias-template class template argument
  deduction implementation is incomplete for Uni20's tensor aliases.
- Git (for cloning the repository and fetching dependencies)
- BLAS and LAPACK libraries are essential; any library that implements the standard Fortran interface will work.
- oneTBB 2022.3 or newer. CMake fetches the pinned source release when a
  compatible system installation is unavailable.
- Python 3.11 or newer with development headers if you want to build the
  Python bindings.

> **Note:** Uni20 prefers system installations of `fmt`, `TBB`, Google Benchmark, and other optional dependencies when compatible versions are available. Otherwise CMake can fetch missing dependencies from source during configuration.

## Dependencies

```bash
apt-get install libopenblas-dev liblapack-dev
```

Optional developer packages such as `libbenchmark-dev`, `libfmt-dev`, and
`libgtest-dev` can also be installed from the system, but Uni20 can fetch them
automatically when needed. Install `libtbb-dev` only when it provides oneTBB
2022.3 or newer; Ubuntu 24.04's 2021.11 package is too old.

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

### CUDA Configuration And Runtime Initialization

Configure a CUDA build with the CUDA language and backend enabled:

```bash
cmake -S . -B build-cuda -DUNI20_ENABLE_CUDA=ON
cmake --build build-cuda --target cuda_hello_world_example
./build-cuda/examples/cuda_hello_world_example
```

The hello-world example reports the CUDA toolkit, driver, visible devices, and
Uni20 backend capabilities. It also installs the runtime described below and
exercises each enrolled device's stream/completion resources.

A CUDA application explicitly installs one process-wide resource service near
the start of `main()`:

```cpp
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/tensor/tensor.hpp>

int main()
{
  auto cuda_lifetime = uni20::cuda::initialize({
      .streams_per_device = 8,
  });

  uni20::CudaTensor<uni20::float32, 2> matrix(32, 48);
  // CUDA tensors and tasks must be destroyed before cuda_lifetime.
}
```

Retain the returned `cuda::Runtime` guard for as long as any CUDA Tensor,
buffer, stream, provider lease, or CUDA task exists. The runtime is not passed
through Tensor operations: ordinary extent-only `CudaTensor` construction
uses its configured default device, while `cuda::device_resources(device)`
selects another enrolled device for explicit construction.

`CudaTensor` describes device storage; it does not choose blocking versus
coroutine execution. A direct Tensor matrix product may block while acquiring
an idle cuBLAS handle and stream, then returns after publishing the queued CUDA
work to the buffers' completion ledgers. Wrapping the same type in
`Async<CudaTensor>` selects coroutine-aware dispatch: its `CudaTask` awaits
those resources without blocking a scheduler participant.

By default, `cuda::initialize()` enrolls every visible device, chooses the
first enrolled device as the default, and creates eight actually-idle streams
per device. `RuntimeConfig::device_ordinals`, `default_device`,
`streams_per_device`, and `stream_flags` customize that installation. Only one
runtime may be active in a process. Direct `cuda::DeviceResources`
construction is reserved for focused tests and low-level bring-up.

The CUDA API and Tensor aliases are available only in CUDA-enabled builds. See
the [CUDA Runtime Foundation](backends/cuda/runtime.md) for lifetime and
resource contracts and [CUDA Buffers](backends/cuda/buffers.md) for the
stream-synchronized access model used by backend authors.

### Build Information

Uni20 generates build metadata for both C++ and Python. In C++:

```cpp
#include <uni20/buildinfo.hpp>

auto const info = uni20::build_info::current();
```

For a terminal report:

```bash
cmake --build build --target buildinfo_example
./build/examples/buildinfo_example
```

See [Build Information](development/build_information.md) for the full C++ and
Python API shape.

## Running Tests

Uni20 ships a large GoogleTest-based suite and registers the per-module tests with [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html).

You can run tests using either **CTest** or by executing test binaries directly. The test system supports both **separate** (per-module) test executables and an optional **combined** test binary.

See [Testing](development/testing.md) for detailed configuration options, test
architecture, and best practices.

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

See [Testing](development/testing.md) for a full explanation of these options
and how they affect the build.

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

For more detail see [Python Bindings](python/bindings.md).

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

- **CMake Options:** Use project-specific CMake options (prefixed with
  `UNI20_`) to enable or disable features such as CUDA, MPI, testing, and
  benchmarking.

- **Dependency Management:** External dependencies are managed through CMake
  `FetchContent`. `UNI20_FETCHCONTENT_BASE_DIR` controls build and stamp files.
  `UNI20_FETCHCONTENT_SOURCE` controls whether fetched sources live under the
  build directory or a shared cache rooted at
  `UNI20_FETCHCONTENT_SOURCE_BASE_DIR`.

- **Directory Structure and CTest:** Prefer `ctest --test-dir build`. The
  top-level build tree already includes the discovered tests; no extra
  `--recursive` flag is required.

## Contributing

Contributions to uni20 are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (e.g., `feature/new-backend`).
3. Make your changes and add tests/benchmarks as needed.
4. Submit a pull request with a clear description of your changes.

For further details, see the [Contributor Guide](CONTRIBUTING.md).
