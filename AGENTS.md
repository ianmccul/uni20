# Repository Guidelines for `uni20`

Welcome! This file lists the expectations for changes anywhere in this repository.

## General expectations
- Keep commits focused and well described. Prefer small, logically-scoped changes.
- When adding new functionality, check whether existing documentation in `docs/` or the top-level `README.md` needs an update to stay accurate.
- New features should be accompanied by unit tests (see **Testing** below) when practical.

## Core Development Rules

* **C++ Standard:** C++23 required.
* **Coroutine Safety:**
  - Lambdas defining coroutines **must not** have capture lists.
  - All coroutine parameters must be passed **by value**, not by reference.
  - Any lambda with captures that suspends leads to **undefined behavior** (the lambda object’s stack frame is destroyed).
* **Asynchronous Execution Model:**
  - `Async<T>` is the canonical asynchronous value wrapper.
  - Schedulers (`DebugScheduler`, `TbbScheduler`, etc.) manage task lifetimes; no direct `std::thread` use.
* **Thread Safety:**
  - `EpochQueue`, `ReadBuffer`, `WriteBuffer`, `MutableBuffer` enforce causal access.
  - All shared state modifications must be guarded by atomic operations or appropriate mutexes

## C++ and CUDA sources (`src/`, `tests/`, `bindings/python/`)
- The project is built with C++23; prefer standard-library utilities over bespoke helpers when they exist.
- Follow the existing `.clang-format` configuration. After configuring the build directory run:
  ```bash
  cmake --build build --target clang_format
  ```
  or run `clang-format` manually on touched files before committing.
- Keep header interfaces tight: minimize `#include`s in headers and prefer forward declarations where possible.
- Guard platform-specific code with the appropriate feature-detection macros found in `src/common`.

## Environment setup
- Before any build, ensure required system dependencies are installed:
  - sudo apt-get update
  - sudo apt-get install -y libtbb-dev libbenchmark-dev libfmt-dev libopenblas-dev libgtest-dev
- CMake attempts to use system versions of TBB and google benchmark (if benchmarking is configured), but will use FetchContent if not available
- CMake always uses FetchContent to fetch the reference version of kokkos/mdspan

## Build configuration
- Use out-of-tree builds (`cmake -S . -B build`). Helpful options:
  - `-DUNI20_BUILD_TESTS=ON` (default) to compile tests.
  - `-DUNI20_ENABLE_WARNINGS=ON` (default) to keep warnings visible.
- The code should compile without warnings when `UNI20_ENABLE_WARNINGS` is `ON` and without relying on `-Werror` suppressions.

## Testing
- After configuring the project, build it with your preferred generator (e.g. `cmake --build build`).
- Run tests with CTest from the build directory:
  ```bash
  ctest --output-on-failure
  ```
- When you add or modify tests, register them via the `add_test_module(...)` helper in the relevant `tests/<module>/CMakeLists.txt` file.

## Python bindings
- Python bindings live in `bindings/python/`. They are C++ source files compiled into an extension; follow the same formatting rules as the rest of the C++ code.
- If you introduce new Python-facing APIs, update any reference material under `docs/` accordingly.

## Documentation
- The `docs/` directory contains developer references. Keep these files synchronized with behavior changes, and prefer Markdown tables and code fences for examples. 
