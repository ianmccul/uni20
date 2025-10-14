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
  - All shared state modifications must be guarded by atomic operations or appropriate mutexes.  

## C++ and CUDA sources (`src/`, `tests/`, `bindings/python/`)
- The project is built with C++23; prefer standard-library utilities over bespoke helpers when they exist.  
- Follow the existing `.clang-format` configuration. After configuring the build directory run:  
  ```bash
  cmake --build build --target clang_format
````

or run `clang-format` manually on touched files before committing.

* Keep header interfaces tight: minimize `#include`s in headers and prefer forward declarations where possible.
* Guard platform-specific code with the appropriate feature-detection macros found in `src/common`.

## C++ Coding Style

* Use `int const& x` style — `const` follows the type.
* When calling member functions from within other members, use `this->foo()` to clarify scope.
* Use trailing underscores (`_`) on private member variables, except for simple aggregates.
* Use `constexpr`, `consteval`, and concepts from C++23 wherever they simplify code or improve correctness.

## Environment setup

* Before any build, ensure required system dependencies are installed:

  * `sudo apt-get update`
  * `sudo apt-get install -y libtbb-dev libbenchmark-dev libfmt-dev libopenblas-dev libgtest-dev`
* CMake attempts to use system versions of TBB and google benchmark (if benchmarking is configured), but will use FetchContent if not available.
* CMake always uses FetchContent to fetch the reference version of kokkos/mdspan.

## Build configuration

* Use out-of-tree builds (`cmake -S . -B build`). Helpful options:

  * `-DUNI20_BUILD_TESTS=ON` (default) to compile tests.
  * `-DUNI20_ENABLE_WARNINGS=ON` (default) to keep warnings visible.
* The code should compile without warnings when `UNI20_ENABLE_WARNINGS` is `ON` and without relying on `-Werror` suppressions.

## Testing

* After configuring the project, build it with your preferred generator (e.g. `cmake --build build`).
* Run tests with CTest from the build directory:

  ```bash
  ctest --output-on-failure
  ```
* When you add or modify tests, register them via the `add_test_module(...)` helper in the relevant `tests/<module>/CMakeLists.txt` file.

## Python bindings

* Python bindings live in `bindings/python/`. They are C++ source files compiled into an extension; follow the same formatting rules as the rest of the C++ code.
* If you introduce new Python-facing APIs, update any reference material under `docs/` accordingly.

## Documentation

* The `docs/` directory contains developer references. Keep these files synchronized with behavior changes, and prefer Markdown tables and code fences for examples.

---

## Doxygen Documentation Policy (for automated agents)

This section defines how automated tools should detect, modify, and validate documentation in Uni20 source files.

### 1. Comment Type Rules

* `///` is the **canonical Doxygen form** for function, class, and member documentation.
  Tools must treat contiguous `///` lines as a single documentation block immediately preceding a declaration.

* `/** ... */` **may** appear for:

  * File- or module-level overviews (e.g., containing `\file`, `\ingroup`, `\defgroup`).
  * Long multi-paragraph or LaTeX-heavy doc blocks.
    Agents **may preserve or reformat** these, but should prefer converting routine member docs to `///`.

* `/* ... */` and `//` are **non-Doxygen implementation comments.**
  They are free for agents to clean, rewrite, or insert to clarify logic, lifetime, or invariants.
  These comments do not appear in generated documentation.

### 2. Formatting Expectations

* Every Doxygen block must begin with `\brief`. Do *not* follow this with a blank line, unless readability demands it. Remove existing blank lines where where possible.
* Always include `\param`, `\tparam`, and `\return` when applicable.
* Maintain this tag order:
  `\brief`, `\details`, `\pre`, `\post`, `\throws`, `\note`, `\warning`, `\tparam`, `\param`, `\return`, `\ingroup`.
* Preserve indentation relative to the documented entity.

### 3. Enforcement Logic

When cleaning or generating documentation, automated agents must:

1. **Insert missing documentation**

   * If a public declaration has no Doxygen block, synthesize one automatically.
   * The generated block must include a meaningful `\brief` inferred from the symbol name, type, or nearby comments.
   * Only fall back to a placeholder such as:

     ```cpp
     /// \brief [TODO] Document this function.
     ```

     if no meaningful inference is possible.

2. **Normalize comment form**

   * Convert `/** ... */` comments on individual members to `///` form unless the block contains multi-paragraph or LaTeX-heavy content.
   * File- or module-level headers using `/** ... */` should remain block-style.
   * Use `/** ... */` **only** for `\defgroup` or file-level documentation.
   * All ordinary macros, functions, and classes must use `///` Doxygen comments.
   * Convert misplaced `/** ... */` to `///` while preserving all tags and text.

3. **Maintain implementation commentary**

   * Preserve and, when useful, rewrite `//` or `/* ... */` comments to improve clarity or correctness.
   * Agents may insert new internal comments where explanation would aid maintainability.
   * Implementation comments must never be upgraded to Doxygen unless the entity is part of the public API.

4. **Ensure completeness**

   * Every Doxygen block must contain at least a `\brief` and all applicable tags (`\param`, `\tparam`, `\return`, etc.).
   * Agents should fill in missing parameter or return documentation where it can be inferred safely.

5. **File and module headers**

   * When generating or repairing top-of-file documentation, use a block comment:

     ```cpp
     /**
      * \file tensor_ops.hpp
      * \ingroup core
      * \brief Tensor addition and contraction routines.
      */
     ```

6. **Internal vs. Public Documentation**

Uni20 does not currently define a stable "public API" boundary.  
All code should be documented, including internal helpers, but documentation must distinguish between *conceptual interfaces* and *implementation internals*.

- Use `\ingroup internal` or `\internal ... \endinternal` for functions, classes, or templates not intended for external use.
- Internal documentation follows the same Doxygen formatting rules as all other code.
- Documentation generators may exclude internal content using `INTERNAL_DOCS = NO` in the Doxygen configuration.
- Codex and other agents may automatically infer internal scope from namespaces such as `uni20::internal`, `detail`, or directories named `internal` or `detail`.
