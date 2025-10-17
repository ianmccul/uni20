# Uni20 Contributor Guide

> This document defines the reproducible environment, coding rules, and documentation standards for Uni20.
> It serves both human developers and automated agents (e.g., Codex, Copilot Workspace, ChatGPT Code Interpreter).
> **Agents must execute the “Mandatory Environment Setup” section before running CMake.**

---

## 1. Mandatory Environment Setup (must run before CMake)

```bash
sudo apt-get update -y && \
sudo apt-get install -y \
    cmake ninja-build g++ \
    libopenblas-dev liblapack-dev
```

**Notes for hosted environments:**

* Do **not** modify `CMakeLists.txt` to bypass dependency detection.
* If BLAS/LAPACK detection fails, re-run the install step above.
* Codex-like agents should always execute this block before attempting a build.

**Optional packages**

These packages are automatically fetched from github if they are not detected.
```
sudo apt-get install -y libtbb-dev libbenchmark-dev libfmt-dev libgtest-dev
```

---

## 2. General Expectations

* Keep commits focused, small, and logically scoped.
* Update documentation in `docs/` or `README.md` whenever APIs change.
* Add or update unit tests for new functionality.
* Commit messages should clearly describe what changed and why.

---

## 3. Core Development Rules

### 3.1 C++ Standard

* Uni20 requires **C++23**.
* Use `int const& x` style — `const` follows the type.
* When calling member functions from within other members, use `this->foo()` to clarify scope.
* Use trailing underscores (`_`) on private member variables, except for simple aggregates.
* Use `constexpr`, `consteval`, and concepts from C++23 wherever they simplify code or improve correctness.

---

### 3.2 Coroutine Safety

**Rule:**
Lambdas that define coroutines **must not have capture lists.**

```cpp
// ❌ Wrong
auto f = [x]() -> Async<int> { co_return x + 1; };

// ✅ Correct
auto f = [](int x) -> Async<int> { co_return x + 1; };
```

**Why:**
When a coroutine suspends, the lambda object is destroyed, invalidating any captured variables.
The coroutine’s heap-allocated state frame stores only its parameters and locals — not captures.
Capturing any variable risks **dangling references or use-after-free** upon resumption.
Passing values explicitly ensures safety and reproducible async semantics.

> **Agents must enforce:** no coroutine lambda may have a capture list, even if immediately invoked.

---

**Rule:**
All coroutine parameters must be passed **by value**, not by reference.

**Why:**
Coroutines decouple lifetime from their caller’s stack.
A suspended coroutine may outlive its caller, so reference parameters can dangle.
Passing by value ensures lifetime independence and safe resumption.

---

### 3.3 Asynchronous Execution Model

* `Async<T>` is the canonical async value wrapper.
* Task schedulers (`DebugScheduler`, `TbbScheduler`, etc.) manage task lifetimes.
* **Do not** use raw `std::thread` or manual synchronization primitives.

**Why:**
Schedulers coordinate task causality and epoch ordering.
Direct threading bypasses dependency tracking, leading to data races or missed wakeups.

---

### 3.4 Thread Safety

* Access to shared state must go through `EpochQueue`, `ReadBuffer`, `WriteBuffer`, or `MutableBuffer`.
* Mutations must be atomic or mutex-protected.

**Why:**
These primitives enforce *causal consistency*: all reads and writes occur in dependency order, ensuring determinism across async tasks.

---

## 4. Testing

```bash
ctest --test-dir build --output-on-failure
```

* Add or modify tests in `tests/<module>/`.
* Register new tests using `add_test_module(...)` in the relevant CMakeLists.txt.
* Keep tests deterministic; avoid random seeds without `REQUIRE_SEED`.

---

## 5. Python Bindings

* Source files live under `bindings/python/`.
* Follow the same C++ style and coroutine safety rules.
* Update API documentation in `docs/` when adding or modifying bindings.

---

## 6. Documentation

* All developer docs reside in `docs/`.
* Use Markdown tables and fenced code blocks for clarity.
* Sync all docs with behavior and API changes.

---

## 7. Doxygen Documentation Policy

**Purpose:** define how tools detect, modify, and validate documentation.

### 7.1 Comment Types
* `///` is the **canonical Doxygen form** for function, class, and member documentation.
  Tools must treat contiguous `///` lines as a single documentation block immediately preceding a declaration.

* `/** ... */` **may** appear for:

  * File- or module-level overviews (e.g., containing `\file`, `\ingroup`, `\defgroup`).
  * Long multi-paragraph or LaTeX-heavy doc blocks.
    Agents **may preserve or reformat** these, but should prefer converting routine member docs to `///`.

* `/* ... */` and `//` are **non-Doxygen implementation comments.**
  They are free for agents to clean, rewrite, or insert to clarify logic, lifetime, or invariants.
  These comments do not appear in generated documentation.

### 7.2 Formatting Rules

* Every Doxygen block must begin with `\brief`. Do *not* follow this with a blank line, unless readability demands it. Remove existing blank lines where where possible.
* Always include `\param`, `\tparam`, and `\return` when applicable.
* Only emit `\return` for callable entities (functions, lambdas, or overloaded operators)
  whose declaration includes parentheses and is not a constructor or destructor.
* Do NOT add `\return` for:
  - typedefs, using-aliases
  - structs, classes, enums, concepts
  - variables or constants
* Remove any spurious \return lines previously added to such declarations.
* Maintain this tag order:
  `\brief`, `\details`, `\pre`, `\post`, `\throws`, `\note`, `\warning`, `\tparam`, `\param`, `\return`, `\ingroup`.
* Preserve indentation relative to the documented entity.

### 7.3 Enforcement

When cleaning or generating documentation:

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
   * When a description line (for `\brief`, `\details`, `\note`, `\warning`, etc.) wraps, indent the following lines so that the first character of text aligns under the first character of the text on the previous line.
     - Example:

       ```cpp
       /// \details Real numbers are unchanged by conjugation, so the value is returned verbatim.
       ///          The overload is `constexpr`, enabling compile-time evaluation for literal arguments.
       ```
   * When adding or normalizing \ingroup tags:
     - Apply \ingroup only to top-level declarations (functions, classes, aliases, concepts, etc.).
     - Do not repeat \ingroup on members, nested types, or typedefs that reside within a grouped entity.
     - Only reapply \ingroup if a nested entity belongs to a *different* group than its parent.

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
