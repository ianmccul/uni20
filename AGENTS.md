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
    libopenblas-dev liblapack-dev libtbb-dev libbenchmark-dev libfmt-dev libgtest-dev
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
* Always run the tests, unless explicitly instructed otherwise.

### 2.1 Development-Stage API Design

* Uni20 is still in active design. Unless the user explicitly asks for
  compatibility with a released interface, prefer the clearest correct design
  over preserving existing development names, aliases, or field layouts.
* Do not add compatibility shims, duplicate legacy fields, or vague transitional
  names solely because current in-tree development code used them. Rename the
  call sites, examples, tests, and docs in the same scoped change.
* Avoid framing unfinished in-tree paths as "production" versus "prototype"
  unless the repository already defines that boundary. Name the exact algorithm,
  backend, or execution path instead.
* If a better API shape becomes clear while implementing a feature, choose that
  shape and update the nearby guidance/tests rather than layering a compatibility
  wrapper around the older design.

### 2.2 Agent-Assisted Development

* Interactive maintainer-agent development is the default workflow. A maintainer
  decision made during the working session is sufficient authority to proceed;
  a separate issue, worktree, or work log is not required unless requested.
* When semantics remain unresolved, present the alternatives and obtain a
  maintainer decision before encoding one. Do not treat current code as the
  intended contract merely because it is implemented.
* Keep durable decisions in canonical subsystem documentation. Keep verification
  in the commit or pull-request summary, and do not commit raw transcripts or
  temporary agent working notes.
* Use independent review in proportion to numerical, ownership, concurrency,
  symmetry, backend, and architectural risk. Deterministic evidence remains the
  primary verification mechanism.
* See `docs/development/agent_assisted_development.md` and
  `docs/development/code_review.md` for the current workflow and review
  guidance.

---

## 3. Core Development Rules

### 3.1 C++ Standard

* Uni20 requires **C++23**.
* Supported compiler floors are **GCC 13** and upstream **Clang 19**. Clang 18
  is not supported because its C++20 alias-template CTAD implementation is
  incomplete for Uni20's tensor aliases.
* Use `int const& x` style — `const` follows the type.
* When calling member functions from within other members, use `this->foo()` to clarify scope.
* Use trailing underscores (`_`) on private member variables, except for simple aggregates.
* Use C++20 designated initializers for aggregate configuration objects when
  they improve readability. For aggregate config structs intended for
  designated initialization, give every optional/defaulted field an explicit
  default member initializer (`= ...` or `= {}`). Treat fields without explicit
  defaults as required. In C++ aggregate list-initialization, omitted
  non-reference members are already initialized from `{}` when no default member
  initializer exists, so `-Wmissing-field-initializers` is a style/API signal
  rather than an uninitialized-memory warning: a warning should mean either a
  required field was omitted or the aggregate should spell out its default.
* Use `constexpr`, `consteval`, and concepts from C++23 wherever they simplify code or improve correctness.
* Uni20 uses the Kokkos reference `mdspan` implementation, in namespace `stdex::`.
* Uni20 uses square brackets `[]` for multi-dimensional indexing of tensors. **Do not** define `MDSPAN_USE_PAREN_OPERATOR`, use `[]` instead. This may mean adding brackets when code like `[a,b]` is used in a macro invocation, especially in TRACE and gtest macros.
* Spell complex scalar types as `uni20::complex<T>` in Uni20 code, tests,
  examples, and docs. `uni20::complex<T>` is intentionally an alias to
  `std::complex<T>`, not a wrapper. Direct `std::complex<T>` spellings should
  be limited to the alias definition, explicit alias tests/docs, or narrow
  external interop boundaries.
* Use `uni20::numeric_limits<T>` in scalar-generic Uni20 algorithms. It
  delegates to `std::numeric_limits<T>` for ordinary arithmetic types and is the
  project customization point for extension or library scalar types with
  missing/incomplete standard-library limits.

---

### 3.2 Coroutine Safety

**Rule:**
Lambdas that define async coroutines **must not have capture lists and must be defined static.**

```cpp
// ❌ Wrong
auto f = [x]() -> AsyncTask { foo(x); co_return; };

// ✅ Correct
auto f = [](int x) static -> AsyncTask { foo(x); co_return; };
```

**Why:**
Async coroutine lambdas are lambdas that return anything derived from `BasicAsyncTask`
(so `AsyncTask`, `CudaTask`, etc). This is a wrapper for a `coroutine_handle`, and
that coroutine handle has a lifetime that is distinct from the lambda itself. Captured
values are stored inside the lambda closure object, and hence go out of scope as soon
as the lambda variable is destroyed. They are not copied onto the coroutine stack frame.
If the lifetime if the coroutine is longer than the lifetime of the lambda,
then any use of the captured variable inside the coroutine will be referring to something that no longer
exists (a dangling reference). Passing values explicitly puts them into the coroutine frame ensures safety
and reproducible async semantics. This is ensured by making the lambda `operator()`` function `static`,
using the C++23 `static` lambda modifier.

> **Agents must enforce:** no coroutine lambda that is passed into the scheduler may have a capture list. To ensure this, always use the `static` modifier on the lambda function.
> Use the codebase formatting convention `static -> AsyncTask` for coroutine lambdas.

The same considerations apply to function objects that return coroutine handles, where it is clear that the function object
cannot contain state that might have a shorter lifetime than the coroutine itself.

---

### 3.3 Asynchronous Execution Model

* `Async<T>` is the canonical async value wrapper.
* Task schedulers (`DebugScheduler`, `TbbScheduler`, etc.) manage task lifetimes.
* **Do not** use raw `std::thread` or manual synchronization primitives.
* In oneTBB contexts, use **application thread** for a thread created by the
  application and **worker thread** for a thread managed by oneTBB. Use
  **host thread** only for host/device distinctions. Mention oneTBB's
  **external thread** or legacy **master thread** terminology only when
  translating upstream documentation.

**Why:**
Schedulers coordinate task causality and epoch ordering.
Direct threading bypasses dependency tracking, leading to data races or missed wakeups.

---

### 3.4 Thread Safety

* Access to shared state must go through `EpochQueue`, `ReadBuffer`, or `WriteBuffer`.
* Mutations must be atomic or mutex-protected.

**Why:**
These primitives enforce *causal consistency*: all reads and writes occur in dependency order, ensuring determinism across async tasks.

---

### 3.5 Symmetry and Block-Sparse Tensor Invariants

* Quantum-number metadata is part of the tensor type semantics. Do **not** drop
  `LocalSpace`, `BlockSpace`, `QNum`, or leg-orientation metadata when moving
  between MPS, MPO, environment, SVD, TensorContraction, CUDA, or MPI layers.
* Symmetry-aware MPS/MPO/DMRG code must preserve explicit block structure. Do
  **not** silently flatten a block-sparse tensor into one dense matrix unless
  the API name and documentation make the conversion explicit, for example a
  debug/reference helper named `to_dense_reference`, `materialize_dense_debug`,
  or similar.
* There is no dense fallback for a symmetry-typed path. A dense calculation is
  a distinct no-symmetry model/path, for example the dense Heisenberg executable
  whose local states both carry the identity charge, or an explicit conversion
  that changes the symmetry group and exits the U(1) path.
* Dense debug/reference projections may exist only as terminal diagnostics:
  they must be explicitly named, documented as leaving the symmetry-aware
  execution path, and must not feed back into U(1) MPS/MPO/DMRG state.
  Production U(1) DMRG paths must reject unsupported block-sparse operations
  rather than silently using a symmetry-erasing dense implementation.
* Every block-sparse operation must validate and/or construct blocks through
  the applicable selection rule. For the first U(1) MPS prototype:
  `ThreeLegBlockMatrix` uses `q_column = q_row + q_local`, local operator
  coefficients use `q_bra = q_ket + q_operator`, and sparse MPO entries use
  `q_left_virtual + q_ket = q_right_virtual + q_bra`.
* TensorContraction worklists generated from symmetry-aware tensors must carry
  logical block keys and placement metadata. If a temporary dense bridge is
  still required, isolate it behind an explicitly named adapter and add tests
  proving that quantum-number sector information is preserved at the adapter
  boundary.
* Agents must treat loss of symmetry metadata as a correctness bug, not an
  optimization tradeoff. If maintaining symmetry metadata is impossible for a
  requested change, stop and report the limitation instead of adding an
  implicit dense path.

**Why:**
Symmetry sectors define which tensor blocks exist and which contractions are
legal. Losing that metadata changes the mathematical problem, hides invalid
states, and prevents the CUDA/MPI block distribution strategy needed for large
DMRG calculations.

---

### 3.6 Mdspan Accessor Semantics

* Do not infer direct memory readability or writeability from
  `data_handle_type` alone. A pointer-shaped data handle only identifies a
  storage handle; the mdspan accessor defines the value semantics of
  `access(...)`.
* A Uni20 accessor that presents a read-only semantic view must declare a const
  `element_type`, even when `access(...)` returns a calculated value rather than
  a reference. Do not encode read-only behavior only in the handle type or
  `reference` alias. `MutableMdspanLike` uses const `element_type` together with
  indexed assignment validity to reject ordinary mutation. An opaque
  device-memory accessor without assignable element semantics does not model
  `MutableMdspanLike` or `MutableDeviceMdspanLike`; resolve it to an accessor with
  the required reference semantics before mutation.
* A tensor view's const interface must resolve an mdspan with const
  `element_type`. Mutable access belongs on the non-const `mdspan()` overload;
  shallow-const descriptors must not make `TensorView const&` writable.
* Direct provider calls such as BLAS/LAPACK may bypass the accessor and read the
  storage through raw pointers only when the accessor is known to be
  `stdex::default_accessor<T>` or when the wrapper explicitly understands and
  lowers the accessor semantics, for example Uni20's conjugating accessor into a
  BLAS transpose/conjugation flag.
* Treat custom accessors, transform accessors, scaling accessors, zip accessors,
  and other proxy accessors as semantic views that require explicit lowering,
  materialization, or a generic elementwise path. A pointer data handle does not
  make them BLAS-addressable.
* Uni20's lazy conjugation follows the C++26 `std::linalg::conjugated_accessor`
  model from WG21 P3050R3, but the user-facing helper is `uni20::conj(span)`.
  Do not use `std::conj` to decide real-mdspan behavior: `uni20::conj` is the
  project customization point that preserves real scalar type semantics.

**Why:**
Mdspan layouts and accessors are not just tags. A view can preserve the original
pointer while changing every value observed through `access(...)`; bypassing
that accessor silently changes the mathematical operation.

---

## 4. Testing

```bash
ctest --test-dir build --output-on-failure
```

* Add or modify tests in `tests/<module>/`.
* Register new tests using `add_test_module(...)` in the relevant CMakeLists.txt.
* Keep tests deterministic; avoid random seeds without `REQUIRE_SEED`.

---

## 5. CUDA Profiling

When profiling TensorContraction CUDA benchmarks with Nsight Systems, always
disable CUDA event tracing explicitly:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cusolver,mpi,openmp \
  --mpi-impl=openmpi \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --backtrace=fp \
  --cuda-event-trace=false \
  --cuda-memory-usage=true \
  --cudabacktrace=memory,sync,other \
  -o profiling/<report-name> \
  <command>
```

**Why:** Nsight Systems defaults `--cuda-event-trace` to `auto`. On CUDA 12.8+
drivers this may behave like enabled for TensorContraction workloads. These
workloads create many `cudaEventRecord` and `cudaStreamWaitEvent` calls, so CUDA
event tracing can add massive profiler-induced overhead and false dependencies.
In local profiling on a single-GPU `L=20, m=512` DMRG run, event tracing caused a
short two-sweep profile to time out after only 58 benchmark rows, while the same
run with `--cuda-event-trace=false` completed all 77 rows.

Use `--cuda-event-trace=true` only for a targeted CUDA event investigation, and
then run a much smaller benchmark with an explicit timeout. Do not use Nsight's
implicit `auto` setting for TensorContraction profiling.

---

## 6. Python Bindings

* Source files live under `bindings/python/`.
* Follow the same C++ style and coroutine safety rules.
* Update API documentation in `docs/` when adding or modifying bindings.

---

## 7. Documentation

* All developer docs reside in `docs/`.
* Use Markdown tables and fenced code blocks for clarity.
* Sync all docs with behavior and API changes.
* When linking to a repository directory whose index is `README.md`, link to
  `directory/`, not `directory/README.md`, so GitHub shows the directory listing
  before the rendered index. Link directly to `README.md` only when a fragment
  identifies a specific section.
* When changing scalar aliases, scalar traits, scalar numeric-limits behavior,
  or scalar spelling policy, update `docs/tensor/scalar_policy.md`.
* When changing Krylov algorithms, supported scalar types, public Krylov
  parameters, default values, or internal convergence/restart tuning, update
  `docs/krylov/algorithms.md`. If scalar support changes, also update
  `docs/krylov/precision_validation.md`.
* When adding or removing dense BLAS/LAPACK wrappers, provider routine coverage,
  or quarantined dense helper probes, update
  `docs/linalg/dense_blas_lapack_coverage.md`. Keep provider-wide coverage
  inventories out of the Krylov algorithm contract.

---

## 8. Doxygen Documentation Policy

**Purpose:** keep generated API documentation useful without forcing noisy,
mechanical comments into ordinary implementation code.

### 8.1 Comment Types
* `///` is the preferred Doxygen form for ordinary function, class, alias,
  concept, and member documentation. Treat contiguous `///` lines immediately
  preceding a declaration as that declaration's documentation block.

* `/** ... */` **may** appear for:

  * File- or module-level overviews (e.g., containing `\file`, `\ingroup`, `\defgroup`).
  * Long multi-paragraph or LaTeX-heavy doc blocks.
  * Existing comments where converting form would create churn without improving
    the documentation.

* `/* ... */` and `//` are **non-Doxygen implementation comments.**
  They are free for agents to clean, rewrite, or insert to clarify logic, lifetime, or invariants.
  These comments do not appear in generated documentation.

### 8.2 Formatting Rules

* New or substantially edited public API Doxygen blocks should begin with a
  concise `\brief`.
* Do not add blank lines after `\brief` unless they improve readability for a
  multi-paragraph block.
* Include `\param`, `\tparam`, and `\return` when they clarify behavior,
  requirements, ownership, lifetime, or non-obvious semantics. Do not add
  tautological tags solely to satisfy a checklist.
* Only emit `\return` for callable entities (functions, lambdas, or overloaded operators)
  whose declaration includes parentheses and is not a constructor or destructor.
* Do NOT add `\return` for:
  - typedefs, using-aliases
  - structs, classes, enums, concepts
  - variables or constants
* Remove any spurious \return lines previously added to such declarations.
* When several tags are present, use this order:
  `\brief`, `\details`, `\pre`, `\post`, `\throws`, `\note`, `\warning`, `\tparam`, `\param`, `\return`, `\ingroup`.
* Preserve indentation relative to the documented entity.

### 8.3 Enforcement

When cleaning or generating documentation:

1. **Document intentionally**

   * Add Doxygen for new or changed public APIs, conceptual interfaces,
     cross-module contracts, and declarations with important invariants.
   * Internal helpers should be documented when their behavior, lifetime, or
     invariants are not obvious from the code.
   * Do not synthesize placeholder comments such as `[TODO] Document this
     function`.
   * Avoid documentation-only churn outside the files and declarations touched
     for the task.

2. **Normalize comment form**

   * Prefer `///` for new ordinary declaration docs.
   * File- or module-level headers using `/** ... */` may remain block-style.
   * Preserve existing block comments unless the surrounding edit already makes
     cleanup worthwhile.
   * When a description line (for `\brief`, `\details`, `\note`, `\warning`, etc.) wraps, indent the following lines so that the first character of text aligns under the first character of the text on the previous line.
     - Example:

       ```cpp
       /// \details Real numbers are unchanged by conjugation, so the value is returned verbatim.
       ///          The overload is `constexpr`, enabling compile-time evaluation for literal arguments.
       ```

3. **Maintain implementation commentary**

   * Preserve and, when useful, rewrite `//` or `/* ... */` comments to improve clarity or correctness.
   * Agents may insert new internal comments where explanation would aid maintainability.
   * Do not upgrade implementation comments to Doxygen unless the declaration is
     part of the documented API surface or carries a useful invariant.

4. **Use grouping sparingly**

   * Define groups at module or file boundaries, not on every declaration.
   * Add `\ingroup` only where it materially improves generated navigation.
   * Do not repeat `\ingroup` on members, nested types, or declarations already
     covered by the surrounding file or type documentation.

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
Treat headers under `src/uni20/` as potentially developer-facing, but do not
document every internal helper mechanically.

- Use `\internal ... \endinternal` only when generated documentation must hide
  or distinguish a non-public declaration.
- Namespaces such as `uni20::internal`, `detail`, or directories named
  `internal` or `detail` already signal internal scope; do not add redundant
  internal grouping tags unless they help the generated docs.
