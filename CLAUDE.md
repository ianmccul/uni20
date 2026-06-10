# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**uni20** is an early-stage C++23 tensor-network library. The load-bearing
subsystem is an epoch-ordered **async runtime**; a near-term goal is a minimal
real-valued **U(1) DMRG** built on it. `AGENTS.md` is the full contributor guide
(environment setup, coding rules, symmetry invariants, doc policy) and is the key
document to read alongside this file.

## Build, test, format

```bash
# Configure (out-of-source). Defaults to Release; pass -G Ninja if available.
cmake -S . -B build

# Build everything
cmake --build build -j

# Run the full test suite (per-module GoogleTest binaries registered with CTest)
ctest --test-dir build --output-on-failure

# Filter tests by name (regex over CTest test names)
ctest --test-dir build --output-on-failure -R IterationPlan
ctest --test-dir build -N                       # list discovered tests

# Run a single module's test binary directly, with gtest filters
./build/tests/common/uni20_common_tests --gtest_filter=TraitsTest.*
./build/tests/uni20_tests --gtest_filter='...'  # optional combined binary

# Format all tracked C/C++ sources (repo-root aware; run from anywhere)
scripts/format-sources.sh
```

Common configure toggles (see `CMakeLists.txt` ~line 88–112 for the full list):
`UNI20_BUILD_TESTS` (ON), `UNI20_BUILD_COMBINED_TESTS` (ON), `UNI20_BUILD_PYTHON`
(ON), `UNI20_ENABLE_CUDA` (OFF), `UNI20_ENABLE_MPI` (OFF),
`UNI20_SANITIZE=address,undefined,...`.

System packages (`apt`): `cmake ninja-build g++ libopenblas-dev liblapack-dev
libtbb-dev libbenchmark-dev libfmt-dev libgtest-dev`. **Do not edit `CMakeLists.txt`
to bypass dependency detection** — missing optional deps (fmt, TBB, Benchmark,
GTest, nanobind) are auto-fetched via FetchContent.

### Adding tests

Tests live in `tests/<module>/` and are registered with
`add_test_module(<name> SOURCES ... LIBS ...)` (defined in
`cmake/Uni20TestHelpers.cmake`). This creates `uni20_<name>_tests` and, when
`UNI20_BUILD_COMBINED_TESTS=ON`, folds the sources into the combined `uni20_tests`
binary. Keep tests deterministic; no random seeds without `REQUIRE_SEED`.

## Architecture

Source lives under `src/uni20/<module>/`. The layering (see
`docs/architecture_diagram.md`) is roughly: Tensor/TensorView → mdspan utilities;
Level1 ops → Kernel dispatch → BLAS / CPU-linalg backends (CUDA/cuSOLVER backends
are stubs).

**Async runtime** (`src/uni20/async/`, docs in `docs/async/`) is the core of the
library. Correctness comes from **epoch ordering, not manual locking**. `Async<T>`
is the canonical async value; reads/writes go through `ReadBuffer<T>`/`WriteBuffer<T>`
gated by `EpochContext`/`EpochQueue`. The same code runs deterministically under
`DebugScheduler` or concurrently under `TbbScheduler`/`TbbNumaScheduler`.
- Never use raw `std::thread` or manual sync primitives — go through the buffers.
- `Async<T>()` is *unconstructed* (not readable until a writer constructs it);
  `Async<T>(args...)` is constructed and immediately readable.

Future heterogeneous (GPU/MPI) execution follows a **device-first** design — see
`docs/roadmap.md` §4.0, `docs/ordering_and_backend_lowering.md`, and
`docs/storage_kind_and_location.md`.

Other modules: `backend/` (BLAS/CUDA/cuSOLVER wrappers), `kernel/` (kernel
dispatch), `linalg/`, `tensor/`, `level1/`, `mdspan/`, `storage/`, `tags/`,
`common/`, `core/`, `expokit/`. Python bindings use **nanobind** under
`bindings/python/`.

## Code conventions (from AGENTS.md — read it for the full set)

- **C++23**. `const` follows the type: `int const& x`. Trailing `_` on private
  members (except simple aggregates). Use `this->foo()` when calling members.
- Tensors use the Kokkos reference `mdspan` in namespace `stdex::`. Use **square
  brackets `[]`** for multi-dim indexing — do **not** define
  `MDSPAN_USE_PAREN_OPERATOR`. (Watch for `[a,b]` inside TRACE/gtest macros.)
- **Coroutine safety (critical):** async coroutine lambdas (anything returning a
  type derived from `BasicAsyncTask`, e.g. `AsyncTask`, `CudaTask`) **must have
  no capture list and must use the `static` modifier** — pass all values as
  parameters. Convention: `[](int x) static -> AsyncTask { ... }`. Captures would
  dangle because the coroutine outlives the lambda closure.
- **Symmetry metadata is part of the type, not an optimization** (`AGENTS.md` §3.5).
  Never silently drop `LocalSpace`/`BlockSpace`/`QNum`/leg-orientation or flatten a
  block-sparse tensor to dense on a symmetry-typed path. There is **no dense
  fallback** for a symmetry path; dense reference helpers must be explicitly named
  (e.g. `to_dense_reference`) and must not feed back into U(1) state. Treat loss of
  symmetry metadata as a correctness bug — if a change can't preserve it, stop and
  report rather than adding an implicit dense path.

## Docs & formatting policy

Developer docs live in `docs/` (index: `docs/README.md`). Keep docs in sync with
API changes. Doxygen: `///` is canonical for member/function/class docs (start
every block with `\brief`); `/** ... */` only for file/module headers and
`\defgroup`. Tag order: `\brief \details \pre \post \throws \note \warning \tparam
\param \return \ingroup`. Use `\ingroup internal` / `uni20::internal` / `detail`
for non-public code. Full policy in `AGENTS.md` §8.

## CUDA profiling

When profiling event-heavy CUDA workloads with Nsight Systems, pass
`--cuda-event-trace=false` (Nsight's `auto` behaves like enabled on CUDA 12.8+ and
adds large overhead / false dependencies for event-heavy workloads). Full `nsys`
invocation in `AGENTS.md` §5.
