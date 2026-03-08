# Uni20 Architecture and Roadmap

This document is the current planning baseline for Uni20.

It consolidates and replaces the old local design drafts:

- `docs/tensor_design.md`
- `docs/Uni20TensorArchitecture.md`
- `docs/ReferenceCountingBasicTensor.md`

## 1. Current Architecture Snapshot (2026-03)

### 1.1 Tensor and mdspan stack

Current code shape:

- `BasicTensor` owns storage and currently inherits `TensorView` (`src/uni20/tensor/basic_tensor.hpp`).
- `TensorView` is the core non-owning view abstraction (`src/uni20/tensor/tensor_view.hpp`).
- mdspan integration and concepts live in `src/uni20/mdspan/`.
- Level-1 tensor kernels live in `src/uni20/level1/`.

### 1.2 Async runtime and AD

Current code shape:

- `Async<T>`, `ReadBuffer<T>`, `WriteBuffer<T>` in `src/uni20/async/`.
- Epoch ordering via `EpochQueue` / `EpochContext`.
- Scheduler implementations: `DebugScheduler`, `TbbScheduler`, `TbbNumaScheduler`.
- Reverse-mode AD via `Var<T>` and `ReverseValue<T>`.

### 1.3 Numerical backend layer

Current code shape:

- BLAS backend wrappers under `src/uni20/backend/blas/`.
- CPU linalg backend under `src/uni20/linalg/backends/cpu/`.
- CUDA/cuSOLVER directories exist but are partial/stub-oriented.

### 1.4 Python bindings

Current code shape:

- Python extension built with `nanobind` under `bindings/python/`.

## 2. What Is Working Well

- Clear module split under `src/uni20/` (`tensor`, `mdspan`, `level1`, `async`, `linalg`, `backend`).
- Deterministic async sequencing model with strong test coverage in `tests/async/`.
- Practical AD model integrated with the same async runtime.
- Build and dependency setup now has stronger diagnostics and reproducibility controls.

## 3. Main Gaps

### 3.1 Tensor/view lifetime model

Today, `BasicTensor` is storage-owning and `TensorView` is non-owning.
For more advanced async and slicing workflows, we still need a clearer long-term lifetime-sharing strategy.

### 3.2 Assignment semantics for reference-like tensor views

Async write semantics now support trait-based dispatch (`assignment_semantics_of<T>`), but tensor-side
specializations and conventions are not yet finalized.

### 3.3 Expression-level execution model

Uni20 currently favors explicit operation calls. A unified expression model (for fusion/lazy lowering)
is still a roadmap item.

### 3.4 Heterogeneous execution maturity

CPU and BLAS are usable today. CUDA/cuSOLVER and broader heterogeneous scheduling remain incomplete.

## 4. Roadmap

## Phase A — Stabilize tensor lifetime semantics

Goals:

- Define the preferred ownership-sharing model for tensor/view workflows.
- Keep current APIs usable while introducing explicit lifetime-safe view patterns.

Deliverables:

- Design note + tests covering view lifetime guarantees.
- Clear guidance for async-safe tensor/view handoff patterns.

## Phase B — Finalize async assignment behavior for tensor types

Goals:

- Decide and document which tensor-related types are `rebind` vs `write_through`.
- Ensure `co_await writer = rhs` behavior is unambiguous for tensor views.

Deliverables:

- Trait specializations for the chosen tensor proxy/view types.
- Tests mirroring real tensor/view write-through and explicit `rebind(...)` paths.
- Documentation update in `docs/async/`.

## Phase C — Introduce expression-layer roadmap implementation

Goals:

- Add a minimal expression node layer where it brings clear value (fusion/reduced temporaries).
- Keep backend lowering aligned with existing `level1`, `kernel`, and `linalg` code.

Deliverables:

- Initial expression API proposal and one implemented vertical slice.
- Benchmarks demonstrating no regression and targeted improvements.

## Phase D — Async-aware tensor expression execution

Goals:

- Bridge expression evaluation with async scheduling in a controlled way.
- Preserve explicit sequencing guarantees from the existing buffer model.

Deliverables:

- Prototype async evaluation path for selected expression nodes.
- Error/cancellation behavior documented and covered by tests.

## Phase E — Backend expansion and parity

Goals:

- Improve CUDA/cuSOLVER path maturity.
- Keep backend dispatch behavior transparent and testable.

Deliverables:

- Backend capability matrix in docs.
- Backend-specific correctness tests and representative performance benchmarks.

## 5. Guardrails

- Maintain C++23 and existing coroutine safety rules (captureless `static` coroutine lambdas).
- Keep async determinism and epoch ordering invariants intact.
- Prefer incremental, test-backed refactors over broad rewrites.
- Update docs in the same change whenever semantics shift.

## 6. Related Docs

- `docs/architecture_diagram.md`
- `docs/async/README.md`
- `docs/async/reverse_mode_ad.md`
- `docs/async/buffers_and_awaiters.md`
- `docs/testing.md`
