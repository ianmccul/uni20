# Uni20 Architecture and Roadmap

This document is the current planning baseline for Uni20.

Revised 2026-06 to adopt **device-first** sequencing (see §4.0): GPU and MPI are
designed into the foundational abstractions from the start rather than layered on
after a CPU-only baseline.

It consolidates and replaces the old local design drafts:

- `docs/tensor_design.md`
- `docs/Uni20TensorArchitecture.md`
- `docs/ReferenceCountingBasicTensor.md`

## 1. Current Architecture Snapshot (2026-03)

### 1.1 Tensor and mdspan stack

Current code shape:

- `BasicTensor` owns storage by composition and exposes resolved mdspans.
- The Tensor-view concept family includes readable, mutable, and rank-constrained
  forms in `src/uni20/tensor/concepts.hpp`.
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
- Future backend work should follow the compile-time capability, runtime `try_*`,
  and fallback pattern described in `docs/backend_dispatch.md`.

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

Today, `BasicTensor` owns storage and models the tensor-level concepts directly.
There is no general concrete non-owning tensor adaptor yet. Advanced async and
slicing workflows still need an explicit lifetime-sharing design.

### 3.2 Assignment semantics for reference-like tensor views

Async write semantics now support trait-based dispatch (`assignment_semantics_of<T>`), but tensor-side
specializations and conventions are not yet finalized.

### 3.3 Expression-level execution model

Uni20 currently favors explicit operation calls. A unified expression model (for fusion/lazy lowering)
is still a roadmap item.

### 3.4 Heterogeneous execution maturity

CPU and BLAS are usable today. CUDA/cuSOLVER and broader heterogeneous scheduling remain incomplete.
Under device-first sequencing (§4.0) these are built out incrementally on top of device-aware
foundations, not deferred to a final phase.

### 3.5 Storage location and ordering ownership

Two foundational abstractions are not yet expressed in core:

- **Storage memory kind vs location.** Storage has a compile-time *kind* axis
  (`StoragePolicy`) but no runtime *location* (device ordinal / MPI rank). See
  `docs/storage_kind_and_location.md`.
- **Ordering ownership.** The Async scheduler should own all ordering; CUDA events and
  MPI requests are derived lowerings, with a polymorphic completion-token abstraction.
  See `docs/ordering_and_backend_lowering.md`.

### 3.6 Block-sparse tensor data model

There is no real block-sparse tensor class yet beyond embryonic experiments, and it
is the linchpin the rest of the execution stack consumes. The design — two-level
(lightweight `mdspan` block + block-sparse container), typed legs
(BlockSpace/LocalSpace), and a single **layout** object (device / MPI-rank map +
coalescing-aware memory plan) — is captured in `docs/block_sparse_tensor.md` and
refined into the symmetry-typed `BlockTensor` in `docs/block_tensor.md`. Its
companion `docs/block_coalescing.md` covers single-axis GEMM grouping,
`docs/kernel_dispatch.md` the backend-list dispatch, and
`docs/execution_architecture.md` ties the data model, dispatch, and scheduling
together under a mechanism/policy split with a concrete build order. The empirical
findings that informed these are in `docs/tensorcontraction_integration_findings.md`.

## 4. Roadmap

### 4.0 Sequencing principle: device-first

Earlier planning assumed a CPU-first sequence — make everything work on the CPU,
then add GPU schedulers, then add MPI. We are replacing that with **device-first**
sequencing: tensor data (and other data) conceptually lives on a *device*, and
GPU + MPI are designed into the foundational abstractions from the start. The
implementations still land incrementally — kernels and schedulers are built out bit
by bit — but the *interfaces* must not bake in host-only or single-device
assumptions, because those are the assumptions that are expensive to remove later.

Three abstractions are load-bearing and must be device-aware from the outset:

1. **Storage memory kind vs location.** Memory *kind* (host / device / unified) is a
   compile-time property of the type that selects legal kernels; *location* (which
   device ordinal, which MPI rank) is a runtime value attached to storage. See
   `docs/storage_kind_and_location.md`.
2. **Completion / dependency tokens.** The Async scheduler owns ordering; CUDA
   events and MPI requests are derived performance lowerings. The token abstraction
   must be polymorphic over {CPU done, CUDA event, MPI request} from day one. See
   `docs/ordering_and_backend_lowering.md`.
3. **Multi-buffer operation acquisition.** Acquiring read/write access for a whole
   operation as one transaction; see `docs/gpu_epoch_design_draft.md`.

Everything else — actual NCCL/MPI wiring, device schedulers, individual kernels —
stays incremental and test-backed. The synchronous-blocking device path is correct
by construction and serves as the correctness oracle for the optimized path, so a
backend is correct from its first kernel.

The phases run on two tracks that proceed largely in parallel. The device-first
change is that the heterogeneous track's *foundations* (Phase H1) start early and are
not deferred to a final "backend expansion" phase.

### Track 1 — Tensor and async semantics

#### Phase A — Stabilize tensor lifetime semantics

Goals:

- Define the preferred ownership-sharing model for tensor/view workflows.
- Keep current APIs usable while introducing explicit lifetime-safe view patterns.

Deliverables:

- Design note + tests covering view lifetime guarantees.
- Clear guidance for async-safe tensor/view handoff patterns.

#### Phase B — Finalize async assignment behavior for tensor types

Goals:

- Decide and document which tensor-related types are `rebind` vs `write_through`.
- Ensure `co_await writer = rhs` behavior is unambiguous for tensor views.

Deliverables:

- Trait specializations for the chosen tensor proxy/view types.
- Tests mirroring real tensor/view write-through and explicit `rebind(...)` paths.
- Documentation update in `docs/async/`.

#### Phase C — Introduce expression-layer roadmap implementation

Goals:

- Add a minimal expression node layer where it brings clear value (fusion/reduced temporaries).
- Keep backend lowering aligned with existing `level1`, `kernel`, and `linalg` code.

Deliverables:

- Initial expression API proposal and one implemented vertical slice.
- Benchmarks demonstrating no regression and targeted improvements.

#### Phase D — Async-aware tensor expression execution

Goals:

- Bridge expression evaluation with async scheduling in a controlled way.
- Preserve explicit sequencing guarantees from the existing buffer model.

Deliverables:

- Prototype async evaluation path for selected expression nodes.
- Error/cancellation behavior documented and covered by tests.

### Track 2 — Heterogeneous execution

#### Phase H1 — Device-aware foundations (start early)

Goals:

- Add the runtime **location** axis to storage alongside the compile-time **kind**
  axis, without disturbing existing host code (`docs/storage_kind_and_location.md`).
- Define a polymorphic **completion-token** abstraction and make epoch publish/wait
  operate over it, with the trivial CPU implementation first
  (`docs/ordering_and_backend_lowering.md`).
- Establish the multi-buffer operation-acquisition transaction
  (`docs/gpu_epoch_design_draft.md`).

Deliverables:

- Storage location representation + tests; symmetry/block-key preservation under
  (re)placement.
- Completion-token interface with the report-done-after-submission contract and
  buffer-release-on-completion semantics documented and tested.

#### Phase H2 — Incremental device backends

Goals:

- Improve CUDA/cuSOLVER path maturity via the compile-time capability + runtime
  `try_*` + fallback pattern (`docs/backend_dispatch.md`).
- Bring up each kernel synchronous-blocking first (validated against the oracle),
  then add event-based async lowering edge by edge.
- Keep public operations generic while backend adapters decide lowering.

Deliverables:

- Backend capability matrix in docs.
- Backend dispatch traits and `try_*` adapters for selected vertical slices.
- Backend-specific correctness tests (differential against the synchronous baseline)
  and representative performance benchmarks.

#### Phase H3 — Distributed execution

Goals:

- Lower cross-rank transfers to nonblocking MPI with a unique tag per edge and an
  active progress engine (`docs/ordering_and_backend_lowering.md`).
- Introduce the placement/cost policy as a strategy layer *above* the ordering layer
  (informed by the RABC scheduling prototype).

Deliverables:

- MPI/NCCL transfer lowering with deterministic per-edge tag derivation and progress
  integration into the schedulers.
- Placement-policy interface emitting transfers into the async DAG; correctness held
  by the ordering layer regardless of placement.

## 5. Guardrails

- Maintain C++23 and existing coroutine safety rules (captureless `static` coroutine lambdas).
- Keep async determinism and epoch ordering invariants intact.
- Prefer incremental, test-backed refactors over broad rewrites.
- Update docs in the same change whenever semantics shift.

## 6. Related Docs

- `docs/architecture_diagram.md`
- `docs/backend_dispatch.md`
- `docs/ordering_and_backend_lowering.md`
- `docs/storage_kind_and_location.md`
- `docs/gpu_epoch_design_draft.md`
- `docs/async/README.md`
- `docs/async/reverse_mode_ad.md`
- `docs/async/buffers_and_awaiters.md`
- `docs/testing.md`
