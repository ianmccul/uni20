# Uni20 Architecture Status: AI Guidance

- **Audience:** remote assistants, coding agents, and reviewers
- **Authority:** non-normative retrieval summary
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-20
- **Canonical sources:** `AGENTS.md`, `docs/about.md`, `docs/architecture/overview.md`,
  `docs/roadmap.md`, canonical subsystem guides, source, and tests

## Answer rule

- Separate documented intent, current implementation, and roadmap work.
- Treat operation/backend coverage as operation-specific.
- Do not infer a stable public API; Uni20 remains in active design.
- When this summary conflicts with canonical docs, source, or tests, report the drift.

## Implemented vertical slices

### Dense Tensor and backend dispatch

- Owning dense `Tensor` types use compile-time rank and runtime extents by default.
- Column-major, row-major, and strided owner forms exist.
- Generated tensors, lazy conjugation, reshape, explicit materialization,
  elementwise overwrite/update, reductions, and a substantial dense linalg surface exist.
- Tensor operations lower through storage-derived backend selectors to resolved
  mdspans and operation-value dispatch.
- CPU reference, BLAS, and initial LAPACK paths are active; backend and scalar
  coverage varies by operation.
- Do not describe a bare mdspan as carrying enough policy for default top-level dispatch.

### Async runtime

- `Async`, `EpochQueue`, `ReadBuffer`, and `WriteBuffer` are implemented core types.
- `DebugScheduler`, `TbbScheduler`, `TbbNumaScheduler`, `DebugCudaScheduler`,
  and `TbbCudaScheduler` are implemented. The CUDA-capable schedulers admit host
  work and route CUDA tasks to an explicit bound device; Tensor-storage-driven
  initial admission is not yet wired.
- Exception/cancellation propagation, scheduler-aware waits, task diagnostics,
  and DAG snapshots exist.
- Async correctness comes from epoch ordering, not scheduler timing.

### Async Tensor operations

- Async wrappers reuse synchronous Tensor front ends and backend dispatch.
- Implemented support includes owner-retaining conjugating/reshape aliases,
  elementwise overwrite/update, matrix products with immediate or async scalar
  parameters, and preserving/consuming `eigh` and exact/truncating SVD wrappers.
- Multi-output operations return independent async outputs.
- Do not generalize operation-specific alias checks into general overlap safety.

### Reverse-mode AD

- Async value-level `Var<T>` and `ReverseValue<T>` foundations are implemented.
- Reverse mode is dataflow-based, not tape-replay-based.
- Tensor-linalg differentiation is not yet integrated through the Tensor operation layer.
- Retained named intermediates may require explicit gradient finalization under the
  current API; consult `reverse_mode_ad.md`.

### Krylov

- Matrix-free symmetric/Hermitian Lanczos, nonsymmetric Arnoldi, generalized
  problems, Krylov exponential action, and an independent Taylor reference exist.
- Projected dense work uses Uni20 Tensor/linalg dispatch.

### Presentation and diagnostics

- Semantic reports, terminal/plain/ASCII rendering, width-aware tables,
  structured diagnostics, exhaustive mdspan formatting, and bounded mdspan
  previews are implemented.
- Trace uses bounded previews for tensor/mdspan-like values.

## Partial foundations and roadmap

### Symmetry and block sparsity

- Quantum-number, U(1), block-space, local-space, and selection-rule foundations exist.
- A complete symmetry-aware `BlockTensor` and lowering pipeline remain design work.
- Symmetry metadata is part of correctness. Never introduce an implicit dense fallback.

### Python

- Current bindings are a nanobind smoke module exposing `greet()` and build metadata.
- Tensor operations, NumPy/DLPack interop, native async values, packaging, and
  notebook display are future work.

### CUDA and distributed execution

- CUDA has a tested low-level runtime foundation: device discovery, scoped
  device selection, stream-pool leases, completion tokens, typed
  `CudaBuffer<T>`, scoped `ReadAccess<T>`/`WriteAccess<T>`, stream-ordered
  allocation when available, and structured diagnostics.
- These primitives are low-level infrastructure, not a stable public Tensor
  API. Their buffer-access and completion-ledger semantics are current and
  tested.
- `CudaTask`, `DebugCudaScheduler`, and `TbbCudaScheduler` are implemented.
  CUDA Tensor storage, Tensor kernels, storage-driven scheduler admission,
  non-blocking resource awaiters, provider-resource management, and distributed
  execution remain open work.
- Blocking versus non-blocking CUDA submission is current design direction and
  belongs in the Tensor storage policy, not in backend selector state.
- Preserve explicit causality and lifetime reasoning, but do not present the current
  Tensor lowering model as implemented.

### Expression/fusion layer

- There is no general Tensor expression/fusion subsystem.
- Prefer explicit operations until a measured sequence justifies fusion.

## Build caution

- Uni20 uses modern CMake, system-package discovery, and `FetchContent`.
- Do not infer dependency targets, transitive linkage, or provider wiring without
  inspecting the relevant CMake files.

## Push-back triggers

- A proposal relies on scheduler timing rather than epoch causality.
- A proposal assumes `Async` proves arbitrary alias safety.
- A proposal bypasses mdspan accessor semantics based only on pointer-shaped handles.
- A proposal claims complete CUDA Tensor, Python Tensor, distributed, or `BlockTensor`
  support.
- A proposal silently erases symmetry metadata.
