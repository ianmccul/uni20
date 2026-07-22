# Uni20 Roadmap

**Status:** current planning baseline, updated 2026-07.

This document describes what remains after the dense Tensor, kernel-dispatch,
Krylov, and first async Tensor vertical slices came together. For a description
of implemented user-visible capabilities, start with [About Uni20](about.md).
For exact operation contracts, use the linked canonical subsystem guides.

Uni20 remains an active-design C++23 library. The roadmap therefore names
engineering outcomes and invariants rather than promising a stable API or fixed
release dates.

## Current Baseline

The following foundations are implemented and tested.

### Tensor and view semantics

- `Tensor<Element, Rank, ...>` is the concrete owner with runtime extents by
  default and explicit column-major, row-major, or strided layout policy.
- Tensor-level concepts separate readable, mutable, owning, strided, and
  rank-constrained objects from resolved mdspan operands.
- Generated `full`, `zeros`, `ones`, and generalized `eye` tensors avoid dense
  storage until an operation requires materialization.
- `conj`, `reshape_view`, explicit ordered reshape views, materializing
  `reshape`, `copy`, `make_tensor`, and in-place conjugation have documented
  ownership and accessor semantics.
- Variadic elementwise overwrite and update operations lower through
  callable-carrying dispatch values to an accessor-respecting CPU reference
  backend.
- `ScalarTensor` provides a rank-zero owner, while inner product and stable
  Euclidean norm expose separate storage-preserving and host-scalar result
  contracts. Full and axis-selective sums use the same distinction.
- All-async elementwise overwrite and update wrappers retain callable state,
  construct overwrite outputs when possible, and preserve one-writer update
  semantics.
- All-async sums support storage-preserving and host-scalar results, deferred
  output construction, and fixed-shape mutable alias outputs.
- Async conjugating and reshape aliases retain the source owner and share its
  exact epoch queue.
- Canonical contiguous Tensors transfer between pageable host storage and CUDA
  storage through explicit blocking `to_host`/`to_device` boundaries. Device
  and peer copies retain CUDA completion in the buffer ledger, with a dedicated
  non-blocking Async CUDA-to-CUDA overload.

See [Tensor Operations](tensor/operations.md),
[Generated Tensors and Reshape](tensor/creation_and_reshape.md), and
[Async Storage](async/storage.md).

### Dense operation dispatch

- Operation values identify backend-independent work such as elementwise
  transforms, copy, GEMM, GEMV, matrix exponential, eigensystems, and Schur
  operations. Values may carry immutable options or callable state.
- `kernel_accepts_types` performs compile-time tri-state capability probing.
- `try_kernel` performs runtime layout/accessor checks and returns a structured
  clean-decline reason.
- `dispatch_kernel` walks an ordered backend list and produces structured
  errors when no backend succeeds.
- Host vector storage selects LAPACK, BLAS when configured, and the CPU
  reference backend in a fixed order.
- Runtime diagnostics can report every candidate, decline reason, and selected
  backend without changing the operation interface.
- Tensor front ends own output shape and allocation policy, then pass resolved
  mdspans to leaf kernels.

Current vertical slices include accessor-respecting variadic elementwise
transforms, copy/materialization, inner product, stable Euclidean norm, GEMM,
GEMV, matrix initialization and exponential, self-adjoint and nonsymmetric
eigensystems, exact and truncating SVD, Schur operations, and tridiagonal
eigensystems. Backend and scalar coverage is operation-specific rather than
uniform.

See [Kernel Dispatch](architecture/kernel_dispatch.md),
[Mdspan Linear Algebra Dispatch](linalg/mdspan_dispatch.md), and
[Mdspan BLAS/LAPACK Wrappers](linalg/blas_lapack_wrappers.md).

### Async execution

- `Async<T>`, epoch queues, read buffers, and exclusive mutable write buffers
  define causal dataflow and storage lifetime.
- `DebugScheduler`, `TbbScheduler`, and `TbbNumaScheduler` execute the host task
  model. `DebugCudaScheduler` deterministically executes host and multi-device
  CUDA tasks from one queue. `TbbCudaScheduler` combines one host arena with one
  worker-only arena per enrolled CUDA device; device-arena observers establish
  and restore device state for every activation.
- TBB waits can make scheduler progress from application or worker threads, and
  `run_all()` waits for scheduler quiescence.
- Exceptions and cancellation propagate to output epochs; multi-output tasks
  can publish independent results or the same failure.
- Task-registry snapshots, presentation reports, optional stacktraces, signal
  triggers, watchdog controls, and Graphviz output support diagnosis.
- Async fixed-output GEMM and matrix-product overwrite/update,
  preserving/consuming self-adjoint
  `eigh`, exact and truncating SVD wrappers, plus full and axis-selective sums,
  schedule the existing synchronous Tensor operations.

See the [Async Documentation Index](async/) and
[Async Tensor Kernel Authoring](async/kernel_authoring.md).

### CUDA runtime and first Tensor provider path

- `cuda::initialize(...)` installs a scoped process-wide runtime with one
  canonical `DeviceResources` instance for every enrolled device.
- Device discovery, scoped device selection, actually-idle stream pools,
  completion tokens, typed buffers, and per-buffer completion ledgers are
  implemented and tested on multi-device systems.
- CUDA tasks can await streams and generic provider resources without blocking
  a scheduler participant. cuBLAS adds pooled handle-plus-stream execution
  leases over those primitives.
- `CudaTensor` owns opaque device storage and selects `CublasBackend`.
  Async GEMM and matrix-product overwrite/update await a device-bound `CudaTask`, which
  acquires a handle-plus-stream lease and lowers through staged CUDA mdspans,
  synchronized buffer access, and checked `S/D/C/ZGEMM` provider calls.
- Ordinary `CublasBackend` uses blocking resource admission. Async lowering
  uses generic `co_dispatch_kernel`; its optional cuBLAS `try_kernel_task`
  implementation suspends while the same execution resources are unavailable.
  Both paths share operand preparation and provider execution.

See [CUDA Runtime Foundation](backends/cuda/runtime.md),
[CUDA Buffers](backends/cuda/buffers.md), and
[CUDA Kernel Dispatch](backends/cuda/kernel_dispatch.md).

### Krylov and numerical validation

- Native matrix-free Lanczos and Arnoldi solvers cover symmetric/Hermitian,
  generalized, and nonsymmetric problems.
- Krylov and Taylor exponential-action algorithms provide independent paths for
  validation.
- Dense projected subspace work uses `DenseMatrix` and the normal linalg
  dispatch layer rather than a private dense algebra stack.
- Matrix Market fixtures, convergence/residual tests, provider comparisons,
  and optional MPLAPACK binary128 probes exercise precision-sensitive paths.

See [Krylov Algorithms](krylov/algorithms.md) and
[Krylov Precision Validation](krylov/precision_validation.md).

### Presentation and development infrastructure

- Semantic presentation documents render to terminal, plain text, strict
  ASCII, width-aware tables, and mdspan/tensor previews.
- Kernel-dispatch and async diagnostics use the presentation layer.
- CI covers Debug, Release, and Clang builds, while a separate pull-request
  workflow validates generated Doxygen documentation.
- Source-module READMEs and contributor/review guidance now provide a navigable
  development surface.

## Immediate Priorities

These are the next implementation tracks. They can proceed in parallel when
their contracts do not conflict.

### 1. Broaden dense linalg vertical slices

Use the existing GEMM/GEMV/eigh shape as the template, not a second wrapper
hierarchy.

- Audit the remaining BLAS/LAPACK wrappers for consistent operand descriptors,
  checked provider failure behavior, scalar promotion, and workspace policy.
- Give ordinary callers allocating, overwrite, update, consuming, or in-place
  Tensor APIs as appropriate to each operation.
- Keep mdspan leaf functions fixed-output and explicit about layout/accessor
  requirements.
- Add provider and CPU-reference kernels through operation-tag dispatch where a
  useful independent reference exists.
- Continue routing Krylov projected operations through these public paths so the
  solver layer exercises the same kernels as applications.
- Add async wrappers one operation at a time after output construction,
  consumption, aliasing, and exception behavior are explicit.
- Maintain real/complex, row/column layout, zero/singleton extent, and optional
  binary128 coverage in proportion to each operation's supported domain.

### 2. Complete common Tensor operations and structural views

- Build synthetic-diagonal views and trace lowering on the implemented
  one-or-more-axis CPU sum reduction.
- Extend scalar-result allocation beyond the current static storage-policy
  model when device/context-bearing storage requires it.
- Add slicing/indexing descriptors with const-correct mdspan access and explicit
  layout/striding semantics.
- Extend the owner-retaining async alias model to slices and future structural
  views without allowing descriptors to retarget accidentally.
- Add async copy/materialization and reshape operations where their overwrite or
  value semantics are unambiguous.
- Keep semantic accessor views such as conjugation read-only; treat writable
  component views such as future `real`/`imag` as true structural slices.
- Introduce subrange dependency tracking only if whole-owner epoch ordering
  becomes a demonstrated bottleneck. Correct conservative ordering is the
  baseline.

Dynamic-rank tensors remain a separate future descriptor design. They should
not add a second meaning to the current fixed-rank mdspan-based `Tensor`.

### 3. Harden async operation composition

- Exercise multi-output success, failure, cancellation, and unobserved-branch
  behavior as more linalg operations become async.
- Preserve the distinction between independent async values and aliases bound
  to an owner's lifetime and epoch queue.
- Add coroutine kernel implementations incrementally through
  `try_kernel_task`. Backends without one continue through ordinary
  `try_kernel` inside `co_dispatch_kernel`; allocation, mutation, and
  consumption remain operation-specific Tensor concerns.
- Improve deadlock and task-provenance diagnostics without adding meaningful
  cost when instrumentation is disabled.
- Integrate future CUDA/MPI external waits with watchdog state so a legitimate
  device or communication wait is not diagnosed as stalled CPU dataflow.

### 4. Implement the symmetry-aware block tensor

The symmetry layer has quantum-number and block-space foundations but no
complete block-sparse Tensor execution path.

- Implement the `BlockTensor` data model with typed legs, explicit legal block
  keys, placement metadata, and one layout/memory plan.
- Generate dense block work only after applying the applicable selection rules.
- Lower legal block operations into the existing raw dense operation and kernel
  dispatch layers.
- Preserve quantum numbers, local spaces, orientations, and logical block keys
  through worklists and backend lowering.
- Keep any dense projection explicitly diagnostic and prevent it from feeding
  back into the symmetry-aware calculation.
- Use coalescing and grouped GEMM only as optimizations over a tested blockwise
  reference path.

See [BlockTensor Design](symmetry/block_tensor.md),
[Raw Primitives and Symmetric Lowering](symmetry/raw_primitives_and_lowering.md),
and [Block Coalescing](symmetry/block_coalescing.md).

### 5. Rebuild the complete DMRG vertical slice in pure Uni20

The `tensorcontraction-integration` branch is a functional reference, not
discarded prototype work. It demonstrates dense and U(1) two-site DMRG,
matrix-free Lanczos, block-sparse environments and centers, truncating SVD,
resident CUDA execution, and MPI-aware block placement. The goal is behavioral
and performance parity through Uni20's current architecture without retaining
the external TensorContraction implementation.

- Establish a dense CPU two-site DMRG path first, using `Tensor`,
  `Async<Tensor>`, dispatched dense kernels, matrix-free Krylov, and
  `truncated_svd`.
- Rebuild MPS, MPO, environment, model, and sweep operations over explicit
  Uni20 ownership, tensor-view, and async contracts.
- Replace branch-specific block containers with the symmetry-aware
  `BlockTensor` and preserve every quantum-number and leg-orientation invariant.
- Reproduce the integration branch's U(1) Heisenberg and U(1)xU(1)
  Fermi-Hubbard numerical checks and sweep diagnostics.
- Lower the effective-Hamiltonian R/A/B/C operation through Uni20 kernel
  dispatch, placement, device-completion, and communication abstractions.
- Recover resident CUDA and MPI execution incrementally, using captured
  R/A/B/C fixtures and branch benchmark results as regression evidence.
- Treat successful parity as migration of capability, not a source-level port:
  reuse algorithms and validated conventions while replacing the bridge's
  scheduler, ownership, storage, and backend boundaries.

See [Tensor-Network Documentation](tensor_network/),
[TensorContraction Integration Findings](tensor_network/contraction_integration_findings.md),
and [R/A/B/C Contraction Scheduling](tensor_network/rabc_contraction_scheduling.md).

## Heterogeneous Execution Track

Uni20 follows a device-aware design rule: host-only assumptions must not become
part of core Tensor, storage, dispatch, or async contracts. This does not mean
that incomplete CUDA code takes priority over every host operation. It means
that host vertical slices must leave the correct extension points.

The [Deployment Environment](architecture/deployment_environment.md) records
the scheduled multi-node, x86-64/AArch64, and multi-generation GPU systems that
motivate these constraints.

### Extend device-aware foundations

- Preserve the implemented split between compile-time storage memory kind and
  runtime location such as CUDA device ordinal or MPI rank.
- Keep CUDA events and buffer completion ledgers as backend completion evidence;
  `EpochQueue` remains the causal ordering model for `Async<T>`.
- Add explicit host/device and cross-device transfer operations without hiding
  synchronization or movement inside ordinary backend fallback.
- Add storage-driven initial scheduler admission after its ownership and
  process-wide scheduler-lifetime contract is defined.
- Keep backend selector state immutable unless a concrete stream,
  communicator, precision, or algorithm use case demonstrates a need for
  stateful values.

### Incremental CUDA and cuSOLVER

- Bring up one operation at a time through the same capability and runtime
  attempt mechanism used by host kernels.
- Use the current `Async<CudaTensor>` matrix-product path as the non-blocking
  correctness baseline. Keep resource admission in the CUDA coroutine and
  provider submission in the non-suspending backend leaf.
- Extend provider coverage through cuBLAS, cuSOLVER, and reference CUDA kernels
  only after each Tensor operation's output, aliasing, and failure contracts are
  explicit.
- Do not allow ordinary backend fallback to copy device operands to host.
  Emergency transfer-based implementations must be explicit composite
  operations whose cost and synchronization are visible.

### Distributed execution

- Represent placement and communication policy above the ordering layer.
- Lower cross-rank edges to nonblocking MPI/NCCL operations with deterministic
  identity and active progress.
- Preserve block and symmetry metadata across placement and transfer.
- Optimize throughput without making correctness depend on one placement
  strategy.

See [Storage Kind and Location](architecture/storage_kind_and_location.md),
[Ordering Ownership](architecture/ordering_and_backend_lowering.md), and
[CUDA Buffer Completion Lowering](backends/cuda/epoch_design_draft.md). Future
distributed planning and commitment constraints are recorded in
[Distributed Kernel Dispatch](architecture/distributed_kernel_dispatch.md).

## Later Integration Work

### Tensor-level reverse-mode AD

Extend the existing async value-level AD machinery only after the Tensor
operation set has explicit mutation, consumption, and alias contracts. Linalg
rules must cover real and complex conventions, multi-output decompositions,
absent gradients, graph pruning, and failure propagation.

### Python and notebook interface

Bind stable C++ Tensor operations through checked dynamic dispatch boundaries.
Python should validate user errors before entering invariant-enforced C++ code,
translate recoverable diagnostic exceptions, and use the presentation model for
plain and rich notebook output. Packaging and dtype promotion policy belong to
this track.

### Expression and fusion layer

Do not build a generic expression system merely to provide operator syntax.
Introduce expression nodes only when a measured operation sequence benefits
from fusion, reduced temporaries, or a backend-specific compound kernel. The
existing explicit Tensor operations and async DAG remain the semantic baseline.

## Definition of a Complete Operation Slice

A substantial new Tensor/linalg operation is complete when the relevant items
below are addressed:

1. The mathematical, shape, alias, ownership, and failure contracts are
   documented.
2. The Tensor front end chooses allocating, overwrite, update, consuming, or
   in-place semantics deliberately.
3. Resolved mdspan operands use operation-tag dispatch and structured clean
   decline.
4. At least one deterministic implementation exists, with an independent test
   oracle where practical.
5. Provider-specific paths have scalar, layout, workspace, and provider-error
   coverage.
6. Async lowering, when exposed, owns buffers and ordinary parameters for the
   coroutine lifetime and routes failures to every output.
7. A focused example or test demonstrates the complete path rather than only a
   leaf helper.

## Guardrails

- Maintain C++23 and the captureless `static` coroutine-lambda rule.
- Treat accessor semantics, storage domain, and symmetry metadata as
  correctness constraints.
- Keep backend decline side-effect free; execution failure is not fallback.
- Make allocation, materialization, transfer, synchronization, and dense
  projection explicit.
- Prefer incremental vertical slices with independent numerical evidence over
  broad interface scaffolding.
- Update current guides and examples in the same change when behavior moves.
  Update AI guidance only when a durable repository-wide convention or
  invariant changes.
