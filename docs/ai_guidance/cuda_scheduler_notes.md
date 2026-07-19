# CUDA Design Guidance

- **Audience:** design assistants, coding agents, and reviewers
- **Authority:** non-normative retrieval summary
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-20
- **Status:** active design work; low-level runtime primitives and device-bound
  debug/oneTBB schedulers exist, but Tensor storage, storage-driven scheduler
  admission, resource awaiters, and provider kernels are not implemented
- **Canonical sources:** current maintainer decisions, `docs/backends/cuda/`,
  inspected CUDA source, and focused tests

## Answer rule

- Treat CUDA Tensor execution, storage-driven scheduler admission, and
  provider-resource management as roadmap work unless current source and
  canonical docs say otherwise.
- Current low-level source is useful implementation evidence for runtime
  primitives, but not a stable public Tensor API.
- Do not infer Tensor-storage, provider-resource, or coroutine contracts from the
  shape of low-level buffer and stream primitives.
- When docs and source disagree, report the drift rather than choosing silently.

## Current state

Uni20 has a tested low-level CUDA runtime foundation:

- device discovery, capability caching, and scoped device restoration;
- reference-counted stream-pool leases with actually-idle reuse;
- immutable completion/event tokens;
- typed move-only `cuda::CudaBuffer<T>` allocations;
- scoped `ReadAccess<T>` and `WriteAccess<T>` objects, obtained through
  `read_synchronized_with(stream)` and `write_synchronized_with(stream)`, that
  install event waits and publish completions;
- `cudaMallocAsync`/`cudaFreeAsync` use when stream-ordered memory pools are
  supported, with `cudaMalloc`/`cudaFree` fallback;
- typed `CudaTask` admission through deterministic `DebugCudaScheduler` and
  parallel `TbbCudaScheduler` implementations bound to a validated device;
- oneTBB arena-entry device selection and restoration, including nested entry
  into another device arena;
- structured CUDA diagnostics through Uni20's presentation layer.

These primitives and scheduler mechanisms are low-level infrastructure rather
than a Tensor API. The `CudaBuffer<T>` completion-ledger and scoped-access
semantics are current and tested. Tensor storage, storage-driven scheduler
admission, and resource acquisition may add higher-level ownership without
weakening those contracts.

Do not claim that any of the following are settled merely because related code exists:

- final CUDA Tensor storage or mdspan accessor shape;
- blocking versus non-blocking CUDA storage policy names;
- coroutine resource acquisition;
- storage-driven scheduler selection and device-context ownership;
- provider-handle/workspace ownership;
- Tensor storage and dispatch integration;
- error-recovery behavior after deferred device failure.

## Blocking versus non-blocking channel direction

Current design direction separates CUDA submission channels from backend
selection:

- **blocking channel:** resource acquisition may wait on the calling thread.
- **non-blocking channel:** resource acquisition suspends through a Uni20
  scheduler while waiting for streams, provider handles, workspace, or other
  scarce resources.

The channel belongs in the Tensor storage policy, conceptually like
`CudaStorage<blocking_channel>` versus `CudaStorage<nonblocking_channel>`, not
in an ad-hoc backend selector. Backend lists remain ordinary storage-derived
lists such as `cuda_reference`, `cublas`, `cusolver`, and future provider
backends.

`Async<Tensor<..., CudaStorage<blocking_channel>>>` is possible C++, but it is
a dubious policy combination. Async CUDA front ends should accept only the
non-blocking CUDA storage policy for resources that may wait for capacity,
unless a future operation documents why blocking is intentional.

Per-call streams, provider handles, and workspaces remain operation-local
resource leases. They may be passed to CUDA backend attempts as internal lowered
operands or execution context, but they should not replace the backend selector.

## Constraints that remain useful during design review

The following are reasoning constraints, not claims that the current implementation
already satisfies them completely.

### Causality

- Uni20 async causality must remain explicit.
- CUDA events should lower established dependencies to device execution; they
  should not become an accidental second, inconsistent causal system.
- Correctness must not rely on scheduler timing or undocumented stream affinity.
- Conflicting accesses require an explicit ordering model.

### Ownership and lifetime

- Stream, event, buffer, provider-handle, and workspace lifetimes must be explicit.
- A host-side handle becoming unowned does not by itself prove queued device work
  has completed.
- Callback-thread behavior must not outlive or access destroyed scheduler/runtime state.
- Device buffers and views must not outlive their allocation owner.

### Coroutine safety

- Uni20 coroutine lambdas remain captureless and preferably `static`.
- Required state must enter through parameters or explicit owned task state.
- Do not resume arbitrary continuation code directly from CUDA-owned callback threads
  without a documented scheduler handoff.
- Resource acquisition must eventually address cancellation and partial-acquisition safety.

### Failure handling

- Immediate API failure and deferred device-execution failure are different cases.
- A design must explain how affected async outputs become failed or cancelled.
- It must not depend on a callback that CUDA may never invoke after a terminal context error.

## Open questions

Treat these as active design questions unless current maintainer decisions say otherwise:

- whether stream reuse should use callbacks, polling, an event service, or another mechanism;
- how `TbbCudaScheduler` instances are owned by device contexts and selected
  from Tensor storage placement;
- how stream, provider handle, workspace, and allocator resources are acquired together;
- whether future CUDA tasks need promise state beyond the implemented typed
  `CudaTask` wrapper and shared basic promise;
- how buffer hazards map onto the existing async epoch model;
- how multi-device execution and device placement are represented;
- how CUDA Tensor storage participates in backend selection;
- how deferred errors poison or retire device-local resources;
- which primitives should be public, internal, or replaced entirely.

## Push-back triggers

- Presenting current CUDA primitives as stable API.
- Treating implementation shape as maintainer-approved architecture.
- Assuming event or stream affinity proves causality.
- Assuming host handle destruction proves device completion.
- Recommending a broad CUDA resource-scheduling design without identifying
  unresolved ownership, cancellation, and failure-routing questions.
- Claiming CUDA Tensor execution, cuBLAS/cuSOLVER integration, or distributed execution
  is complete.
- Putting blocking/non-blocking channel state or per-call resource leases into an
  ad-hoc backend selector.
