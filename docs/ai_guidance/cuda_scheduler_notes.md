# CUDA Design Guidance

- **Audience:** design assistants, coding agents, and reviewers
- **Authority:** non-normative retrieval summary
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-18
- **Status:** active design work; even the low-level primitives are still evolving
- **Canonical sources:** current maintainer decisions, `docs/backends/cuda/`,
  inspected CUDA source, and focused tests

## Answer rule

- Treat all CUDA architecture as provisional unless a maintainer-approved document
  explicitly marks a contract as settled.
- Current source shows implementation experiments and constraints, not necessarily
  final public semantics.
- Do not infer scheduler, Tensor-storage, provider-resource, or coroutine contracts
  from the shape of the present primitives.
- When docs and source disagree, report the drift rather than choosing silently.

## Current state

Uni20 has in-progress low-level CUDA work around topics such as:

- device selection and restoration;
- stream ownership and reuse;
- completion/event representation;
- typed device-buffer ownership;
- scoped read/write access experiments;
- structured CUDA diagnostics.

These are **not a stable foundation**. Their ownership model, API shape, lifecycle,
and interaction with the async runtime may still change.

Do not claim that any of the following are settled merely because related code exists:

- the final stream-pool state machine;
- the final event/completion-token model;
- the final buffer hazard representation;
- the final `DeviceContext` responsibilities;
- coroutine resource acquisition;
- CUDA scheduler structure;
- provider-handle/workspace ownership;
- Tensor storage and dispatch integration;
- error-recovery behavior after deferred device failure.

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
- how device-local scheduling integrates with oneTBB or another host scheduler;
- how stream, provider handle, workspace, and allocator resources are acquired together;
- whether CUDA task types differ from ordinary async task types;
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
- Recommending a broad CUDA scheduler design without identifying unresolved ownership,
  cancellation, and failure-routing questions.
- Claiming CUDA Tensor execution, cuBLAS/cuSOLVER integration, or distributed execution
  is complete.
