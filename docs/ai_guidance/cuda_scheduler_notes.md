# CUDA Scheduler Guidance

- **Audience:** design assistants, coding agents, and reviewers
- **Authority:** non-normative summary
- **Status:** partly implemented runtime foundation; scheduler design remains
  incomplete
- **Canonical sources:** `docs/backends/cuda/runtime.md`,
  `docs/backends/cuda/kernel_dispatch.md`,
  `docs/backends/cuda/epoch_design_draft.md`,
  `docs/backends/cuda/cusolver.md`,
  `docs/async/scheduler_migration.md`, and current CUDA source

This file summarizes the constraints agents must preserve while extending the
CUDA runtime. Do not revive older stream-affinity, virtual-stream, or lazy-event
proposals without new profiling evidence and a maintainer decision.

## Implemented Foundation

The current CUDA runtime layer provides:

- structured `CudaRuntimeError` diagnostics through the presentation layer;
- temporary device selection with restoration;
- move-only CUDA stream and non-timing event ownership;
- shared operation-completion tokens;
- a device-local stream pool with `idle`, `leased`, and `pending` states;
- stream return through `cudaLaunchHostFunc` after actual device completion.

It does not yet provide CUDA Tensor storage, Tensor kernels, queued coroutine
acquisition, `CudaTask` lowering, or a complete CUDA scheduler.

## Required Stream-Pool Contract

A stream is available only when its prior queued work has completed. The absence
of a current host submitter is not sufficient.

The required operation shape is:

```text
acquire an actually idle stream
install dependency event waits
enqueue CUDA work
record one operation completion event
enqueue the stream-return host function
consume the lease
mark the slot idle only when the host function runs
```

The pool is both a scheduling resource and admission control. Exhaustion should
eventually suspend the requesting coroutine, limiting queued device work, live
intermediates, workspaces, and allocator pressure.

The CUDA host function must remain lightweight and must not call CUDA APIs. It
may update pool state and arrange scheduler notification. It must not directly
run arbitrary coroutine continuation work.

## Event Policy

Use one event-based model uniformly:

- every submitted operation records one non-timing completion event;
- multi-output operations share one completion token;
- readers and writers publish that token into their buffer epoch state;
- later operations acquire any idle stream and install
  `cudaStreamWaitEvent` dependencies;
- correctness never depends on producer/consumer stream affinity.

Do not cache streams on buffers, park implicit stream tails, or prefer streams
based on dependency affinity. While a producer is running its stream is not
idle; once the stream is idle, the producer is already complete.

For tiny-operation overhead, prefer batching, coalescing, provider batched APIs,
or CUDA graphs.

## Device Context

The eventual `CudaDeviceContext` should own device-local resources:

- one logical scheduler arena for the device;
- the idle stream pool;
- event allocation/recycling;
- CUDA memory-pool configuration;
- exclusive provider handle and workspace pools;
- completion/error monitoring;
- runtime diagnostics and counters.

A oneTBB arena owns concurrency slots, not fixed workers. Attach a
`task_scheduler_observer` to establish the arena's CUDA device whenever a
worker or application thread enters and restore its previous device on exit.
Do not model provider handles as permanently worker-owned thread state.

Provider handles with mutable stream state must never be shared concurrently.
Acquire streams, handles, and workspaces as one composite device-local request,
then run the complete backend walk and provider call without another
`co_await`. Raw handles remain internal to their RAII leases and backend
adapters. Provider-resource reuse at host return versus device completion is an
explicit provider/routine policy.

Host-intensive APIs initially run on the same per-device scheduler. Add a
separate bounded provider lane only if profiling demonstrates starvation or
latency problems.

## Buffer Epochs

GPU synchronization belongs to the memory allocation or storage epoch, not to
an incidental operation object:

- reads wait on the latest writer completion;
- writes wait on the latest writer and all outstanding readers;
- reads publish reader completions;
- writes publish a new writer completion and supersede prior readers;
- buffer storage remains alive until every retained completion token is no
  longer needed.

The CPU async DAG establishes causal readiness before CUDA submission. CUDA
events lower already-submitted memory dependencies; they must not wait for
producer work whose event has not yet been recorded.

## Coroutine Integration

The coroutine promise's scheduler pointer is execution-routing state, but a
coroutine's promise type is fixed at frame creation. Global `schedule()` may
route `AsyncTask` and `CudaTask` differently. If CUDA execution requires state
in `CudaTaskPromise`, an ordinary `AsyncTask` enters CUDA by creating and
awaiting a `CudaTask`; it cannot migrate and become one. The parent resumes
through the scheduler recorded in its own promise.

Same-task-type scheduler migration is a separate capability. For example, a
`CudaTask` may move between device schedulers only if its promise device state
and scheduler route are updated consistently. Do not conflate heterogeneous
nested-task routing with live-task migration.

Future resource acquisition must be:

- cancellation-safe;
- fair enough to avoid starvation;
- able to admit composite stream/handle/workspace requests without hold-and-wait;
- able to resubmit through the scheduler currently recorded by the task;
- safe when callbacks occur on CUDA-owned threads;
- compatible with deterministic single-stream debug execution.

Do not resume a coroutine directly from `cudaLaunchHostFunc`. The callback
should notify a scheduler-neutral bridge that resubmits through the task's
recorded scheduler.

General CUDA Graph capture is unsupported initially. If added later, begin and
end capture in one non-suspending device-scheduler transaction after all
resources have been acquired. Never `co_await` while capture is active.

## Errors

Immediate CUDA API failures raise structured errors at the submission point.
Deferred execution errors must eventually fail every affected async output and
mark the device context or stream pool unusable when CUDA reports a poisoned
context.

`cudaLaunchHostFunc` is not guaranteed to run after a CUDA context error.
Therefore the final scheduler cannot rely solely on pool-return callbacks for
error recovery or waiter progress. The completion/error service must have a
terminal-context path that releases or fails pending waiters rather than
deadlocking.

## TensorContraction Evidence

The `origin/tensorcontraction-integration` branch remains useful implementation
evidence for:

- per-device contexts;
- affine stream and cuBLAS-lane leases;
- non-timing event pooling;
- scratch-buffer reuse;
- stream-ordered allocation and free;
- memory-pool retention and allocation caching;
- runtime counters and environment configuration;
- multi-device and MPI/CUDA workload behavior.

Reuse those lessons, not the bridge's architecture or fail-fast environment
parsing. New native Uni20 code should use the current async, dispatch,
diagnostics, and presentation contracts.
