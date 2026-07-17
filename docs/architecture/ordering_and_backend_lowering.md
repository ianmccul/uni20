# Ordering Ownership and Backend Lowering

**Status:** active design note extending the implemented async ordering model
to future CUDA and MPI lowering.

Draft design note: how the async runtime relates to device (CUDA) and
distributed (MPI) execution. Design direction, not implemented behavior.

Related notes:

- `docs/async/runtime_model.md` — CPU epoch/causality model.
- `docs/architecture/storage_kind_and_location.md` — storage memory kind (type) vs location (runtime).
- `docs/backends/cuda/epoch_design_draft.md` — conservative CUDA buffer-tail
  completion lowering.
- `docs/backends/cuda/runtime.md` — CUDA stream ownership and pools.
- `docs/backends/cuda/kernel_dispatch.md` — host execution routes for
  lightweight and hybrid provider calls.
- `docs/architecture/backend_dispatch.md` — compile-time capability / runtime `try_*` dispatch.
- `docs/architecture/kernel_dispatch.md` — the `backend_list` walk and scheduler integration.

## Summary

The async scheduler owns ordering. CUDA streams/events and nonblocking MPI are
performance machinery layered below it: they exist to keep the host
non-blocking and the device/network queues full, not to provide correctness.
As a debugging option, every CUDA call could run on the default stream with no
events at all, synchronizing after each kernel, and the program would still be
correct — the async layer already dispatches tasks in dependency order.

## Why CUDA is a separate layer

The async DAG and a CUDA stream/event graph look similar, which suggests
building one out of the other. That fails on construction order:
`cudaEventRecord` / `cudaStreamWaitEvent` must be called in execution order —
you cannot wait on an event that has not been recorded yet. The uni20 graph is
built *out of order*: `FutureValue` lets a consumer be registered before its
producer exists, and reverse-mode AD constructs the graph backwards. So CUDA
events cannot be the dependency model, and CUDA is a lowering layer instead.

The async scheduler is exactly the component that separates construction from
execution: it accepts edges in any order and resolves a forward execution
order. Backend lowering — recording CUDA events, posting MPI operations —
happens only during that forward pass. By then every producer has already been
dispatched, so waiting on its event is safe.

The division of labour: the async layer guarantees **ordering**; CUDA events
provide **synchronization** (completion detection — buffer lifetime, host
reads) and cross-stream concurrency. Events are never needed for ordering.

## The synchronous baseline

Dispatching in dependency order and blocking after every kernel is correct by
construction. That gives a free correctness oracle:

- Any divergence between the synchronous baseline and the event-optimized path
  is a bug in the *lowering*, never in the algorithm. The debug modes in
  `../backends/cuda/epoch_design_draft.md` (`SynchronousDeviceDebug`,
  `LegacyDefaultStreamDebug`) define correct behavior, and the optimized path
  is validated against them by differential testing.
- A new backend can be brought up with zero events — suspend after each
  kernel — and is correct from the first kernel. Events are added edge by edge
  where they pay. Adding an event changes timing, never results.

(The MPI baseline is different — see below. Blocking-everywhere can deadlock,
so the MPI reference mode is "nonblocking post-all-then-wait with unique
tags", not "block each operation".)

## Two clocks: submission order vs completion order

When a DAG node fires, the guarantee is that its dependencies have been
*submitted* to their backends — not that the data is ready. Submission order
is enough for ordering (stream FIFO and events re-serialize the work on the
device), but two rules follow from the gap between the clocks:

- **Report done after submission.** A device/MPI dispatch coroutine may report
  "done" to the scheduler only after its completion event is recorded / its
  MPI operation is posted. If the event record is deferred (say, to a later
  host callback), a consumer can try to wait on an event that does not exist
  yet — a load-dependent race.
- **Buffer lifetime follows completion, not coroutine return.** A coroutine
  may return while its device work is still pending, so the buffers it read
  and wrote must stay alive until the work actually completes. Buffer release
  is therefore tied to completion events — "token-pins-storage". The first
  `cuda::Buffer` implementation enforces this by retaining its latest writer
  and unfinished reader completions and synchronizing them before freeing the
  allocation.

A producer cannot know whether its consumer is same-layer or cross-layer, so
it always does the cheap thing — record a completion event at submission — and
the consumer chooses its own wait strategy.

## CUDA lowering rules

Use one uniform event-based dependency model:

- **Every submitted operation records one completion event**, not one event per
  buffer. Multi-output operations share that token.
- **Every operation acquires an actually idle stream from the pool.** The
  scheduler does not preserve producer/consumer stream affinity.
- **Dependencies use `cudaStreamWaitEvent`.** If a selected stream happens to
  make a wait redundant, that may be optimized locally without making affinity
  part of the runtime contract.
- **Stream return follows actual completion.** A `cudaLaunchHostFunc` at the
  stream tail marks the slot idle and eventually wakes a queued acquirer.
- **GPU→host edges** suspend the consuming *task*, resumed by a completion
  callback (e.g. `cudaLaunchHostFunc`); they do not block a host thread. This
  is the one residual that never disappears: the host genuinely needs
  completed data.
- **Host-intensive provider calls** run as non-suspending jobs after a
  `CudaTask` is routed to its per-device scheduler and completes composite
  resource admission. One scheduler participant is occupied until the host API
  returns; device completion remains represented by the usual event token. A
  separate provider lane is optional and must be justified by profiling.
- **Tiny kernels** can cost more in event overhead than they compute. The
  preferred remedies are batching, coalescing, or CUDA graphs rather than
  stream-affinity bookkeeping.

## MPI lowering

Because the async layer owns ordering, MPI's in-order message matching is
unnecessary. Give every DAG edge its own tag and matching becomes independent
of posting order; MPI reduces to "transfer buffer X for edge E, signal
completion" — the same shape as a CUDA event. Two separate moves get there:

- **Unique tag per edge** removes the ordering/matching coupling (a send can
  never match the wrong receive).
- **Nonblocking `Isend`/`Irecv`** removes deadlock. An acyclic DAG alone does
  *not* prevent blocking-MPI deadlock — two independent edges can
  circular-wait under blocking rendezvous sends. (A genuine DAG cycle
  deadlocks the scheduler itself; that is a graph bug the MPI layer cannot
  repair.)

What stays genuinely different from CUDA:

1. **Two-sided.** Both MPI ranks must post; there is no one-sided "record an
   event" (absent MPI RMA, a different model).
2. **Deterministic tag agreement.** Both MPI ranks must derive the same tag
   for an edge without communicating, so graph construction (including
   reverse-mode AD) must be deterministic across ranks.
3. **Finite tag space.** `MPI_TAG_UB` is only guaranteed ≥ 32767, so tags need
   to be unique only among concurrently in-flight transfers — recycled with
   headroom, or widened via the `(source, tag)` pair or extra communicators.
4. **Active progress.** Something must pump `MPI_Test`/`MPI_Wait` to complete
   requests and resume the awaiting tasks (a progress thread, or
   progress-on-poll).

### CUDA vs MPI as lowering targets

| Concern | CUDA | MPI |
|---|---|---|
| Ordering primitive | recorded event (one-sided) | tag matching (two-sided) |
| Made order-independent by | scheduler + stream order | scheduler + unique tag per edge |
| Deadlock-free baseline | blocking-each (unconditional) | nonblocking post-all-then-wait |
| Completion signal | event + host callback | request + progress engine |
| Separate scheduler role | thin: stream pool + callback bridge | active: progress engine |

## Open questions

- Where does placement policy (which device or MPI rank a block lives on) sit
  relative to this layer? It is a scheduling concern above hazard tracking;
  the ordering layer must stay correct for any placement chosen.
- How are completion callbacks integrated with the existing schedulers
  (`DebugScheduler`, `TbbScheduler`, `TbbNumaScheduler`) to resume suspended
  tasks?
- Is the "block this small kernel" decision static (per kernel category) or
  adaptive (measured)?
