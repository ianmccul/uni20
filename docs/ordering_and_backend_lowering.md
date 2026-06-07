# Ordering Ownership and Backend Lowering

This is a draft design note. It records design direction for how the Uni20 async
runtime relates to device (CUDA) and distributed (MPI/NCCL) execution. It is not a
description of current implemented behavior.

Related notes:

- `docs/async/runtime_model.md` — CPU epoch/causality model.
- `docs/storage_kind_and_location.md` — storage memory kind (type) vs location (runtime).
- `docs/gpu_epoch_design_draft.md` — GPU per-buffer hazard model (`GpuEpochQueue`).
- `docs/cuda_runtime_design_notes.md` — CUDA stream ownership and pools.
- `docs/backend_dispatch.md` — compile-time capability / runtime `try_*` dispatch.

## Summary

The Async scheduler owns **all** ordering and correctness. CUDA streams/events and
nonblocking MPI are **derived performance lowerings** of the dependency graph, not
part of the correctness model. Concretely: you could delete every CUDA event and
stream and synchronize after every kernel — making device work blocking — and the
program would still be correct, because the Async scheduler already enforces the
dependency order. The device/distributed machinery exists only to keep the host
non-blocking and to queue as much work onto the GPU / network as possible.

This note explains why ordering *cannot* live in the CUDA layer, what follows from
treating events as a pure optimization, and how the same principle maps onto MPI.

## Why ordering cannot live in the CUDA layer

CUDA conflates graph **construction** with graph **execution**. `cudaEventRecord`
and `cudaStreamWaitEvent` are imperative, execution-order calls: the record-before-
wait rule is a constraint on the order in which you *make the API calls*. There is
no separate "build the graph, then run it" phase.

Uni20's async graph needs the opposite. Reverse-mode AD and `FutureValue` build the
computational graph in **reverse time order**: an edge "B depends on A" can be
registered before A's producing node even exists. CUDA cannot express "wait on a
producer that has not been recorded yet," so CUDA events can never be the
dependency model.

The Async scheduler is precisely the component that separates the two concerns: it
accepts dependency edges in arbitrary construction order and resolves them into a
forward execution order. Backend lowering (emitting CUDA events, posting MPI
operations) happens **only during that forward execution pass**. This is also why a
consumer can safely wait on a producer's CUDA event: by the time an edge is lowered,
the scheduler is walking forward and the producer has already been dispatched, so
the event is guaranteed to have been recorded. All the reverse-time graph building
lives in the build phase and is gone by lowering time.

> **Principle:** CUDA events and MPI requests are a backend lowering of a subset of
> DAG edges, emitted at dispatch time. They are never the dependency model.

## Ordering ownership and the synchronous baseline

Because the fully synchronous version (dispatch in DAG order, block to completion
after each kernel) is correct *by construction*, it is a free correctness oracle:

- Any divergence between the synchronous baseline and the event-optimized path is,
  by definition, a bug in the **lowering** — never in the algorithm.
- This gives the debug modes in `gpu_epoch_design_draft.md`
  (`SynchronousDeviceDebug`, `LegacyDefaultStreamDebug`) a principled status: they
  are not just conveniences, they *define* correct behavior, and the optimized path
  is validated against them by differential testing.

For MPI the baseline differs (see below): blocking-everywhere is not unconditionally
safe, so the MPI reference mode is "nonblocking post-all-then-wait with unique
tags," not "block each operation."

## Incremental, monotone backend bring-up

Since events are a *partial* optimization, the device path is correct from the first
kernel, before any scheduling sophistication exists:

- Bring up a new backend with zero events — suspend after each kernel — and get it
  correct.
- Then add async lowering edge by edge where it pays.

Adding an event never changes results, only timing. This monotonicity is the
foundation for building kernel backends incrementally: an unimplemented or naive
device kernel degrades performance, not correctness.

## Two clocks: submission order vs completion order

The CPU scheduler tracks **submission** order, not completion. When a DAG node
fires, the guarantee is that every operation it depends on has already been
*submitted* to its lower layer — not that the data is ready. This is enough,
because the lower layer (CUDA stream order + events, MPI matching) re-establishes
the actual ordering once work is submitted.

This rests on one contract:

> **Report-done-after-submission:** a device/MPI dispatch coroutine may report
> "done" to the Async scheduler only after it has recorded its completion token
> (CUDA event) / posted its operation — never before.

If any kernel defers its event record to a later host callback and reports done
early, the "event guaranteed already recorded" property silently breaks and
produces load-dependent races.

The two clocks must be decoupled in storage lifetime:

- **Scheduler runs on submission order.** A coroutine may enqueue device work and
  return immediately; the DAG then advances while the work is still pending.
- **Storage/buffer lifetime runs on completion order.** The buffers a coroutine
  read and wrote must stay alive *past its return*, until the work actually
  completes. Buffer RAII therefore releases on **completion-token completion**, not
  on coroutine return. (This matches the `reader_events` / `writer_event` retention
  in `gpu_epoch_design_draft.md`.)

A producer cannot know whether its consumer is same-layer or cross-layer, so it
always does the cheap thing — record a token at submission. The consumer chooses
its wait strategy.

## CUDA lowering rules

Lowering should spend events only where ordering is not already free:

- **Intra-stream ordering is free.** Consecutive ops on one stream are FIFO-ordered
  by CUDA; assign a dependency chain to a single stream and emit **zero** events.
- **Cross-stream / cross-device joins** are the only edges that need an event — one
  completion event per operation, not per buffer.
- **GPU→host edges** lower to a *task suspension*, not a host-thread block: the
  consumer coroutine suspends until a completion callback (e.g. `cudaLaunchHostFunc`)
  resumes it. This is the minimal residual that does not disappear in the fully
  relaxed path, because the host genuinely needs completed data.
- **Tiny kernels** where event overhead exceeds the kernel may be *faster* blocking
  / default-stream-serialized. The lowering should be allowed to choose "block this
  one" as a performance decision. (This is the same phenomenon as the Nsight
  event-tracing overhead documented in `AGENTS.md` §5: event machinery is not free.)

## MPI lowering

Once the Async scheduler owns ordering, the MPI layer no longer needs MPI's in-order
matching. That frees every DAG edge to use its own tag, which makes matching
**posting-order-independent**. MPI then collapses to "transfer buffer X for edge E,
signal completion" — structurally the same pure lowering as a CUDA event.

Two independent moves give this, and they are distinct:

- **Unique tag per edge** removes the ordering/matching coupling (posting order
  cannot mismatch a send to the wrong receive).
- **Nonblocking `Isend`/`Irecv`** removes deadlock.

Note that an acyclic DAG alone does *not* prevent a blocking-MPI deadlock: two
independent parallel edges can still circular-wait under blocking rendezvous
`Send`. It is the nonblocking move, not acyclicity, that removes it. A genuine DAG
cycle deadlocks the scheduler itself and is a graph bug the MPI layer cannot repair.

The **irreducible MPI residue** — what stays different from CUDA after this — is:

1. **Two-sided.** Both ranks must post; there is no one-sided "record an event"
   (absent MPI RMA, which is a different model).
2. **Deterministic tag agreement.** Both ranks must derive the *same* tag for edge E
   without communicating, which requires globally-deterministic edge identity.
   Graph construction (including reverse-mode AD) must be replicated/deterministic
   across ranks so tags agree.
3. **Finite tag space.** `MPI_TAG_UB` is only guaranteed ≥ 32767, so "a unique tag
   per communication" must mean *unique among concurrently in-flight transfers*,
   recycled with headroom, or widened via the `(source, tag)` pair and/or multiple
   communicators.
4. **Active progress.** Something must pump `MPI_Test`/`Wait` to complete requests
   and resume the awaiting tasks (a progress thread or progress-on-poll).

### CUDA vs MPI as lowering targets

| Concern | CUDA | MPI |
|---|---|---|
| Ordering primitive | recorded event (one-sided) | tag matching (two-sided) |
| Made order-independent by | scheduler + stream order | scheduler + unique tag per edge |
| Deadlock-free baseline | blocking-each (unconditional) | nonblocking post-all-then-wait |
| Completion signal | event + host callback | request + progress engine |
| Separate scheduler role | thin: stream pool + callback bridge | active: progress engine |

## Open questions

- Where does the placement/cost policy (which device or rank a block lives on) sit
  relative to this layer? It is a scheduling concern *above* hazard tracking; the
  ordering layer must remain correct for any placement the policy chooses. See
  `docs/rabc_contraction_scheduling.md` for the prototype's placement strategy.
- How are completion callbacks integrated with the existing schedulers
  (`DebugScheduler`, `TbbScheduler`, `TbbNumaScheduler`) to resume suspended tasks?
- Should the lowering's "block this small kernel" decision be static (per kernel
  category) or adaptive (measured)?
