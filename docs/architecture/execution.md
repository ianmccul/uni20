# Execution Architecture: Mechanism, Policy, and Build Order

**Status:** active forward design, not a description of complete current
heterogeneous execution.

This is the connective overview for how the uni20
execution stack fits together: the data model, kernel dispatch, and device
scheduling as **mechanism**, with placement and coalescing as **policy** layered
on top. It records the decisions reached while integrating the TensorContraction
prototype and the open questions that remain. It is design direction, not a
description of current implemented behavior.

Related notes:

- `docs/symmetry/block_sparse_tensor.md` — the tensor + layout (the linchpin data model).
- `docs/symmetry/block_tensor.md` — the symmetry-typed `BlockTensor` refinement of the data model.
- `docs/symmetry/block_coalescing.md` — single-axis GEMM grouping.
- `docs/architecture/backend_dispatch.md` — the `maybe_can_*` / `try_*` / generic dispatch pattern.
- `docs/architecture/kernel_dispatch.md` — the ordered backend-list generalization and scheduler integration.
- `docs/architecture/ordering_and_backend_lowering.md` — ordering ownership; two-clocks lifetime.
- `docs/architecture/storage_kind_and_location.md` — memory kind vs location.
- `docs/backends/cuda/runtime.md` — stream ownership; idle-stream `co_await` pool.
- `docs/backends/cuda/kernel_dispatch.md` — lightweight CUDA submission versus
  host-intensive provider scheduling.
- `docs/async/scheduler_migration.md` — typed initial scheduler admission,
  shared rescheduling, and nested-task continuation semantics.
- `docs/backends/cuda/cusolver.md` — solver-provider resource model.
- `docs/backends/cuda/epoch_design_draft.md` — GPU per-buffer hazard model.
- `docs/tensor_network/rabc_contraction_scheduling.md` — the R/A/B/C apply and its cost model.

## Summary

uni20 should go straight to **CPU + CUDA + MPI**, designing the abstractions for
all three from the start while implementing backends incrementally against the
synchronous CPU oracle. The architecture separates **mechanism** (block-sparse
tensor + layout, kernel dispatch, device scheduling) from **policy** (placement,
coalescing, "tail on CPU"). All policy lives in one place: a planner whose oracle
is the R/A/B/C cost model. The mechanism layers must be policy-free, so the planner
can make any choice. Of the three mechanism pieces, only the **tensor + layout**
has no existing foundation, and the other two depend on its layout abstraction, so
it is both the biggest gap and the highest-leverage piece to build first.

## Mechanism vs policy

The guiding constraint is *uni20 should facilitate any choice we want to make*.
That requires a clean split:

- **Mechanism (policy-free):** the tensor layout can map any block to any
  device / MPI rank; dispatch can run any backend; scheduling handles any device
  mix. None of these bake in a placement assumption.
- **Policy (one layer):** the planner decides coalescing groups and the
  device / MPI-rank map, using the cost model. Decisions like "the small-block tail runs on the CPU"
  are planner output, not properties of the tensor, dispatch, or scheduler.

This is why the cost-model calibration work matters beyond the prototype: it is the
planner's oracle, and the planner is the only place policy is allowed to live.

## The three mechanism pieces and their dependency order

1. **Foundation — buffer-with-subviews + completion token.** An extension of the
   existing epoch/buffer model: storage tracked as a buffer with strided
   sub-ranges, and a device completion token carried in the buffer's epoch. This is
   the seam under both the tensor (blocks as sub-views) and scheduling (deferred
   GPU sync). It also houses the token-pins-storage lifetime rule.
2. **Tensor + layout (the gap).** See `../symmetry/block_sparse_tensor.md`, refined into the
   symmetry-typed `BlockTensor` in `../symmetry/block_tensor.md`. Design the layout
   type first (device / MPI-rank map + coalescing-aware memory plan), then the
   block-sparse container on top. This is the only piece with no existing
   foundation, and dispatch/scheduling/MPI/planner all consume its layout.
3. **Kernel dispatch.** The `backend_dispatch.md` pattern already defines the
   shape (`maybe_can_*` compile-time capability, `try_*` runtime attempt, generic
   fallback as the correctness oracle), generalized to an ordered backend list in
   `kernel_dispatch.md`. What this stack adds: dispatch reads the
   layout for device selection, emits ops into the appropriate scheduler with a
   completion token, picks a synchronization mode (below), and treats
   batched/coalesced kernels as backend capabilities.
4. **CUDA / MPI scheduling.** An extension of the existing scheduler set and the
   CUDA runtime notes, with the scheduler shapes and sync modes below.

Build order is monotone and oracle-validated (per `ordering_and_backend_lowering.md`):
foundation → tensor+layout → CPU dispatch validated against the synchronous oracle
→ GPU/MPI scheduling and backends, edge by edge. The CPU path is a **required
backend**, not a later addition; the core async-DAG apply on CPU kernels is the
first end-to-end apply that goes through the real runtime (the TensorContraction
bridge bypasses it), and step 4 is what eventually retires the bridge.

## Scheduling: lightweight submission and provider execution

Uni20 allows multiple schedulers. The CUDA design uses one logical scheduler
arena per device. Distinct initial-admission interfaces prevent scheduler-family
mixups: `IAsyncScheduler` accepts `AsyncTask`, while `ICudaScheduler` accepts
`CudaTask`. Both concrete types share `BasicAsyncTaskPromise` and become the
same internal `BasicTask` after admission, so buffer wakeups and nested
continuations use one rescheduling contract.

An ordinary coroutine enters the CUDA task domain by `co_await`ing a newly
created `CudaTask` already bound to the appropriate device scheduler. The
parent remains an `AsyncTask` and resumes through its own scheduler when the
CUDA child completes. The concrete task type controls only initial admission;
the selected scheduler is runtime state in the shared promise. Explicit
live-task migration remains a separate future capability.

A oneTBB arena limits simultaneous participation but does not own fixed worker
threads. A device-arena observer establishes and restores the CUDA device as
workers or application threads enter and leave. Device resources therefore
belong to `cuda::DeviceContext` pools, not permanently to workers.

CUDA host calls then fall into two execution classes:

- **Lightweight submission.** An ordinary kernel launch, asynchronous copy, or
  similar call briefly occupies its caller while enqueuing device work. The
  stream remains leased until completion, but the host thread is released after
  submission.
- **Hybrid provider execution.** A cuSOLVER, cuTensorNet, or similar API may
  keep its caller active while it performs substantial CPU orchestration and
  launches many kernels. Initially these calls occupy one participant in the
  same device scheduler and use exclusively leased device-local handles.

The provider job itself is an ordinary non-coroutine function. The coroutine
may suspend while waiting for a composite stream/handle/workspace request, but
does not suspend during the backend walk or provider call. A separate provider
execution lane remains an optional profiling-driven refinement. See
`../backends/cuda/kernel_dispatch.md`.

CUDA completion monitoring and the MPI progress engine follow the same broad
rule: notification may originate on a service or callback thread, but the
continuation is submitted through the scheduler currently recorded by its
promise rather than executed on the notifying thread.

### Awaitable resources do double duty

`co_await` for a free stream is both **load-balancing** (never pile work onto a
stream you cannot observe) and **admission control / backpressure** (when all
streams are leased, suspending bounds in-flight work, hence live intermediates,
hence GPU memory). At large `m` the stream-pool size is a memory-safety knob, not
only a performance one.

### Acquisition policy must be deterministic-capable

Resource acquisition (which stream/lane you get) introduces nondeterminism, which
collides with the property that the same code runs deterministically under
`DebugScheduler` (the synchronous correctness oracle). The acquisition policy must
therefore be a scheduler concern, not baked into call sites: under `DebugScheduler`
it resolves deterministically (lowest-index free stream, single lane, synchronous
completion) so differential testing against the relaxed path stays valid.

## Completion consumption modes

The buffer-epoch completion token is the always-on correctness substrate.
Consumers use it in two ways:

- **Deferred (default substrate).** The launch returns immediately after
  recording one completion event. The token is attached to the output buffer's
  epoch; later GPU consumers install event waits and host consumers await
  completion. This rides the *same* epoch-on-buffer mechanism as CPU ordering,
  with no parallel dependency system. It makes the token-pins-storage rule
  mandatory: no coroutine frame is left holding the inputs alive.
- **Explicit completion await.** For GPU→host edges and control-flow that
  genuinely needs op-level sequencing (a norm, Lanczos coefficients), suspend
  the coroutine on the token.

Record one event for every submitted operation. This keeps dependency lowering
uniform and avoids stream-affinity state; batching and coalescing amortize event
cost for tiny operations. Deferred sync can delocalize CUDA errors to a later
completion boundary, so the synchronous `DebugScheduler` path remains the
error-localizing oracle.

## What the TensorContraction exercise informed

The prototype was an *informing* exercise, and it has informed the design (durable
findings recorded in project memory; constants reusable as planner inputs):

- **Cost-model crossover.** The flop/launch/byte crossover is one knob behind
  three decisions: placement (which device), coalescing (merge or not; see
  [Block Coalescing](../symmetry/block_coalescing.md)), and lowering granularity
  (stream chain vs CPU batch vs graph). Calibrated GV100 constants (untraced):
  ~57 TFLOP/s effective, ~5 µs
  kernel launch, ~166 µs segment transition, ~28 GB/s GPU↔GPU peer; CPU↔GPU is
  PCIe (~12 GB/s, asymmetric) as a separate device-class link constant.
- **Model sophistication is not the lever.** A 4-parameter input-anchored proxy
  ranks placements about as well as elaborate typed-hypergraph models
  (LOO Kendall-τ ≈ 0.71–0.79, ~3 % top-1 regret); more features overfit at small
  sample sizes. The win came from *diverse calibration data*, not model complexity.
- **The tail is the real performance structure.** Roughly half the kernel launches
  carry ~0 % of flops and ~1 % of bytes, and that fraction does not shrink with
  `m`. The remedies — coalescing at the source, CPU offload, and graph/batched
  submission — all attack per-op overhead, and the CPU is a required device anyway,
  so tail placement falls out of the heterogeneous cost model with no bespoke code.
- **Tracing has catastrophic (~5×) overhead**, so calibration must time untraced
  and trace once for connectivity (recorded in memory and `AGENTS.md` §5 / the
  Nsight `--cuda-event-trace=false` rule).

## Decisions made

- Go straight to CPU + CUDA + MPI: design seams for all three, implement
  monotonically against the synchronous oracle.
- Mechanism (tensor+layout, dispatch, scheduling) is policy-free; placement and
  coalescing are the planner's policy, with the cost model as oracle.
- Build order: buffer-with-subviews + token → tensor+layout → CPU dispatch →
  GPU/MPI; the CPU async-DAG apply is the first real apply and the route to
  retiring the bridge.
- Lightweight CUDA submission and bounded hybrid-provider execution over one
  per-device scheduler, heterogeneous nested-task routing, and optional
  same-task-type migration; awaitable device-local resources with
  deterministic-capable acquisition.
- Deferred buffer-access sync is the default substrate (GPU token = buffer epoch
  token); explicit `co_await` for host-consuming edges; one completion event
  recorded for every submitted operation.

## Open questions

- **Planner ownership.** How much does the generic dependency scheduler discover
  vs a thin apply-specific planner that decides coalescing + placement and emits
  block-granularity nodes into a general runtime that owns ordering, lowering, and
  replay-caching? (Current lean: thin domain planner, general runtime.)
- **Replay caching.** Whether to cache a lowered schedule per DMRG site and reuse
  it across sweeps (the portable analog of a CUDA graph); gated by a cheap
  graph-instantiation-cost microbench, and viable only with cross-sweep reuse
  because typical Lanczos replay counts are ~4 and the environment pass runs once.
- **Where the tail actually runs.** CPU offload vs same-device coalescing vs
  batched submission — a planner policy decision the mechanism must facilitate
  equally.
- **Scheduler taxonomy per node.** The concrete set of schedulers (CPU compute,
  one execution scheduler per CUDA device, optional profiling-driven provider
  lanes, MPI progress) and how they are configured per device class.
- The piece-specific open questions in `../symmetry/block_sparse_tensor.md` and
  `../symmetry/block_coalescing.md`.
