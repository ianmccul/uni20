# Execution Architecture: Mechanism, Policy, and Build Order

This is a draft design note. It is the connective overview for how the uni20
execution stack fits together: the data model, kernel dispatch, and device
scheduling as **mechanism**, with placement and coalescing as **policy** layered
on top. It records the decisions reached while integrating the TensorContraction
prototype and the open questions that remain. It is design direction, not a
description of current implemented behavior.

Related notes:

- `docs/block_sparse_tensor.md` — the tensor + layout (the linchpin data model).
- `docs/block_tensor.md` — the symmetry-typed `BlockTensor` refinement of the data model.
- `docs/block_coalescing.md` — single-axis GEMM grouping.
- `docs/backend_dispatch.md` — the `maybe_can_*` / `try_*` / generic dispatch pattern.
- `docs/kernel_dispatch.md` — the ordered backend-list generalization and scheduler integration.
- `docs/ordering_and_backend_lowering.md` — ordering ownership; two-clocks lifetime.
- `docs/storage_kind_and_location.md` — memory kind vs location.
- `docs/cuda_runtime_design_notes.md` — stream ownership; idle-stream `co_await` pool.
- `docs/cuda_cusolver_architecture.md` — solver-lane resource model.
- `docs/gpu_epoch_design_draft.md` — GPU per-buffer hazard model.
- `docs/rabc_contraction_scheduling.md` — the R/A/B/C apply and its cost model.

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
2. **Tensor + layout (the gap).** See `block_sparse_tensor.md`, refined into the
   symmetry-typed `BlockTensor` in `block_tensor.md`. Design the layout
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

## Scheduling: two scheduler shapes

uni20 allows multiple schedulers; device work falls into two shapes that hold
resources differently:

- **Synchronous-blocking (e.g. cuSOLVER).** The worker thread blocks for the op's
  duration, so thread-occupancy == op-duration and the resource (a per-thread
  cuSOLVER handle/context) is held for that time. Pool size = max concurrent
  blocking ops. See `cuda_cusolver_architecture.md`.
- **Async-submission (e.g. CUDA kernels).** The worker only submits (briefly), then
  the work runs on the GPU without it. The **stream is the leased resource, held
  until completion**, decoupled from the thread, which is freed after submission.
  `co_await acquire_idle_stream()` (see `cuda_runtime_design_notes.md`) gates stream
  acquisition.

Both share one suspend/resume mechanism: a coroutine stores a pointer to its
scheduler, so a suspended coroutine is resubmitted to its home scheduler when
ready. Work originates on a CPU scheduler, runs on a cuSOLVER/CUDA/MPI scheduler,
and resumes on its home scheduler — not on the service thread. cuSOLVER lanes, the
CUDA completion monitor, and the MPI progress engine are all instances of "suspend
here, resume on home".

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

## Synchronization modes (per kernel)

Two modes, chosen by the kernel based on its typical consumer; the buffer-epoch
completion token is the always-on correctness substrate:

- **Deferred (default substrate).** The launch returns immediately; the completion
  token is attached to the output buffer's epoch; synchronization happens lazily at
  the next access through `ReadBuffer`/`WriteBuffer`. This rides the *same*
  epoch-on-buffer mechanism as CPU ordering — the GPU event simply becomes the
  buffer epoch's completion token, with no parallel sync system. It is ideal for
  GPU→GPU dataflow and for the tail (many tiny kernels on one stream, FIFO-ordered
  with zero events, one terminal sync). It makes the token-pins-storage rule
  mandatory: no coroutine frame is left holding the inputs alive.
- **Explicit await (`co_await` a nested `CudaTask`).** For GPU→host edges and
  control-flow that genuinely needs op-level sequencing (a norm, Lanczos
  coefficients), suspend the coroutine on the token.

Keep events lazy: rely on same-stream FIFO and record a cross-stream event only
when a cross-stream/host consumer actually materializes (the producer is already
dispatched by lowering time, so the record-before-wait order holds). Deferred sync
delocalizes CUDA errors to the next access; the synchronous `DebugScheduler` path
is the antidote and error-localizer.

## What the TensorContraction exercise informed

The prototype was an *informing* exercise, and it has informed the design (durable
findings recorded in project memory; constants reusable as planner inputs):

- **The cost model's flop/launch/byte crossover is one knob behind three
  decisions** — placement (which device), coalescing (merge or not,
  `block_coalescing.md`), and lowering granularity (stream chain vs CPU batch vs
  graph). Calibrated GV100 constants (untraced): ~57 TFLOP/s effective, ~5 µs
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
- Two scheduler shapes (sync-blocking, async-submission) over one
  suspend/resume-on-home mechanism; awaitable resources with deterministic-capable
  acquisition.
- Deferred buffer-access sync is the default substrate (GPU token = buffer epoch
  token); explicit `co_await` for host-consuming edges; events recorded lazily.

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
  per-device CUDA submission, per-device cuSOLVER, MPI progress) and how they are
  configured per device class.
- The piece-specific open questions in `block_sparse_tensor.md` and
  `block_coalescing.md`.
