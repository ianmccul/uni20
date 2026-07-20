# Scheduler Routing and Task Domains

**Status:** concrete host and CUDA promise types, promise-neutral suspended-task
routing, same-domain scheduler inheritance, cross-domain explicit routing, and
deterministic unified host/CUDA execution are implemented. The oneTBB CUDA
scheduler remains device-bound pending the unified TBB scheduler checkpoint.

This note separates three related mechanisms: initial scheduler admission,
rescheduling after suspension, and nested continuation routing. CUDA backend and
resource policy are described in
[CUDA Kernel Dispatch and Device Scheduling](../backends/cuda/kernel_dispatch.md).

Related documents:

- [Coroutine Primer](coroutines_primer.md) introduces coroutine ownership and
  nesting.
- [Schedulers](schedulers.md) describes host scheduler behavior.
- [oneTBB Execution Primer](tbb_execution_primer.md) distinguishes an arena,
  its slots, and the physical threads that participate.

## Concrete Promises, Shared Execution

Uni20 uses distinct public task and promise types over one shared promise
implementation and one internal ownership representation:

```text
AsyncTask --> AsyncTaskPromise --\
                                  +--> TaskPromiseBase
CudaTask  --> CudaTaskPromise  --/

TaskHandle  = non-owning erased coroutine identity + TaskPromiseBase pointer
              + immutable concrete-promise domain tag
BasicTask   = move-only ownership claim for either concrete task kind
```

The concrete type controls initial scheduler admission:

- `IAsyncScheduler::schedule(AsyncTask&&)` admits ordinary host tasks;
- `ICudaScheduler::schedule(CudaTask&&, int)` admits CUDA tasks and binds an
  explicit device;
- one scheduler object may implement both interfaces without erasing the task
  domain.

Each concrete promise constructs its declared public task directly. After
admission, schedulers and awaiters store and route the task as `BasicTask`.
Both concrete task types consequently use the same tested implementation for:

- coroutine ownership and cancellation;
- buffer and epoch awaiters;
- exception propagation;
- scheduler recording and rescheduling;
- nested-task continuations and cross-scheduler return.

`TaskHandle` always obtains its paired coroutine pointer, promise pointer, and
domain tag from the original typed coroutine handle. It is non-owning and is
valid only while some runtime ownership path keeps the coroutine frame alive.
Code must not recreate a `std::coroutine_handle<TaskPromiseBase>` from a frame
address: promise inheritance does not make that conversion valid. `BasicTask`
carries the actual ownership claim wherever generic runtime code does not need
the declared task kind.

This arrangement keeps initial admission and declared task identity type-safe
without duplicating the coroutine runtime or adding virtual promise dispatch.
The erased handle retains the concrete promise's domain so continuation routing
can reject a direct cross-domain transfer. CUDA-aware code may narrow a
CUDA-tagged promise to inspect its device. Neither property overrides the
runtime scheduler binding.

## Internal Rescheduling

`IScheduler` is the internal route for a task that has already been admitted.
Its private `reschedule(BasicTask&&)` operation is callable only by task
machinery. Public scheduler interfaces derive from it and add their typed
initial-admission operations.

The shared promise records an `IScheduler*`:

1. Initial admission records the selected scheduler.
2. A buffer or epoch awaiter owns the suspended `BasicTask`.
3. When the task becomes viable, `BasicTask::reschedule()` submits it
   through the recorded `IScheduler`.
4. The scheduler resumes the same coroutine on its own execution domain.

The TBB scheduler implementations accept initial and resumed activations with
`task_group::defer()` followed by non-blocking `task_arena::enqueue()`. A
publishing or completion-service thread therefore does not enter the target
arena. Because defer registers the activation first, a `run_all()` sequenced
after admission returns cannot miss it. Failure to allocate or publish this
internal activation is fatal and reports scheduler/device context rather than
returning an unowned coroutine handle.

The scheduler route is runtime continuation state, not a numerical backend tag.
Tensor storage and backend dispatch still determine where the operands reside
and which kernel is legal.

## Nested Entry and Return

When a coroutine awaits another Uni20 task, an unbound child inherits the
parent scheduler only when both tasks have the same domain. A same-domain CUDA
child may also inherit its parent's device. Cross-domain nesting requires the
child to have been explicitly admitted or bound to a scheduler before it is
awaited. Entry then follows this rule:

| Prepared child route | Action |
|---|---|
| different scheduler | enqueue on the child scheduler |
| different task domain | enqueue through the recorded scheduler |
| same scheduler and domain, scheduler rejects direct transfer | enqueue through the recorded scheduler |
| same scheduler and domain, scheduler accepts direct transfer | transfer directly |

When the child finishes:

| Parent continuation route | Action |
|---|---|
| scheduler, domain, or scheduler-specific route differs | resubmit the continuation |
| scheduler accepts the same-domain route | transfer directly to the continuation |

Scheduler-pointer equality is therefore necessary but not sufficient for
symmetric transfer. The same predicate governs nested task entry, a task
returned by a forwarding awaiter, and final continuation return.

The common promise base and promise-neutral continuation handle make this work
for all four parent/child combinations of `AsyncTask` and `CudaTask`. In
particular, an `AsyncTask` parent may await a `CudaTask` child already bound to a
device scheduler. The child runs in the CUDA domain and the parent later resumes
on its original host scheduler.

The parent does not become a `CudaTask`, and the child does not carry the parent
onto the CUDA scheduler after completion.

## Current CUDA Schedulers

`DebugCudaScheduler` is the deterministic first unified implementation. It
derives from `DebugScheduler`, implements CUDA admission, and uses one runnable
queue for ordinary and CUDA tasks. Ordinary activations run in the host domain.
Each CUDA activation reads its bound device from `CudaTaskPromise`, establishes
that device with `cuda::ScopedDevice`, and restores the calling thread before
the next activation. One scheduler can therefore drive host work and CUDA work
for every visible device, including from `get_wait()`.

Tests cover direct CUDA admission, resumption after buffer suspension,
multi-device tasks in one queue, explicit host/CUDA nesting in both directions,
same-device direct transfer, cross-device resubmission, exception propagation,
cancellation, and calling-thread device restoration. A separate non-CUDA death
test proves that an unbound cross-domain child cannot inherit a scheduler.

`TbbCudaScheduler` is the parallel implementation. Each instance owns one
oneTBB arena and one `task_scheduler_observer` for its validated CUDA device.
The observer saves and selects the device whenever a worker or application
thread enters the arena, then restores the previous device on exit. Nested
participation in another device arena uses a thread-local selection stack, so
returning from device 1 to an outer device-0 arena restores device 0 before the
outer coroutine continues.

The TBB CUDA scheduler implements CUDA admission and `IScheduler`
rescheduling, but remains a separate device-bound scheduler in this checkpoint.
Its `run_all()` has
the same activation-quiescence meaning as `TbbScheduler`: externally suspended
tasks may become runnable and submit a later activation.

Tests cover direct TBB admission, one scheduler per visible device,
out-of-order resumption after two suspension points, non-blocking resumption
into a saturated arena, concurrent arena-participant device selection before
and after suspension, calling-thread restoration, and nested entry into a
different device arena. Multi-device cases skip when fewer than two devices are
visible.

## Device Selection

`CudaTaskPromise` contains an optional CUDA device ordinal. The device is
unbound when the coroutine frame is created and is fixed before first resume:

- typed initial admission binds the device selected by the CUDA scheduler;
- an unbound CUDA child nested under a bound CUDA parent inherits the parent's
  device;
- repeating the same pre-start binding is harmless;
- rebinding to another device, or binding after execution starts, is invalid.

The device ordinal is CUDA-specific promise state. Generic task machinery sees
only the immutable domain tag stored in `TaskHandle`; after checking that tag,
CUDA-aware code may narrow the common promise pointer to `CudaTaskPromise`.
Streams, provider handles, workspaces, and buffer leases remain operation-local
RAII values rather than task-global state.

A higher-level CUDA scheduling function may later select the correct execution
route from an explicit device or Tensor storage. That policy should remain
above the shared coroutine machinery.

## Live-Task Migration

Explicit migration of an already-running task is not implemented. If it is
needed, it should be a distinct suspension operation such as:

```cpp
co_await schedule_on(target_scheduler);
```

The awaiter would transfer the suspended `BasicTask` to another compatible
scheduler by changing the route in the shared promise and submitting exactly
one activation. It must not change the coroutine's concrete return type, move
Tensor storage, or silently change operand placement.

Before adding this capability, define and test:

- scheduler and device-context lifetime;
- cancellation and exception behavior during transfer;
- task-group activation and quiescence accounting;
- whether device-local resource leases may cross migration (the default answer
  should be no);
- diagnostics for creation scheduler versus current scheduler.

No current CUDA lowering requires live migration. A CPU parent can enter a CUDA
domain by awaiting a separately admitted `CudaTask`, which keeps the transition
explicit and returns naturally to CPU control flow.

## Lifetime and Quiescence

A task remembers which scheduler should run its next activation, but it does
not own that scheduler or keep it alive. The scheduler must remain alive until
every task routed through it has completed, or has been cancelled so that it
can never become runnable again. The same rule applies to resource waiters and
completion callbacks that may publish a later activation.

Application schedulers should normally be long-lived runtime services, often
lasting until process shutdown. A test may use a stack-local scheduler, but it
must finish or cancel every routed task before leaving the scheduler's scope
and ensure that no external callback can submit a later resumption.

`run_all()` is scheduler-local activation quiescence. It does not imply that an
externally suspended task has completed, nor does it drain every scheduler in a
multi-device process. In particular, waiting for the current oneTBB task group
does not make it safe to destroy a scheduler while a coroutine remains
suspended on an epoch, resource, or external event. Device-context shutdown
will need an explicit contract that diagnoses outstanding routed tasks and
resource waiters.

## Remaining Tests and Work

The unified debug tests establish the target task-domain route. The next
checkpoints should add:

- one TBB scheduler with host execution plus one CUDA arena per enrolled device;
- global or context-level typed CUDA admission through that unified scheduler;
- exception and cancellation propagation across a host/CUDA scheduler boundary;
- scheduler destruction diagnostics with outstanding suspended tasks;
- task-registry diagnostics that identify the current scheduler/device domain;
- live-task migration tests only if a concrete use case justifies that API.

## Open Choices

- How a CUDA device context exposes typed initial task admission.
- Whether scheduler identity should be recorded separately for diagnostics.
- What operation drains or shuts down the unified host/device runtime.
