# Scheduler Routing and Task Domains

**Status:** concrete host and CUDA initial-admission types, shared suspended-task
routing, cross-scheduler nested continuations, and deterministic plus oneTBB
device-bound CUDA schedulers are implemented. Explicit live-task migration
remains future work.

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

## Concrete Admission, Shared Execution

Uni20 deliberately uses two concrete task types over one coroutine promise and
one internal task representation:

```text
AsyncTask ----\
               +--> BasicTask = BasicAsyncTask<BasicAsyncTaskPromise>
CudaTask -----/
```

The concrete type exists only at initial scheduler admission:

- `IAsyncScheduler::schedule(AsyncTask&&)` admits ordinary host tasks;
- `ICudaScheduler::schedule(CudaTask&&)` admits CUDA tasks;
- neither scheduler interface can accidentally admit the other concrete type.

After admission, a task is stored and routed as `BasicTask`. Both concrete task
types consequently use the same tested implementation for:

- coroutine ownership and cancellation;
- buffer and epoch awaiters;
- exception propagation;
- scheduler recording and rescheduling;
- nested-task continuations and cross-scheduler return.

`BasicAsyncTaskPromise::get_return_object()` cannot know the coroutine
function's declared return type. It therefore returns an internal move-only
`BasicTaskReturnObject`. That proxy has private conversions to `AsyncTask` and
`CudaTask`; private construction keys prevent unrelated code from converting
or relabelling task types. An unconverted proxy releases its coroutine ownership
normally.

This arrangement keeps initial admission type-safe without introducing a
second promise hierarchy or a type-erased activation object.

## Internal Rescheduling

`IScheduler` is the internal route for a task that has already been admitted.
Its private `reschedule(BasicTask&&)` operation is callable only by task
machinery. Public scheduler interfaces derive from it and add their typed
initial-admission operations.

The shared promise records an `IScheduler*`:

1. Initial admission records the selected scheduler.
2. A buffer or epoch awaiter owns the suspended `BasicTask`.
3. When the task becomes viable, `BasicAsyncTask::reschedule()` submits it
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

When a coroutine awaits another Uni20 task, the child retains its concrete type
only long enough to establish the initial route. Entry follows this rule:

| Child route | Action |
|---|---|
| unset | inherit the parent route and transfer directly |
| same as parent | transfer directly |
| different from parent | enqueue the child on its recorded scheduler and suspend the parent |

When the child finishes:

| Parent continuation route | Action |
|---|---|
| same as the child execution route | transfer directly to the continuation |
| different | resubmit the continuation through its recorded scheduler |

The common promise makes this work for both `AsyncTask` and `CudaTask`. In
particular, an `AsyncTask` parent may await a `CudaTask` child already bound to a
device scheduler. The child runs in the CUDA domain and the parent later resumes
on its original host scheduler.

The parent does not become a `CudaTask`, and the child does not carry the parent
onto the CUDA scheduler after completion.

## Current CUDA Schedulers

`DebugCudaScheduler` is the deterministic first implementation of
`ICudaScheduler`. It:

- is constructed for one validated `cuda::Device`;
- runs queued tasks on the calling thread;
- uses `cuda::ScopedDevice` to establish its device for `run()` and `run_all()`;
- restores the calling thread's previous CUDA device before returning;
- preserves the ordinary debug-scheduler meaning of `run_all()`: run until no
  task remains runnable in that scheduler, not until externally suspended work
  becomes viable.

Tests cover direct CUDA admission, rescheduling after a buffer suspension, one
scheduler per visible device with out-of-order multi-device resumption, and an
`AsyncTask` parent returning to its CPU scheduler after a CUDA child. The
multi-device case skips when fewer than two devices are visible.

`TbbCudaScheduler` is the parallel implementation. Each instance owns one
oneTBB arena and one `task_scheduler_observer` for its validated CUDA device.
The observer saves and selects the device whenever a worker or application
thread enters the arena, then restores the previous device on exit. Nested
participation in another device arena uses a thread-local selection stack, so
returning from device 1 to an outer device-0 arena restores device 0 before the
outer coroutine continues.

The TBB scheduler implements the same `ICudaScheduler` admission and
`IScheduler` rescheduling contracts as the debug scheduler. Its `run_all()` has
the same activation-quiescence meaning as `TbbScheduler`: externally suspended
tasks may become runnable and submit a later activation.

Tests cover direct TBB admission, one scheduler per visible device,
out-of-order resumption after two suspension points, non-blocking resumption
into a saturated arena, concurrent arena-participant device selection before
and after suspension, calling-thread restoration, and nested entry into a
different device arena. Multi-device cases skip when fewer than two devices are
visible.

## Device Selection

`CudaTask` does not contain a device ordinal or device context. Device selection
belongs to the chosen CUDA scheduler and the Tensor storage/device domain.
Callers that construct CUDA work must bind the task to the scheduler matching
its operands before the task can execute.

This avoids duplicating placement state in the promise and prevents transient
resources from becoming task-global state. Streams, provider handles,
workspaces, and buffer leases remain operation-local RAII values.

A higher-level CUDA scheduling function may later select the correct
`ICudaScheduler` from an explicit device or Tensor storage. That policy should
remain above the shared coroutine machinery.

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

The implemented debug and TBB tests establish the fundamental cross-domain
route. The next checkpoints should add:

- global or context-level typed CUDA admission that selects a device scheduler;
- exception and cancellation propagation across a host/CUDA scheduler boundary;
- scheduler destruction diagnostics with outstanding suspended tasks;
- task-registry diagnostics that identify the current scheduler/device domain;
- live-task migration tests only if a concrete use case justifies that API.

## Open Choices

- How a CUDA device context exposes typed initial task admission.
- Whether scheduler identity should be recorded separately for diagnostics.
- What operation drains or shuts down a multi-scheduler device context.
- Whether any future task domain genuinely requires a specialized promise;
  adding one would require a new common internal routing design and should be
  justified by persistent task state that cannot live in the scheduler or
  coroutine locals.
