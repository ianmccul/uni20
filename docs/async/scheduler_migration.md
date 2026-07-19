# Scheduler Routing, Nested Task Domains, and Promise Specialization

**Status:** active design exploration. Uni20 supports scheduler-aware nesting
between canonical `AsyncTask` coroutines today. Explicit live-task migration
and heterogeneous promise types are not yet complete runtime features.

This note separates generic async-runtime capability from CUDA and other
backend policies. It also records the choices that must be resolved before a
coroutine can move safely between schedulers.

Related documents:

- [Coroutine Primer](coroutines_primer.md) introduces current task ownership
  and nesting.
- [Schedulers](schedulers.md) describes current scheduler behavior.
- [oneTBB Execution Primer](tbb_execution_primer.md) distinguishes an arena,
  its slots, and the physical threads that participate.
- [CUDA Kernel Dispatch and Device Scheduling](../backends/cuda/kernel_dispatch.md)
  applies scheduler routing to one execution context per CUDA device.

## Capability Is Not Policy

The async runtime may provide all of the following without requiring every
operation to use them:

- explicit transfer of a suspended coroutine to another scheduler;
- nested `co_await` between coroutine tasks;
- scheduler inheritance when a nested task starts;
- different promise types derived from a common async-task promise base;
- promise-specific metadata or await transformations;
- scheduler-aware return from a nested task to its continuation.

Backend lowering decides whether to use those capabilities. For example, a
CUDA Tensor wrapper may create a `CudaTask` whose promise selects a device
scheduler, and an ordinary `AsyncTask` may `co_await` that task. If CUDA needs no
special promise state, a same-type scheduler transfer remains another possible
lowering. Neither encoding belongs in the generic dependency or kernel-dispatch
contract.

Three mechanisms must remain distinct:

1. **Type-directed initial scheduling:** overloads of global `schedule()` can
   route `AsyncTask`, `CudaTask`, and future task types differently.
2. **Nested task-domain transition:** one coroutine can `co_await` a newly
   created task with another promise type, retaining the outer coroutine as its
   continuation.
3. **Live-task scheduler migration:** a suspended task can change scheduler only
   within the capabilities and invariants of its existing, fixed promise type.

A coroutine cannot change its promise type after creation. If CUDA execution
requires state stored in `CudaTaskPromise`, an ordinary `AsyncTask` must create
and await a `CudaTask`; it cannot migrate and become one.

## Current Implemented Mechanics

The canonical host task is now a concrete `AsyncTask` derived from
`BasicAsyncTask<BasicAsyncTaskPromise>`. This preserves the shared task
ownership implementation while giving initial scheduler admission a distinct
task type.

Its relevant behavior is:

1. `BasicAsyncTaskPromise::sched_` records the scheduler used when a suspended
   task becomes runnable again.
2. Epoch and buffer wakeups call `BasicAsyncTask::reschedule()`, which reads
   that pointer and submits the task to the recorded scheduler.
3. `co_await` on another `AsyncTask` is explicitly supported. The inner task
   stores the outer coroutine as its continuation and inherits the outer task's
   scheduler only when it does not already have one.
4. A child on the same scheduler starts through symmetric coroutine transfer.
   A child with a different selected scheduler is submitted there.
5. At final suspend, the inner task transfers directly to a continuation on the
   same scheduler, or resubmits a continuation whose scheduler differs.

A oneTBB `task_group` tracks a scheduled resumption, called an **activation** in
this note. When that activation resumes a coroutine and the coroutine suspends,
the activation returns and its task-group work is complete. The suspended
coroutine remains owned by its awaiter or dependency object, not continuously
by the task group.

This means scheduler migration probably does not move persistent task-group
membership. It changes where the next activation is submitted. Tests must
confirm this interpretation before the migration API is implemented.

## Existing Alternate-Promise Scaffolding

`BasicAsyncTask<Promise>` and `IsAsyncTaskPromise` show the intended extension
point, and `cuda_task.hpp` sketches a derived CUDA promise. Scheduler interfaces
now separate two operations:

- `IScheduler` is the internal route used to resume an already-bound host task;
- `IAsyncScheduler` adds initial `AsyncTask` submission and host-side wait
  controls.

The path is not yet operational as a heterogeneous task system:

- the internal `IScheduler` route still accepts the canonical host task's basic
  state rather than a future CUDA task state;
- ordinary awaiters accept only canonical `AsyncTask` ownership;
- nested `BasicAsyncTask<Promise>::await_suspend()` currently assumes matching
  inner and outer promise types;
- continuation storage and final-suspend code are expressed in terms of the
  canonical promise;
- the CUDA promise sketch does not yet provide a complete CUDA-task return
  object and scheduling contract.

Treat `CudaTask` as evidence of intended capability, not as current policy or a
usable scheduler-selection mechanism.

## Type-Directed Initial Scheduling

Global scheduling can overload on the concrete task type:

```cpp
schedule(make_cpu_task(...));          // AsyncTask -> configured CPU scheduler
schedule(make_cuda_task(device, ...)); // CudaTask  -> scheduler for that device
```

The CUDA scheduler selection may use state in `CudaTaskPromise`, such as a
`cuda::DeviceContext*` or device ordinal. The scheduler then owns a type-erased
activation for queueing purposes without erasing or replacing the coroutine's
actual promise.

The same routing applies when a task is first entered through nested `co_await`
rather than global `schedule()`. A child with an explicit promise-selected route
must be enqueued there; it must not start executing inline on the parent's
incompatible scheduler.

## Same-Type Scheduler Transfer

Live-task migration, when supported by a promise type, should be expressible
independently of CUDA:

```cpp
co_await schedule_on(target_scheduler);
```

The awaiter conceptually:

1. receives ownership of the current suspended task;
2. changes the scheduler route recorded in its promise;
3. submits exactly one activation to the target scheduler;
4. returns without resuming the coroutine inline.

A `CudaTask` may use this mechanism to move between device schedulers if its
promise state can be updated consistently:

```cpp
co_await on_cuda_device(device);
```

That wrapper can update both the CUDA promise's device execution context and its
scheduler route before resubmission. It does not change the task into another
promise type, move Tensor storage, or silently change operand placement.

Whether ordinary `AsyncTask` is accepted by a CUDA scheduler is a separate
design choice. It is valid only if CUDA execution needs no `CudaTaskPromise`
state and the CUDA scheduler satisfies the `AsyncTask` promise contract.

## Nested Entry And Return Routing

Nested coroutine entry and continuation return can follow one symmetric rule.

When an outer coroutine awaits an inner task:

| Inner route | Action |
|---|---|
| unset | inherit the outer route and transfer directly |
| same as outer/current route | transfer directly |
| different from outer/current route | enqueue the inner activation on its recorded scheduler and suspend the outer task |

When the inner task finishes:

| Continuation route | Action |
|---|---|
| same as the scheduler executing the inner task | transfer directly to the continuation |
| different | enqueue the continuation activation on its recorded scheduler and return a no-op coroutine |

This preserves the current symmetric-transfer fast path whenever execution does
not cross a scheduler boundary. It also gives an optional specialized promise a
precise role: it may establish an initial route before first execution. An
ordinary task leaves the route unset and inherits its caller or initial
scheduler.

The scheduler route is not a backend tag. It is runtime continuation-routing
state. Device and storage validation remain separate.

## Migration-Scope Choices

### Choice A: coroutine-local routing

Every coroutine promise owns its scheduler route. A nested task inherits the
caller's route when it starts, but a later migration changes only the nested
task's promise.

When the nested task completes:

- if the continuation records the same scheduler, use direct symmetric
  transfer;
- if the continuation records another scheduler, enqueue it there and return a
  no-op coroutine from final suspend.

Example:

```text
CPU AsyncTask parent
  -> co_await CudaTask child routed to CUDA device 0
       -> child migrates to CUDA device 1 without changing promise type
       -> child completes on CUDA scheduler 1
  -> parent resumes on its recorded CPU scheduler
```

If subsequent work should remain in the CUDA task domain, put that control flow
inside the `CudaTask` or create another CUDA child. An `AsyncTask` parent does
not become a `CudaTask` when its child migrates.

Advantages:

- migration is local and visible in the coroutine that requests it;
- a helper cannot silently change its caller's execution policy;
- nested CUDA work can return naturally to CPU control flow;
- same-scheduler nesting retains the current zero-queue symmetric-transfer fast
  path.

Costs:

- final suspend becomes scheduler-aware;
- a cross-scheduler return requires an enqueue rather than direct transfer;
- continuations need enough type-erased promise information to query their
  scheduler and transfer ownership safely.

This is the current preferred direction.

### Choice B: migrate the continuation chain

A migration changes the scheduler route of the current coroutine and every
suspended continuation above it. Direct symmetric transfer can then continue
on the new scheduler.

This is cheaper at a cross-scheduler return but has undesirable semantics: a
deep helper can move all of its callers, and updating a chain of suspended
promises creates complicated ownership and cancellation interactions. This
choice should be rejected unless a concrete workload demonstrates that
coroutine-local routing is inadequate.

### Choice C: type-directed initial scheduling

`AsyncTask` and `CudaTask` can imply different initial scheduler families. This
is a useful and legitimate role for distinct promise types:

- global `schedule()` can route by task type;
- a `CudaTaskPromise` can carry the runtime device context used to choose one
  scheduler among several CUDA device schedulers;
- an `AsyncTask` can await the CUDA task and resume later on its own scheduler;
- task-specific promise state remains available throughout the child coroutine.

This complements rather than replaces same-type scheduler migration. A live
task cannot change its return or promise type, and a `CudaTask` may still need
to migrate between CUDA devices or execution lanes.

## Promise-Type Design Choices

### Generic `AsyncTask` only

Store all routing in the common promise and use explicit `schedule_on(...)`.
This has the smallest public task vocabulary and is sufficient only if CUDA
does not require persistent task-specific promise state.

### Specialized promises

A derived promise can carry persistent task metadata that genuinely belongs to
the coroutine rather than an individual operation, for example:

- a default device execution context;
- task-kind diagnostics;
- provider-specific cancellation or error metadata;
- restricted `await_transform` behavior.

Transient streams, provider handles, workspaces, and buffer leases should remain
ordinary coroutine locals with RAII lifetime. They do not belong permanently in
the promise.

### Working direction

Implement heterogeneous nested-task and scheduler plumbing without assuming
that an `AsyncTask` can become a CUDA task. Keep same-type migration as a
separate capability. Use `CudaTaskPromise` when device execution genuinely
requires persistent promise state; otherwise retain the option of ordinary
`AsyncTask` on a device scheduler.

## Required Runtime Refactoring

The current checkpoint deliberately completes only the host `AsyncTask` split:

1. `AsyncTask` is a concrete task type over the existing basic ownership
   implementation.
2. `IScheduler` exposes only private internal rescheduling to task machinery.
3. `IAsyncScheduler` defines public initial admission for `AsyncTask` plus the
   host-side pause and wait controls.

A future `CudaTask` checkpoint must decide how its basic task state participates
in internal rescheduling before adding an initial CUDA scheduler interface. Do
not introduce a loosely typed activation wrapper merely to erase that decision.
Heterogeneous nested `await_suspend`, continuation ownership, and final-suspend
routing must then be implemented and tested together.

## Scheduler Lifetime And Quiescence

A promise may record a scheduler only while that scheduler remains alive. This
is already a requirement for suspended current tasks and becomes more visible
when tasks can migrate into device contexts.

The initial contract should require device execution contexts and their
schedulers to outlive every task, resource waiter, and completion callback that
can route work to them. Destruction must diagnose outstanding references rather
than leave dangling scheduler pointers.

`TbbScheduler::run_all()` currently waits for activations in its task group. It
does not make externally suspended coroutines disappear, and migration should
not silently strengthen it into whole-program quiescence. Required questions
are therefore:

- does `run_all()` wait only for activations already submitted to that
  scheduler, as it does now;
- do tests need a separate multi-scheduler drain or device-context shutdown
  operation;
- how should diagnostics show a task whose creation scheduler differs from its
  current scheduler route?

## Required Tests

Before CUDA lowering relies on heterogeneous routing or migration, add
deterministic tests for:

- nested tasks on one scheduler retaining direct transfer;
- global `schedule()` routing `AsyncTask` and `CudaTask` to different scheduler
  families;
- an `AsyncTask` awaiting a `CudaTask` and returning to the original scheduler;
- an ordinary task migrating from one `DebugScheduler` to another;
- a child migrating while its parent remains on the original scheduler;
- a parent migrating before awaiting a child, with the child inheriting the new
  route;
- multiple nested migrations and return through each continuation;
- exception and cancellation propagation across a scheduler boundary;
- a heterogeneous derived-promise child awaited by a canonical `AsyncTask`;
- `DebugScheduler` and `TbbScheduler` combinations in both directions;
- scheduler destruction with outstanding routed or suspended tasks;
- task-registry diagnostics showing creation and current scheduler identities.

CUDA-specific tests then add per-device arena observation, resource acquisition,
and multi-device routing.

## Open Choices

- Representation of the common internal rescheduling operand, if `CudaTask`
  cannot use the current host-task basic state directly.
- Whether the generic API is `schedule_on`, `transfer_to`, or another explicit
  execution-routing name.
- How scheduler lifetime is retained or registered by a promise.
- Whether creation scheduler identity is stored separately for diagnostics.
- Whether a specialized promise may override scheduler inheritance when first
  awaited.
- Which persistent CUDA execution state, if any, justifies `CudaTaskPromise`
  beyond type-directed scheduler routing itself.
