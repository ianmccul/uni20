# CUDA Kernel Dispatch and Device Scheduling

**Status:** active design note. The low-level CUDA runtime foundation and
device-bound debug/oneTBB schedulers are implemented, but CUDA Tensor storage,
CUDA kernel backends, resource awaiters, device-context integration, and live
coroutine scheduler migration are not yet implemented.

This note defines how Uni20 kernel dispatch should interact with CUDA
execution. It distinguishes the logical CUDA execution context from the
physical oneTBB workers that happen to execute its tasks, and it distinguishes
lightweight CUDA submission from provider APIs that occupy a CPU thread for a
material interval.

Related notes:

- [Kernel Dispatch](../../architecture/kernel_dispatch.md) defines the generic
  `kernel_accepts_types` / `try_kernel` / backend-list contract.
- [Execution Architecture](../../architecture/execution.md) describes scheduler
  and storage-policy layering.
- [Ordering and Backend Lowering](../../architecture/ordering_and_backend_lowering.md)
  defines submission order, completion order, and token-pins-storage.
- [CUDA Runtime](runtime.md) defines streams, completion tokens, and the
  actually-idle stream pool.
- [CUDA Buffer Completion Lowering](epoch_design_draft.md) defines typed
  buffers and scoped read/write guards.
- [cuSOLVER Architecture](cusolver.md) records solver-specific resource and
  workspace considerations.
- [oneTBB Execution Primer](../../async/tbb_execution_primer.md) explains arena
  slots, worker participation, and scheduler observers.
- [Scheduler Routing, Nested Task Domains, and Promise Specialization](../../async/scheduler_migration.md)
  distinguishes type-directed scheduling, heterogeneous nesting, and same-type
  migration.

## Summary

Uni20 should have one logical CUDA execution context per device. Entering that
context establishes the device on whichever oneTBB worker or application
thread is currently participating in its arena. Streams, provider handles,
workspaces, and memory resources belong to the device context and are leased
explicitly; they do not belong permanently to physical workers. Completion
events are created and recorded on their producer streams rather than leased
from a pool.

CUDA leaf-kernel dispatch remains an ordinary, non-suspending C++ call:

```text
CPU AsyncTask
  -> co_await a newly created CudaTask for device D
     -> task-aware scheduling routes CudaTask to device-D scheduler
     -> co_await required device-D resources
     -> invoke the backend walk without suspension
     -> enqueue device work
     -> record and publish completion
  -> resume AsyncTask through its own recorded scheduler
```

`CudaTask` and `AsyncTask` have distinct concrete promise types over the shared
`TaskPromiseBase` implementation. The concrete type identifies the declared
task kind and controls type-safe initial admission, while the selected scheduler
is recorded in the common promise state and remains authoritative for execution.
Device placement belongs to the CUDA scheduler/context and Tensor storage, not
to task-specific promise state. A live task does not change concrete task type
when it crosses a scheduler boundary.

Lightweight and host-intensive CUDA calls use the same device scheduler and
resource model initially. A host-intensive provider call occupies one device
scheduler participant until its host API returns. Handle and stream pool
capacity bound provider concurrency. A separate provider lane or scheduler is
an optional later optimization if profiling shows that provider calls impede
lightweight submission or other host work.

## Blocking And Non-Blocking Channels

CUDA operations should be described in terms of submission channels:

- **blocking:** resource acquisition may wait on the calling thread. This is
  appropriate for non-async C++ APIs, tests, command-line examples, and bring-up
  code.
- **non-blocking:** resource acquisition suspends through a Uni20 scheduler
  while waiting for streams, provider handles, workspace, or other scarce
  resources. This requires a scheduler, but it does not require a multi-threaded
  scheduler: `DebugScheduler` and a one-slot `TbbScheduler` are valid
  non-blocking execution environments.

This channel choice is a Tensor storage-policy parameter. It is orthogonal to
Tensor ownership and to whether a value is wrapped in `Async<T>`. A concrete
`Tensor<..., CudaStorage<blocking_channel>>` and
`Tensor<..., CudaStorage<nonblocking_channel>>` are distinct storage domains
even if they share the same underlying allocation primitive. That distinction
can naturally flow into the mdspan handle/accessor type seen by dispatchable
kernels.

A non-async C++ API may still choose the non-blocking channel internally if it
enters a scheduler and waits at a defined boundary. Conversely,
`Async<Tensor<..., CudaStorage<blocking_channel>>>` is possible as C++, but it
is a dubious policy combination: lowering it through blocking resource
acquisition would occupy the scheduler participant instead of suspending the
coroutine. Async CUDA front-ends should therefore accept only the non-blocking
CUDA storage policy for resources that may wait for capacity, unless a future
operation explicitly documents why blocking is intentional.

Implementation should make the invalid combination hard to spell. The channel
should not be carried by an ad-hoc backend selector. Blocking front-ends accept
the blocking storage policy; async lowering accepts the non-blocking storage
policy. Mixing the two is not an important use case, and disallowing it keeps
accidental scheduler blocking out of async code.

The per-call stream, handle, and workspace leases are still operation-local.
They should be acquired by the front-end lowering for the storage policy and
passed to CUDA backends as an internal lowered operand or execution context.
They should not be stored in the backend selector. The selector remains the
ordinary backend list associated with the storage/default execution policy, for
example `cuda_reference`, `cublas`, `cusolver`, and future provider backends.
Both channels should converge on the same non-suspending leaf backend once the
complete resource set has been acquired.

## Per-Device Execution Context

The implemented `cuda::DeviceContext` currently owns the validated device,
short-lived buffer-state mutex, and idle stream pool. It should grow to own all
execution resources associated with one device:

```cpp
namespace uni20::cuda
{
struct DeviceContext
{
  Device device;
  TbbScheduler scheduler;
  StreamPool streams;
  HandlePool<CusolverHandle> cusolver_handles;
  HandlePool<CublasHandle> cublas_handles;
  // Memory resources, workspaces, diagnostics, and provider-specific pools.
};
} // namespace uni20::cuda
```

This is conceptual structure, not a required concrete aggregate. In
particular, the retained CUDA primary context and error-monitoring service may
need separate lifetime wrappers.

Each submission initially allocates its own non-timing completion event.
`Stream::record_completion()` guarantees that the event is created on the
producer stream's device, and `Stream::wait_on()` permits same- or cross-device
consumers. Introduce event pooling only if measurements show that allocation is
a material cost.

The device ordinal comes from Tensor storage/location policy. A kernel with
multiple device operands must either validate that they share a device or be an
explicit transfer/collective operation. Backend selection must not silently
copy operands between devices.

### An arena has slots, not fixed workers

A oneTBB `task_arena` owns a task domain and a concurrency limit. It does not
own a permanent private set of worker threads. For example, an arena with
concurrency two permits at most two simultaneous participants, but different
oneTBB workers may occupy those slots over time. An application thread may also
enter a reserved slot through `task_arena::execute()`.

Attach a `task_scheduler_observer` to each device arena:

- on arena entry, save the participating thread's prior CUDA device and select
  the arena's device;
- on arena exit, restore the prior device;
- initialize the CUDA primary context and other fallible device state before
  enabling observation;
- do not throw from observer callbacks;
- optionally verify the current device at task entry in debug builds.

The observer establishes a logical device-affinity interval while a thread
participates in the arena. It does not pin worker identity. Public CUDA resource
wrappers still retain their own device ordinals and use explicit device guards
when they may be called outside the device scheduler.

## Task Types, Nesting, And Scheduler Migration

The scheduler pointer in the shared Uni20 coroutine promise determines where a
viable task is submitted. Initial admission remains type-specific:
`IAsyncScheduler` accepts `AsyncTask`, while `ICudaScheduler` accepts `CudaTask`.
A CUDA scheduling front-end selects the per-device scheduler from explicit
device or Tensor-storage placement.

An ordinary CPU coroutine enters that CUDA task domain by awaiting a newly
created task:

```cpp
co_await cuda_operation(device, operands...); // returns CudaTask
```

The outer `AsyncTask` remains a distinct coroutine with its original promise and
scheduler route. The nested `CudaTask` is scheduled on its device context. When
it completes, the outer continuation is submitted through the scheduler
recorded in the outer promise.

Both task types use the same promise, but their concrete return types remain
fixed at creation. Scheduler migration is a separate future capability. A
`CudaTask` could migrate between compatible device schedulers by updating the
common scheduler route consistently; that does not move its Tensor operands or
change their device.

Required migration invariants are:

- migration occurs only at an explicit suspension point;
- migration never changes the coroutine's promise or task type;
- the target scheduler and `cuda::DeviceContext` outlive the migrated task;
- ownership of the suspended task passes exactly once from the old scheduler
  route to the new one;
- cancellation and exceptions remain attached to the same coroutine and async
  outputs;
- generic buffer/epoch access may remain live across migration, but a task must
  not migrate while holding a device-local stream, handle, or workspace lease;
- scheduler migration does not move Tensor storage or change its device;
- callbacks never resume the coroutine directly on CUDA-owned threads; they
  submit it through the scheduler currently recorded by the task.

The exact awaiter and scheduler API remain an open implementation design. In
particular, scheduler migration must be reconciled with activation accounting,
wait/quiescence semantics, diagnostics, and nested coroutine continuation
routing before implementation. Heterogeneous nesting is required independently:
a `CudaTask` with an explicit device route must not inherit and execute on an
incompatible parent scheduler merely because it was entered through `co_await`.

## Two CUDA Host-Call Classes

### Lightweight submission

A lightweight call performs bounded host work and primarily enqueues device
work. Typical examples are:

- an ordinary CUDA kernel launch;
- `cudaMemcpyAsync`;
- many cuBLAS operations;
- event record and stream-wait operations.

These calls execute directly in the device scheduler after logical dependencies
and resource leases are ready. They must not perform a host synchronization
before returning.

### Host-intensive provider execution

A provider call may execute a substantial CPU algorithm while also launching
and coordinating device kernels. Examples can include:

- cuSOLVER factorizations and decompositions;
- cuTensorNet optimization, preparation, and contraction APIs;
- selected cuQuantum operations;
- other vendor APIs shown by profiling to occupy the caller materially.

For example, a cuSOLVER SVD can keep the calling CPU thread active while
launching thousands of kernels. The call may therefore occupy a device
scheduler participant for much longer than an ordinary kernel launch.

Classification is operation- and provider-specific. Library branding alone is
not sufficient. The first implementation uses the device scheduler directly;
only measurements should justify introducing a separate bounded provider
execution lane.

## Awaitable Resource Leases

Provider handles are device-local resources, not thread-local coroutine state.
An internal coroutine may await an exclusive RAII handle lease, provided the
handle is documented to permit sequential use from different host threads.
The lease may move with the suspended coroutine between physical workers in the
same device arena.

Acquisition should normally be composite. The acquired resource bundle is not a
backend selector. It is an internal lowered operand used by CUDA backend
attempts:

```cpp
auto resources = co_await storage_domain.acquire(cusolver_request{
    .stream = true,
    .handle = true,
    .workspace_bytes = workspace_bytes,
});

dispatch_kernel(storage_domain.backends(), svd_op{}, resources, request);
```

Once the complete resource set has been acquired, the backend walk and provider
call are ordinary non-coroutine code. No `co_await` occurs while invoking the
provider. Raw provider handles remain internal to their leases and backend
adapters; they are not user-facing coroutine results.

A provider with a verified host-thread-affine handle contract requires a
provider-specific execution adapter. Do not impose permanent thread affinity on
all CUDA resources because one future provider might require it.

## Composite Resource Admission

A caller must not acquire one scarce resource and then suspend indefinitely
waiting for another. Holding a stream while waiting for a provider handle or
workspace, for example, creates avoidable starvation and deadlock risks.

The device context should queue one composite request and make it runnable when
all required resources are available:

```text
queued device request
        |
        +-- actually-idle stream available
        +-- compatible provider handle available
        +-- required workspace available
        |
        v
non-suspending backend walk and provider call
```

Stream-pool capacity remains device admission control. Handle-pool and
workspace limits add provider-specific admission control without tying those
resources to scheduler workers.

## Non-Suspending Leaf Dispatch

`try_kernel(...)` remains an ordinary function. It executes with a CUDA stream
and scoped buffer guards supplied through the backend context. A CUDA backend
attempt may:

1. validate all remaining runtime preconditions;
2. acquire scoped read/write buffer guards that install device-event
   dependencies on its stream;
3. invoke the CUDA runtime or provider API;
4. optionally record an operation-level completion event;
5. leave buffer-completion publication to guard destruction;
6. return `KernelAttempt::success`.

It must not `co_await`, queue unfinished host-side submission work, or return
success before the scoped guards have enough lifetime to publish completions at
the appropriate stream tail. Before the CPU async buffers that made the
operation runnable are released, the CUDA scoped guards must have been
destroyed or otherwise released so their tokens are retained by the affected
storage epochs.

The strong clean-decline contract remains unchanged. A CUDA backend may decline
only before it changes provider state, enqueues work, consumes an operand,
commits output, or produces another externally visible side effect. Failure
after that point is an operation error and must not fall through to another
backend.

## Submission And Completion Boundaries

A CUDA operation has at least two relevant boundaries:

1. **Submitted:** the host call has returned, its completion event has been
   recorded, and the affected storage epochs retain that token.
2. **Completed:** the recorded device work and any requested transfer have
   finished.

The scheduler participant and provider handle may become available at the
submitted boundary. The stream, operands, outputs, workspaces, and any
provider-specific state referenced by device work remain retained until the
provider contract permits release, normally no earlier than completion.

An `Async<GpuTensor>` result can become logically ready at the submitted
boundary: its metadata, storage, and producer completion token are fixed.
Another GPU operation can consume it by installing an event wait without
waiting on the host.

An `Async<CpuTensor>` produced by a device-to-host transfer, a C++ scalar, or
another host-visible result becomes ready only at the completed boundary.

## Provider-Resource Reuse

The lifetime of a handle lease is not necessarily the lifetime of the
operation's device work. The device context must support provider-specific
policies:

- release the handle after the host API returns; or
- retain the handle until device completion.

The first policy permits another operation to submit while prior device work
remains pending. It is appropriate only when the provider permits handle reuse
and all operation-specific workspace/state has independent lifetime.

The second policy is conservative for providers or routines whose state may
remain associated with queued device work. Initial implementations should use
the conservative policy until provider documentation, focused tests, and
profiling justify earlier reuse.

## Relationship To Generic Kernel Dispatch

The generic backend walk does not become a coroutine and does not know about
scheduler migration or resource-wait queues.

The async Tensor wrapper performs execution routing and resource acquisition,
then calls `dispatch_kernel(...)` with the storage policy's ordinary backend
list plus an internal resource-lease operand. The backend selector should not
be rewritten into a per-call resource object. If an ordered backend list
contains a CUDA provider candidate and fallback candidates, the whole relevant
backend walk runs after resource admission so runtime decline can continue
without moving operands across another suspension boundary.

A queued request may be cancelled cleanly before resource admission. Once the
backend has changed provider state or enqueued work, cancellation cannot turn
the attempt into a backend decline or reclaim retained resources early. The
submitted work must reach a completion/error boundary.

## CUDA Graph Capture

General CUDA Graph stream capture is unsupported in the initial runtime. The
ordinary CUDA coroutine path must not begin capture, accept an already-capturing
stream, or suspend while capture is active.

This restriction does not require permanent thread affinity. A future graph API
can pre-acquire its stream, handles, workspaces, and descriptors, then run one
non-suspending capture transaction on the device scheduler:

```text
begin capture -> submit captured calls -> end capture
```

The complete transaction remains on one participating thread. No `co_await`
occurs between begin and end. Graph support should be added only for workloads
with enough stable repetition to amortize capture and instantiation; typical
DMRG contractions and short Krylov matvec sequences may not meet that threshold.

## Blocking Adapters And Debug Policy

A non-async C++ API may block without requiring a second numerical
implementation. It can use the blocking channel directly, or enter the selected
device scheduler and use the non-blocking channel internally before waiting at
the requested submission or completion boundary. In both cases, it should
invoke the same non-suspending leaf backend as async CUDA lowering.

A fully synchronized debug policy waits after every submitted operation. This
is the error-localizing correctness baseline described by the CUDA epoch notes.
It should wrap the same dispatch and submission paths rather than define a
parallel collection of CUDA kernels.

## Error Propagation

Immediate provider and CUDA API errors occur inside the leaf backend attempt.
They fail the requesting async outputs through the normal structured exception
path.

Deferred execution errors are detected at a later CUDA synchronization or
completion-service boundary. They must fail every affected output epoch and
mark a poisoned device context terminal when recovery is impossible.

Diagnostics should include:

- operation name;
- selected backend/provider;
- CUDA device and device-scheduler identity;
- stream, handle-pool, or workspace lease identity where useful;
- whether failure occurred before submission or during completion;
- structured vendor and CUDA status;
- source location and stack information where available.

Rendering uses Uni20's presentation layer. Runtime layers retain structured
diagnostic data rather than preformatting one terminal-only string.

## Initial Implementation Order

1. Use the implemented shared-promise `CudaTask`, `ICudaScheduler`, and
   cross-scheduler nested continuation routing as the scheduler contract.
2. Integrate the implemented `TbbCudaScheduler` and its device-establishing
   arena observer into `cuda::DeviceContext`.
3. Add typed CUDA admission that selects the correct scheduler from explicit
   device or Tensor-storage placement.
4. Specify live-task scheduler migration separately, including activation
   accounting, cancellation, exceptions, waits, and quiescence.
5. Add cancellation-safe composite acquisition for idle streams and one
   provider handle/workspace pool.
6. Add scheduler-neutral completion notification that resubmits through the
   scheduler currently recorded by the suspended task.
7. Implement one lightweight copy or kernel path.
8. Implement one host-intensive cuSOLVER operation on the same device scheduler
   and profile whether a separate provider lane is justified.
9. Validate multi-device isolation and explicit migration between two device
   contexts.

## Open Questions

- Should typed CUDA admission select its `cuda::DeviceContext` through an
  explicit factory, Tensor storage, or another scheduling customization?
- Does any future task domain need a specialized promise, rather than state in
  its scheduler/context and coroutine-local resource leases?
- Which same-type scheduler migrations are useful after heterogeneous nested
  task routing is available?
- Confirm that a oneTBB task group owns only a scheduled activation and that no
  persistent task-group membership needs to migrate with a suspended coroutine.
- Should `run_all()` retain its current activation-quiescence meaning, and what
  separate operation drains or shuts down a multi-scheduler device context?
- How should device scheduler destruction prove that no migrated tasks or
  resource waiters still reference its context?
- Which provider/routine families permit handle release at host-call return?
- Should CPU-intensive providers eventually receive a separate bounded lane
  within the device context?
- Which completion mechanism detects terminal CUDA context failure when a
  `cudaLaunchHostFunc` callback cannot run?
