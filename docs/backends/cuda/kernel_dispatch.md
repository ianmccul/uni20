# CUDA Kernel Dispatch and Device Scheduling

**Status:** active design note. The low-level CUDA runtime foundation,
`CudaStorage`, and unified host/multi-device debug and oneTBB schedulers are
implemented. Scoped process-wide resource initialization and canonical
per-device resources are implemented; scheduler enrollment into that runtime,
general CUDA Tensor kernel coverage, and live coroutine scheduler migration are
not yet implemented. Generic stream/provider-resource awaiters and non-blocking
async Tensor-to-cuBLAS matrix-product lowering are implemented.

This note defines how Uni20 kernel dispatch should interact with CUDA
execution. It distinguishes a logical CUDA device activation from the physical
oneTBB workers that happen to execute its tasks, and it distinguishes
lightweight CUDA submission from provider APIs that occupy a CPU thread for a
material interval.

Related notes:

- [Kernel Dispatch](../../architecture/kernel_dispatch.md) defines the generic
  `kernel_accepts_types` / `try_kernel` / optional `try_kernel_task` /
  backend-list contract.
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

Uni20 has one logical CUDA activation domain per device. Entering that domain
establishes the device on whichever oneTBB worker or application thread is
currently participating in its arena. Streams, provider handles, workspaces,
and memory resources belong to the canonical per-device `DeviceResources` and
are leased explicitly; they do not belong permanently to physical workers.
Completion events are created and recorded on their producer streams rather
than leased from a pool.

CUDA leaf-kernel dispatch remains an ordinary, non-suspending C++ call:

```text
CPU AsyncTask
  -> co_dispatch_kernel performs the ordered backend walk
     -> CublasBackend prepares operands and returns a CudaTask for device D
  -> co_await that CudaTask
     -> task-aware scheduling routes CudaTask to device-D scheduler
     -> co_await required device-D resources
     -> invoke the prepared non-suspending provider leaf
     -> enqueue device work
     -> record and publish completion
  -> resume AsyncTask through its own recorded scheduler
```

`CudaTask` and `AsyncTask` have distinct concrete promise types over the shared
`TaskPromiseBase` implementation. The concrete type identifies the declared
task kind and controls type-safe initial admission, while the selected scheduler
is recorded in the common promise state and remains authoritative for execution.
The selected device ordinal is stored only in `CudaTaskPromise`: initial
admission binds it from Tensor placement and an unbound nested CUDA task may
inherit it from a CUDA parent. A live task does not change concrete task type or
promise type. Device affinity may change only while the task is suspended at an
explicit device-selection boundary.

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
  scheduler: `DebugCudaScheduler` and a one-slot `TbbCudaScheduler` are valid
  non-blocking execution environments.

The first implemented storage policy is `CudaStorage`, exposed
conveniently through `CudaTensor`. It supplies opaque CUDA storage that can
participate in the non-blocking channel, but the operation entry point still
selects resource-admission behavior. An ordinary direct Tensor operation may
block while acquiring resources; an `Async<Tensor>` lowering must suspend
instead. Both paths share `CudaBuffer`, operand preparation, and provider
execution.

A direct C++ API uses ordinary `dispatch_kernel` and may block during resource
admission. An async CUDA operation uses `co_dispatch_kernel` while holding its
epoch buffers, so a backend-provided task can suspend on resource scarcity
rather than occupying a scheduler participant.

The channel should not be carried by an ad-hoc backend selector. The same
storage-selected numerical backend remains applicable to both entry points;
the ordinary backend adapter uses blocking admission, while coroutine dispatch
uses an optional task-producing customization for that backend and operation.

The per-call stream, handle, and workspace leases are operation-local and must
not be stored in the backend selector. Ordinary `CublasBackend` GEMM prepares
the operands, blocks for an execution lease, and submits the prepared call.
Async matrix-product lowering instead calls generic `co_dispatch_kernel`.
`CublasBackend::try_kernel_task` prepares the same operands and returns a CUDA
task that suspends for the execution lease before invoking the prepared cuBLAS
leaf. Backends without this optional hook use their ordinary blocking
`try_kernel` implementation directly inside the dispatch coroutine.

## Per-Device Resources

The scoped process-wide `cuda::Runtime` owns one canonical
`cuda::DeviceResources` for every enrolled device. Each resource set owns the
validated device, idle stream pool, and a lazy type-indexed provider-resource
registry. Each buffer separately owns the mutex protecting its completion
ledger. The resource service can contain structures such as:

```cpp
namespace uni20::cuda
{
struct DeviceResources
{
  Device device;
  StreamPool streams;
  cublas::ExecutionPool cublas;
  ResourcePool<CusolverHandleSlot> cusolver_handles;
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
`IAsyncScheduler` accepts `AsyncTask`, while `ICudaScheduler` accepts a
`CudaTask` either with explicit affinity or through its default CUDA device.
One scheduler object may implement both interfaces. A CUDA task may later use
`co_await cuda::set_device(device)` to establish or change affinity at an
explicit suspension point.

An ordinary CPU coroutine enters that CUDA task domain by awaiting a newly
created task:

```cpp
co_await cuda_operation(device, operands...); // returns CudaTask
```

The outer `AsyncTask` remains a distinct coroutine with its original promise and
scheduler route. A scheduler-unbound nested `CudaTask` inherits that route only
when the unified scheduler accepts its CUDA domain and established device
affinity. The cuBLAS GEMM task binds the operand device before nesting, so admission goes
directly to the correct device activation. When it completes, the outer
continuation is submitted through the scheduler recorded in the outer promise.

Both task promises share `TaskPromiseBase`, but their concrete promise and
return types remain fixed at creation. Device migration within one unified
scheduler is implemented; scheduler migration remains a separate future
capability. Neither operation moves Tensor operands or changes their placement.

Required migration invariants are:

- migration occurs only at an explicit suspension point;
- migration never changes the coroutine's promise or task type;
- the target scheduler and `cuda::DeviceResources` outlive the migrated task;
- ownership of the suspended task passes exactly once from the old scheduler
  route to the new one;
- cancellation and exceptions remain attached to the same coroutine and async
  outputs;
- generic buffer/epoch access may remain live across migration, but a task must
  not migrate while holding a device-local stream, handle, or workspace lease;
- scheduler migration does not move Tensor storage or change its device;
- callbacks never resume the coroutine directly on CUDA-owned threads; they
  submit it through the scheduler currently recorded by the task.

The exact scheduler-migration API remains an open implementation design. It must
be reconciled with activation accounting, wait/quiescence semantics,
diagnostics, and nested coroutine continuation routing before implementation.
Heterogeneous nesting is required independently: a `CudaTask` with an explicit
device route must not inherit and execute on an incompatible parent scheduler
merely because it was entered through `co_await`.

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

Acquisition should return the complete resource bundle required by leaf
dispatch. The acquired bundle is not a backend selector. In non-blocking async
lowering it is an internal operand passed directly to the provider-ready leaf:

```cpp
auto execution = co_await cublas::acquire_execution(cublas_pool);
invoke_cublas_gemm(execution, output, alpha, lhs, rhs, beta);
```

Once the complete resource set has been acquired, the backend walk and provider
call are ordinary non-coroutine code. No `co_await` occurs while invoking the
provider. Raw provider handles remain internal to their leases and backend
adapters; they are not user-facing coroutine results.

A provider with a verified host-thread-affine handle contract requires a
provider-specific execution adapter. Do not impose permanent thread affinity on
all CUDA resources because one future provider might require it.

## Ordered Provider Admission

The first implementation does not atomically reserve every provider resource.
Instead, it uses one documented order: acquire the scarcer provider handle,
then wait for an actually-idle stream. A provider-specific awaiter presents the
completed pair as one move-only execution lease to leaf dispatch.

This avoids permanently pairing handles and streams, which would strand stream
capacity whenever a handle is idle. It also avoids the dangerous reverse order
of holding a stream while waiting for a provider handle. Stream-only operations
do not acquire handles, and provider operations do not retain buffer guards
while waiting, so the ordering has no resource cycle. Handle-local workspace
caches may grow inside the handle slot without adding another acquisition
edge.

## Non-Suspending Leaf Dispatch

`try_kernel(...)` remains an ordinary function. Direct GEMM acquires a CUDA
execution lease and scoped buffer guards inside the accepted backend attempt.
Async lowering performs side-effect-free preparation followed by awaitable
resource admission before entering the same non-suspending prepared leaf. A
CUDA backend attempt may:

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

The scheduler participant becomes available at the submitted boundary. The
initial cuBLAS implementation conservatively retains its handle until a host
callback reaches the submitted stream tail. The stream, operands, outputs,
workspaces, and any provider-specific state referenced by device work remain
retained until their individual contracts permit release.

An `Async<GpuTensor>` result can become logically ready at the submitted
boundary: its metadata, storage, and producer completion token are fixed.
Another GPU operation can consume it by installing an event wait without
waiting on the host.

An `Async<CpuTensor>` produced by a device-to-host transfer, a C++ scalar, or
another host-visible result becomes ready only at the completed boundary.

## Provider-Resource Reuse

The lifetime of a handle lease is not necessarily the lifetime of the
operation's device work. The per-device resources must support provider-specific
policies:

- release the handle after the host API returns; or
- retain the handle until device completion.

The first policy permits another operation to submit while prior device work
remains pending. It is appropriate only when the provider permits handle reuse
and all operation-specific workspace/state has independent lifetime.

The second policy is conservative for providers or routines whose state may
remain associated with queued device work. The implemented cuBLAS execution
lease uses this policy. Earlier handle reuse requires provider documentation,
focused tests, and profiling.

## Relationship To Generic Kernel Dispatch

Ordinary `dispatch_kernel` remains non-coroutine code and invokes only
`try_kernel`. `co_dispatch_kernel` performs the same ordered type and runtime
backend walk from within a coroutine. For each backend/operation pair it awaits
`try_kernel_task` when that optional customization exists; otherwise it invokes
ordinary `try_kernel` inline.

The cuBLAS GEMM customization completes runtime operand preparation before
creating its deferred task, so a decline remains clean. Once the task is
awaited, resource admission and provider failures are terminal rather than
eligible for fallback. This permits coroutine implementations to be added one
backend and operation at a time without trivial `co_*` wrappers for blocking
kernels.

Clean cancellation before resource admission is required but is not yet
implemented for the intrusive resource wait queues. Until that support lands,
a queued task and its pool must remain alive until admission resumes the task.
Once the backend has changed provider state or enqueued work, cancellation
cannot turn the attempt into a backend decline or reclaim retained resources
early. The submitted work must reach a completion/error boundary.

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

## Next Implementation Order

1. Enroll the selected unified CUDA scheduler with the scoped process runtime
   without making scheduler state part of `DeviceResources`.
2. Add automatic initial CUDA admission from Tensor-storage placement when the
   device is available without inspecting a pending async value.
3. Specify live-task scheduler migration separately, including activation
   accounting, cancellation, exceptions, waits, and quiescence.
4. Define cancellation and runtime-shutdown behavior for queued resource
   waiters.
5. Implement one lightweight CUDA Tensor copy or elementwise kernel path.
6. Implement one host-intensive cuSOLVER operation on the same device scheduler
   and profile whether a separate provider lane is justified.
7. Validate multi-device isolation and explicit migration between two device
   resource sets.

## Open Questions

- How should typed CUDA admission combine Tensor storage placement with an
  explicitly selected scheduler override?
- Which same-type scheduler migrations are useful after heterogeneous nested
  task routing is available?
- Confirm that a oneTBB task group owns only a scheduled activation and that no
  persistent task-group membership needs to migrate with a suspended coroutine.
- Should `run_all()` retain its current activation-quiescence meaning, and what
  separate operation drains or shuts down a multi-device scheduler?
- How should device scheduler destruction prove that no migrated tasks or
  resource waiters still reference it?
- Which provider/routine families permit handle release at host-call return?
- Should CPU-intensive providers eventually receive a separate bounded lane
  within the device scheduler?
- Which completion mechanism detects terminal CUDA context failure when a
  `cudaLaunchHostFunc` callback cannot run?
