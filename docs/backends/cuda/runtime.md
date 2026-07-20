# CUDA Runtime Foundation

**Status:** the low-level ownership, completion-token, idle-stream-pool,
provider-resource-pool, device-buffer, scoped stream-access, and CUDA-task
resource-awaiting primitives are implemented in `src/uni20/backend/cuda/`.
The async layer provides deterministic and oneTBB unified host/multi-device
schedulers, including per-activation device selection and restoration. The
first provider consumer is the cuBLAS handle/stream execution pool and GEMM
leaf. `CudaAsyncStorage` now connects `Tensor` ownership to `CudaBuffer`; CUDA
Tensor GEMM now lowers through opaque mdspans to `CublasBackend`. General CUDA
Tensor operations and non-blocking async GEMM resource admission are not yet
implemented.

This document defines the resource-management contract beneath future CUDA
Tensor kernels and async lowering.

## Device Selection

`uni20::cuda::Device` is the validated identity used to refer to one visible
CUDA device. `Device::get(ordinal)` validates the ordinal and initializes a
process-wide immutable `DeviceCapabilities` snapshot on first use. Repeated
lookups return cheap values and share the same cached snapshot. `Device::count()`
and `Device::enumerate()` provide process-visible discovery, while
`Device::current()` resolves the calling thread's selected device.

The capability snapshot currently records stable hardware and CUDA-runtime
properties needed by later storage and execution policy: UUID, PCI location,
compute capability, memory size, multiprocessor/thread/shared-memory limits,
copy-engine and concurrent-kernel support, managed-memory properties,
stream-priority support, cooperative launch, and stream-ordered memory-pool
support. Provider-library availability and operation coverage are not hardware
capabilities and remain backend-specific runtime checks.

Device discovery does not create streams, schedulers, provider handles, memory
pools, or allocations, and does not change the calling thread's selected device.
`CudaAsyncTensor` construction takes an explicit `DeviceContext`, and its
`CudaBufferView` mdspan handles retain the buffer whose context records the
allocation device.

CUDA runtime calls that operate on device-local resources must select the
resource's device. `uni20::cuda::ScopedDevice` temporarily selects a device and
restores the calling thread's previous device when it leaves scope.

The planned runtime has one logical execution context and one scheduler arena
per CUDA device. A oneTBB arena does not own a fixed set of workers, so the
device arena will use a `task_scheduler_observer` to select the device whenever
a worker or application thread enters and restore its previous device on exit.
This gives every task executing in that arena the same logical device without
requiring physical thread affinity.

Public resource wrappers must still retain their owning device and must not rely
on ambient device state. They can be constructed, destroyed, or used by code
outside the device scheduler, where `ScopedDevice` remains the correctness
mechanism.

## Stream And Completion Ownership

`uni20::cuda::Stream` is a reference-counted lease of one `StreamPool` slot.
Streams are acquired from the pool rather than constructed directly. Copies
share the same leased slot, and the slot is returned to the pool only after the
last `Stream` reference is destroyed and all previously queued CUDA work has
completed.

`Stream::record_completion()` creates a non-timing CUDA event on the stream's
own device, records the current stream tail, and returns an immutable shared
`uni20::cuda::Completion` token. There is no public unrecorded-event state and
no opportunity to associate an event with the wrong producer device.

Consumers install dependencies in the direction that changes the stream:

```cpp
Completion completion = producer.record_completion();
consumer.wait_on(completion);
```

The completion retains the producer device and its private event. A consumer
stream may belong to another device because CUDA supports cross-device event
waits. Once every delayed consumer has installed its wait or otherwise released
the token, destroying the final `Completion` reference destroys the host event
handle without waiting for device completion; CUDA retains any device-side
state still needed by already-enqueued waits.

Dependency events are allocated per completion initially. Add an event pool only
if measurement shows that event creation and destruction are a material
submission cost. Timing-capable events belong in a separate profiling API rather
than dependency tracking.

## Actually-Idle Stream Pool

A stream slot has three states:

```text
idle -> leased -> pending -> idle
```

- `idle`: all previously submitted work and the pool-return host function have
  completed; the stream may be leased.
- `leased`: one or more scoped `Stream` handles reference the stream and may
  enqueue waits and work.
- `pending`: the final stream handle has been destroyed, but queued work has not
  completed.

`StreamPool::try_acquire()` returns only an `idle` stream. It never returns a
stream merely because no host submitter currently owns it. This prevents an
unrelated operation from being queued behind a long-running kernel and
acquiring a false dependency.

Destroying the final `Stream` handle performs the following transition:

1. Mark the slot `pending`.
2. Enqueue a lightweight `cudaLaunchHostFunc` at the current stream tail.
3. When the host function runs, mark the slot `idle`.

The host function performs no CUDA calls. It only updates pool bookkeeping.
An exhausted non-blocking acquisition queues an intrusive FIFO waiter. The pool
callback publishes the newly available stream and reschedules the owning task
through the scheduler recorded by its coroutine promise; it does not resume
coroutine work directly on CUDA's callback thread. Registration uses an
explicit handshake so readiness racing with `await_suspend()` cannot resume the
coroutine before suspension ownership has been published.

There is no explicit stream `submit()` step. Recording operation completions is
done by `Stream::record_completion()` or by scoped buffer access guards; stream
pool admission is controlled only by the lifetime of `Stream` handles.

## Streams As Admission Control

The stream pool is a natural throttle:

- it bounds simultaneously executing or queued CUDA operations;
- it limits live intermediate storage and allocator pressure;
- exhausted acquisition can suspend a coroutine instead of allowing unbounded
  device submission.

The intended awaitable API is conceptually:

```text
auto stream = co_await cuda::acquire_stream(context.streams());
{
  auto output_access = output.write_synchronized_with(stream);
  auto input_access = input.read_synchronized_with(stream);
  enqueue CUDA work using the access pointers;
}
```

The pool exposes immediate `try_acquire()`, blocking `acquire()`, and the
CUDA-task-only `cuda::acquire_stream(pool)` awaiter. The CUDA execution model
therefore has two explicit submission channels above the same pool:

- **blocking:** resource acquisition may wait on the calling thread. This is
  suitable for non-async C++ calls, bring-up code, and tests.
- **non-blocking:** resource acquisition suspends through a Uni20 scheduler
  instead of blocking the calling thread. This requires a scheduler, but the
  scheduler may be `DebugCudaScheduler`, a one-slot `TbbCudaScheduler`, or a
  larger unified device scheduler.

The implemented `CudaAsyncStorage` policy identifies CUDA storage intended for
the non-blocking channel and uses `CudaBuffer<T>` as its allocation primitive.
The first ordinary Tensor GEMM path deliberately uses blocking pool admission
at its synchronous C++ boundary, while leaving device execution asynchronous.
Async Tensor lowering must instead await exhausted resource pools before
entering the same non-suspending provider leaf. A future blocking-only CUDA
storage policy should remain a distinct storage domain; wrapping such a Tensor
in `Async<T>` would almost always be a policy mistake.

The storage policy exposes a distinct mdspan accessor type so CUDA kernel
dispatch can recognize the non-blocking channel at type level. The shared
`CudaBufferView<T>` handle carries buffer identity and element offset without a
raw pointer. Per-call leases
such as streams, provider handles, and workspaces remain operation-local; they
are not part of the backend selector.

A small configurable pool is expected. A reasonable initial heuristic is around
twice the maximum useful device concurrency, but measured workload behavior
should determine defaults.

## Provider Resources And Ordered Acquisition

`cuda::ResourcePool<Resource>` owns a fixed set of preconstructed provider
resources. `ResourceLease<Resource>` is move-only and returns its slot to the
oldest queued waiter on release. The pool provides immediate, blocking, and
CUDA-task awaitable acquisition through `cuda::acquire_resource(pool)`.

Provider handles are not permanently paired with streams. The implemented
`cublas::ExecutionPool` acquires one scarce handle first, then waits for an
actually-idle stream, and returns both in one `cublas::ExecutionLease`. This
ordering avoids stranding streams behind idle handles and bounds the number of
provider operations waiting for stream capacity. Provider acquisition must
finish before buffer access guards are created.

`cuda::DeviceContext::provider_resource<Resource>(...)` lazily constructs one
instance of each concrete provider-resource type and destroys those resources
before its stream pool. `cublas::execution_pool(context)` uses that registry so
repeated GEMM calls reuse the same handles. This is the current explicit-context
implementation; the planned process-wide CUDA runtime can become the canonical
owner without changing the Tensor or backend call shape.

There is no resource cycle under the implemented contract:

- provider operations always acquire handle before stream;
- stream-only operations never wait for a provider handle;
- no buffer guard is retained while waiting for either resource.

The cuBLAS handle is conservatively returned from a host callback at the
operation stream tail. The stream remains governed independently by its
reference-counted pool lease. Another operation may therefore reuse the handle
with a different idle stream while an unrelated reference still retains the old
stream slot.

## Event-Based Dependencies

The runtime deliberately does not attempt stream affinity or same-stream
continuation:

- every submitted operation records a completion event;
- each operation acquires an independently selected idle stream;
- data dependencies are installed with `cudaStreamWaitEvent`;
- correctness never depends on receiving a producer's stream for its consumer.

While a producer is running, its stream is not available. Once it is available,
the producer is already complete, so preserving affinity has little value.
Avoiding affinity also removes complexity from multi-input joins, fairness,
cancellation, and scheduler policy.

If event overhead becomes significant for tiny operations, the preferred
solutions are operation coalescing, batched kernels, or CUDA graphs. Reintroduce
stream-affinity state only if profiling demonstrates a real need.

## Device Buffers And Scoped Access

For an introductory explanation and complete first-use examples, start with
[CUDA Buffers](buffers.md). This section records how buffers fit into the wider
runtime.

`uni20::cuda::DeviceContext` currently owns a validated device, its idle stream
pool, and a mutex for short buffer-state snapshots and publication.
`uni20::cuda::CudaBuffer<T>` is a move-only owner of one typed CUDA allocation
associated with that context. It uses `cudaMallocAsync` when the device supports
stream-ordered memory pools and records that allocation as the buffer's current
writer completion; otherwise it falls back to `cudaMalloc`. The context must
outlive its buffers and every active stream handle acquired from its stream
pool.

A buffer retains the latest exclusive-writer completion and the submitted
reader completions since that writer. It does not reproduce an `EpochQueue`:
existing async epochs or synchronous program order establish causality, while
the retained completions represent unfinished device work.

The useful mental model is an ordinary mutable value with delayed device
completion. Concurrent reads are valid; a write cannot overlap another write or
any live read. Guard acquisition validates that ordinary value rule with a
reader count and a single-writer flag. It does not queue the caller or wait for
another host guard to release. Guard release records its stream tail in the
completion ledger, allowing the next causally ordered operation to be submitted
without waiting for the GPU to catch up.

Scoped access is used as follows:

```cpp
auto stream = context.streams().acquire();
{
  auto out = output.write_synchronized_with(stream);
  auto a = lhs.read_synchronized_with(stream);
  auto b = rhs.read_synchronized_with(stream);

  launch_on(stream, out.data(), a.data(), b.data());
}
```

`read_synchronized_with(stream)` waits the stream on the latest writer
completion and returns a `ReadAccess<T>` exposing `T const*`.
`write_synchronized_with(stream)` waits the stream on the latest writer and
every unfinished reader, then returns a `WriteAccess<T>` exposing `T*`. Guard
destruction records a completion event at the current stream tail and briefly
locks the context state to publish it. Concurrent readers do not wait for one
another; a following writer waits for every unfinished reader. The full causal
and completion contract is in
[CUDA Buffer Completion Lowering](epoch_design_draft.md).

Explicit `guard.release()` performs the same completion publication before
lexical destruction. Access construction is synchronous bounded host work and
has no coroutine-awaiter form. Only acquisition of potentially unavailable
resources such as streams, provider handles, and workspaces has separate
blocking and non-blocking channels.

The current allocation path uses CUDA's default stream-ordered pool when
available, but it does not yet configure release thresholds, prime pools, or
route tensor storage through an allocator policy. That broader direction is in
[Memory Allocation](memory_allocation.md).

## Error Reporting

Checked CUDA runtime calls raise `uni20::cuda::CudaRuntimeError`, which carries:

- the CUDA status and symbolic name;
- the operation that failed;
- the CUDA runtime description;
- the device ordinal when known;
- Uni20 source-location and stacktrace context.

`diagnostic_report(CudaRuntimeError const&)` renders the failure through Uni20's
presentation layer. Native C++ mode aborts with that diagnostic; Python mode can
preserve the structured exception.

RAII cleanup failures and invalid stream-pool state transitions are logic failures and
remain fail-fast `CHECK`/`PANIC` paths. Destructors cannot safely switch into the
recoverable exception policy.

`cudaLaunchHostFunc` is not guaranteed to run after a CUDA context error. A
future CUDA completion/error service must therefore treat a poisoned context as
terminal for its stream pool and propagate failure to affected async outputs;
it must not wait forever for a pool-return callback.

## Next Layer

The next CUDA runtime checkpoints should add:

1. Typed CUDA Tensor storage and a device mdspan/accessor contract built over
   `cuda::CudaBuffer` without making device memory host-indexable.
2. Move the implemented stream and provider pools behind the future canonical
   process-wide per-device resource service.
3. Typed initial `CudaTask` admission that selects the scheduler matching Tensor
   storage placement. Shared-promise nested-await and continuation routing are
   already implemented and covered by both CUDA schedulers' tests.
4. Live-task scheduler migration as a separate capability where useful.
5. Define explicit cancellation and shutdown behavior for tasks queued on
   resource pools.
6. Extend handle slots with provider-specific reusable workspace caches.
7. Deferred CUDA execution-error propagation into async output epochs.
