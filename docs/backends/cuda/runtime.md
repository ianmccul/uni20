# CUDA Runtime Foundation

**Status:** the low-level ownership, completion-token, idle-stream-pool, device
buffer, and scoped stream-access primitives are implemented in
`src/uni20/backend/cuda/`. CUDA Tensor storage, CUDA kernels, coroutine
awaiters, and a CUDA-specific scheduler are not yet implemented.

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
The next Tensor-storage checkpoint will place a `Device` in a typed CUDA storage
domain so allocation and resolved views retain explicit placement.

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
Future coroutine integration will also arrange for the oldest stream waiter to
be rescheduled through the scheduler recorded by its coroutine promise; it must
not resume arbitrary coroutine work directly on CUDA's callback thread.

There is no explicit stream `submit()` step. Recording operation completions is
done by `Stream::record_completion()` or by scoped buffer access guards; stream
pool admission is controlled only by the lifetime of `Stream` handles.

## Streams As Admission Control

The stream pool is a natural throttle:

- it bounds simultaneously executing or queued CUDA operations;
- it bounds lane-local handles and workspaces;
- it limits live intermediate storage and allocator pressure;
- exhausted acquisition can suspend a coroutine instead of allowing unbounded
  device submission.

The intended awaitable API is conceptually:

```text
auto stream = co_await device.acquire_stream();
{
  auto output_access = output.write(stream);
  auto input_access = input.read(stream);
  enqueue CUDA work using the access pointers;
}
```

The exact awaiter and cancellation interface remains part of the future
`CudaTask`/scheduler design. The current pool exposes both nonblocking
`try_acquire()` and blocking `acquire()`. The blocking path is suitable for
bring-up and synchronous-looking CUDA backends; a coroutine must eventually
suspend rather than call it while waiting for capacity.

A small configurable pool is expected. A reasonable initial heuristic is around
twice the maximum useful device concurrency, but measured workload behavior
should determine defaults.

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

`uni20::cuda::DeviceContext` currently owns a validated device, its idle stream
pool, and a mutex for short buffer-state snapshots and publication.
`uni20::cuda::Buffer<T>` is a move-only owner of one typed `cudaMalloc`
allocation associated with that context. The context must outlive its buffers
and every active stream handle acquired from its stream pool.

A buffer retains the latest exclusive-writer completion and the submitted
reader completions since that writer. It does not reproduce an `EpochQueue`:
existing async epochs or synchronous program order establish causality, while
the retained completions represent unfinished device work.

Scoped access is used as follows:

```cpp
auto stream = context.streams().acquire();
{
  auto out = output.write(stream);
  auto a = lhs.read(stream);
  auto b = rhs.read(stream);

  launch_on(stream, out.data(), a.data(), b.data());
}
```

`read(stream)` waits the stream on the latest writer completion and returns a
`ReadBuffer<T>` exposing `T const*`. `write(stream)` waits the stream on the
latest writer and every unfinished reader, then returns a `WriteBuffer<T>`
exposing `T*`. Guard destruction records a completion event at the current
stream tail and briefly locks the context state to publish it. Concurrent
readers do not wait for one another; a following writer waits for every
unfinished reader. The full causal and completion contract is in
[CUDA Buffer Completion Lowering](epoch_design_draft.md).

The current allocation path deliberately uses `cudaMalloc`/`cudaFree` while
the lifetime semantics are established. It is not the intended hot-path
allocator; future Tensor storage should use configured stream-ordered pools as
described in [Memory Allocation](memory_allocation.md).

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
   `cuda::Buffer` without making device memory host-indexable.
2. A device-local scheduler plus eventual provider-handle and workspace pools
   in `DeviceContext`.
3. Arena-entry device establishment and restoration through a
   `task_scheduler_observer`.
4. Heterogeneous `CudaTask` scheduling and nested-await routing, including
   promise state, continuation return, cancellation, wait, and quiescence
   semantics.
5. Same-task-type scheduler migration as a separate capability where useful.
6. Cancellation-safe coroutine acquisition using the implemented transaction
   and completion-publication rules.
7. A scheduler-neutral completion notification bridge that resubmits through
   the scheduler currently recorded by the suspended task.
8. Deferred CUDA execution-error propagation into async output epochs.
