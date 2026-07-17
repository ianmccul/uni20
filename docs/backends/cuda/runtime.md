# CUDA Runtime Foundation

**Status:** the low-level ownership, completion-token, and idle-stream-pool
primitives are implemented in `src/uni20/backend/cuda/`. CUDA Tensor storage,
CUDA kernels, coroutine awaiters, and a CUDA-specific scheduler are not yet
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

## Stream And Event Ownership

`uni20::cuda::Stream` and `uni20::cuda::Event` are move-only RAII owners of
CUDA runtime handles:

- copying is disabled;
- moving transfers ownership;
- streams are nonblocking by default;
- dependency events use `cudaEventDisableTiming`;
- destruction selects the owning device and destroys the CUDA handle.

Timing-capable events belong in explicit profiling APIs, not dependency
tracking.

`uni20::cuda::Completion` is a shared opaque completion token backed by an
event. One submitted operation records one completion event, regardless of how
many buffers it reads or writes. Every output and every input read epoch may
share that token.

## Actually-Idle Stream Pool

A stream slot has three states:

```text
idle -> leased -> pending -> idle
```

- `idle`: all previously submitted work and the pool-return host function have
  completed; the stream may be leased.
- `leased`: one host submitter owns the stream and may enqueue waits and work.
- `pending`: submission has ended, but queued work has not completed.

`StreamPool::try_acquire()` returns only an `idle` stream. It never returns a
stream merely because no host submitter currently owns it. This prevents an
unrelated operation from being queued behind a long-running kernel and
acquiring a false dependency.

Submitting a lease performs the following transition:

1. Record one non-timing completion event at the current stream tail.
2. Mark the slot `pending`.
3. Enqueue a lightweight `cudaLaunchHostFunc` after the completion event.
4. Consume the lease and return the completion token.
5. When the host function runs, mark the slot `idle`.

The host function performs no CUDA calls. It only updates pool bookkeeping.
Future coroutine integration will also arrange for the oldest stream waiter to
be rescheduled through the scheduler recorded by its coroutine promise; it must
not resume arbitrary coroutine work directly on CUDA's callback thread.

An unused lease may be returned explicitly without submission. The caller must
not use that path after enqueueing CUDA work or a stream wait.

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
install dependency event waits;
enqueue CUDA work;
auto completion = stream.submit();
```

The exact awaiter and cancellation interface remains part of the future
`CudaTask`/scheduler design. The current pool exposes nonblocking
`try_acquire()` so the resource lifecycle can be tested independently.

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

RAII cleanup failures and invalid lease-state transitions are logic failures and
remain fail-fast `CHECK`/`PANIC` paths. Destructors cannot safely switch into the
recoverable exception policy.

`cudaLaunchHostFunc` is not guaranteed to run after a CUDA context error. A
future CUDA completion/error service must therefore treat a poisoned context as
terminal for its stream pool and propagate failure to affected async outputs;
it must not wait forever for a pool-return callback.

## Next Layer

The next CUDA runtime checkpoint should add:

1. A device context owning a device-local scheduler, stream pools, and eventual
   event/handle/workspace pools.
2. Arena-entry device establishment and restoration through a
   `task_scheduler_observer`.
3. Heterogeneous `CudaTask` scheduling and nested-await routing, including
   promise state, continuation return, cancellation, wait, and quiescence
   semantics.
4. Same-task-type scheduler migration as a separate capability where useful.
5. Cancellation-safe composite resource acquisition.
6. A scheduler-neutral completion notification bridge that resubmits through
   the scheduler currently recorded by the suspended task.
7. Deferred CUDA execution-error propagation into async output epochs.
