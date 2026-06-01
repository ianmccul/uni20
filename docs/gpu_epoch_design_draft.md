# GPU Epoch Design Draft

This is a draft design note.  It is intended to workshop the final uni20 GPU
dependency model before replacing or rewriting the existing CUDA scheduler notes.
Broader CUDA resource-management notes, including future idle-aware stream
leasing, live in `cuda_runtime_design_notes.md`.

The immediate implementation target is still single-threaded TensorContraction
integration.  The API and invariants should nevertheless be shaped so that the
same model can later be made thread-safe without changing semantics.

## Goals

- Track GPU buffer read/write ordering with explicit per-epoch CUDA events.
  Kernel launches are assumed to be causally ordered by the CPU async layer with
  respect to event and stream synchronization.
- Treat CUDA streams as transient execution resources.  A stream is leased for
  one access plan, used to enqueue work, and returned to the device stream pool
  after a non-timing completion event has been recorded.
- Make CUDA events the durable dependency contract between buffer epochs.  The
  first implementation should not cache streams on buffers or try to preserve
  stream affinity across operations.
- Preserve a clean path to a multi-threaded scheduler by making buffer access
  acquisition conceptually atomic.  The path to a threaded implementation should
  be close to "add locking around critical sections", with minimal refactoring.
  Avoid potential TOCTOU problems from the outset.
- Keep the first implementation single-threaded and avoid mutex/locking
  machinery until it is needed.

## Analogy To CPU Epochs

The CPU async runtime has an `EpochQueue`/`EpochContext` model.  A logical value
has an ordered chain of epochs.  Readers attach to the readable epoch, writers
advance the value to a later epoch, and RAII handles release their participation.
In the current CPU implementation, new epochs are appended through
`next_epoch_`; `EpochQueue::current_`/`latest()` is therefore the newest epoch at
the tail of the epoch chain.  Earlier epochs closer to the head are the ones
that can currently be running or finishing.  CUDA's "stream tail" is separate
terminology, but it has the same "newest enqueued position" sense.

The GPU model is similar, but CUDA changes the mechanics:

- A CUDA stream functions as an execution queue.
- CUDA events are concrete cross-stream readiness tokens.  We do not need to
  encode event ordering ourselves; the CUDA API does that for us.  We still need
  to track which buffer generation owns each writer or reader completion event.
- Multiple kernels may concurrently read the same device buffer.
- GPU writes must be ordered; we cannot model the CPU model's unordered
  multi-writer accumulation mode, as there is no such functionality in CUDA.
- A GPU operation with multiple outputs naturally shares one completion point,
  because all outputs become valid at the same stream tail.  This is different
  to CPU kernels, which can release write buffers independently.

This simplifies the GPU version.  We do not need to track a full `EpochQueue`;
we only need the current per-buffer epoch state.  Prior operations on a stream
are handled by the CUDA runtime while an operation is being enqueued.  Once the
operation is published, the buffer's epoch state is represented by CUDA events,
not by ownership of the stream used to create them.

The object is called `GpuEpochQueue` in this document to emphasize the analogy
with the CPU `EpochQueue`: the queue owns the current read/write epoch state for
one GPU buffer, and access handles are the GPU analogues of CPU read/write
buffers.

## Core Objects

`GpuEpochQueue`

The synchronization state for one GPU buffer.  Using the convention above, it is
closest in spirit to the current tail of an `EpochQueue`: it tracks the latest
writer generation and outstanding reader generations for that buffer.

State:

- `generation`: monotonically incremented for each write acquisition.  This is
  useful for debugging even if CUDA events/streams provide the real ordering.
- `writer_event`: non-timing CUDA event for the latest completed writer
  generation, or null only for an initialized buffer whose contents are already
  known to be valid without device work.
- `reader_events`: non-timing CUDA events for readers of the current generation
  that have been published and must complete before the next writer.
- `active_readers`: count or debug set of read handles acquired but not yet
  published.
- `writer_active`: debug flag for an acquired but unpublished writer.

`GpuEvent`

A small RAII wrapper around a non-timing `cudaEvent_t` acquired from the
`CudaDeviceContext` event pool.  Dependency events should be created with
`cudaEventDisableTiming`; timing-capable events belong only in explicit
profiling or benchmarking APIs.

`StreamSlot`

A reusable CUDA execution lane owned by a `CudaDeviceContext`.  It contains a
CUDA stream and pool bookkeeping only.  Device-library handles such as cuBLAS,
cuSOLVER, or cuQuantum are separate thread-local/per-device resources.  A stream
slot does not belong to a buffer epoch after an access plan has been published.

`ConcreteStreamLease`

An RAII handle that leases a concrete `StreamSlot` from a `CudaDeviceContext` so
CUDA work can be enqueued.  When the access handle publishes, it records the
completion event into this stream and returns the stream slot directly to the
pool.  Correctness must not depend on receiving the same stream for a later
operation.

`GpuAccessPlan`

The transaction returned by atomically acquiring read/write access for one GPU
operation.  In the first implementation it owns a scheduler-selected
`ConcreteStreamLease` and publishes one completion event for the operation.
External-stream mode can be added later with the same epoch rules, but should
not be part of the initial TensorContraction prototype.

## Access Rules

Reads are compatible with other reads:

- A read waits for the latest writer generation.
- A read does not wait for other readers.
- A read publishes a reader completion handle so later writers know when it is
  safe to modify the buffer.
- A read does not increment the generation counter.

Writes are ordered and exclusive:

- A write waits for the latest writer generation.
- A write waits for all outstanding reader generations.
- A write increments the generation counter.
- A write clears or supersedes old reader generations.
- A write publishes the new writer completion handle.

This gives the usual ordering table:

- read/read: compatible, no ordering required between readers.
- write/read: reader waits for prior writer.
- read/write: writer waits for outstanding readers.
- write/write: later writer waits for prior writer.

## Atomic Acquisition

The operation that must be conceptually atomic is access acquisition plus
dependency installation, not a standalone `waitOn(event)`.

For a future multi-threaded scheduler, acquisition should be one critical
section over the relevant `GpuEpochQueue`s and stream-pool state:

1. Snapshot writer and reader generations for all input/output buffers.
2. Select or accept a stream slot.
3. Emit any required `cudaStreamWaitEvent` dependencies on the selected stream.
4. Reserve the new read/write intents and return an access handle.

The single-threaded TensorContraction implementation can perform these steps
without locks, but it should keep this transaction boundary in the API.

## Correctness Baseline

The conceptual correctness baseline is a fully synchronized GPU execution mode:

```text
enqueue one CUDA operation
synchronize all affected devices
publish CPU-visible completion
enqueue the next operation
```

If every CUDA call were followed by synchronization of all relevant devices, then
all GPU side effects would be complete before the next operation was submitted.
In that degenerate mode, no CUDA events, stream waits, or GPU epoch machinery
would be required for correctness.

The GPU epoch system is an optimization over that baseline.  It replaces global
barriers with precise dependency handles:

- writer completion handles instead of synchronizing after writes;
- reader completion handles instead of synchronizing after reads;
- non-timing CUDA events instead of host-side synchronization;
- `cudaStreamWaitEvent` for precise writer/readers ordering.

The invariant is:

- For every legal CPU async schedule, asynchronous GPU execution must be
  observationally equivalent to the fully synchronized execution.

This suggests two debugging modes.

`LegacyDefaultStreamDebug` should be the preferred first-line debug path.  It
submits all GPU work to the CUDA legacy default stream, disables dependency
event record/wait where possible, and relies on default-stream ordering to
serialize device work.  It is deterministic, much less intrusive than explicit
device synchronization after every operation, and close to the existing
TensorContraction serial CUDA diagnostic mode.

`SynchronousDeviceDebug` is the stricter reference path.  It forces a
device-wide or all-device synchronization after each submitted GPU operation and
disables most event-based scheduling.  This mode should be slow and should
be used as a debugging hammer for memory lifetime bugs, host/device transfer
bugs, or cases where default-stream serialization is not strong enough to
isolate the issue.

## CPU/GPU Boundary

The CPU async scheduler owns logical causality.  The GPU epoch scheduler owns
device-side memory hazards for work that is already causally ready to submit.

For `Tensor<T, GpuStorage>`, CPU-side validity means:

- the tensor object is logically constructed;
- metadata such as shape, layout, and index structure is valid on the CPU;
- the `GpuStorage` object exists and owns a valid device allocation or deferred
  allocation handle;
- the storage can accept dependency-aware GPU access requests.

It does not mean:

- all prior kernels touching the storage have completed;
- the device data is idle;
- CPU code may dereference or randomly access the elements.

`GpuStorage` is opaque to ordinary CPU code.  A CPU async task may `co_await`
logical tensor dependencies and then submit GPU work once the tensor object and
its metadata are valid.  At that point, the producing GPU work for each input
has already been submitted and has a fixed GPU dependency handle: a CUDA event
or a known initialized state that requires no wait.

This gives the launch protocol:

```text
CPU async task:
  co_await logical Tensor dependencies
  acquire GpuStorage read/write epochs
  enqueue CUDA kernels/copies with required waits
  publish output GPU epoch handles
  return or suspend according to the host-side API
```

The CUDA layer should not accept dependencies on future producer events that have
not yet been submitted or fixed.  Arbitrary task ordering, reverse submission,
and backprop causality belong in the CPU async scheduler.  CUDA streams and
events should only order already-submitted GPU work and memory hazards between
causally ready operations.

This boundary avoids making the CUDA scheduler a general DAG executor.  It also
avoids artificial CUDA deadlocks from stream reuse: a GPU operation is submitted
only after every dependency it may wait on has a fixed producer position in the
GPU execution graph.  Stream-slot reuse only has to preserve stream ordering
while an access plan is active; after publication the durable dependency is the
recorded event rather than the stream slot itself.

Host access to GPU-resident data is always an explicit scheduled operation.  A
readback is modeled as a GPU read followed by an asynchronous transfer or
conversion into CPU storage:

```text
Tensor<T, GpuStorage>
  -> acquire GPU read epoch
  -> enqueue cudaMemcpyAsync D2H or conversion kernel
  -> co_await host completion
  -> Tensor<T, CpuStorage>
```

Likewise, CPU-to-GPU materialization is an explicit upload/write operation:

```text
Tensor<T, CpuStorage>
  -> allocate/acquire Tensor<T, GpuStorage>
  -> enqueue cudaMemcpyAsync H2D or conversion kernel
  -> publish GPU write epoch
```

Unified memory should not be implicit in `GpuStorage`.  If uni20 supports it
later, it should be a distinct storage type such as `Tensor<T, UnifiedStorage>`
with its own coherence and ownership rules.  Randomly accessing the same storage
from CPU and GPU is not part of the `GpuStorage` model.

## Writer Workflow

A writer is exclusive and advances the generation.

1. Acquire write access from the `GpuEpochQueue`.
2. Validate that no writer is already active.
3. Validate that no readers are active.  If readers are still live, acquiring a
   writer is a usage error; the queue should detect this rather than block.
4. Lease a stream from `CudaDeviceContext`.
5. Make that stream wait on the current queue readiness:
   - the previous `writer_event`, if the previous phase was a writer;
   - all published `reader_events`, if the previous phase was readers.
6. Launch write kernels or copies into the leased stream.
7. On publish, record one non-timing completion event into the stream.
8. Store that event as the queue's new `writer_event`.
9. Increment `generation`.
10. Clear old reader events and return the stream lease to the pool.

Conceptually:

```text
old epoch complete -> writer stream work -> writer_event
```

## Reader Workflow

A reader is shared and does not advance the generation.

1. Acquire read access from the `GpuEpochQueue`.
2. Validate that no writer is active.
3. Lease a stream from `CudaDeviceContext`.
4. Make that stream wait on the current `writer_event` for this generation.
5. Launch read-only kernels or copies into the leased stream.
6. On publish, record one non-timing completion event into the stream.
7. Append that event to the queue's `reader_events`.
8. Return the stream lease to the pool.

Multiple readers of the same generation all wait on the same writer completion
event and then run independently:

```text
writer_event -> reader_1 stream work -> reader_1_event
             -> reader_2 stream work -> reader_2_event
             -> reader_3 stream work -> reader_3_event
```

The next writer waits on all reader completion events:

```text
reader_1_event \
reader_2_event  -> next writer stream work -> next_writer_event
reader_3_event /
```

## RAII Access Handles

There are two access families.

Stream-owned handles:

- `GpuReadStream`
- `GpuWriteStream`

These are returned when the scheduler selects and owns the stream slot.  They are
RAII handles analogous to a mutex lock.  Destruction may publish completion and
release the stream slot because the completion point is well-defined: an event
recorded at the current tail of the owned stream slot.

Example behavior:

```text
GpuWriteStream::~GpuWriteStream():
  if active:
    record completion event on slot.stream
    publish event to GpuEpochQueue
    release handle
```

Explicit `publish()` should be available for early release; the destructor is the
safe fallback.

Event/external-stream handles:

- `GpuReadEvent`
- `GpuWriteEvent`

These are used when the caller already has a stream from elsewhere.  They do not
own the stream slot, so their destructor should not do CUDA work.  The caller
must explicitly publish a completion handle or event.  In debug builds, the
destructor should assert if the handle is dropped while still active.

Example behavior:

```text
GpuWriteEvent::~GpuWriteEvent():
  DEBUG_CHECK(!active)
```

This avoids hiding scheduling boundaries in destructors for externally managed
streams.

## Writer Acquisition Modes

Stream mode:

```text
GpuWriteStream acquireWriteStream(buffer)
```

The context/scheduler selects a stream slot, synchronizes that stream with the
buffer's existing writer/readers, increments the generation, and returns an RAII
handle.  When the handle publishes, the new writer generation is represented by
the completion event recorded in the owned stream.

External event mode:

```text
GpuWriteEvent acquireWriteEvent(buffer, external_stream)
```

The context computes the dependencies that `external_stream` must satisfy before
writing.  The caller enqueues work on the external stream and must explicitly
publish the completion handle or event.  The generation still increments during
acquisition, because a writer is always a new ordered generation.

## Reader Acquisition Modes

Stream mode:

```text
GpuReadStream acquireReadStream(buffer)
```

The context/scheduler selects a stream slot and synchronizes it with the latest
writer generation.  The reader does not increment the generation.  When the
handle publishes, it adds a reader completion handle to the buffer.

External event mode:

```text
GpuReadEvent acquireReadEvent(buffer, external_stream)
```

The context computes the dependency needed for `external_stream` to see the
latest writer.  The caller enqueues read-only work and explicitly publishes the
reader completion handle or event.

## Multi-Buffer Operations

Most real kernels operate on several buffers.  The acquisition API should handle
the whole operation rather than acquiring each buffer independently.

For an operation with read buffers `R` and write buffers `W`:

1. Acquire read access to all `R`.
2. Acquire write access to all `W`.
3. Select one stream slot from the device pool.
4. Enqueue all CUDA work for the operation.
5. Record one shared completion event for the operation.
6. Publish that event to all output writer generations.
7. Publish that event as a reader completion for read buffers if the operation
   read them.

If the operation has multiple outputs, those outputs should usually share the
same completion event.

## Open Questions

- Should reader completion events be coalesced further in multi-buffer access
  plans, or is one shared event per access plan sufficient?
- Should `GpuReadStream` publish on destruction unconditionally, or should debug
  builds require explicit publication to make scheduling boundaries visible?
- How much of this should be prototyped inside TensorContraction before the real
  uni20 CUDA scheduler exists?
- How should NCCL/MPI remote-storage dependencies map onto the same epoch model?
