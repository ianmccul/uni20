# GPU Epoch Design Draft

This is a draft design note.  It is intended to workshop the final uni20 GPU
dependency model before replacing or rewriting the existing CUDA scheduler notes.

The immediate implementation target is still single-threaded TensorContraction
integration.  The API and invariants should nevertheless be shaped so that the
same model can later be made thread-safe without changing semantics.

## Goals

- Track GPU buffer read/write ordering without eager per-buffer CUDA events.
- Let same-stream continuation use CUDA stream order with no event or wait.
- Materialize CUDA events only at real ordering boundaries: cross-stream handoff
  or stream-slot repossession.
- Preserve a clean path to a multi-threaded scheduler by making buffer access
  acquisition conceptually atomic.
- Keep the first implementation single-threaded and avoid mutex/locking
  machinery until it is needed.

## Analogy To CPU Epochs

The CPU async runtime has an `EpochQueue`/`EpochContext` model.  A logical value
has an ordered chain of epochs.  Readers attach to the readable epoch, writers
advance the value to a later epoch, and RAII handles release their participation.

The GPU model is similar, but CUDA changes the mechanics:

- CUDA stream order is itself an execution queue.
- CUDA events are concrete cross-stream readiness tokens.
- Multiple kernels may concurrently read the same device buffer.
- GPU writes should be ordered; we do not need the CPU model's unordered
  multi-writer accumulation mode.
- A GPU operation with multiple outputs naturally shares one completion point,
  because all outputs become valid at the same stream tail.

For this reason, the GPU model should keep an epoch context per buffer, but its
internal completion token should be "stream tail or event" rather than a CPU
coroutine latch.

## Core Objects

`GpuEpochContext`

The synchronization state for one GPU buffer.  It is closest in spirit to the
head of an `EpochQueue`: it tracks the latest writer generation and outstanding
reader generations for that buffer.

State:

- `generation`: monotonically incremented for each write acquisition.  This is
  useful for debugging even if CUDA events/streams provide the real ordering.
- `writer_done`: completion token for the latest writer generation.
- `readers_done`: coalesced completion tokens for outstanding readers of the
  current generation.

`GpuEpochToken`

An internal completion token.  It is shared by all buffer generations completed
by the same operation or coalesced batch.

States:

- `StreamTail(slot, slot_generation)`: the epoch is complete at the current tail
  of a stream slot, provided that stream slot has not been repossessed.
- `Event(cudaEvent_t)`: the stream tail has been materialized into a CUDA event.

`StreamSlot`

A reusable CUDA execution lane owned by a `CudaDeviceContext`.  It contains a
CUDA stream and device-local library handles such as cuBLAS.  A stream slot may
have one or more `GpuEpochToken`s parked on its tail.

`GpuAccessPlan`

The transaction returned by atomically acquiring read/write access for one GPU
operation.  It owns either a scheduler-selected stream slot or an externally
provided stream dependency contract.

## Access Rules

Reads are compatible with other reads:

- A read waits for the latest writer generation.
- A read does not wait for other readers.
- A read publishes a reader completion token so later writers know when it is
  safe to modify the buffer.
- A read does not increment the generation counter.

Writes are ordered and exclusive:

- A write waits for the latest writer generation.
- A write waits for all outstanding reader generations.
- A write increments the generation counter.
- A write clears or supersedes old reader generations.
- A write publishes the new writer completion token.

This gives the usual ordering table:

- read/read: compatible, no ordering required between readers.
- write/read: reader waits for prior writer.
- read/write: writer waits for outstanding readers.
- write/write: later writer waits for prior writer.

## Atomic Acquisition

The operation that must be conceptually atomic is access acquisition, not
`waitOn(event)`.

For a future multi-threaded scheduler, acquisition should be one critical
section over the relevant `GpuEpochContext`s and stream-slot state:

1. Snapshot writer and reader generations for all input/output buffers.
2. Select or accept a stream slot.
3. Repossess/finalize stream tails if the selected stream slot must be used for
   unrelated work.
4. Emit any required `cudaStreamWaitEvent` dependencies on the selected stream.
5. Reserve the new read/write intents and return an access handle.

The single-threaded TensorContraction implementation can perform these steps
without locks, but it should keep this transaction boundary in the API.

## Stream-Tail Tokens

The central optimization is that a completion token does not need to be a CUDA
event immediately.

If work completes at the tail of stream slot `S`, the produced generation can be
represented by:

```text
StreamTail(S)
```

If a later compatible operation also runs on `S`, no event and no wait are
required.  CUDA stream order is sufficient.

If a different stream must consume the result, or if `S` must be repossessed for
unrelated work, the token is finalized:

```text
Event(E) where E = cudaEventRecord(S)
```

All buffer generations sharing the same `GpuEpochToken` now share the same event.

## Stream Repossession

Stream repossession is the GPU analogue of work-stealing overhead.

A stream slot may be parked at the tail of one or more active `GpuEpochToken`s.
If the scheduler wants to reuse that stream slot for unrelated work, it must
first finalize all parked stream-tail tokens for that slot.  This records CUDA
events even if no future consumer eventually waits on them.  That cost is
required because, after unrelated work is enqueued, the original stream tail
position can no longer be recovered.

Invariant:

- A stream slot must not accept unrelated work until every active token parked on
  that stream slot has either been continued compatibly or finalized to an event.

## RAII Access Handles

There are two access families.

Stream-owned handles:

- `GpuReadStream`
- `GpuWriteStream`

These are returned when the scheduler selects and owns the stream slot.  They are
RAII handles analogous to a mutex lock.  Destruction may publish completion and
release the stream slot because the completion token is well-defined: the tail of
the owned stream slot.

Example behavior:

```text
GpuWriteStream::~GpuWriteStream():
  if active:
    publish StreamTail(slot)
    release handle
```

Explicit `publish()` should be available for early release; the destructor is the
safe fallback.

Event/external-stream handles:

- `GpuReadEvent`
- `GpuWriteEvent`

These are used when the caller already has a stream from elsewhere.  They do not
own the stream slot, so their destructor should not do CUDA work.  The caller
must explicitly publish a completion token or event.  In debug builds, the
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
handle.  When the handle publishes, the new writer generation is represented by a
virtual stream-tail token.

External event mode:

```text
GpuWriteEvent acquireWriteEvent(buffer, external_stream)
```

The context computes the dependencies that `external_stream` must satisfy before
writing.  The caller enqueues work on the external stream and must explicitly
publish the completion token or event.  The generation still increments during
acquisition, because a writer is always a new ordered generation.

## Reader Acquisition Modes

Stream mode:

```text
GpuReadStream acquireReadStream(buffer)
```

The context/scheduler selects a stream slot and synchronizes it with the latest
writer generation.  The reader does not increment the generation.  When the
handle publishes, it adds a reader completion token to the buffer.

External event mode:

```text
GpuReadEvent acquireReadEvent(buffer, external_stream)
```

The context computes the dependency needed for `external_stream` to see the
latest writer.  The caller enqueues read-only work and explicitly publishes the
reader completion token or event.

## Multi-Buffer Operations

Most real kernels operate on several buffers.  The acquisition API should handle
the whole operation rather than acquiring each buffer independently.

For an operation with read buffers `R` and write buffers `W`:

1. Acquire read access to all `R`.
2. Acquire write access to all `W`.
3. Select one stream slot, preferably one that allows same-stream continuation
   for the most dependencies.
4. Enqueue all CUDA work for the operation.
5. Publish one shared completion token to all output writer generations.
6. Publish reader completion tokens for read buffers if a later writer must wait
   for the reads.

If the operation has multiple outputs, those outputs should usually share the
same `GpuEpochToken`.

## Open Questions

- How should reader tokens be coalesced: one token per stream slot, one token per
  operation, or one shared token per access plan?
- Should `GpuReadStream` publish on destruction unconditionally, or only if a
  later writer actually needs a reader dependency?
- How should stream-slot compatibility be defined for continuing a parked stream
  tail: same output buffer, same access plan, or explicit scheduler decision?
- How much of this should be prototyped inside TensorContraction before the real
  uni20 CUDA scheduler exists?
- How should NCCL/MPI remote-storage dependencies map onto the same epoch model?
