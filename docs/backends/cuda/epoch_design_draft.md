# CUDA Buffer Completion Lowering

**Status:** typed CUDA buffers and scoped read/write guards are implemented by
`uni20::cuda::CudaBuffer<T>`, `uni20::cuda::ReadAccess<T>`, and
`uni20::cuda::WriteAccess<T>`. The filename is retained for existing links; this
document supersedes the earlier `GpuEpochQueue` proposal.

This note defines how an ordering already established by ordinary C++ control
flow or Uni20's async runtime is lowered to CUDA stream synchronization. CUDA
does not maintain a second epoch chain.

The implemented guards expose explicit release and enforce the live-access
contract described below.

Related notes:

- [CUDA Buffers](buffers.md) is the introductory guide for kernel and provider
  authors using this contract.
- [Async Runtime Model](../../async/runtime_model.md) defines `EpochQueue`,
  readers, writers, and causal readiness.
- [CUDA Runtime](runtime.md) defines streams, completion tokens, and the
  actually-idle stream pool.
- [Ordering and Backend Lowering](../../architecture/ordering_and_backend_lowering.md)
  defines the submission/completion split.
- [CUDA Kernel Dispatch](kernel_dispatch.md) covers future coroutine and
  provider-resource integration.

## Causal Contract

CUDA submission builds a forward-only partial order. A stream can wait for a
completion that has already been recorded, but this layer cannot register a
consumer before its producer exists. Out-of-order construction belongs to
`EpochQueue`, not to CUDA events.

Treat `CudaBuffer<T>` like an ordinary mutable value whose storage happens to
be on a GPU:

- any number of reads may overlap;
- a write must not overlap another write or any read;
- the buffer must not be moved or destroyed while an access guard refers to it;
- synchronous code establishes these rules through ordinary program order;
- `Async<CudaBuffer<T>>` establishes them through its CPU `ReadBuffer` and
  `WriteBuffer` epochs.

Releasing a CUDA access guard does not wait for the GPU operation to finish. It
records a completion at the current stream tail and publishes that token to the
buffer. A causally later operation can therefore be submitted immediately: its
stream waits asynchronously for the published completion even when the GPU has
a large backlog.

The caller owns access ordering:

- synchronous code submits conflicting accesses in causal program order;
- an async CUDA operation destroys or otherwise releases its CUDA scoped access
  guards before releasing the CPU `ReadBuffer` or `WriteBuffer` that made the
  operation runnable;
- `EpochQueue` therefore ensures that a writer is not submitted until every
  preceding reader has published its CUDA completion;
- concurrent reads are legal and need not be ordered with one another;
- concurrent conflicting accesses remain invalid and are not repaired by CUDA
  bookkeeping.

This means the CUDA buffer state represents only unfinished operations from the
already-ordered past. It needs no generation number, current epoch, runnable
queue, task wakeup mechanism, or value ownership.

## Current Objects

`cuda::DeviceResources`

: Owns one validated `Device`, an actually-idle `StreamPool`, and lazily
  constructed provider resources. The installed CUDA runtime normally owns one
  canonical instance per enrolled device. The resource set must outlive its
  buffers and streams.

`cuda::CudaBuffer<T>`

: A move-only owner of one typed CUDA allocation. Its raw device pointer
  is exposed only through scoped access guards. Its completion ledger consists
  of the latest exclusive-writer completion and reader completions submitted
  since that writer. Live guard counts validate ordinary read/write access
  rules; they do not order callers or make them wait. The buffer does not own an
  `EpochQueue` or task wakeup state. A per-buffer mutex protects short ledger
  snapshots and publication; independent buffers do not share it.

`cuda::ReadAccess<T>`

: A scoped read-only guard constructed by
  `buffer.read_synchronized_with(stream)`. Construction installs the
  latest-writer wait on `stream` and exposes `T const*`. Destruction records and
  publishes a reader completion at the stream tail.

`cuda::WriteAccess<T>`

: A scoped read/write guard constructed by
  `buffer.write_synchronized_with(stream)`. Construction waits on the latest
  writer and every unfinished reader, then exposes `T*`. Destruction records and
  publishes an exclusive-writer completion at the stream tail. In-place
  operations use one write guard; they do not add a separate read guard for the
  same buffer.

`cuda::Stream`

: A reference-counted lease of one `StreamPool` slot. Access guards keep a copy
  so they can record completions. The final stream reference schedules pool
  return after all queued work and access-completion events finish.

`cuda::Completion`

: An immutable shared token for one recorded non-timing CUDA event. Retaining a
  completion in buffer state does not share or reference-count the buffer
  itself.

## Access Rules

The buffer state is equivalent to:

```cpp
Completion writer_completion;
std::vector<Completion> reader_completions;
std::size_t live_read_guards = 0;
bool live_write_guard = false;
```

Each access guard owns one live host-side access token. Moving a guard transfers
that token and leaves the source inert. Explicit `release()` or destruction
returns the token after publishing the guard's device completion.

A read access:

1. rejects acquisition when `live_write_guard` is set;
2. increments `live_read_guards`;
3. waits its stream for `writer_completion`, when present;
4. does not wait for other readers;
5. appends its release completion to `reader_completions`, then decrements
   `live_read_guards`.

An exclusive write access:

1. rejects acquisition when `live_write_guard` is set or
   `live_read_guards != 0`;
2. sets `live_write_guard`;
3. waits its stream for `writer_completion` and every unfinished reader
   completion;
4. publishes itself as the new `writer_completion` and clears the prior reader
   completions;
5. clears `live_write_guard`.

These checks diagnose invalid host access; they do not introduce a reservation
queue. A conflicting acquisition never waits for another guard to release and
never suspends a coroutine. Correctly ordered code releases the predecessor
guard first, making its recorded completion immediately available to the
successor.

Completed readers are pruned opportunistically before their retained vector
needs to grow and when a writer snapshots its dependencies. This bounds event
retention for long-lived read-mostly inputs without introducing an epoch or
generation counter.

## Scoped Access

A typical operation is queued as:

```cpp
auto stream = resources.streams().acquire();
auto out = output.write_synchronized_with(stream);
auto a = lhs.read_synchronized_with(stream);
auto b = rhs.read_synchronized_with(stream);

launch_on(stream, out.data(), a.data(), b.data());

a.release();
b.release();
out.release();
```

Lexical destruction provides the same completion publication when explicit
early release is unnecessary.

Each access guard performs these construction steps:

1. Validate the supplied stream handle. The stream may belong to another device
   when the intended CUDA operation can legally use the allocation; obtaining
   synchronized access does not itself grant foreign-device pointer access.
2. Under the buffer state mutex, validate and acquire its live read or write
   token, then copy the predecessor completions required by the access.
3. Release the state mutex.
4. Enqueue waits for the copied predecessor completions.
5. Expose only the pointer permitted by the guard type.

Explicit release or guard destruction records one completion at the stream tail
and briefly locks the buffer state to publish that completion and return the
live access token. Publication and token return are one state transition: a
causally later conflicting acquisition sees either the live predecessor guard
or its recorded completion. The state mutex is not held
while the caller launches CUDA work. Independent operations and compatible
readers can therefore queue and execute concurrently.

The construction snapshot and release publication are intentionally not one
transaction. Their correctness follows from the causal contract: a conflicting
successor cannot enter acquisition until its predecessors have released, while
compatible readers do not need to observe one another.

The blocking `StreamPool::acquire()` path is the bring-up path for the blocking
CUDA submission channel. The non-blocking channel will suspend through a Uni20
scheduler while stream resources are unavailable, then use the same access
construction, launch, and release-publication rules. CUDA buffer access itself
never needs an awaiter: once the stream and any provider resources have been
acquired, `read_synchronized_with(stream)` and
`write_synchronized_with(stream)` perform only bounded host work.
Async CUDA lowering must use the non-blocking channel for resources that can be
unavailable; using blocking resource acquisition inside an async operation is a
scheduler-policy error, not an optimization choice.

## Failure And Cleanup

Scoped stream and buffer access are RAII resources.

- If dependency installation fails during guard construction, no pointer is
  exposed and the live access token is returned.
- If publication storage fails during guard destruction, the stream is
  synchronized before destruction continues. Omitting the completion is then
  safe because the device access has finished, and the live token can be
  returned without publishing a false dependency.
- Moving, resetting, or destroying a buffer with live access tokens is a
  fail-fast contract violation.
- Destroying the final `Stream` reference enqueues the stream-pool return
  callback at the current stream tail; the slot becomes idle only when that
  callback runs.
- Destroying a buffer waits for its writer and retained reader completions
  before `cudaFreeAsync` for stream-ordered allocations, or `cudaFree` for the
  synchronous fallback.
- CUDA cleanup failures remain fail-fast because destructors cannot safely use
  the recoverable error policy.

`CudaBuffer` uses the stream-ordered allocator when device capabilities allow
it, but higher-level tensor storage still needs the allocator policy described
in [Memory Allocation](memory_allocation.md), while preserving the same
completion and lifetime contract.

## Non-Goals

The current layer does not provide:

- Tensor or mdspan storage/accessor integration;
- typed element access or host `operator[]` for device memory;
- coroutine-aware stream acquisition;
- provider-handle or workspace acquisition;
- cross-device buffer operations;
- CUDA execution-error monitoring after successful host submission.

Those capabilities belong to later storage, scheduler, and backend checkpoints.
