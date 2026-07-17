# CUDA Buffer Completion Lowering

**Status:** typed CUDA buffers and scoped read/write guards are implemented by
`uni20::cuda::Buffer<T>`, `uni20::cuda::ReadBuffer<T>`, and
`uni20::cuda::WriteBuffer<T>`. The filename is retained for existing links; this
document supersedes the earlier `GpuEpochQueue` proposal.

This note defines how an ordering already established by ordinary C++ control
flow or Uni20's async runtime is lowered to CUDA stream synchronization. CUDA
does not maintain a second epoch chain.

Related notes:

- [Async Runtime Model](../../async/runtime_model.md) defines `EpochQueue`,
  readers, writers, and causal readiness.
- [CUDA Runtime](runtime.md) defines streams, completion tokens, and the
  actually-idle stream pool.
- [Ordering and Backend Lowering](../../architecture/ordering_and_backend_lowering.md)
  defines the submission/completion split.
- [CUDA Kernel Dispatch](kernel_dispatch.md) covers future coroutine and
  provider-resource integration.

## Causal Contract

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

`cuda::DeviceContext`

: Owns one validated `Device`, an actually-idle `StreamPool`, and a mutex for
  short buffer-state snapshots and publication. The mutex is not held while a
  backend or provider call executes. The context must outlive its buffers and
  streams.

`cuda::Buffer<T>`

: A move-only owner of one typed `cudaMalloc` allocation. Its raw device pointer
  is exposed only through scoped access guards. Its private state consists of
  the latest exclusive-writer completion and reader completions submitted since
  that writer. It does not own an `EpochQueue` or task wakeup state.

`cuda::ReadBuffer<T>`

: A scoped read-only guard constructed by `buffer.read(stream)`. Construction
  installs the latest-writer wait on `stream` and exposes `T const*`. Destruction
  records and publishes a reader completion at the stream tail.

`cuda::WriteBuffer<T>`

: A scoped read/write guard constructed by `buffer.write(stream)`. Construction
  waits on the latest writer and every unfinished reader, then exposes `T*`.
  Destruction records and publishes an exclusive-writer completion at the stream
  tail. In-place operations use one write guard; they do not add a separate read
  guard for the same buffer.

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
```

A read access:

1. waits for `writer_completion`, when present;
2. does not wait for other readers;
3. appends its guard-destruction completion to `reader_completions`.

An exclusive write access:

1. waits for `writer_completion` and every unfinished reader completion;
2. publishes itself as the new `writer_completion`;
3. clears the prior reader completions.

Completed readers are pruned opportunistically before their retained vector
needs to grow and when a writer snapshots its dependencies. This bounds event
retention for long-lived read-mostly inputs without introducing an epoch or
generation counter.

## Scoped Access

A typical operation is queued as:

```cpp
auto stream = context.streams().acquire();
{
  auto out = output.write(stream);
  auto a = lhs.read(stream);
  auto b = rhs.read(stream);

  launch_on(stream, out.data(), a.data(), b.data());
}
```

Each access guard performs these construction steps:

1. Validate that the stream and buffer belong to the same CUDA device.
2. Under the context state mutex, copy the predecessor completions required by
   the read or write access.
3. Release the state mutex.
4. Enqueue waits for the copied predecessor completions.
5. Expose only the pointer permitted by the guard type.

Each guard destructor records one completion at the stream tail and briefly
locks the context state to publish that completion. Neither the state mutex nor
any buffer lock is held while the caller launches CUDA work. Independent
operations and compatible readers can therefore queue and execute concurrently.

The construction snapshot and destruction publication are intentionally not one transaction. Their
correctness follows from the causal contract: a conflicting successor cannot
enter acquisition until its predecessors have published, while compatible
readers do not need to observe one another.

The blocking `StreamPool::acquire()` path is the bring-up path for
synchronous-looking CUDA backends. A future coroutine awaiter will suspend while
stream resources are unavailable, then use the same access construction, launch,
and guard-destruction publication rules.

## Failure And Cleanup

Scoped stream and buffer access are RAII resources.

- If dependency installation fails during guard construction, no pointer is
  exposed.
- If publication storage fails during guard destruction, the stream is
  synchronized before destruction continues. Omitting the completion is then
  safe because the device access has finished.
- Destroying the final `Stream` reference enqueues the stream-pool return
  callback at the current stream tail; the slot becomes idle only when that
  callback runs.
- Destroying a buffer waits for its writer and retained reader completions
  before `cudaFree`.
- CUDA cleanup failures remain fail-fast because destructors cannot safely use
  the recoverable error policy.

The first `Buffer` uses `cudaMalloc`/`cudaFree` to establish semantics. Hot-path
storage should later use the stream-ordered allocator described in
[Memory Allocation](memory_allocation.md), while preserving the same completion
and lifetime contract.

## Non-Goals

The current layer does not provide:

- Tensor or mdspan storage/accessor integration;
- typed element access or host `operator[]` for device memory;
- coroutine-aware stream acquisition;
- provider-handle or workspace acquisition;
- cross-device buffer operations;
- CUDA execution-error monitoring after successful host submission.

Those capabilities belong to later storage, scheduler, and backend checkpoints.
