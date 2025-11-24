# Async primitives

This document summarizes the building blocks that make up Uni20's asynchronous value system. It complements the scheduler
references by focusing on how values, tasks, and cancellation interact.

## Core containers

### Async<T>
- A move-only wrapper around a value of type `T` that coordinates access through `EpochQueue`.
- Construction options:
  - Default construction leaves the value uninitialized but ready for queued reads/writes.
  - In-place construction (`Async<T>{std::in_place, ...}`) forwards arguments to the stored `T`.
  - Deferred construction via `async::deferred` keeps the control block alive while delaying pointer installation.
- Access patterns:
  - `read()`, `mutate()`, and `write()` yield awaitable buffers that coordinate through epochs to maintain
    writer→reader ordering.
  - `emplace()` returns an `EmplaceBuffer` that performs placement-new into the storage, useful when `T` requires
    custom construction.
  - `get_wait()` blocks until pending writers finish, optionally driving a scheduler explicitly when deterministic
    progress is required.

### ReverseValue<T>
- Wraps an `Async<T>` for reverse-mode workflows. It prepends epochs so gradient writes can flow opposite to forward
  evaluation order without violating read/write sequencing.

## Tasks and scheduling

### AsyncTask
- A move-only wrapper around a coroutine handle produced by `AsyncTaskPromise`.
- Awaiting an `AsyncTask` transfers execution into the awaited coroutine and resumes the awaiting coroutine once the
  awaited coroutine completes.
- Ownership is reference-counted via the promise. Helper utilities such as `reschedule` and `make_sole_owner` enforce
  that only a single owner can hand control back to schedulers.

### Schedulers
- `DebugScheduler` and `TbbScheduler` satisfy the `IScheduler` interface. They pick up `AsyncTask` instances enqueued
  by buffers, epoch transitions, or explicit rescheduling requests.
- NUMA-aware schedulers can query and set preferred nodes via the task promise (`preferred_numa_node` and
  `set_preferred_numa_node`).

## Cancellation and error propagation

- When a coroutine cannot produce its value, `BasicAsyncTask::cancel_if_unwritten()` marks the handle so that the next
  resume conveys cancellation instead of normal completion. Awaiters test this flag to propagate cancellation to
  dependent tasks without resuming the coroutine body again.
- `BasicAsyncTask::release_handle()` and destructor paths destroy coroutines only when cancellation is intentional or
  the coroutine has finished, preventing dangling continuations.
- Readers resumed after a cancellation observe the cancellation flag and surface an error or exception rather than an
  uninitialized value.

## Buffer types

- `ReadBuffer<T>`: co_awaitable handle that yields a const reference once its epoch's writer finishes.
- `MutableBuffer<T>`: grants mutable access while preserving epoch ordering, allowing in-place updates.
- `WriteBuffer<T>` / `EmplaceBuffer<T>`: treat the storage as uninitialized until completion. These buffers mark the
  writer as required so readers can wait for completion before proceeding.

## Debugging helpers

- `NodeInfo` records DAG metadata (address, type name, global index) for values participating in async graphs. It is
  used by debug schedulers and visualization tools.
- `EpochQueue::NodeInfo` integration can be enabled with `UNI20_DEBUG_DAG` to trace epoch transitions and task
  rescheduling.
