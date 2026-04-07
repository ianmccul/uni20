# Async Runtime Model

This document explains the core semantics of `Async<T>` and epoch-based causality.

If you only remember one thing: Uni20 async correctness comes from epoch ordering,
not from manual locking in kernel code.

## Glossary

| Term | Meaning in Uni20 |
|---|---|
| coroutine | a resumable function that can suspend at `co_await` |
| task | one coroutine instance managed by a scheduler (`AsyncTask`) |
| scheduler | decides when *ready* tasks run (`DebugScheduler`, `TbbScheduler`, ...) |
| storage | where the `T` object lives (`shared_storage<T>`) |
| epoch | one "version step" of an `Async<T>` timeline: writer phase then reader phase |
| buffer | a typed capability to read or write at a specific epoch (`ReadBuffer<T>`, `WriteBuffer<T>`) |
| phase | internal epoch state (`Pending`, `Started`, `Writing`, `Reading`, `Finished`) |

## Design Intent

The runtime separates three concerns:

- storage lifetime (`shared_storage<T>`)
- causal ordering (`EpochQueue` / `EpochContext`)
- execution policy (`IScheduler` implementations)

This split is why the same async code can run deterministically with `DebugScheduler`
or concurrently with TBB schedulers.

## Core Objects and Ownership

| Object | Owns | Purpose |
|---|---|---|
| `Async<T>` | `shared_storage<T>`, `EpochQueue` handle | user-facing async value |
| `ReadBuffer<T>` | `EpochContextReader<T>` | read gate for one epoch |
| `WriteBuffer<T>` | `EpochContextWriter<T>` | write gate for one epoch |
| `AsyncTask` | coroutine handle ownership token | scheduler-managed coroutine lifetime |
| `EpochContext` | phase state + waiter sets | enforces writer/read transitions |

Key ownership fact:

- buffers keep queue/storage alive even if the originating `Async<T>` object is moved or destroyed

## `Async<T>` Construction States

`Async<T>()` and `Async<T>(args...)` are intentionally different.

| Construction form | Storage state | Initial epoch readability |
|---|---|---|
| `Async<T>()` | unconstructed | not readable until writer constructs value |
| `Async<T>(args...)` | constructed | readable immediately |

This is the major semantic change from older docs.

Why this exists:

- it avoids hidden default-construction of large `T` values
- it makes "has a value" an explicit event in the dataflow
- it mirrors `std::optional<T>`-style "constructed vs not constructed" semantics, but with shared ownership

## Epoch Flow

Each epoch proceeds through phases:

- `Pending`
- `Started`
- `Writing`
- `Reading`
- `Finished`

Operationally, think in this sequence:

1. writer(s) acquire write phase
2. active writer completes and releases
3. readers run
4. when readers drain, next epoch can advance

Causality guarantee:

- `writer_n -> readers_n -> writer_{n+1}`

You can think of it visually as:

```text
Epoch n:     Writing  ->  Reading  ->  Finished
                          |
                          v
Epoch n+1:   Pending  ->  Started  ->  Writing  -> ...
```

## What `co_await` Means Here

When you `co_await` a buffer:

- if the epoch is ready, coroutine continues immediately
- otherwise coroutine suspends, and the task is queued in epoch waiters
- epoch transition later re-schedules that task

The scheduler decides when resumed tasks run, but not whether ordering is legal.
Epoch logic defines legality.

Practical mental model:

- `co_await reader` means "do not run this line until the value is readable"
- `co_await writer = value` means "do not run this line until I am the active writer, then commit a value"

## Copy/Move Semantics of `Async<T>`

### Move

Move construction/assignment transfers handle ownership for storage + queue.
It does not copy values.

### Copy construction/assignment

These are value-level operations.

- copy construction schedules transfer from source value into a fresh destination timeline
- copy assignment resets destination timeline first, then schedules value transfer

This is deliberate: copying does not clone dependency graph internals.

## Waiting and Blocking

`wait()` blocks until value is readable.
`get_wait()` blocks until value is readable and returns the materialized value.

- under `DebugScheduler`, waiting helps drive runnable tasks and can emit deadlock diagnostics
- under TBB schedulers, waiting cooperates with TBB execution model

Blocking API is a bridge for thread-bound callers; coroutine code should prefer awaitable composition.

## Invariants You Can Rely On

- default `Async<T>` has unconstructed value until first construction path
- mutable write access (`co_await writer`) requires already-constructed storage
- `writer.emplace(...)` is always valid construction/reconstruction path
- proxy assignment (`co_await writer = value`) is also valid first-write construction path
- `take_release()` is the explicit "move out and release writer" path
- epoch ordering is deterministic regardless of scheduler execution order
- TaskRegistry state transitions are tracked at coroutine handle/promise level

## Common Consequences in Real Code

- first write can use `co_await writer = value`
- read-modify-write kernels should release read gates before awaiting conflicting writes
- deadlocks are typically dependency-shape bugs, not scheduler bugs

## Related References

- Buffer-level API details: `buffers_and_awaiters.md`
- Error and cancellation flow: `exceptions_and_cancellation.md`
- Scheduler behavior: `schedulers.md`
- Debug triage: `task_registry_debug.md`
- Fast lookup table: `quick_reference.md`
