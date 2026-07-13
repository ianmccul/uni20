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
| buffer | a typed capability for shared reading or exclusive mutation at a specific epoch (`ReadBuffer<T>`, `WriteBuffer<T>`) |
| await path | the temporary adaptor selected by expressions such as `maybe()`, `storage()`, or `take()` |
| access proxy | borrowed or owning result that exposes a ready value or its storage |
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
| `WriteBuffer<T>` | `EpochContextWriter<T>` | exclusive mutable gate for one epoch |
| `AsyncTask` | coroutine handle ownership token | scheduler-managed coroutine lifetime |
| `EpochContext` | phase state + waiter sets | enforces writer/read transitions |

Key ownership fact:

- buffers keep their storage and selected epoch context alive even if the originating `Async<T>` object is moved or destroyed
- async aliases retain their parent storage and use the parent's exact queue

## Buffers Are Capabilities

There are two ordinary buffer capabilities:

- `ReadBuffer<T>` owns shared read participation in one epoch.
- `WriteBuffer<T>` owns exclusive mutable participation in one epoch.

`Write` describes the exclusive producer role in the epoch protocol, not a
write-only memory permission. After awaiting a `WriteBuffer<T>`, a coroutine
may inspect and mutate an existing `T`, construct or replace an absent value,
or move the value out. Ordinary in-place mutation therefore needs one writer
and no separate reader for the same queue.

The several awaiter and proxy classes do not represent additional buffers.
They implement choices such as borrowed versus owning lifetime, optional read,
cancellation behavior, direct storage access, or consuming `take()`. Repeated
awaits on one buffer continue to use the same selected epoch.

For each queue used by an ordinary operation, the supported access set is:

- one or more readers and no writer, or
- one writer and no separate readers.

A reader and writer from the same queue select different epochs. Advanced code
can intentionally copy a read value, release that reader, and then await the
later writer, but this is a sequential two-epoch transition rather than
simultaneous read/write access.

## Queue Enrollment Threading Contract

`EpochQueue` is one causal timeline, not a concurrent registration data
structure. Calls that enroll a new access in that timeline must be externally
serialized. This includes `Async<T>::read()` and `Async<T>::write()` calls made
through the parent or through any alias sharing its queue.

Once a read or write buffer has been created, the buffer and its task may run
concurrently with other scheduled work. The selected `EpochContext` provides
the synchronization needed during execution. Concurrently mutating the queue
head to create buffers is outside the API contract; sharing a queue between
aliases does not broaden that contract.

Using one queue for a parent and its aliases is deliberately conservative. It
may serialize operations on disjoint slices, but it prevents two independent
timelines from authorizing conflicting access to the same storage. Finer-grain
parallelism requires explicit subrange hazard tracking rather than independent
queues over aliased bytes.

## Async<T> Construction States

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

1. the writer acquires the write phase
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

## What co_await Means Here

When you `co_await` a buffer:

- if the epoch is ready, coroutine continues immediately
- otherwise coroutine suspends, and the task is queued in epoch waiters
- epoch transition later re-schedules that task

The scheduler decides when resumed tasks run, but not whether ordering is legal.
Epoch logic defines legality.

Practical mental model:

- `co_await reader` means "do not run this line until the value is readable"
- `co_await writer = value` means "do not run this line until I am the active writer, then commit a value"

## Copy/Move Semantics of Async<T>

### Move

Move construction/assignment transfers handle ownership for storage + queue.
It does not copy values.

### Copy construction/assignment

These are value-level operations for ordinary values.

- copy construction schedules transfer from source value into a fresh destination timeline
- copy assignment resets destination timeline first, then schedules value transfer

This is deliberate: copying does not clone dependency graph internals.

`async_value_kind_of<T>` classifies ordinary payloads as
`async_value_kind::value`. Alias descriptor types declare `async_alias_tag`,
which selects `async_value_kind::shared_alias`. Copying
`Async<AliasDescriptor>` is then a structural handle copy: storage, lifetime
owner, and epoch queue remain shared. This exception is necessary because
copying an alias onto a fresh timeline would lose ordering with the bytes it
references.

## Waiting and Blocking

`wait()` blocks until value is readable.
`get_wait()` blocks until value is readable and returns the materialized value.

- under `DebugScheduler`, waiting helps drive runnable tasks and can emit deadlock diagnostics
- under TBB schedulers, waiting cooperates with TBB execution model

Blocking API is a bridge for thread-bound callers; coroutine code should prefer awaitable composition.

## Invariants You Can Rely On

- default `Async<T>` has unconstructed value until first construction path
- dereferencing or converting a write proxy to `T&` requires already-constructed storage
- `writer.emplace(...)` is always valid construction/reconstruction path
- proxy assignment constructs empty storage and otherwise evaluates
  `stored_value = source`
- `take_release()` is the explicit "move out and release writer" path
- epoch ordering is deterministic regardless of scheduler execution order
- TaskRegistry state transitions are tracked at coroutine handle/promise level

## Common Consequences in Real Code

- first write can use `co_await writer = value`
- read-modify-write kernels use one `WriteBuffer` for exclusive access
- separate read and write buffers on one queue require an explicit two-epoch algorithm and early reader release
- deadlocks are typically dependency-shape bugs, not scheduler bugs

## Related References

- Buffer-level API details: `buffers_and_awaiters.md`
- Error and cancellation flow: `exceptions_and_cancellation.md`
- Scheduler behavior: `schedulers.md`
- Debug triage: `task_registry_debug.md`
- Fast lookup table: `quick_reference.md`
