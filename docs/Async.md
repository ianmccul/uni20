# Async primitives

This document summarizes the building blocks that make up Uni20's asynchronous value system. It complements the scheduler
references by focusing on how values, tasks, and cancellation interact. The async layer is intentionally generic—`Async<Tensor>`
is the primary target, but `Async<T>` can wrap any type and schedule operations that run as coroutines. Existing helpers already
enqueue async math on their underlying objects, and new operations fit by writing small coroutines that return
[`AsyncTask`](../src/async/AsyncTask.hpp). The main purpose of `Async<T>` is to simplify writing multi-threaded asyncronous code by tracking dependencies and timelines and syncronization without the need for explicit threading and locks.

## Key classes
- An [`Async<T>`](../src/async/Async.hpp) is the handle that sequences every operation through an `EpochQueue` and a scheduler
  instead of raw threads. Each call to `read()`, `write()`, or `mutate()` obtains a buffer awaitable that enqueues the work.
- [`AsyncTask`](../src/async/AsyncTask.hpp) is a wrapper for a coroutine handle. An `AsyncTask` can be *scheduled* via a scheduler. It operates on **Buffer** objects (see below) that manage operations on the objects contained in an `Async<T>`. An `AsyncTask` has an ownership model, such that it owns the coroutine handle, and can transfer ownership of that handle to the scheduler. It can also manage cancellation of coroutines, and exceptions.
- [`ReverseValue<T>`](../src/async/ReverseValue.hpp) is similar to `Async<T>` but it reverses the direction of time: you can read the value (and pass it on to other async operations) before that value is available.
- [`Dual<T>`](../src/async/Dual.hpp) carries forward and reverse `Async` values as a pair for reverse-mode autodifferentiation.

## Epoch Queue

The `EpochQueue` manages sequencing and time ordering of operations on an `Async<T>`. Epochs are implicitly numbered, and execution moves forwards in time. A single Epoch consists of a *writer* task, and any number of *readers*. Once the writer task has completed, the readers are scheduled (in parallel). Once all of the readers at that epoch have finished, the `EpochQueue` moves onto the next Epoch, and schedules the writer, and so on. We can think of an `EpochQueue` as managing reads and writes to one specific object (which will typically be a tensor, but could be anything).

Execution proceeds as a sequence of barriers: once the writer of an epoch completes, its readers are eligible for scheduling; once all readers complete, the queue advances to the next epoch. There is no explicit dependency graph. Instead, causal dependencies are enforced *implicitly* by coroutine suspension and resumption:
 * When a coroutine `co_await`s a buffer, it suspends until the required value is ready.
 * When the awaited buffer becomes ready, the coroutine is resumed by the scheduler.

This design yields the same causal guarantees as a DAG scheduler but with much lower overhead and simpler semantics. The dependencies do not need to be specified at the construction of each task, but are specified by coroutine itself.

### Async<T> overview
- A wrapper around a value of type `T` that coordinates access through `EpochQueue`.
- Construction options:
  - Conversion constructor to intialize an `Async<T>` from an existing object of type `T`, or something implicitly or explicitly convertible to `T`.
  - In-place construction (`Async<T>{std::in_place, ...}`) forwards arguments to the stored `T`, in the same way as in-place construction for `std::optional` and other `std` classes.
  - Default construction leaves the value unconstructed.
- Copy construction schedules a copy of the value rather than cloning the async timeline:
  - `Async(const Async&)` creates a fresh storage buffer and epoch queue, then enqueues an asynchronous read of the
    source value followed by a write into the new instance's initial epoch.
  - The copy does **not** duplicate epoch histories, coroutine handles, or other stateful dependencies; ownership of
    storage is not shared between the two instances.
  - Equivalently, Use `async_assign` when you want to explicitly schedule the copy of an `Async<T>`'s value.
- Access patterns:
  - `read()`, `mutate()`, and `write()` yield awaitable buffers that coordinate through epochs to maintain
    writer→reader ordering.
  - `emplace()` returns an `EmplaceBuffer` that performs placement-new into the storage. This enables delayed initialization of `Async<T>` for types that do not have a default constructor.
  - `get_wait()` blocks until pending writers finish, optionally driving a scheduler explicitly when deterministic
    progress is required.

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

- `BasicAsyncTask::cancel_if_unwritten()` sets a cancellation flag on the promise that indicates that it is expected that the `BasicAsyncTask::written()` function will be called on it; otherwise the task will be cancelled.
  Awaiters observe this flag on their next resume and propagate cancellation without re-entering the
  coroutine body. This is how a writer can exit early before touching the storage.
- Coroutine destruction depends on exclusive ownership of the handle, not on whether cancellation occurred. Destroying
  an `AsyncTask` drops one reference to the coroutine; the coroutine frame is destroyed only after the last owner
  releases it.
- `BasicAsyncTask::release_handle()` transfers ownership of the coroutine handle to the caller when they are the sole
  owner, enabling manual lifetime management when cooperative scheduling or inspection is required.
- Awaiters resumed after a cancellation see the cancellation flag and surface an error or exception instead of producing
  an uninitialized value. Exception plumbing is still maturing, but the expectation is that stored exceptions propagate
  along the same await-resume path once hook points are fully wired.

## Buffer types

Asyncronous coroutines work by obtaining a *buffer*, which is an *awaitable* object that gives direct access to the object of an `Async<T>`. The buffer is obtained from member functions of `Async<T>`, `read()`, `write()`, `mutate()`, `emplace()`.

The buffer object is typically passed into a coroutine (by value, so it exists on the coroutine stack frame), and is used in a `co_await` expression to return a reference to the underlying object. This acts as a syncronization point: if the object is not ready for access in the task dependency graph, the coroutine will suspend and only resume once all of the dependencies have finished. So when the `co_await` has returned, it is safe to access the object.

- `read()` returns a `ReadBuffer<T>`. `co_await` on a `ReadBuffer<T>` gives either a value or a const reference to the object.
- `mutate()` returns a `MutableBuffer<T>`. `co_await` on a `MutableBuffer<T>` returns a reference to the object, that can be read or modified.
- `write()` returns a `WriteBuffer<T>`. This returns a reference to the value, but in principle should only be written, not read from (eg it might be in some moved-from state).
- `emplace()` returns `EmplaceBuffer<T>`. This is used to in-place construct the object by passing the constructor parameters to the `co_await` statement. The syntax is `co_await std::move(buffer)(arg1, arg2, ...)`. We need to `move` the buffer because this is a once-only operation, once the object is constructed the buffer is in an invalid moved-from state.

## Debugging helpers

- `NodeInfo` records DAG metadata (address, type name, global index) for values participating in async graphs. It is
  used by debug schedulers and visualization tools.
- `EpochQueue::NodeInfo` integration can be enabled with `UNI20_DEBUG_DAG` to trace epoch transitions and task
  rescheduling.

## See also
- Buffer types: [`ReadBuffer`](../src/async/buffers/ReadBuffer.hpp), [`MutableBuffer`](../src/async/buffers/MutableBuffer.hpp),
  [`WriteBuffer`](../src/async/buffers/WriteBuffer.hpp), [`EmplaceBuffer`](../src/async/buffers/EmplaceBuffer.hpp).
- Scheduler overview: [`docs/Scheduler.md`](./Scheduler.md).
- Debug DAG tracing: enable `UNI20_DEBUG_DAG` to emit `NodeInfo` edges via `EpochQueue`.
