# Async primitives

This document summarizes the building blocks that make up Uni20's asynchronous value system. It complements the scheduler
references by focusing on how values, tasks, and cancellation interact. The async layer is intentionally generic—`Async<Tensor>`
is the primary target, but `Async<T>` can wrap any type and schedule operations that run as coroutines. Existing helpers already
enqueue async math on their underlying objects, and new operations fit by writing small coroutines that return
[`AsyncTask`](../src/async/AsyncTask.hpp). The main purpose of `Async<T>` is to simplify writing multi-threaded asynchronous code by tracking dependencies and timelines and synchronization without the need for explicit threading and locks.

## Key classes
- An [`Async<T>`](../src/async/Async.hpp) is the handle that sequences every operation through an `EpochQueue` and a scheduler
  instead of raw threads. Each call to `read()`, `write()`, or `mutate()` obtains a buffer awaitable that enqueues the work.
- [`AsyncTask`](../src/async/AsyncTask.hpp) is a wrapper for a coroutine handle. An `AsyncTask` can be *scheduled* via a scheduler. It operates on **Buffer** objects (see below) that manage operations on the objects contained in an `Async<T>`. An `AsyncTask` has an ownership model, such that it owns the coroutine handle, and can transfer ownership of that handle to the scheduler. It can also manage cancellation of coroutines, and exceptions.
- [`ReverseValue<T>`](../src/async/ReverseValue.hpp) is similar to `Async<T>` but it reverses the direction of time: you can read the value (and pass it on to other async operations) before that value is available.
- [`Dual<T>`](../src/async/Dual.hpp) carries forward and reverse `Async` values as a pair for reverse-mode autodifferentiation.

## Epoch Queue

The `EpochQueue` manages sequencing and time ordering of operations on an `Async<T>`. Epochs are implicitly numbered, and execution moves forwards in time. A single Epoch consists of a *writer* task, and any number of *readers*. Once the writer task has completed, the readers are scheduled (in parallel).

Once all of the readers at that epoch have finished, the `EpochQueue` moves onto the next Epoch, and schedules the writer, and so on. We can think of an `EpochQueue` as managing reads and writes to one specific object (which will typically be a tensor, but could be anything). When the final reader of an epoch signals completion, the `EpochQueue` advances to the next epoch and enqueues its writer for scheduling. There is no explicit dependency graph. Instead, causal dependencies are enforced *implicitly* by coroutine suspension and resumption:
 * When a coroutine `co_await`s a buffer, it suspends until the required value is ready.
 * When the awaited buffer becomes ready, the coroutine is resumed by the scheduler.

This design yields the same causal guarantees as a DAG scheduler but with much lower overhead and simpler semantics. The dependencies are not specified at task creation, but arise naturally from the points in the coroutine where `co_await` is invoked. Each `co_await` expresses a dependency that will suspend execution until its prerequisite value is ready.

### Async<T> overview
- A wrapper around a value of type `T` that coordinates access through `EpochQueue`.
- Construction options:
  - Conversion constructor to initialize an `Async<T>` from an existing object of type `T`, or something implicitly or explicitly convertible to `T`.
  - In-place construction (`Async<T>{x, ...}`) forwards multiple arguments to the `T` constuctor.
  - Default construction leaves the value unconstructed.
- Copy construction schedules a copy of the value rather than cloning the async timeline:
  - `Async(const Async&)` creates a fresh storage buffer and epoch queue, then enqueues an asynchronous read of the
    source value followed by a write into the new instance's initial epoch.
  - The copy does **not** duplicate epoch histories, coroutine handles, or other stateful dependencies; ownership of
    storage is not shared between the two instances.
  - Equivalently, Use `async_assign` when you want to explicitly schedule the copy of an `Async<T>`'s value.
- Access patterns:
  - `read()`, `mutate()`, and `write()` yield awaitable buffers (`ReadBuffer<T>` and `WriteBuffer<T>`) that coordinate through epochs to maintain
    writer→reader ordering.
  - `co_await buffer.emplace(...)` performs placement-new into storage. This enables delayed initialization of `Async<T>` for types that do not have a default constructor.
  - `get_wait()` blocks until pending writers finish, optionally driving a scheduler explicitly when deterministic
    progress is required.

## Tasks and scheduling

### **AsyncTask**

An `AsyncTask` represents a coroutine handle that can be scheduled, awaited, or destroyed according to RAII ownership rules. It is the fundamental executable unit in Uni20’s asynchronous layer. `AsyncTask` is an alias for a `BasicAsyncTask<BasicAsyncTaskPromise>`, it is expected in the future that there will be other variants, such as a `GPUTask` and so on.

* `AsyncTask` is a **move-only** type wrapping a coroutine handle produced by an `AsyncTaskPromise`.
* Holding an `AsyncTask` means **owning** the coroutine handle. Submitting it to a scheduler **transfers ownership** to that scheduler.
* Implementation-wise, the ownership is *implemented* with a reference count in the promise object, but this count is normally `1`. In some cases it can be larger than 1, but this is not important in typical use.
* As an advanced technique, it is possible to `co_await` on an `AsyncTask`. In this case, executation is immediately transferred into the awaited coroutine. When that coroutine completes, control returns to the outer coroutine. This enables nesting of tasks.

### Schedulers
- `DebugScheduler` and `TbbScheduler` satisfy the `IScheduler` interface. They pick up `AsyncTask` instances enqueued
  by buffers, epoch transitions, or explicit rescheduling requests.
- NUMA-aware schedulers can query and set preferred nodes via the task promise (`preferred_numa_node` and
  `set_preferred_numa_node`).

## Cancellation and error propagation

- `BasicAsyncTask::cancel_if_unwritten()` sets a cancellation flag on the promise that indicates that it is expected that the `BasicAsyncTask::written()` function will be called on it; otherwise the task will be cancelled. Awaiters observe this flag on their next resume and propagate cancellation without re-entering the coroutine body. This is how a writer can exit early before touching the storage.
- Awaiters resumed after a cancellation see the cancellation flag and surface an error or exception instead of producing
  an uninitialized value. Exception plumbing is still maturing, but the expectation is that stored exceptions propagate
  along the same await-resume path once hook points are fully wired.

## Buffer types

asynchronous coroutines work by obtaining a *buffer*, which is an *awaitable* object that gives direct access to the object of an `Async<T>`. The buffer is obtained from member functions of `Async<T>`, `read()`, `write()`, and `mutate()`.

The buffer object is typically passed into a coroutine (by value, so it exists on the coroutine stack frame), and is used in a `co_await` expression to return a reference to the underlying object. This acts as a synchronization point: if the object is not ready for access in the task dependency graph, the coroutine will suspend and only resume once all of the dependencies have finished. So when the `co_await` has returned, it is safe to access the object.

- `read()` returns a `ReadBuffer<T>`. `co_await` on a `ReadBuffer<T>` gives either a value or a const reference to the object.
- `write()` and `mutate()` both return a `WriteBuffer<T>`. `mutate()` additionally requires the value to already be initialized.
- `co_await buffer.emplace(arg1, arg2, ...)` in-place constructs (or reconstructs) the value in the target storage.

## Compound awaiters

Uni20 defines several helper awaiters that extend how `co_await` can be used inside coroutines. These utilities live in [`awaiters.hpp`](../src/async/awaiters.hpp) and provide higher-level control and composition patterns on top of the core `Async<T>` and buffer primitives. The usual way to use these is via helper functions.

### `AllAwaiter`

* `AllAwaiter` waits for **all** of the provided awaitables to complete before resuming the coroutine.
    It generalizes `co_await` to a tuple of awaitables and is used by the async task factory functions when a coroutine must wait for multiple dependencies simultaneously.
* `co_await all(a, b, c)` suspends until all `a`, `b`, and `c` are ready, then resumes and returns a tuple of their results (or references).

Each child awaiter is awaited independently; the parent resumes only when every sub-awaitable reports readiness. This is more efficient than waiting on N different buffers separately, since the coroutine only suspends once.

### `TryAwaiter`

* `try_await()' Wraps another awaiter and returns an `std::optional<T>` result. `co_await`ing on a `TryAwaiter` does not suspend, but tests to see if the buffer is available immediately. If the buffer is ready the value is returned in the `std::optional{value}`. Otherwise, if the buffer is not yet ready it produces an empty value.
* This allows non-blocking “try” semantics — e.g., attempt to read a buffer if ready, otherwise skip or fallback. A typical use is where the coroutine needs to wait for two or more objects, and it can usefully do some preprocessing work as soon as it has any of the objects, and we don't know in which order they will become ready.

Usage:

```cpp
  auto maybe_val = co_await try_await(buffer);
  if (maybe_val) use(*maybe_val);
```

### `WriteToAwaiter`

* Performs a write of a value into a `WriteBuffer<T>` or compatible awaitable.
  It is constructed with a target buffer and a value (or reference) and, when awaited, ensures the write happens only when the buffer is ready.
* this is a safer way to use a `WriteBuffer`.

Helper:

```cpp
co_await write_to(write_buffer, std::move(value));
```

## See also
- Buffer types: [`ReadBuffer`](../src/async/buffers.hpp), [`WriteBuffer`](../src/async/buffers.hpp).
- Scheduler overview: [`docs/Scheduler.md`](./Scheduler.md).
- Debug DAG tracing: enable `UNI20_DEBUG_DAG` to emit `NodeInfo` edges via `EpochQueue`.
