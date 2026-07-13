# Buffers, Await Paths, and Access Proxies

This document is the practical reference for `ReadBuffer<T>` and `WriteBuffer<T>`.

Use this when implementing kernels or reviewing coroutine correctness.

## The Short Version

`Async<T>` has only two ordinary access capabilities:

- `ReadBuffer<T>`: shared read access to one epoch
- `WriteBuffer<T>`: exclusive mutable access to one epoch

`Write` names the buffer's exclusive producer role in the epoch protocol; it
does not mean write-only memory access. Once its epoch is active, a
`WriteBuffer<T>` may inspect and mutate the existing value, construct an absent
value, replace the value, or move the value out. An in-place operation therefore
uses one `WriteBuffer<T>`, not a `ReadBuffer<T>` followed by a `WriteBuffer<T>`
for the same timeline.

Types such as `ReadMaybeAwaiter`, `OwningWriteAwaiter`, and `TakeAwaiter` are
temporary await-path adaptors. They change what an await returns or who retains
the epoch handle across suspension. They are not additional buffer classes and
do not create additional epochs.

The API has three layers:

| Layer | Examples | Role |
|---|---|---|
| access capability | `ReadBuffer<T>`, `WriteBuffer<T>` | selects an epoch and owns its reader/writer participation |
| await path | plain `co_await`, `maybe()`, `or_cancel()`, `storage()`, `take()` | chooses readiness, cancellation, ownership, or consumption behavior |
| access result | `T const&`, `WriteAccessProxy<T>`, owning proxies | exposes the value or storage after the epoch becomes ready |

### Why there is one WriteBuffer type

The writer handle and its exception-sink registration belong to one selected
epoch. Value access, storage access, and consuming the value are different ways
to await that same handle, not different scheduling capabilities. Keeping one
`WriteBuffer<T>` makes that identity explicit and avoids duplicating lifetime
and exception-routing logic across several buffer types.

The write side can be read as this tree:

```text
WriteBuffer<T>                         one writer capability
  co_await writer                      borrowed WriteAccessProxy<T>
  co_await writer.transfer()           owning OwningWriteAccessProxy<T>
  co_await writer.storage()            borrowed shared_storage<T>&
  co_await writer.transfer().storage() owning storage proxy
  co_await writer.take()               consume value, buffer retains writer
  co_await writer.take_release()       consume value and release writer
```

The awaiter classes are the short-lived objects between the expression on the
left and the result on the right. Most application and kernel code should not
name their concrete types.

### One synchronization domain per operation

For one `EpochQueue`, an ordinary operation should acquire either:

- any number of `ReadBuffer<T>` capabilities and no writer, or
- one `WriteBuffer<T>` capability and no separate reader.

Multiple reads are compatible. A writer is exclusive and also supplies the
read access needed for mutation. A separate read and write on the same queue
represent two different epochs; they are not a way to obtain ordinary
read/write access simultaneously. Some low-level code deliberately performs
such a two-epoch transition by copying what it needs and releasing the reader
before awaiting the writer, but tensor kernels should not use that as their
default mutation pattern.

## ReadBuffer<T>

`ReadBuffer<T>` owns reader participation in one selected epoch. The modifiers
below choose different await paths through that same capability.

### Lifetime and ownership notes

- copy-constructible
- movable
- copy-assignment deleted
- move-assignment available
- `transfer()` is the explicit named-buffer path to owning access

### Await behavior

| Form | Return type | Notes |
|---|---|---|
| `co_await reader` | `T const&` | non-owning read access |
| `co_await reader.transfer()` | `OwningReadAccessProxy<T>` | owning read access (`get()`, `get_release()`, `release()`) |

### Read modifiers

| Modifier | Return behavior | Typical use |
|---|---|---|
| `co_await reader.maybe()` | `T const*` (nullable) | probe optional/cancelled paths without exception |
| `co_await reader.transfer().maybe()` | `std::optional<OwningReadAccessProxy<T>>` | ownership-preserving optional read |
| `co_await reader.or_cancel()` | `T const&` or throws `task_cancelled` | cancellation-aware borrowed read |
| `co_await reader.transfer().or_cancel()` | `OwningReadAccessProxy<T>` or throws `task_cancelled` | cancellation-aware owning read |

### Early release

`reader.release()` (or owning proxy `release()`) releases the reader gate before object destruction.

Use it to shorten a read lifetime after copying everything the coroutine needs.
It is mandatory only for advanced code that deliberately transitions from a
read epoch to a later write epoch on the same queue.

## WriteBuffer<T>

`WriteBuffer<T>` owns exclusive mutable participation in one write epoch. It
does not imply that the stored `T` already exists: the write epoch may either
mutate a constructed value or construct/replace the value.

### Lifetime and ownership notes

- move-only
- not copyable
- lvalue `co_await` returns a non-owning write proxy
- rvalue `co_await` returns an owning write proxy
- `transfer()` is the explicit named-buffer path to owning access

### Mutation uses one WriteBuffer

For an existing value, await the writer and use the returned proxy as exclusive
read/write access:

```cpp
auto increment = [](WriteBuffer<int> value) static -> AsyncTask {
  auto access = co_await value;
  access.get() += 1;
  co_return;
};
```

Do not acquire a separate `ReadBuffer` for the same `Async<T>` merely to inspect
the old value. For operations involving tensor aliases, either derive a safe
view while holding the one writer, materialize an input, or reject the aliasing
case according to that operation's contract.

### Await behavior

| Form | Return type | Notes |
|---|---|---|
| `co_await writer` | `WriteAccessProxy<T>` | non-owning proxy; convertible to `T&` |
| `co_await writer.transfer()` | `OwningWriteAccessProxy<T>` | owning proxy |
| `co_await writer.storage()` | `shared_storage<T>&` | explicit storage control |
| `co_await writer.transfer().storage()` | `OwningStorageAccessProxy<T>` | owning storage access |
| `co_await writer.take()` | `T` | move out + destroy stored value |
| `co_await writer.transfer().take()` | `T` | move out + destroy stored value, then release writer |
| `co_await writer.take_release()` | `T` | move out + destroy stored value, then release writer |
| `co_await writer.transfer().take_release()` | `T` | owning shorthand for take + release |

The named awaiter classes implementing these rows are transport objects. For
example, `writer.storage()` creates a `StorageAwaiter`, while
`writer.transfer().storage()` creates an `OwningStorageAwaiter`. User code
normally names the returned storage proxy, not the awaiter itself.

The lvalue forms borrow the writer held by `WriteBuffer<T>`. The `transfer()`
forms move that writer through the awaiter into an owning result, making the
result independent of the original buffer object's lifetime.

### Initialization rule

For default `Async<T>`, first write is safe through proxy assignment:

```cpp
co_await writer = value;
```

This calls proxy `operator=`, which uses `emplace(...)` semantics internally when needed.

### Assignment semantics trait

Write-proxy assignment is type-driven via:

- `uni20::async::assignment_semantics_of<T>`
- `uni20::async::assignment_semantics_v<T>`

Default behavior is `assignment_semantics::rebind`:

- `co_await writer = rhs` reconstructs/rebinds the stored object (`emplace(...)` path).

Types can opt into `assignment_semantics::write_through` by specialization:

```cpp
namespace uni20::async {
template <>
struct assignment_semantics_of<MyProxyType>
    : std::integral_constant<assignment_semantics, assignment_semantics::write_through> {};
}
```

For `write_through` types:

- `co_await writer = rhs` assigns through the existing object
- storage must already be constructed
- use `proxy.rebind(...)` (or `proxy.emplace(...)`) for explicit retarget/reconstruction

### Why += / -= can initialize

Uni20 intentionally allows write-proxy `+=` and `-=` to initialize unconstructed storage.
This differs from normal C++ value semantics, but it is useful for async dataflow accumulation:

- gradient accumulation where the first contribution should create the value
- tensor reductions where the destination may not be pre-constructed
- simpler kernels (no explicit constructed/unconstructed branching)

Think of this as **initialize-or-accumulate** semantics for write buffers, scoped to async epoch writes.

### Write proxy helpers

Write proxies support:

- `take()`: move out and destroy current stored value
- `take_release()`: `take()` plus immediate writer release
- `release()`: explicit early release when done writing
- `rebind(...)`: explicit reconstruct/rebind path

This is available for both:

- `WriteAccessProxy<T>` from `co_await writer`
- `OwningWriteAccessProxy<T>` from `co_await writer.transfer()`

These are the write-side analogue of read-proxy `get_release()`.

Bad first-write pattern:

```cpp
int& out = co_await writer;
out = value;
```

Good first-write pattern:

```cpp
co_await writer = value;
```

### Why you sometimes see co_await out = value;

In C++, unary `co_await` binds tighter than assignment, so:

```cpp
co_await out = value;
```

means:

```cpp
(co_await out) = value;
```

This is a convenient shorthand for:

```cpp
auto proxy = co_await out;
proxy = value;
```

In current Uni20 semantics this is also valid for first write to default-constructed `Async<T>`.
The invalid pattern is binding a mutable reference before construction:

```cpp
int& ref = co_await out; // may throw buffer_write_uninitialized
```

Use proxy assignment or proxy `emplace(...)` instead.

## Repeated Await on Same Buffer Object

Repeated `co_await` on the same buffer object is supported and tested for:

- plain read/write await
- `maybe()`
- `or_cancel()`
- `storage()`
- `take()`
- `take_release()`
- repeated proxy assignment / arithmetic updates

These repeated awaits all use the same epoch capability. They do not create new
readers, writers, or queue entries. After `transfer()` or `release()`, the
original buffer no longer owns an active capability and must not be reused.

## Exception Sink Hooks on Buffers

Buffers can register exception sinks with the current coroutine promise.

- `WriteBuffer` coroutine parameters are auto-registered as sinks
- explicit registration is available via `propagate_exceptions_to(...)`

This is how unhandled exceptions are forwarded to downstream epochs.

## Helper Awaiters (awaiters.hpp)

| Helper | Purpose |
|---|---|
| `all(a, b, ...)` | await all inputs and return tuple of results |
| `try_await(x)` | non-blocking readiness probe |
| `write_to(writer.transfer(), value)` | concise deferred write helper |

## Common Kernel Pattern

For different input and output timelines:

```cpp
auto fused = [](ReadBuffer<int> in_a, ReadBuffer<int> in_b, WriteBuffer<int> out) static -> AsyncTask {
  auto a = co_await in_a.transfer();
  auto b = co_await in_b.transfer();
  int result = a.get() + b.get();
  a.release();
  b.release();
  co_await out = result;
  co_return;
};
```

This shape is a good default: acquire inputs, compute, optionally release
readers early, then write a distinct output timeline.

For mutation of one timeline, use only its writer:

```cpp
auto double_in_place = [](WriteBuffer<int> value) static -> AsyncTask {
  auto access = co_await value;
  access.get() *= 2;
  co_return;
};
```

An operation that receives both a reader and writer for the same queue needs a
specific two-epoch algorithm and release point. Do not infer that pattern from
the fact that `ReadBuffer` and `WriteBuffer` are both available.

## Related References

- Runnable await-path map: `examples/async/async_buffer_await_paths_example.cpp`
- Runtime model: `runtime_model.md`
- Exception routing details: `exceptions_and_cancellation.md`
- Fast lookup: `quick_reference.md`
