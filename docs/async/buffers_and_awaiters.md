# Buffers and Awaiters

This document is the practical reference for `ReadBuffer<T>` and `WriteBuffer<T>`.

Use this when implementing kernels or reviewing coroutine correctness.

## Why Buffers Exist

`Async<T>` itself is a container. Access permissions are expressed through buffers:

- `ReadBuffer<T>`: read permission for one epoch
- `WriteBuffer<T>`: write permission for one epoch

This makes dependency order explicit in function signatures.

## `ReadBuffer<T>`

### Lifetime and ownership notes

- copy-constructible
- movable
- copy-assignment deleted
- move-assignment available

### Await behavior

| Form | Return type | Notes |
|---|---|---|
| `co_await reader` | `T const&` | non-owning read access |
| `co_await std::move(reader)` | `OwningReadAccessProxy<T>` | owning read access (`get()`, `get_release()`, `release()`) |

### Read modifiers

| Modifier | Return behavior | Typical use |
|---|---|---|
| `co_await reader.maybe()` | `T const*` (nullable) | probe optional/cancelled paths without exception |
| `co_await std::move(reader).maybe()` | `std::optional<OwningReadAccessProxy<T>>` | ownership-preserving optional read |
| `co_await reader.or_cancel()` | `T const&` or throws `task_cancelled` | cancellation-aware borrowed read |
| `co_await std::move(reader).or_cancel()` | `OwningReadAccessProxy<T>` or throws `task_cancelled` | cancellation-aware owning read |

### Early release

`reader.release()` (or owning proxy `release()`) releases the reader gate before object destruction.

Use it when a coroutine reads first and later awaits a writer on a related timeline.

## `WriteBuffer<T>`

### Lifetime and ownership notes

- move-only
- not copyable
- lvalue `co_await` returns a non-owning write proxy
- rvalue `co_await` returns an owning write proxy

### Await behavior

| Form | Return type | Notes |
|---|---|---|
| `co_await writer` | `WriteAccessProxy<T>` | non-owning proxy; convertible to `T&` |
| `co_await std::move(writer)` | `OwningWriteAccessProxy<T>` | owning proxy |
| `co_await writer.storage()` | `shared_storage<T>&` | explicit storage control |
| `co_await std::move(writer).storage()` | `OwningStorageAccessProxy<T>` | owning storage access |
| `co_await writer.take()` | `T` | move out + destroy stored value |
| `co_await std::move(writer).take()` | `T` | move out + destroy stored value, then release writer |
| `co_await writer.take_release()` | `T` | move out + destroy stored value, then release writer |
| `co_await std::move(writer).take_release()` | `T` | owning shorthand for take + release |

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

### Why `+=` / `-=` can initialize

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
- `OwningWriteAccessProxy<T>` from `co_await std::move(writer)`

These are the write-side analogue of read-proxy `get_release()`.

Bad first-write pattern:

```cpp
auto& out = co_await writer;
out = value;
```

Good first-write pattern:

```cpp
co_await writer = value;
```

### Why you sometimes see `co_await out = value;`

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
auto& ref = co_await out; // may throw buffer_write_uninitialized
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

## Exception Sink Hooks on Buffers

Buffers can register exception sinks with the current coroutine promise.

- `WriteBuffer` coroutine parameters are auto-registered as sinks
- explicit registration is available via `propagate_exceptions_to(...)`

This is how unhandled exceptions are forwarded to downstream epochs.

## Helper Awaiters (`awaiters.hpp`)

| Helper | Purpose |
|---|---|
| `all(a, b, ...)` | await all inputs and return tuple of results |
| `try_await(x)` | non-blocking readiness probe |
| `write_to(std::move(writer), value)` | concise deferred write helper |

## Common Kernel Pattern

```cpp
auto fused = [](ReadBuffer<int> in_a, ReadBuffer<int> in_b, WriteBuffer<int> out) static->AsyncTask {
  auto a = co_await std::move(in_a);
  auto b = co_await std::move(in_b);
  int result = a.get() + b.get();
  a.release();
  b.release();
  co_await out = result;
  co_return;
};
```

This shape is a good default: acquire inputs, compute, release readers, then write output.

## Related References

- Runtime model: `runtime_model.md`
- Exception routing details: `exceptions_and_cancellation.md`
- Fast lookup: `quick_reference.md`
