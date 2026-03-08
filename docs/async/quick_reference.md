# Async Quick Reference

This page is intentionally compact. Use it for fast lookup during implementation,
code review, and AI-agent prompting.

For explanations, see:

- `getting_started.md`
- `coroutines_primer.md`
- `cookbook.md`
- `runtime_model.md`
- `exceptions_and_cancellation.md`

## Core Types

| Type | Purpose | Key methods |
|---|---|---|
| `Async<T>` | Async value + epoch queue | `read()`, `write()`, `get_wait()`, `move_from_wait()` |
| `ReadBuffer<T>` | Read gate for one epoch | `co_await reader`, `maybe()`, `or_cancel()`, `release()` |
| `WriteBuffer<T>` | Write gate for one epoch | `co_await writer`, `storage()`, `take()`, `take_release()`, `release()` |
| `AsyncTask` | Move-only coroutine handle owner | schedule via `schedule(...)` |
| `IScheduler` | Scheduler interface | `schedule`, `pause`, `resume`, wait hooks |

## Canonical Coroutine Pattern

```cpp
auto kernel = [](ReadBuffer<int> in, WriteBuffer<int> out) static->AsyncTask {
  auto owned = co_await std::move(in);
  int v = owned.get();
  owned.release();
  co_await out = v + 1;
  co_return;
};
```

Rules:

- coroutine lambdas are captureless
- coroutine lambdas are `static`
- pass buffers as coroutine parameters

## Read/Write Semantics

### `Async<T>` initialization

- `Async<T>()`: storage exists, value is unconstructed
- `Async<T>(args...)`: value is constructed immediately

### `ReadBuffer<T>` await results

| Expression | Result |
|---|---|
| `co_await reader` (lvalue) | `T const&` |
| `co_await std::move(reader)` | `OwningReadAccessProxy<T>` |
| `co_await reader.maybe()` | `T const*` |
| `co_await std::move(reader).maybe()` | `std::optional<OwningReadAccessProxy<T>>` |
| `co_await reader.or_cancel()` | `T const&` or `task_cancelled` |
| `co_await std::move(reader).or_cancel()` | `OwningReadAccessProxy<T>` or `task_cancelled` |

### `WriteBuffer<T>` await results

| Expression | Result |
|---|---|
| `co_await writer` | `WriteAccessProxy<T>` (convertible to `T&`) |
| `co_await std::move(writer)` | `OwningWriteAccessProxy<T>` |
| `co_await writer.storage()` | `shared_storage<T>&` |
| `co_await std::move(writer).storage()` | `OwningStorageAccessProxy<T>` |
| `co_await writer.take()` | `T` moved out, then storage destroyed |
| `co_await std::move(writer).take()` | `T` moved out, then storage destroyed and writer released |
| `co_await writer.take_release()` | `T` moved out, then storage destroyed and writer released |
| `co_await std::move(writer).take_release()` | `T` moved out, then storage destroyed and writer released |

Critical write rule:

- first write on default `Async<T>` can use `co_await writer = value` for default `rebind` types
- `co_await writer += x` and `co_await writer -= x` also emplace when unconstructed
- `co_await writer` may still throw `buffer_write_uninitialized` if you request mutable-reference-style access before initialization

Assignment semantics trait:

- `assignment_semantics_of<T>` defaults to `assignment_semantics::rebind`
- specialize to `assignment_semantics::write_through` for reference-proxy types
- `write_through` requires already-constructed storage; use proxy `rebind(...)` for explicit retarget/reconstruction

## Async Ops (`async_ops.hpp`)

| Helper | Typical use |
|---|---|
| `async_assign(src, dst)` | copy-like value propagation |
| `async_move(src, dst)` | move-like value transfer |
| `async_binary_op(...)` | schedule `out = op(a, b)` |
| `async_compound_op(...)` | schedule in-place-style update to async lhs |

Helper awaiters:

- `all(a, b, ...)`
- `try_await(awaitable)`
- `write_to(std::move(writer), value)`

## Exceptions and Cancellation

Main exception types (`async_errors.hpp`):

- `task_cancelled`
- `buffer_read_uninitialized`
- `buffer_write_uninitialized`
- `async_value_uninitialized`

Unhandled exception behavior:

- promise captures `std::exception_ptr`
- forwards exception to registered sink epochs
- downstream awaiters rethrow on access

Sink registration:

- automatic for `WriteBuffer` coroutine parameters
- explicit via `co_await propagate_exceptions_to(buf1, buf2, ...)`

## Scheduler Choice

| Scheduler | When to use |
|---|---|
| `DebugScheduler` | deterministic tests and semantics debugging |
| `TbbScheduler` | general parallel execution |
| `TbbNumaScheduler` | NUMA-aware execution |

## TaskRegistry Debugging

### Build-time

- `-DUNI20_DEBUG_ASYNC_TASKS=ON`
- `-DUNI20_ENABLE_STACKTRACE=ON`

If `<stacktrace>` is unavailable, build continues with degraded stacktrace output.

### Runtime verbosity

Environment variable: `UNI20_DEBUG_ASYNC_TASKS`

- `none`, `off`, `false`, `0` -> no dump
- `basic`, `on`, `true`, `1` -> basic dump
- `full`, `all`, `verbose`, `2` -> full dump

Unknown/unset currently defaults to `basic`.

## Fast Troubleshooting

| Symptom | Likely cause | First check |
|---|---|---|
| `buffer_write_uninitialized` | requested mutable reference before construction | replace with proxy assignment (`co_await writer = value`) or proxy `emplace(...)` |
| reader blocks forever | read/write dependency cycle or missing release | verify epoch ordering and `reader.release()` placement |
| cancellation unexpectedly propagates | `or_cancel()` path used and no value present | inspect upstream writer and exception sink routing |
| deadlock dump at shutdown | unresolved suspended tasks | inspect `TaskRegistry::dump()` task/epoch links |

## Where to Look (Code and Tests)

Core code:

- `src/uni20/async/async.hpp`
- `src/uni20/async/buffers.hpp`
- `src/uni20/async/epoch_context.hpp`
- `src/uni20/async/async_task_promise.hpp`
- `src/uni20/async/debug_scheduler.hpp`
- `src/uni20/async/tbb_scheduler.hpp`

Behavioral ground truth:

- `tests/async/test_async_basic.cpp`
- `tests/async/test_async_awaiters.cpp`
- `tests/async/test_task_registry.cpp`
- `tests/async/test_tbb_scheduler.cpp`
- `tests/async/test_reverse_value.cpp`
