# Exceptions and Cancellation

This document explains how failures move through async coroutines and epoch contexts.

The key distinction is:

- access errors (uninitialized read/write)
- cancellation signals
- unhandled exceptions escaping coroutine bodies

## Exception Hierarchy

Defined in `src/uni20/async/async_errors.hpp`.

| Type | Meaning |
|---|---|
| `async_error` | base class for async runtime failures |
| `async_state_error` | invalid state/access class |
| `async_storage_missing` | storage metadata unexpectedly missing |
| `async_value_uninitialized` | direct async value access before initialization |
| `async_cancellation` | cancellation-related base class |
| `task_cancelled` | coroutine/task cancellation signal |
| `buffer_error` | buffer-specific state failure |
| `buffer_cancelled` | read buffer cancelled intentionally |
| `buffer_read_uninitialized` | read attempted before any value was constructed |
| `buffer_write_uninitialized` | mutable write ref requested before construction |

## Access-Path Behavior

### Read side

- normal read path on missing value: `buffer_read_uninitialized`
- `or_cancel()` path on missing value: `task_cancelled`

### Write side

- `co_await writer` on unconstructed value: `buffer_write_uninitialized`
- preferred fix: `co_await writer = value` (or proxy `emplace(...)`)

## Unhandled Exception Flow in Coroutines

When a coroutine throws and does not catch:

1. promise catches in `unhandled_exception()`
2. if exception is `task_cancelled`, promise sets cancel-on-resume flag
3. otherwise promise stores `std::exception_ptr`
4. promise forwards exception to registered epoch sinks

At continuation/final-suspend boundaries, stored exceptions are forwarded to continuation promises.

## Exception Sink Registration

### Automatic path

`WriteBuffer` coroutine parameters are auto-registered as sinks.

This covers the normal kernel pattern where output buffers should receive failures
from the coroutine computing them.

### Explicit path

Use:

```cpp
co_await propagate_exceptions_to(buf1, buf2, ...);
```

This is useful when errors should be routed to additional buffers (including read buffers in advanced flows).

### Explicit sink lifetime guard

If an explicit sink is destroyed during stack unwinding before promise
`unhandled_exception()` runs, runtime aborts intentionally (`CHECK`).

This fail-fast behavior prevents silent loss of exception routing in hard-to-debug cases.

## Cancellation Semantics

Cancellation state is represented through promise flagging (`set_cancel_on_resume()`).

Effects:

- cancelled tasks follow cancellation cleanup path at final suspend
- continuations can be destroyed recursively when cancellation is active
- downstream await paths may throw `task_cancelled` depending on awaiter mode

## Practical Guidelines

- use proxy assignment (`co_await writer = value`) for first write to default `Async<T>`
- treat `or_cancel()` as a control-flow choice, not just a convenience API
- register explicit sinks only when routing intent is clear
- keep explicit sink objects alive until coroutine completion

## Testing Focus Areas

Important tests in `tests/async/test_async_basic.cpp` cover:

- handled vs unhandled exception propagation
- automatic routing to write parameters
- explicit routing with `propagate_exceptions_to(...)`
- duplicate explicit registration no-op behavior
- fail-fast path for destroyed explicit sinks during unwind

## Related References

- Buffer API details: `buffers_and_awaiters.md`
- Runtime semantics: `runtime_model.md`
- Debug dump triage: `task_registry_debug.md`
- Fast lookup: `quick_reference.md`
