# C++ Coroutine Primer (For Uni20 Async)

This is a short primer for contributors who have not used C++20/23 coroutines before.
It explains just enough to be productive in `src/uni20/async/` and `tests/async/`.

## What a Coroutine Is (In Practice)

A coroutine is a function that can suspend and later resume. In C++ that happens when the function body uses
`co_await`, `co_yield`, or `co_return`.

For Uni20 async code you will mostly see:

- `co_await`: suspend until some async precondition is satisfied
- `co_return`: finish the coroutine

When a coroutine suspends, its state (locals, parameters, current line) lives in a coroutine frame allocated by the
compiler/runtime. It does not live on the caller's stack.

## What `AsyncTask` Means

In Uni20, an `AsyncTask` is the move-only owner of a coroutine handle.

Key consequences:

- Scheduling transfers ownership of the handle to the scheduler.
- When you `co_await` an awaitable in this system, Uni20 passes ownership of the current task to that awaitable.
  The awaitable either reschedules the task later, or transfers control to a nested task.

This is why Uni20 uses an `AsyncTask`-taking `await_suspend(...)` style rather than directly returning raw
`std::coroutine_handle<>` from awaiters.

## What `co_await` on a Buffer Does

Buffers are the bridge between data dependencies and coroutine suspension:

- `ReadBuffer<T>` is an awaitable gate for reading the value of an `Async<T>` at a particular epoch.
- `WriteBuffer<T>` is an awaitable gate for writing the value at a particular epoch.

When you write:

```cpp
int v = co_await reader;
```

the coroutine suspends until the epoch becomes readable, then resumes and returns the value.

Similarly:

```cpp
co_await writer = v;
```

suspends until the epoch becomes writable, then writes through the proxy returned by `co_await writer`.

## The "Static + Captureless" Rule (Important)

Coroutine lambdas that return `AsyncTask` must be:

- captureless
- `static` (C++23 static lambda call operator)

Correct pattern:

```cpp
schedule([](ReadBuffer<int> in, WriteBuffer<int> out) static->AsyncTask {
  int v = co_await in;
  co_await out = v;
  co_return;
}(a.read(), b.write()));
```

Why this exists:

- Captured values live in the lambda closure object, which often lives on the caller stack.
- The coroutine frame can outlive that closure object.
- If a coroutine suspends, any captured references (and sometimes captured values) can become dangling.

Passing everything as parameters ensures the data lives in the coroutine frame and has the right lifetime.

## Where to Look in the Code

If you are debugging or trying to understand exact behavior, these files are the usual starting points:

- `src/uni20/async/async.hpp`: `Async<T>` container semantics
- `src/uni20/async/buffers.hpp`: `ReadBuffer` / `WriteBuffer` awaiters
- `src/uni20/async/epoch_context.hpp`: epoch phases and reader/writer scheduling
- `src/uni20/async/async_task_promise.hpp`: promise behavior (exceptions, continuations, ownership)
- `src/uni20/async/debug_scheduler.hpp`, `src/uni20/async/tbb_scheduler.hpp`: scheduler behavior

For behavioral ground truth, tests under `tests/async/` are authoritative.
