# Async Cookbook

This page collects common patterns for Uni20 async coroutines.

If you are new to the runtime, start with `getting_started.md` and `coroutines_primer.md`.

## 1) Produce a Value (First Write)

Default `Async<T>` starts unconstructed. First write is usually simplest as proxy assignment.

```cpp
auto produce = [](WriteBuffer<int> out) static->AsyncTask {
  co_await out = 42;
  co_return;
};
```

## 2) Consume a Value

```cpp
auto consume = [](ReadBuffer<int> in) static->AsyncTask {
  int v = co_await in;
  (void)v;
  co_return;
};
```

## 3) Read-Compute-Write (Different Timelines)

When input and output are different `Async<T>` objects, this is the simplest kernel shape:

```cpp
auto kernel = [](ReadBuffer<int> in, WriteBuffer<int> out) static->AsyncTask {
  int v = co_await in;
  in.release();
  co_await out = v + 1;
  co_return;
};
```

## 4) Read-Modify-Write (Same Timeline)

When you read and then later write to the same logical timeline, release reader gates before waiting on a writer.

```cpp
auto add_in_place = [](ReadBuffer<int> in, WriteBuffer<int> out) static->AsyncTask {
  auto owned = co_await in.transfer();
  int v = owned.get();
  owned.release();

  // `+=` now has emplace semantics on unconstructed storage.
  co_await out += v;
  co_await out += 1;

  co_return;
};
```

## 5) Fan-In (Wait for Multiple Inputs)

```cpp
auto sum2 = [](ReadBuffer<int> a, ReadBuffer<int> b, WriteBuffer<int> out) static->AsyncTask {
  auto [va, vb] = co_await all(a, b);
  a.release();
  b.release();
  co_await out = va + vb;
  co_return;
};
```

## 6) Why `co_await out = value;` Works

This is C++ operator precedence. Unary `co_await` binds tighter than assignment:

```cpp
co_await out = value; // means: (co_await out) = value;
```

In current Uni20 semantics this is also valid on first write to default-constructed `Async<T>`.
Proxy assignment uses `emplace(...)` internally when needed.

## 7) Route Exceptions to an Output

Unhandled exceptions in a coroutine can be forwarded to registered sink epochs.

Common pattern: a reader failure should invalidate the output writer epoch.

```cpp
auto kernel = [](ReadBuffer<int> in, WriteBuffer<int> out) static->AsyncTask {
  co_await propagate_exceptions_to(out);
  int v = co_await in;           // if this throws unhandled, out receives the exception
  co_await out = v + 1;
  co_return;
};
```

## 8) Cancellation-Aware Reads

If you want "no value" to behave like cancellation, use `or_cancel()`:

```cpp
auto kernel = [](ReadBuffer<int> in, WriteBuffer<int> out) static->AsyncTask {
  int v = co_await in.or_cancel();  // throws task_cancelled if missing
  co_await out = v;
  co_return;
};
```

## 9) Debugging Deadlocks

Use `DebugScheduler` first. It runs tasks in deterministic order and dumps task/epoch state on deadlock.

Environment variable:

- `UNI20_DEBUG_ASYNC_TASKS=full` for maximum `TaskRegistry` dump output

Docs:

- `task_registry_debug.md`
- `examples/async_buffer_semantics_example.cpp`
