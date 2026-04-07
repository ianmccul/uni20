# Async Getting Started

This guide is for contributors who are new to Uni20 async code.

Goal: after reading this, you should be able to write a small coroutine kernel,
understand why it suspends/resumes, and avoid the most common correctness traps.

## What Problem the Runtime Solves

Uni20 async code expresses dependency order directly in dataflow form.
You do not manually lock data structures or coordinate condition variables.

Instead:

- values live in `Async<T>`
- read/write access goes through `ReadBuffer<T>` and `WriteBuffer<T>`
- coroutines suspend at `co_await` until the epoch is ready
- scheduler controls when suspended tasks resume

Think of each `Async<T>` as a timeline of versions (epochs): one writer phase,
then reader phase, then next epoch.

## Minimal Setup

```cpp
#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/debug_scheduler.hpp>

using namespace uni20::async;
```

For first experiments and most unit tests, use `DebugScheduler`.
Normal application code can use the default global scheduler without `ScopedScheduler`.

```cpp
DebugScheduler sched;
ScopedScheduler scoped(&sched);
```

## Async Expressions (Operator Overloads)

Many operators on `Async<T>` (and mixes of `Async<T>` with scalars) build a DAG by scheduling coroutines.
They do not compute immediately.

```cpp
Async<int> a = 2;
Async<int> b = a + 1;  // schedules a kernel; does not block

int b_value = b.get_wait(); // blocks until the scheduled work completes
(void)b_value;
```

Operator kernels schedule work onto the global scheduler.
If you need a non-default scheduler, use `ScopedScheduler` or `set_global_scheduler(...)`.
If you only need completion and do not need the value, use `b.wait()` instead.

## First Complete Example

```cpp
Async<int> value;

schedule([](WriteBuffer<int> writer) static->AsyncTask {
  co_await writer = 42;
  co_return;
}(value.write()));

schedule([](ReadBuffer<int> reader) static->AsyncTask {
  int v = co_await reader;
  TRACE("Read value =", v);
  co_return;
}(value.read()));

sched.run_all();
```

What happened:

1. `value` starts with unconstructed storage.
2. writer coroutine reaches `co_await writer = 42` and constructs the value.
3. reader coroutine resumes once epoch reaches read phase.
4. `sched.run_all()` drains all scheduled work.

## First Important Semantic Rule

Default `Async<T>` does not default-construct `T`.

First write is safe through proxy assignment:

```cpp
co_await writer = 42;
```

What is unsafe is forcing mutable-reference style access before construction:

```cpp
auto& out = co_await writer;
out = 42;
```

This can throw `buffer_write_uninitialized`.

## Typical Read-Compute-Write Kernel

```cpp
auto increment = [](ReadBuffer<int> in, WriteBuffer<int> out) static->AsyncTask {
  auto owned = co_await in.transfer();
  int v = owned.get();
  owned.release();
  co_await out = v + 1;
  co_return;
};
```

Why `owned.release()` is often useful:

- it releases the reader gate before awaiting writer work on related timelines
- this avoids self-deadlock patterns in read-modify-write compositions

`transfer()` is the explicit owning form for named buffers. Direct `co_await`
on a temporary buffer still uses the owning rvalue path.

## Quick Error Handling

Main buffer access failures:

- `buffer_read_uninitialized`
- `buffer_write_uninitialized`
- `task_cancelled`

Unhandled exceptions thrown inside a coroutine are captured in the promise and
forwarded to registered sink epochs (usually output write buffers).

To explicitly route exceptions to additional buffers, use:

```cpp
co_await propagate_exceptions_to(buffer_a, buffer_b);
```

## Coroutine Style Requirements

For async coroutine lambdas (`AsyncTask` return type):

- no captures
- `static` lambda call operator

Correct pattern:

```cpp
auto task = [](ReadBuffer<int> in, WriteBuffer<int> out) static->AsyncTask {
  int v = co_await in;
  co_await out = v;
  co_return;
};
```

This avoids closure lifetime issues when coroutine lifetime exceeds lambda object lifetime.

## Where to Go Next

- C++ coroutine background: `coroutines_primer.md`
- Runtime semantics and ownership: `runtime_model.md`
- Exact buffer awaiter behavior: `buffers_and_awaiters.md`
- Common patterns: `cookbook.md`
- Exception and cancellation routing: `exceptions_and_cancellation.md`
- Scheduler behavior and tradeoffs: `schedulers.md`
- AD model (`ReverseValue`, `Var`): `reverse_mode_ad.md`
- Fast lookup tables: `quick_reference.md`

## Examples to Read

The `examples/` folder contains runnable async demos that match current semantics:

- `examples/async_example.cpp`: basic reads/writes and `try_await`
- `examples/async_example2.cpp`: async toy IO + expression composition
- `examples/async_ops_example.cpp`: expression DAG composition + `all(...)`
- `examples/async_buffer_semantics_example.cpp`: read/write ownership, cancellation paths, and exception routing
- `examples/async_tbb_reduction_example.cpp`: parallel map-reduce with `TbbScheduler`
- `examples/async_fib_example.cpp`: recursive async composition
