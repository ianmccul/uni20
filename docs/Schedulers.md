# Schedulers in Uni20

Uni20 executes all asynchronous tasks via a *scheduler*.

## Core Semantics

### Ownership of Tasks

* All async coroutines in Uni20 are an instance of `AsyncTask` (this is the **promise type** associated with the coroutine).
* An `AsyncTask` is a move-only wrapper for a `std::coroutine_handle`.
* When you call `schedule(std::move(task))`, you **transfer ownership** of that coroutine to the scheduler.
* After scheduling, the caller must not touch the task again — it no longer owns the coroutine.
* The scheduler now decides *when* and *on which thread* the coroutine is resumed.

### Resumption Path

* When a coroutine suspends (e.g. `co_await`), it arranges for itself to be **rescheduled**.
* `reschedule()` is called internally by the runtime when dependencies are satisfied.
* Both `schedule()` and `reschedule()` enqueue tasks in the same way: the scheduler gains exclusive ownership.

### Pinning

* Each coroutine is pinned to the scheduler on which it was originally scheduled.
* This is tracked in the `sched_` field of the coroutine promise.
* Even if resumed later by external events, it will always return to the same scheduler.

---

## Blocking and Waiting

* Some code paths (e.g. `Async<T>::get_wait()`) require blocking until a value is ready.
* The semantics differ by scheduler:

  * **DebugScheduler**: drives the scheduler in the current thread step by step.
  * **TbbScheduler**: yields or blocks until background worker threads complete the task.

Blocking is only safe because the scheduler owns the coroutine; the caller waits only for completion, not for control.

---

Uni20 allows different schedulers, via the `set_global_scheduler()` function. You can also submit a task directly via the
scheduler `.schedule(AsyncTask&&)` member function.

## Available Schedulers

* [DebugScheduler](DebugScheduler.md)

  * Single-threaded.
  * Deterministic order.
  * Detects deadlocks.
  * Intended for testing and debugging.

* [TbbScheduler](TbbScheduler.md)

  * Multithreaded, built on oneAPI TBB.
  * Uses a `task_arena` and `task_group` internally.
  * Suitable for production workloads.

---

## Using a scheduler

There are some example programs in the `examples/` directory, including:

### `examples/async_example.cpp`

* **What it does:** Demonstrates a simple coroutine pipeline with `Async<int>` values.
* **Scheduler:** Uses the **DebugScheduler**. Tasks are explicitly scheduled and run synchronously inside `run_all()`.
* **Purpose:** Showcases coroutine composition and dependency management in a small, deterministic setting.

### `examples/async_example2.cpp`

* **What it does:** Extends the first example with multiple dependent `Async<int>` computations (chained async arithmetic).
* **Scheduler:** Still uses **DebugScheduler**. The test environment makes it easy to reason about ordering.
* **Purpose:** Shows more complex dependency graphs and verifies that the scheduler correctly propagates execution.

### `examples/async_fib_example.cpp`

* **What it does:** Coroutine-based Fibonacci implementation (`fib(n)` calls `fib(n-1)` and `fib(n-2)` asynchronously).
* **Scheduler:** Runs under the **DebugScheduler** for determinism.
* **Purpose:** Demonstrates recursive coroutine spawning and handling many small tasks efficiently in a controlled environment.

### `examples/async_ops_example.cpp`

* **What it does:** Demonstrates higher-level async operations (like `map`, `zip`, `reduce`) built on top of the core async + scheduler system.
* **Scheduler:** Again, the **DebugScheduler** is used for reproducibility.
* **Purpose:** Serves as a “library-style” showcase — shows that async ops compose naturally when backed by a scheduler.

### `examples/future_example.cpp`

* **What it does:** Uses `FutureValue` and `AsyncTask` to bridge futures and coroutines.
* **Scheduler:** Uses **DebugScheduler** (global). Futures are awaited inside coroutines and scheduled deterministically.
* **Purpose:** Demonstrates integration between `Async` coroutines and future-based APIs.

### `examples/gradient_solver.cpp`

* **What it does:** A more “serious” numerical example. Implements a gradient-based solver using async tasks for staging computations.
* **Scheduler:** Uses **DebugScheduler** for correctness (easy debugging, deterministic ordering).
* **Purpose:** Demonstrates async in a realistic computational workflow, preparing the ground for eventually swapping in TBB for scalability.

### `examples/async_tbb_reduction_example.cpp`

* **What it does:** Computes the **sum of squares** from 1 to N using `square` and `sum` coroutines.
* **Scheduler:** Uses **TbbScheduler** with multiple threads.
* **Purpose:** Demonstrates scalable parallel execution with real concurrency. Unlike the earlier DebugScheduler examples, this actually fans out onto worker threads.

Any of these examples can be easily adapted to use a different scheduler.

## Benchmarking the Schedulers

The **`benchmarks/benchmark_coroutine_overhead.cpp`** program measures the raw overheads of coroutine scheduling in Uni20. It compares:

* **Baseline** – raw loop overhead.
* **Sine** – loop with lightweight floating-point compute (`std::sin`).
* **SimpleAsync** – trivial coroutine chain using **DebugScheduler**.
* **Binary** – replaces the inner loop `x += 1` with `x = x + 1`. This launches two kernels per iteration, one kernel to evaluate for `x+1` and one kernel for the assignment.
* **SimpleAsyncTbb** – same as SimpleAsync, but scheduled on **TbbScheduler**.
* **BinaryTbb** – same as Binary, but scheduled on **TbbScheduler**.

### Example Results (Intel i7-12700KF, GCC, Release build)

| Benchmark      | Time (ns) | CPU (ns) | Iterations  |
| -------------- | --------- | -------- | ----------- |
| Baseline       | 1.43      | 1.43     | 466,777,125 |
| Sine           | 13.7      | 13.7     | 51,389,673  |
| SimpleAsync    | 120       | 120      | 5,670,216   |
| Binary         | 219       | 219      | 2,472,248   |


### Interpretation

* Coroutine suspension and resumption cost about **100–200 ns** in the **DebugScheduler**.
* The **TbbScheduler** adds more overhead (800–900 ns) due to work-stealing and thread pool management.
* The cost is still small compared to realistic compute kernels, where tasks often run for microseconds or milliseconds.
* The Tbb benchmarks slow down when using more threads. Since the dependencies are purely serial, adding more threads is pure overhead.
* I am not sure why BinaryTbb is consistently as fast (or faster) than SimpleAsyncTbb, since the BinaryTbb should have double the number of tasks that are scheduled.  

These benchmarks help quantify:

* The cost of coroutine suspension/resumption.
* The additional scheduling overhead introduced by **DebugScheduler** vs. **TbbScheduler**.

In practice, **TbbScheduler** overhead is expected to be close to optimal, since it defers scheduling to oneTBB’s work-stealing runtime.


## Future Directions

* Scheduler hints (e.g. memory buffer affinity).
* GPU and distributed schedulers.
