# DebugScheduler

The `DebugScheduler` is the simplest scheduler implementation in Uni20.
It executes all coroutines **on the calling thread** in a **deterministic FIFO order**.
This makes it invaluable for testing, debugging, and reasoning about coroutine semantics.

---

## Semantics

### Task Ownership

* When you call `schedule(std::move(task))`, ownership of the coroutine is transferred into the scheduler’s internal queue.
* Tasks are stored in a simple `std::deque<AsyncTask>`.
* The scheduler resumes tasks one at a time on the current thread when `run()` or `run_all()` is called.

### Rescheduling

* When a coroutine suspends, it arranges to call `reschedule(std::move(task))`.
* `reschedule()` also pushes the coroutine back into the scheduler’s queue.
* Because execution is single-threaded, rescheduled tasks never race with other workers.

### Blocking (`get_wait()`)

* If a coroutine attempts to synchronously block on a value, `get_wait()` drives the scheduler in small steps:

  * It checks if any tasks are runnable.
  * If not, it reports a deadlock (see below).
  * Otherwise, it calls `run()` to resume one task.
* This ensures forward progress even when waiting synchronously.

---

## Deadlock Detection

Because `DebugScheduler` controls the entire execution, it can detect a key failure mode:

* If `get_wait()` is called on a value that is not ready,
* **and** there are no runnable tasks left in the queue,
* **then** the system is in deadlock: progress is impossible.

In this case, the scheduler raises a fatal error with a diagnostic.

This is not possible in `TbbScheduler`, where worker threads may be blocked on external events.

---

## Determinism

Unlike multithreaded schedulers:

* Execution order is deterministic, since only one thread is ever used.
* This makes debugging and reproducing bugs straightforward.
* For the same sequence of coroutine operations, `DebugScheduler` always produces the same trace.

---

## API Summary

* `schedule(task)` — enqueue a task for execution.
* `reschedule(task)` — re-enqueue a suspended task (not public).
* `run()` — resume a single task if available.
* `run_all()` — run tasks until the queue is drained.
* `can_run()` — check if there are runnable tasks.
* `done()` — check if all tasks have completed.

---

## Usage

```cpp
using namespace uni20::async;

DebugScheduler sched;
ScopedScheduler guard(&sched);

Async<int> a = 1;
Async<int> b = 2;
Async<int> c = a + b;

int result = c.get_wait();  // scheduler drives progress internally
EXPECT_EQ(result, 3);
```
