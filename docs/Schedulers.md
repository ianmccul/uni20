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

## Future Directions

* Scheduler hints (e.g. memory buffer affinity).
* GPU and distributed schedulers.
* Safe hot-swapping of global schedulers.
