# TbbScheduler

The `TbbScheduler` is the production-ready scheduler in Uni20.
It integrates with [oneAPI Threading Building Blocks (TBB)](https://github.com/uxlfoundation/oneTBB) to provide efficient, multithreaded coroutine execution.

---

## Semantics

### Task Ownership

* As with all Uni20 schedulers, `schedule(std::move(task))` transfers ownership of the coroutine.
* The scheduler enqueues the task into a **TBB task arena** for execution.
* The coroutine is always resumed under the same scheduler instance that originally owned it.

### Rescheduling

* Suspended coroutines reschedule themselves via `reschedule(std::move(task))`.
* These tasks are passed back to the same `TbbScheduler`, which enqueues them again.
* This matches TBB’s natural execution model: the thread that triggers rescheduling is often the best candidate to execute it, thanks to TBB’s affinity heuristics.

### Blocking (`get_wait()`)

* In contrast to `DebugScheduler`, the `TbbScheduler` does **not** step tasks in the current thread.
* Instead, `get_wait()` yields until worker threads complete the task.
* This allows true parallelism but means deadlocks cannot be detected by the scheduler.
* **FIXME** This is not an optimal implementation, since we should instead use some synchronization to only wait until the required task has finished, rather than polling in a loop with `thread::yield()`. It is not clear however whether this is possible using TBB.

---

## Implementation

The scheduler is built on two core TBB primitives:

* **`task_arena`**:
  Controls the number of threads participating in scheduling.
  Each `TbbScheduler` has its own arena, isolating it from other schedulers.

* **`task_group`**:
  Manages a dynamic set of tasks within the arena.
  Each coroutine resumption is submitted via `task_group::run()`.

Tasks are executed by worker threads managed by TBB.
At destruction or when calling `run_all()`, the scheduler waits for all outstanding tasks to complete.

---

## Efficiency

Despite its simple implementation, `TbbScheduler` is close to optimal:

* **Work stealing**: Each TBB worker has its own task deque. Resumed coroutines are pushed locally, preserving cache locality. Idle workers steal from other queues, ensuring load balance.
* **Depth-first preference**: By default, TBB executes the most recently spawned tasks first, which minimizes memory pressure and improves cache utilization — well suited for coroutine graphs.
* **Affinity**: Because rescheduling often occurs on the thread that released dependencies, tasks tend to remain close to their data. This provides “good enough” affinity without explicit hints.
* **Low overhead**: Uni20 hands off raw coroutine handles directly into TBB, avoiding extra indirection.

---

## API Summary

* `schedule(task)` — enqueue a new coroutine into the arena.
* `reschedule(task)` — enqueue a suspended coroutine.
* `run_all()` — block until all tasks owned by this scheduler are complete.
* Destructor — waits for all tasks to complete before destroying the scheduler.

---

## Usage

```cpp
using namespace uni20::async;

TbbScheduler sched{4};          // use 4 worker threads
ScopedScheduler guard(&sched);

Async<int> a = 1;
Async<int> b = 2;
Async<int> c = a + b;

int result = c.get_wait();      // executed on worker threads
EXPECT_EQ(result, 3);
```
