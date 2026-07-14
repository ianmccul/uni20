# oneTBB Execution Primer for Uni20

This document explains the oneTBB concepts used by Uni20's async runtime. It is
not a general oneTBB tutorial. Its purpose is to make the execution and waiting
semantics of `TbbScheduler` understandable, especially when only one thread is
available or synchronous code calls `Async<T>::get_wait()` from inside an async
task.

The most important rule is:

> An arena limits where and how many threads may execute tasks. A task group
> tracks a set of tasks. Neither one automatically makes an ordinary blocking
> function into a scheduler-aware wait.

## Short Mental Model

1. oneTBB schedules **tasks**, not user-selected operating-system threads.
2. A `task_arena` has execution **slots**. Both application threads and
   oneTBB-managed worker threads may occupy slots.
3. `max_concurrency` is the maximum number of occupied arena slots, not the
   number of worker threads.
4. Reserved application-thread slots keep part of an arena available to
   application threads that enter it.
5. `task_arena::execute()` lets an application thread enter an arena and run a
   function there.
6. `task_group::wait()` is a broad scheduler-aware wait for the whole group.
7. `task::suspend()` is a targeted scheduling point that preserves an ordinary
   synchronous call stack while its thread performs other TBB work.

## Vocabulary

| Term | Meaning in this document | Common misunderstanding |
|---|---|---|
| application thread | A thread created by the application, such as `main` | It is not necessarily idle while TBB runs |
| worker thread | A thread managed by oneTBB's scheduler | Arena concurrency is not a worker count |
| task | A schedulable unit of work | It is not an operating-system thread; ordinary execution is non-preemptive |
| arena | A scheduling domain with concurrency and placement constraints | An arena is not a fixed private thread pool |
| slot | Capacity for one thread to execute in an arena | A slot is not an operating-system thread |
| task group | A dynamically populated set of tasks that can be waited for or cancelled together | It does not define arena concurrency |
| resumable task | A TBB task whose synchronous stack is suspended at a `suspend_point` | It is distinct from a C++ coroutine |

Uni20 uses **application thread** and **worker thread** consistently. Current
oneTBB documentation also uses **external thread** for an application thread
that is outside an arena, while older TBB material uses **master thread**. Read
both terms as application thread when consulting upstream documentation.

## Application Threads and Worker Threads

oneTBB owns a process-wide worker pool. An explicit arena does not create a
permanent private set of threads. Instead, threads enter an arena, occupy its
slots, execute eligible tasks, and leave.

An application thread can participate in TBB work. The main example is:

```cpp
arena.execute([&] {
  group.wait();
});
```

The calling thread enters `arena`, executes the function, and may execute tasks
while `group.wait()` is making progress. When the outermost `execute()` returns,
the application thread has left the arena and its previous scheduler context is
restored.

A worker thread and an application thread are different origins for an
operating-system thread, not different kinds of task execution. Either may run
the same task if the arena admits it.

Do not rely on thread-local state surviving a resumable-task suspension. An
ordinary TBB task is non-preemptive, but a suspended task may resume on a
different worker. oneTBB guarantees that an outermost blocking construct such
as `task_arena::execute()` eventually returns on its original calling thread.

### Non-Preemption and Scheduling Points

oneTBB does not time-slice runnable tasks. If several tasks are queued in a
one-slot arena, oneTBB selects one of them without promising FIFO order or
fairness. Ordinary code in that task then runs without TBB preemption.

An arena slot is execution capacity for a participating thread; it is not owned
permanently by a task. Other TBB work can run when the current execution reaches
a scheduler-aware boundary:

- the task returns or completes
- the task waits in a TBB construct such as `task_group::wait()` or a nested
  parallel algorithm, allowing the participating thread to service other work
- the task calls `task::suspend()`, preserving its synchronous stack while TBB
  runs other tasks
- a Uni20 coroutine resumption actually suspends at `co_await` or completes, so
  `resume_and_track()` returns and the surrounding TBB task finishes

An awaitable that is already ready does not suspend merely because the source
contains `co_await`; coroutine execution may continue in the same TBB task.
Likewise, cancellation does not forcibly preempt running user code.

The following are not TBB scheduling points:

- `std::this_thread::yield()`
- sleeping
- blocking on an ordinary condition variable, mutex, file descriptor, or other
  operation unknown to TBB
- a long CPU loop with no TBB wait, task suspension, or coroutine suspension

Those operations retain or block the participating thread. With one arena
participant, no queued TBB task can make progress until control returns to TBB.

## Task Arenas

A `oneapi::tbb::task_arena` defines a scheduling domain:

- tasks submitted from within the arena belong to that arena
- tasks from one arena are not executed in another arena
- the arena limits simultaneous participation through its slot count
- constraints may restrict NUMA node, core type, or threads per core
- construction is lazy; the internal arena is initialized on first use unless
  initialized explicitly

### `execute()`

`task_arena::execute(f)` is the normal way for an application thread to enter
an arena.

- If a slot is available, the calling thread joins the arena and runs `f`.
- If it cannot join, oneTBB may turn `f` into a task, enqueue it, and wait for
  its completion.
- If the caller is already executing in the arena, the call remains in that
  arena.

`execute()` establishes the execution context. It does **not** make arbitrary
blocking code inside `f` scheduler-aware:

```cpp
arena.execute([&] {
  while (!ready())
  {
    // Wrong: this occupies the current stack without giving TBB a scheduling
    // point. With one participating thread, the producer cannot run.
    std::this_thread::yield();
  }
});
```

The body must call a TBB waiting primitive or suspend the current TBB task.

### `enqueue()`

`task_arena::enqueue(f)` submits work without requiring the caller to enter the
arena. It has different progress and exception rules from `task_group::run()`.
In particular, oneTBB may create one worker to service enqueued work even when
`global_control::max_allowed_parallelism` is `1`.

Uni20's `TbbScheduler` does not currently dispatch coroutine handles with
`task_arena::enqueue()`. Its private function named `enqueue_task()` eventually
enters the arena and calls `task_group::run()`. Do not infer oneTBB `enqueue()`
semantics from the Uni20 helper's name.

## Arena Concurrency

### `max_concurrency`

`max_concurrency` limits the total number of threads simultaneously executing
inside an arena. It includes worker and application threads occupying arena
slots.

It is an upper bound, not a promise. The scheduler may use fewer threads due to
the global parallelism limit, unavailable work, affinity restrictions, nested
arenas, or operating-system scheduling.

The value `task_arena::automatic` asks oneTBB to select the concurrency from the
available machine and process configuration.

### Reserved Application-Thread Slots

Reserved slots are arena capacity kept available for application threads.
Workers may use only the non-reserved portion. Application threads enter
through `execute()` and can use reserved capacity.

The public oneTBB documentation normally calls these `reserved_slots`; some
older API declarations and implementation material use
`reserved_for_masters`.

For an arena with maximum concurrency `M` and `R` reserved slots:

- at most `M` threads execute in the arena simultaneously
- at most `M - R` ordinary workers occupy the arena simultaneously
- the reservation does not create application threads
- unused reserved slots do not become ordinary worker capacity

Useful configurations are:

| Construction | Interpretation |
|---|---|
| `task_arena(automatic, 1)` | Automatic total concurrency with one slot available to application threads |
| `task_arena(1, 1)` | One arena participant; ordinary task-group work can run entirely on an application thread |
| `task_arena(4, 1)` | At most four participants; at most three ordinary workers, with one slot reserved for application entry |
| `task_arena(4, 0)` | At most four participants and no slot reserved specifically for application entry |
| `task_arena(4, 4)` | No ordinary worker slots; generally unsuitable for asynchronously submitted work |

The current `TbbScheduler(int max_concurrency)` constructor uses:

```cpp
arena_(max_concurrency, /*reserved_for_application_threads=*/1)
```

The argument is arena concurrency, not a request for that many worker threads.

### `global_control::max_allowed_parallelism`

`oneapi::tbb::global_control` applies process-wide while the control object is
alive. For `max_allowed_parallelism = N`, oneTBB limits active worker threads to
`N - 1`. Multiple overlapping controls use the smallest active value.

With `max_allowed_parallelism = 1`, ordinary TBB work is executed serially by
application threads. The documented exception is work submitted with
`task_arena::enqueue()`, for which oneTBB may still run one worker.

This control is different from an arena limit:

- `global_control` bounds worker participation across the process
- `max_concurrency` bounds simultaneous participation in one arena
- reserved application-thread slots partition arena capacity between workers
  and application-thread entry

For a strict Uni20 single-thread test, use both a global parallelism limit of
one and a `TbbScheduler` arena concurrency of one. Avoid `task_arena::enqueue()`
in the code path under test.

## Arena Constraints

`task_arena::constraints` combines concurrency with placement preferences or
restrictions.

| Constraint | Purpose |
|---|---|
| `max_concurrency` | Maximum simultaneous arena participation |
| `numa_id` | Select the NUMA node on which arena participants execute |
| `core_type` | Prefer or select a processor core type on supported systems |
| `max_threads_per_core` | Limit simultaneous participants per physical core |

Constraints do not:

- allocate or place tensor storage
- move a task between arenas
- create a fixed set of workers
- replace `EpochQueue` dependency ordering
- guarantee that the requested degree of parallelism is available

`TbbNumaScheduler` uses separate constrained arenas. The scheduler must route a
task to the intended arena; oneTBB does not move tasks between those arenas to
balance load.

When constructing `TbbScheduler` from constraints, set `max_concurrency` on the
constraints object.

## Task Groups

A `task_group` tracks dynamically added tasks. Its main operations are:

- `run(f)`: add and schedule a task, then return
- `defer(f)`: create a task handle without scheduling it yet
- `wait()`: wait until all tasks in the group complete or are cancelled
- `run_and_wait(f)`: schedule one task and wait for the group
- `cancel()`: request cancellation through the group's context

The group is an ownership, completion, and cancellation boundary. It is not a
scheduling domain. In Uni20, `task_group::run()` is called from inside
`arena_.execute(...)`, which associates the resulting task execution with that
arena.

`task_group::wait()` is scheduler-aware: the waiting thread may execute
available TBB tasks, including tasks unrelated to that group. It is therefore a
good implementation for `TbbScheduler::run_all()`, whose contract is global
quiescence for that scheduler.

It is not a suitable implementation for a targeted `Async<T>::get_wait()`:

- it waits for every task in the group, not the requested epoch
- when called from a task in the same group, the caller itself is unfinished
- unrelated long-running or externally suspended tasks would delay the wait

## C++ Coroutine Suspension and TBB Task Suspension

Uni20 uses two distinct suspension mechanisms.

| Mechanism | What is preserved | How it suspends | How it resumes |
|---|---|---|---|
| C++ coroutine | Coroutine frame containing parameters and live locals | `co_await` returns control through the coroutine protocol | Uni20 scheduler reschedules the coroutine handle |
| TBB resumable task | The current ordinary synchronous TBB task stack | `oneapi::tbb::task::suspend(callback)` | Any thread calls `oneapi::tbb::task::resume(point)` |

They compose. A TBB task may be in the middle of resuming a Uni20 coroutine;
that coroutine may call an ordinary synchronous Krylov routine; Krylov may call
`get_wait()`; and `get_wait()` may suspend the surrounding TBB task stack. No
Krylov coroutine is required.

### `task::suspend()` and `task::resume()`

The intended shape is:

```cpp
oneapi::tbb::task::suspend(
    [&](oneapi::tbb::task::suspend_point point) {
      register_waiter(point);
    });

// Later, after the condition becomes ready:
oneapi::tbb::task::resume(point);
```

The callback receives the continuation token for the current TBB execution
stack. It should register that token and return promptly. Once suspended, the
thread can participate in other TBB work. `resume(point)` signals that the
saved stack may continue and may be called from another thread.

Uni20's waiter contract should require:

- each `suspend_point` is resumed exactly once
- the callback does not block or recursively wait
- readiness is rechecked while registering the waiter, under the same lock used
  by notification, so completion cannot be lost between the initial check and
  registration
- notification removes ready waiters under the lock and calls `resume()` after
  releasing the lock

Nested synchronous function calls are supported because the whole current task
stack is preserved. A resumed stack may later suspend again. Multiple tasks may
also have independent outstanding suspension points. Uni20 does not need two
simultaneous suspension points for the same already-suspended stack.

## The Single-Thread Case

The single-thread case is not a fallback that bypasses TBB. It is cooperative
scheduling on one application thread.

Consider:

```cpp
oneapi::tbb::global_control serial{
    oneapi::tbb::global_control::max_allowed_parallelism, 1};
TbbScheduler scheduler{1};
```

For Uni20's `task_group::run()` dispatch path, the timeline is:

1. The application thread enters the arena briefly and schedules task A with
   `task_group::run()`.
2. The application thread leaves the arena. A remains pending because there is
   no ordinary worker slot.
3. The application thread calls `get_wait()` and enters the arena with
   `execute()`.
4. The wait suspends its synchronous TBB execution stack.
5. The same application thread executes task A.
6. A makes the requested epoch readable and resumes the saved suspension point.
7. The application thread continues `get_wait()` and leaves `execute()`.

This also works when the wait is nested inside a task:

1. The application thread enters `run_all()` and executes task A while in
   `group.wait()`.
2. A calls synchronous Krylov code.
3. Krylov schedules task B and calls `get_wait()` for a scalar reduction.
4. A's current synchronous stack is suspended.
5. The same application thread executes B.
6. B publishes the result and resumes A.
7. Krylov and A continue normally.

`execute()` alone is insufficient. If step 4 is replaced by a condition-variable
wait, spin loop, or `std::this_thread::yield()`, the sole participating thread
remains trapped in A and B cannot run.

### Comparison with `DebugScheduler`

`task_arena(1, 1)` with worker parallelism disabled is similar to
`DebugScheduler` in important ways:

- only one task body executes at a time
- progress is cooperative rather than preemptive
- the configuration exposes code that incorrectly depends on a spare worker

It is not equivalent to `DebugScheduler`:

- `DebugScheduler` uses Uni20's explicit deterministic queue and batch behavior
- oneTBB task selection order remains unspecified even with one slot
- a TBB wait may execute unrelated eligible tasks from the arena
- `DebugScheduler` provides direct no-runnable-task deadlock diagnostics
- one-slot TBB exercises arena entry, task-group waiting, and resumable-task
  behavior that `DebugScheduler` does not

Uni20 should test both schedulers. The one-slot TBB configuration is a required
execution mode, not merely a performance setting or a substitute for
`DebugScheduler`.

## Uni20 Targeted-Wait Implementation

`TbbScheduler::wait_for()` follows this conceptual structure:

```cpp
void wait_for(WaitRequest const& request)
{
  if (request.is_ready())
    return;

  request.notify_when_ready(wake_current_suspension);
  arena_.execute([&] {
    while (!request.is_ready()) {
      if (runnable_work_exists())
        suspend_until_ready_or_idle(request);
      else
        wait_for_new_work_or_watchdog(request);
    }
  });
}
```

The initial check keeps ready-value access cheap. `execute()` normalizes callers
that are outside the arena and callers already participating in it. The
suspension is targeted to one readiness predicate rather than all tasks in
`tg_`.

There are three distinct levels of targeting:

1. **Targeted completion:** `get_wait()` returns as soon as its particular
   epoch is readable; it does not wait for scheduler-wide quiescence.
2. **Targeted wakeup:** the transition of that epoch to readable should resume
   the registered suspension point directly, rather than relying on polling or
   completion of every TBB task.
3. **Targeted work selection:** choosing only tasks on the dependency path to
   that epoch.

The first two are appropriate Uni20 contracts. The third is not available from
oneTBB's ordinary arena scheduler because TBB does not know the `EpochQueue`
dependency graph. While the wait is suspended, TBB may select any eligible task
in the arena. If it selects a long-running unrelated task in a one-slot arena,
the target wait is delayed until that task reaches a scheduling point.

Uni20's baseline policy is throughput-oriented and work-conserving. It should
not attempt to reconstruct the dependency path and assign higher TBB priority
to tasks merely because an application thread is waiting for one epoch. Krylov
scalar reductions are commonly phase boundaries at which the machine is already
working on the contributing matrix-vector and inner-product operations. Running
other eligible work while that wait is outstanding is acceptable.

The debug `TaskRegistry` can expose an approximate explicit DAG, but that graph
is diagnostic data. It may be disabled, incomplete, or expensive to maintain.
Normal scheduling behavior must not depend on debug instrumentation. Critical-
path prioritization would require always-available dependency metadata and
evidence from profiling that the latency benefit outweighs scheduling overhead
and possible starvation. It is not part of the current design.

`IScheduler::WaitRequest` carries both the readiness predicate and an optional
targeted wakeup-registration function. `EpochContextReader` registers that
wakeup at the epoch's mutex-protected transition to `Reading`. Schedulers that
do not use resumable tasks ignore the registration and retain their existing
predicate-driven behavior.

### Wait watchdog

A suspended wait also has a configurable diagnostic escape path. The scheduler
tracks runnable coroutine execution quanta. Submitting a resumption increments
the count; completing it decrements the count. A coroutine execution quantum
that enters synchronous `get_wait()` is temporarily removed from the count
because its stack is suspended rather than runnable. Paused-queue entries are
also excluded until `resume()` dispatches them.

Zero submitted work is not by itself proof of deadlock. An external operation,
another application thread, or a future GPU or MPI completion may still make an
epoch ready. The useful diagnostic condition is therefore:

1. the requested epoch is still not ready
2. the scheduler has had no runnable execution quantum for a configurable
   interval
3. both conditions still hold when the timeout is serviced

When the runnable count reaches zero, the scheduler resumes the innermost
suspended wait. That stack then waits on a condition variable with the watchdog
deadline. It is safe to occupy the arena slot in this state because no
scheduler-visible work is runnable. Submitting new work increments the count,
notifies the condition variable, and lets the stack return to resumable TBB
suspension before the new work needs the slot.

This arrangement needs neither a timer thread nor a sleeping TBB task. A
sleeping or polling watchdog task would be incorrect because it could occupy
the only slot and delay the dependency that satisfies the wait. On timeout, the
waiting stack prints scheduler state, requests a best-effort `TaskRegistry` DAG
snapshot when debug tracking is available, and raises `async_wait_timeout`
through the normal Uni20 error policy.

Runnable-work accounting should use a generation counter as well as a count so
that timeout servicing can detect a zero-to-nonzero-to-zero transition without
mistaking stale observations for one continuous idle interval. Enqueuing new
work resets the idle deadline. A paused scheduler and a scheduler with no
submitted work should be reported separately because their likely causes are
different.

This watchdog diagnoses a lack of scheduler-visible progress; it does not prove
that the dependency graph contains a cycle. Its timeout must remain configurable
for applications that legitimately wait on slow external completion.
`TbbSchedulerWaitOptions::watchdog_timeout` defaults to 30 seconds and accepts
`std::nullopt` to disable the watchdog.

Resumable stacks can themselves be nested. An idle transition wakes only the
most recently registered nested wait, allowing suspended stacks to unwind from
the inside out. Waking every registered suspension point would allow an
application-thread ancestor to enter its watchdog path while the dependency
task remained suspended below it. Independent application-thread waits may all
enter their watchdog paths when no nested execution stack exists.

No `in_coroutine` test is required. The relevant execution context is the TBB
arena and resumable task stack, not whether some function higher in the call
chain happens to be a C++ coroutine. Uni20 uses thread-local state only to mark
the currently executing scheduler quantum; it explicitly clears that state
before suspension and establishes it again when the stack resumes. No thread
identity or thread-local value is assumed to survive a oneTBB suspension.

## What Suspend/Resume Does Not Fix

Resumable tasks provide progress for valid dependency graphs. They do not make
invalid graphs valid.

- Waiting for an epoch that depends on the current task is still a cycle.
- Acquiring incompatible read and write epochs for aliased storage may still
  deadlock.
- A paused scheduler cannot make progress until resumed.
- A suspension point that is never resumed is a leaked blocked computation.
- Blocking inside the suspension callback can still exhaust scheduler progress.

Use `DebugScheduler` and `TaskRegistry` diagnostics to find dependency cycles.
Use the TBB single-thread configuration to find waits that accidentally depend
on spare workers.

## Uni20 Design Rules

1. Treat arena concurrency as slot capacity, not worker count.
2. Enter the scheduler's arena with `execute()` before an application-thread
   targeted wait.
3. Use `task_group::wait()` for scheduler-wide quiescence, not for one async
   value.
4. Use resumable-task suspension for nested synchronous waits under
   `TbbScheduler`.
5. Do not spin or use `std::this_thread::yield()` while waiting for TBB work.
6. Keep suspend callbacks non-blocking and make waiter registration race-free.
7. Do not rely on thread identity or thread-local state across a suspension.
8. Test scheduler changes with one total participant as well as several
   workers.
9. Keep `task_arena::enqueue()` use explicit because its progress and exception
   behavior differs from the task-group path.
10. Remember that `EpochQueue` defines dependency legality; TBB only executes
    work that those dependencies make ready.

## Recommended Tests

The TBB scheduler test suite should include:

- top-level `get_wait()` with `max_allowed_parallelism = 1` and
  `TbbScheduler{1}`
- nested `get_wait()` from an async task under the same configuration
- two sequential suspensions of the same resumed synchronous stack
- several tasks suspended independently and resumed in different orders
- readiness becoming true during waiter registration
- repeated ready-value access that never enters the suspension path
- a multi-worker stress test for lost wakeups and duplicate resume attempts
- a watchdog timeout with no runnable work, including diagnostic wakeup of the
  suspended application stack
- external or newly submitted work arriving before the watchdog deadline
- runnable work transitioning through zero without triggering a stale timeout
- independent waits timing out or completing concurrently

The single-thread tests should record operating-system thread identifiers and
confirm that the application thread can schedule, suspend, execute the
dependency, and resume without an ordinary worker.

## Official References

These specification pages define the contracts summarized above:

- [oneTBB task arena specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/task_arena/task_arena_cls.html)
- [oneTBB task scheduler specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler.html)
- [oneTBB task group specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/task_group/task_group_cls.html)
- [oneTBB global control specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/scheduling_controls/global_control_cls.html)
- [oneTBB resumable tasks specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/scheduling_controls/resumable_tasks.html)

## Related Uni20 Documentation

- [Async schedulers](schedulers.md)
- [Async runtime model](runtime_model.md)
- [Buffers and awaiters](buffers_and_awaiters.md)
- [Task registry debugging](task_registry_debug.md)
- [Coroutine primer](coroutines_primer.md)
