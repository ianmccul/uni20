# Async Schedulers

Schedulers answer one question: when a ready coroutine should run.

They do not define dependency legality; epochs do that. If a coroutine is not causally ready,
it will still suspend regardless of scheduler choice.

A scheduler owns queued runnable activations, not every coroutine whose promise
records that scheduler. Before admission, the public `AsyncTask` or `CudaTask`
owns the frame. After a running coroutine suspends, its awaiter owns the
`BasicTask` until readiness publishes another activation. The recorded scheduler
pointer is only a non-owning resumption route. Consequently, a scheduler does
not know about a task merely because that task names it, and scheduler
quiescence does not include externally suspended frames.

Current tasks normally retain the scheduler installed when they are first
scheduled. A nested `AsyncTask` with a different scheduler is submitted there,
then returns its continuation to the continuation's recorded scheduler.
Explicit live-task migration is a planned generic capability, not current
behavior. See [Scheduler Routing and Task Domains](scheduler_migration.md).

## Global Scheduler Model

Most async code uses the global scheduler helpers from `debug_scheduler.hpp`:

- `set_global_scheduler(...)`
- `get_global_scheduler()`
- `reset_global_scheduler()`
- `schedule(AsyncTask&&)`
- `ScopedScheduler`

If you do not override it, the global scheduler defaults to an internal `DebugScheduler`.

`ScopedScheduler` is the standard way to override scheduler context in tests.
It changes the selected scheduler but does not own it or extend its lifetime.
Any scheduler supplied to it must remain alive until all tasks routed through
that scheduler have completed or can no longer resume. See
[Lifetime and Quiescence](scheduler_migration.md#lifetime-and-quiescence) for
the full contract, including stack-local test schedulers.

## Choosing a Scheduler

| Scheduler | Strength | Tradeoff | Typical use |
|---|---|---|---|
| `DebugScheduler` | deterministic, simple deadlock diagnostics | single-threaded | semantics tests, debugging |
| `TbbScheduler` | parallel throughput | non-deterministic task interleaving | parallel CPU workloads |
| `TbbNumaScheduler` | NUMA-aware dispatch over per-node TBB arenas | extra dispatch complexity | NUMA-sensitive workloads |
| `DebugCudaScheduler` | deterministic unified host/multi-device execution | calling-thread only | CUDA semantics tests and bring-up |
| `TbbCudaScheduler` | parallel unified host/multi-device execution | non-deterministic task interleaving | CPU work plus asynchronous CUDA submission and provider calls |

## DebugScheduler

Execution model:

- scheduled tasks are stored internally
- `run()` snapshots and executes one runnable batch
- `DebugSchedulerOptions::order` selects `fifo`, `reverse`, or `random` batch
  order; the default is `reverse`, preserving the scheduler's historical
  behavior
- random order is reproducible from `DebugSchedulerOptions::random_seed`,
  which defaults to zero
- `run_all()` drains until queue empty

Wait/deadlock behavior:

- `help_while_waiting(...)` drives runnable tasks while waiting
- if waiting condition is false and no runnable tasks remain, runtime emits `TaskRegistry::dump()` and aborts with deadlock diagnostic

Use `DebugScheduler` as the first tool for dependency bugs.

## TbbScheduler

Execution model:

- tasks are dispatched into oneTBB `task_arena` + `task_group`
- initial admission and rescheduling register activations with
  `task_group::defer()` and publish them through non-blocking
  `task_arena::enqueue()`
- submitting threads do not enter the arena or wait for arena capacity
- allocation or oneTBB admission failure is a fatal scheduler-infrastructure
  error reported with scheduler and, where applicable, CUDA device context
- ready coroutines resume on threads participating in the arena, including
  oneTBB workers and application threads that enter through `task_arena::execute()`
- `run_all()` resumes if paused, then waits for task-group completion of the
  currently submitted activations

The task group owns each scheduled resumption while it runs. If that resumption
suspends the coroutine on an epoch or external event, the TBB task finishes and
the awaiter owns the suspended coroutine until it becomes ready. `run_all()` is
therefore scheduler-activation quiescence, not proof that every coroutine ever
routed through the scheduler has completed.

An admission that happens-before `run_all()` is included because its deferred
task is already registered with the group. If admission and `run_all()` are
unordered and concurrent, either may linearize first; `run_all()` is not a
global barrier against submissions that have not yet been accepted.

Pause/resume:

- `pause()` queues handles without dispatch
- `resume()` drains queued handles and dispatches them

Wait behavior:

- `get_wait()` enters the scheduler arena through `task_arena::execute()`
- the epoch's transition to reader-ready state directly wakes its registered
  suspension point
- oneTBB resumable tasks release the arena slot while dependencies execute
- nested synchronous waits work with one total arena participant
- if runnable scheduler work reaches zero, the innermost suspended wait enters
  the watchdog path
- `TbbSchedulerWaitOptions::watchdog_timeout` defaults to 5 seconds when
  `UNI20_ASYNC_DEBUG=ON` and to `std::nullopt` otherwise
- callers may explicitly provide a timeout or `std::nullopt` in either build mode

The watchdog raises `async_wait_timeout` after one continuous interval with no
scheduler-visible runnable work. It emits a best-effort `TaskRegistry` and
Graphviz diagnostic when async task tracking is compiled in. This is evidence
of stalled progress, not proof of a dependency cycle, because external GPU,
MPI, I/O, or application-thread activity may still produce the value. See
[`tbb_execution_primer.md`](tbb_execution_primer.md) for the complete execution
model and single-participant timeline.

For out-of-band inspection, an instrumented `TaskRegistry` diagnostics service
can consume a signal such as `SIGUSR1` and write a best-effort DAG snapshot. A
signal requests diagnostics but does not raise `async_wait_timeout`: it may
arrive while valid work is still running.

## TbbNumaScheduler

`TbbNumaScheduler` manages multiple `TbbScheduler` arenas, one per NUMA node.

Dispatch policy:

- task with preferred node: dispatch to that node when available
- task without preference: round-robin node selection

Diagnostics:

- `scheduled_count_for(node)` reports dispatch counts used by tests
- tests verify round-robin and preferred-node behavior

## CUDA Schedulers

`DebugCudaScheduler` and `TbbCudaScheduler` implement typed CUDA admission for
`CudaTask`. They share ordinary `BasicTask` rescheduling and continuation
routing with the host schedulers.

- `DebugCudaScheduler` also admits `AsyncTask`. It uses one deterministic queue
  for both domains and selects either the CUDA promise's affinity or the
  scheduler's default device around each CUDA activation. One instance can
  execute tasks for multiple devices and restores the calling thread between
  activations.
- `TbbCudaScheduler` extends `TbbScheduler` with one worker-only arena per
  enrolled CUDA device. The host and device arenas share task-group, pause,
  wait, watchdog, and rescheduling state.
- Each CUDA arena installs a `task_scheduler_observer`. Every participating
  worker saves its previous CUDA device, selects the arena device, and restores
  the previous selection on exit.
- initial CUDA admission and rescheduling enqueue task-group activations without
  making completion-service or publishing threads enter the device arena
- CUDA tasks may begin without device affinity and execute device-neutral work
  through the scheduler's default device arena
- `co_await cuda::set_device(device)` establishes or changes affinity; it
  continues immediately when that device is already current and otherwise
  resubmits the task through the selected device arena
- same-device nested tasks can transfer directly; cross-domain and cross-device
  continuations re-enter through the shared scheduler's routing hook
- `run_all()` and `get_wait()` cover currently submitted host and CUDA
  activations through the shared task group. They do not complete a coroutine
  suspended on an external buffer, CUDA completion, or future resource awaiter.

Tasks running in a CUDA scheduler arena must not call `cudaSetDevice` directly.
Cross-device work re-enters the scheduler through the target device arena; this
keeps the observer authoritative for the whole arena activation.

Debug builds verify the thread's current CUDA device before each oneTBB CUDA
activation. Leaf kernel wrappers do not inspect coroutine promises; they check
their CUDA API results and may validate stream and buffer device compatibility.

An unbound nested task inherits its parent scheduler only within the same task
domain. Crossing between host and CUDA requires explicit prior admission or
route binding. CUDA task admission is not global yet; a later process-wide
runtime will provide the ordinary default scheduler and device contexts.

## Practical Guidance

- start debugging with `DebugScheduler`
- once semantics are stable, validate under `TbbScheduler`
- use `TbbNumaScheduler` when NUMA topology materially affects performance
- use `DebugCudaScheduler` to diagnose CUDA task ordering deterministically
- use `TbbCudaScheduler` for parallel CUDA submission once device placement is explicit

If behavior differs between debug and TBB schedulers, suspect missing dependency edges,
missing releases, or lifetime bugs before suspecting scheduler implementation.

## Related References

- Runtime semantics: `runtime_model.md`
- oneTBB execution and waiting model: `tbb_execution_primer.md`
- Buffer usage and release patterns: `buffers_and_awaiters.md`
- Deadlock/debug output: `task_registry_debug.md`
- Fast lookup: `quick_reference.md`
