# Async Schedulers

Schedulers answer one question: when a ready coroutine should run.

They do not define dependency legality; epochs do that. If a coroutine is not causally ready,
it will still suspend regardless of scheduler choice.

## Global Scheduler Model

Most async code uses the global scheduler helpers from `debug_scheduler.hpp`:

- `set_global_scheduler(...)`
- `get_global_scheduler()`
- `reset_global_scheduler()`
- `schedule(AsyncTask&&)`
- `ScopedScheduler`

If you do not override it, the global scheduler defaults to an internal `DebugScheduler`.

`ScopedScheduler` is the standard way to override scheduler context in tests.

## Choosing a Scheduler

| Scheduler | Strength | Tradeoff | Typical use |
|---|---|---|---|
| `DebugScheduler` | deterministic, simple deadlock diagnostics | single-threaded | semantics tests, debugging |
| `TbbScheduler` | parallel throughput | non-deterministic task interleaving | production parallel work |
| `TbbNumaScheduler` | NUMA-aware dispatch over per-node TBB arenas | extra dispatch complexity | NUMA-sensitive workloads |

## `DebugScheduler`

Execution model:

- scheduled tasks are stored internally
- `run()` executes one batch in deterministic order
- `run_all()` drains until queue empty

Wait/deadlock behavior:

- `help_while_waiting(...)` drives runnable tasks while waiting
- if waiting condition is false and no runnable tasks remain, runtime emits `TaskRegistry::dump()` and aborts with deadlock diagnostic

Use `DebugScheduler` as the first tool for dependency bugs.

## `TbbScheduler`

Execution model:

- tasks are dispatched into oneTBB `task_arena` + `task_group`
- ready coroutines resume on TBB worker threads
- `run_all()` resumes if paused, then waits for task-group completion

Pause/resume:

- `pause()` queues handles without dispatch
- `resume()` drains queued handles and dispatches them

Wait behavior:

- if already inside a TBB arena thread, wait loop yields until ready
- otherwise waits on condition variable notified by task completion

## `TbbNumaScheduler`

`TbbNumaScheduler` manages multiple `TbbScheduler` arenas, one per NUMA node.

Dispatch policy:

- task with preferred node: dispatch to that node when available
- task without preference: round-robin node selection

Diagnostics:

- `scheduled_count_for(node)` reports dispatch counts used by tests
- tests verify round-robin and preferred-node behavior

## Practical Guidance

- start debugging with `DebugScheduler`
- once semantics are stable, validate under `TbbScheduler`
- use `TbbNumaScheduler` when NUMA topology materially affects performance

If behavior differs between debug and TBB schedulers, suspect missing dependency edges,
missing releases, or lifetime bugs before suspecting scheduler implementation.

## Related References

- Runtime semantics: `runtime_model.md`
- Buffer usage and release patterns: `buffers_and_awaiters.md`
- Deadlock/debug output: `task_registry_debug.md`
- Fast lookup: `quick_reference.md`
