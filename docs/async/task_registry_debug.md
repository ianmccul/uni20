# TaskRegistry Debug Instrumentation

`TaskRegistry` is the async runtime introspection layer used for deadlock and
lifecycle diagnostics.

This document explains what it tracks, how to enable it, and how to interpret dumps.

## Enablement

### Build-time switches

- `-DUNI20_DEBUG_ASYNC_TASKS=ON`
- `-DUNI20_ENABLE_STACKTRACE=ON`

`UNI20_ENABLE_STACKTRACE` probes for `<stacktrace>`. If unavailable, build continues
with degraded output and explicit warning.

### Runtime verbosity switch

Environment variable: `UNI20_DEBUG_ASYNC_TASKS`

- `0`, `none`, `off`, `false`, `no` -> no dump
- `1`, `basic`, `on`, `true`, `yes` -> basic diagnostics
- `2`, `full`, `all`, `verbose` -> full diagnostics

Unset or unknown values currently default to `basic`.

## What Is Tracked

### Tasks

Each task record includes:

- task id (monotonic convenience id)
- coroutine handle address
- transition count
- current state (`constructed`, `running`, `suspended`, `leaked`)
- creation timestamp
- last transition timestamp
- creation stacktrace (when available)
- last-transition stacktrace (when available)

### Epoch contexts

Each epoch context record includes:

- epoch id
- epoch address
- generation counter
- phase
- next epoch (id/address when present)
- creation timestamp
- creation stacktrace (when available)

### Associations

Dump output correlates tasks with epochs at dump time by scanning epoch snapshots:

- reader associations
- writer associations

This gives useful suspension context without requiring high-overhead always-on edge tracking.

## Output Conventions

- timestamps are local time with timezone offset
- tasks and epochs are numbered to improve human scanability
- stacktraces are printed when available; when unavailable, output is explicitly marked degraded

## APIs

| API | Purpose |
|---|---|
| `TaskRegistry::dump()` | full global dump (deadlock triage path) |
| `TaskRegistry::dump_epoch_context(epoch, reason)` | focused dump for one epoch |
| `TaskRegistry::dump_mode()` | current runtime mode |

State hooks are invoked from promise/task runtime paths, so transitions are tied to
coroutine-handle operations, not high-level wrapper object usage.

## Typical Triage Workflow

1. run with `UNI20_DEBUG_ASYNC_TASKS=full`
2. inspect suspended tasks and their associated epochs
3. inspect epoch phases and next-epoch links
4. check creation/transition stacktraces for mismatched writer/read ordering

If output volume is too large during exception handling paths, use focused epoch dumps
(`dump_epoch_context`) at throw sites and reserve full dump for deadlock endpoints.

## Related References

- Runtime semantics: `runtime_model.md`
- Exception routing: `exceptions_and_cancellation.md`
- Scheduler deadlock behavior: `schedulers.md`
- Fast lookup: `quick_reference.md`
