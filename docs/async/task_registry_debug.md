# TaskRegistry Debug Instrumentation

`TaskRegistry` is the async runtime introspection layer used for deadlock and
lifecycle diagnostics.

This document explains what it tracks, how to enable it, and how to interpret dumps.

## Enablement

### Build-time switches

- `-DUNI20_DEBUG_ASYNC_TASKS=ON`
- `-DUNI20_DEBUG_DAG=ON`
- `-DUNI20_ENABLE_STACKTRACE=ON` when supported; this is the fresh-build default

CMake defaults `UNI20_ENABLE_STACKTRACE` to the result of a compile-and-link
probe. If unavailable, it defaults to `OFF` and the build continues with
degraded output and an explicit warning. Forcing it `ON` with an unsupported
toolchain is a configuration error; setting it `OFF` explicitly disables
otherwise available stacktrace capture.

`UNI20_DEBUG_DAG` enables async value nodes and coroutine argument dependency capture.
It requires the task registry and therefore enables `UNI20_DEBUG_ASYNC_TASKS`.

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

### Graphviz DAG snapshots

`TaskRegistry::graphviz_dot()` returns a Graphviz DOT document for the current async
runtime context. Internally this is a renderer over a structured
`TaskRegistry::GraphSnapshot` plus `TaskRegistry::GraphDiagnostics`, so callers can
inspect the same data model without parsing DOT. The graph includes:

- task nodes with lifecycle state
- epoch nodes with phase/generation
- async value nodes when `UNI20_DEBUG_DAG=ON`
- dashed coarse argument edges captured from coroutine `ReadBuffer`/`WriteBuffer`
  parameters
- concrete `co_await` edges for awaitables that expose async value metadata
- live epoch wait edges for currently suspended readers and writers
- best-effort diagnostics for blocked tasks, missing writers, and dependency cycles
- optional source provenance from stack walking when `UNI20_HAS_STACKTRACE=1`

Async value node labels show storage identity and current construction state.
`storage=0x...` is the `shared_storage<T>` control-block identity;
`state=constructed` means a `T` object exists at snapshot time,
`state=unconstructed` means the storage exists but does not currently hold a
constructed `T`, and `state=invalid` means no storage control block was available.
Shape-like objects that expose `mdspan().extents()` or `extents()` may also show
`value=shape=(...)`; built-in scalar values show their value. Other constructed
objects omit the `value=...` line by default. Custom types can provide a one-line
`uni20_async_debug_value(T const&)` overload in their own namespace to override
the default.

Async values and coroutine tasks can be labelled explicitly for diagnostic output:

```cpp
Async<double> value;
value.debug_name("residual block");

auto task = kernel(value.read(), output.write());
task.debug_name("apply preconditioner");
scheduler.schedule(std::move(task));
```

These labels are optional. Unlabelled nodes keep the generic `data N` and `task N`
labels, and labels do not affect scheduling, dependency tracking, or coroutine
lifetime.

When stacktrace support is available, task and epoch labels include compact
source locations such as `created_at=examples/foo.cpp:31`,
`scheduled_at=examples/foo.cpp:42`, and `awaiting_at=examples/foo.cpp:57`.
Concrete `co_await` edges can also show the source line where the dependency was
observed. Full stacktraces are kept out of the visible label and emitted as
Graphviz tooltips. The number of frames serialized into snapshots, DOT tooltips,
and text dumps is controlled by `TaskRegistry::StacktraceOptions`, which can be
set programmatically with `TaskRegistry::set_stacktrace_options(...)` or reset
from environment defaults with `TaskRegistry::reset_stacktrace_options()`.
Set `max_frames=0` to keep compact locations while omitting full stacktrace text;
set it to `std::numeric_limits<std::size_t>::max()` for uncapped traces.
This provenance is best-effort and compiler/debug-info dependent. Without
stacktrace support, DAG structure and diagnostics still work; the provenance
fields are simply empty and omitted from DOT labels.

Diagnostic annotations are intentionally conservative. A missing-writer annotation
means a task is currently blocked reading an epoch with no visible writer task or
writer activity in the registry snapshot. A dependency-cycle annotation is inferred
from current blocked-read wait edges plus tracked producer edges; it identifies a
likely cycle to inspect, not a proof of terminal font- or scheduler-dependent
behavior.

Use `TaskRegistry::graphviz_dot_best_effort()` or
`TaskRegistry::dump_graphviz_file_best_effort(path)` for deadlock/debugger paths.
The best-effort path avoids indefinite waits on registry and epoch locks; if a lock
is busy, the DOT output marks that part of the snapshot as unavailable.

Use `TaskRegistry::snapshot()` when program code wants structured task, epoch, and
async-value records. `TaskRegistry::diagnose_snapshot(snapshot)` derives blocked
task, missing-writer, and dependency-cycle annotations from that immutable snapshot.
`TaskRegistry::graphviz_dot(snapshot, diagnostics)` renders a pre-captured snapshot,
which is useful when another subsystem needs both structured diagnostics and a DOT
artifact from the same point in time.

The snapshot APIs are intended for diagnostics at normal program checkpoints,
deadlock handlers, debugger calls, or controlled interruption paths. They are not
async-signal-safe signal handlers.

## Programmatic and External Dump Requests

For normal program control, call one of:

- `TaskRegistry::graphviz_dot()`
- `TaskRegistry::dump_graphviz_file(path)`
- `TaskRegistry::request_graphviz_dump()` followed by
  `TaskRegistry::service_debug_requests()`

`DebugScheduler`, `TbbScheduler`, and `TbbNumaScheduler` service queued dump requests
at scheduler progress points. This lets instrumentation request a dump without
performing file I/O on the requesting path. A caller can still service explicitly
when it needs a specific output directory or deterministic timing.

For external triggers, start the optional diagnostics service:

```cpp
uni20::TaskRegistry::DiagnosticsServiceOptions options;
options.dump_options.output_dir = "/tmp/uni20-dag";
options.dump_options.file_prefix = "run";
options.request_file = "/tmp/uni20-dag/request";
options.signal_number = SIGUSR1; // Linux/POSIX use only
uni20::TaskRegistry::start_diagnostics_service(options);
```

The service owns a background thread. It writes best-effort DOT files when a request
file appears, when the configured signal is received, or when program code calls
`request_graphviz_dump()`. A signal is consumed with `sigtimedwait` on Linux rather
than by doing work inside a signal handler. Setting `UNI20_DEBUG_DAG_SIGNAL`
automatically starts the service during program initialization, allowing the signal
to be blocked before namespace-scope users and worker threads are created. Without
that environment variable, callers must start the service explicitly before creating
threads that should inherit the blocked signal mask.

Default output and service settings can be configured with environment variables:

| Variable | Meaning |
|---|---|
| `UNI20_DEBUG_DAG_OUTPUT_DIR` | Default DOT output directory; defaults to `/tmp` |
| `UNI20_DEBUG_DAG_FILE_PREFIX` | Default DOT filename prefix; defaults to `uni20-dag` |
| `UNI20_DEBUG_DAG_REQUEST_FILE` | Control file consumed by `start_diagnostics_service()` |
| `UNI20_DEBUG_DAG_SIGNAL` | Signal for the service, e.g. `SIGUSR1`, `USR2`, or a number; setting it starts the service during program initialization |
| `UNI20_DEBUG_DAG_POLL_MS` | Diagnostics-service poll interval in milliseconds |
| `UNI20_DEBUG_DAG_STACKTRACE_FRAMES` | Maximum stack frames shown in snapshots, DOT tooltips, and text dumps; `0` hides trace text, `all`/`full`/`unlimited`/`max` disables the cap |
| `UNI20_DEBUG_DAG_STACKTRACE_INTERNAL_FRAMES` | Whether stacktrace text includes internal Uni20/libstdc++ frames; defaults to `true` |

`TaskRegistry::default_graphviz_dump_path()` produces paths of the form
`<dir>/<prefix>.<pid>.<sequence>.dot`.

## Output Conventions

- timestamps are local time with timezone offset
- tasks and epochs are numbered to improve human scanability
- stacktraces are printed when available; when unavailable, output is explicitly marked degraded
- Graphviz source provenance is optional; missing provenance does not mean missing dependency data

## APIs

| API | Purpose |
|---|---|
| `TaskRegistry::dump()` | full global dump (deadlock triage path) |
| `TaskRegistry::dump_epoch_context(epoch, reason)` | focused dump for one epoch |
| `TaskRegistry::snapshot()` | captures the current task/epoch/value graph as structured records |
| `TaskRegistry::snapshot_best_effort()` | captures a structured snapshot without indefinite lock waits |
| `TaskRegistry::diagnose_snapshot(snapshot)` | derives blocked-task, missing-writer, and cycle annotations |
| `TaskRegistry::graphviz_dot()` | returns current task/epoch/value DAG as Graphviz DOT |
| `TaskRegistry::graphviz_dot_best_effort()` | returns a partial DOT snapshot without indefinite lock waits |
| `TaskRegistry::graphviz_dot(snapshot, diagnostics)` | renders a captured snapshot and annotations as DOT |
| `TaskRegistry::dump_graphviz(stream)` | writes DOT to a C stream, stderr by default |
| `TaskRegistry::dump_graphviz_file(path)` | writes DOT to a file |
| `TaskRegistry::dump_graphviz_file_best_effort(path)` | writes a best-effort DOT snapshot to a file |
| `TaskRegistry::default_stacktrace_options()` | returns stacktrace formatting defaults from environment variables |
| `TaskRegistry::stacktrace_options()` | returns active stacktrace formatting options |
| `TaskRegistry::set_stacktrace_options(options)` | updates active stacktrace formatting options |
| `TaskRegistry::reset_stacktrace_options()` | resets stacktrace formatting options from environment variables |
| `TaskRegistry::name_task(handle, label)` | assigns an optional diagnostic label to a task node |
| `TaskRegistry::name_async_value(node, label)` | assigns an optional diagnostic label to an async value node |
| `TaskRegistry::request_graphviz_dump()` | queues a dump request for scheduler/service processing |
| `TaskRegistry::service_debug_requests(options)` | writes queued requests on the calling thread |
| `TaskRegistry::start_diagnostics_service(options)` | starts background signal/control-file servicing |
| `TaskRegistry::stop_diagnostics_service()` | stops the background diagnostics service |
| `TaskRegistry::dump_mode()` | current runtime mode |

State hooks are invoked from promise/task runtime paths, so transitions are tied to
coroutine-handle operations, not high-level wrapper object usage.

## Typical Triage Workflow

1. run with `UNI20_DEBUG_ASYNC_TASKS=full`
2. inspect suspended tasks and their associated epochs
3. inspect epoch phases and next-epoch links
4. check `created_at`, `scheduled_at`, and `awaiting_at` locations in DOT labels
5. check full creation/transition stacktraces for mismatched writer/read ordering

If output volume is too large during exception handling paths, use focused epoch dumps
(`dump_epoch_context`) at throw sites and reserve full dump for deadlock endpoints.

## Related References

- DAG debug example: `dag_debug_examples.md`
- Runtime semantics: `runtime_model.md`
- Exception routing: `exceptions_and_cancellation.md`
- Scheduler deadlock behavior: `schedulers.md`
- Fast lookup: `quick_reference.md`
