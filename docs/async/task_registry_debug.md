# TaskRegistry Debug Instrumentation

`TaskRegistry` is the async runtime introspection layer used for deadlock and
lifecycle diagnostics.

This document explains what it tracks, how to enable it, and how to interpret
presentation reports and Graphviz snapshots.

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

For a complete build-capability check and guided demonstration, build and run:

```bash
cmake -S . -B ./build_codex/build_gcc14_debug_dag \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DUNI20_DEBUG_DAG=ON \
  -DUNI20_ENABLE_STACKTRACE=ON
cmake --build ./build_codex/build_gcc14_debug_dag \
  --target async_diagnostics_guide_example
./build_codex/build_gcc14_debug_dag/examples/async_diagnostics_guide_example
```

GCC 14 is a useful baseline for libstdc++ `std::stacktrace`; CMake still performs
the authoritative compile-and-link capability probe for the selected compiler and
standard library.

### Runtime verbosity switch

Environment variable: `UNI20_DEBUG_ASYNC_TASKS`

- `0`, `none`, `off`, `false`, `no` -> no dump
- `1`, `basic`, `on`, `true`, `yes` -> basic diagnostics
- `2`, `full`, `all`, `verbose` -> full diagnostics

Unset or unknown values currently default to `basic`.

## What Is Tracked

The registry is observational. It stores task identities, transitions, stack
information, and dependency associations, but it never owns a coroutine frame
or keeps one alive. A registry entry for a suspended task does not mean that the
scheduler owns or has queued that task.

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
observed. Full stacktraces are kept out of visible labels and emitted as Graphviz
tooltips. DOT tooltip lines are wrapped to 100 display cells because xdot does
not wrap its Pango tooltip labels itself. The number of frames serialized into
snapshots and DOT tooltips is controlled by `TaskRegistry::StacktraceOptions`,
which can be
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
`task_registry_report(snapshot, diagnostics)` produces a presentation-native
terminal/plain-text report from the same records; it does not require parsing DOT.
`TaskRegistry::graphviz_dot(snapshot, diagnostics)` renders a pre-captured snapshot,
which is useful when another subsystem needs both structured diagnostics and a DOT
artifact from the same point in time.

An exception escaping an `AsyncTask` or `CudaTask` coroutine reaches the shared
`TaskPromiseBase::unhandled_exception()` implementation while that task is still
present in the registry. The promise marks the task failed and records the
exception summary before propagating the exception to writer sinks. Automatic
diagnostics are debug-registry-only and opt-in:

```cpp
auto previous = uni20::TaskRegistry::coroutine_exception_diagnostics_options();
uni20::TaskRegistry::set_coroutine_exception_diagnostics_options({
    .enabled = true,
    .write_graphviz = true,
    .dump_options = {.output_dir = "/tmp/uni20-failures", .file_prefix = "failure"},
});

// Schedule and observe the async work. Restore `previous` afterward.
```

The policy is process-wide rather than thread-local because a coroutine may resume
on a different worker thread. Only the coroutine that originates an exception
triggers automatic capture; downstream coroutines that merely rethrow the injected
exception are still marked failed in later live snapshots but do not each write a
duplicate file. “Unhandled” here has the standard coroutine meaning of escaping
one coroutine body. The exception may still be intentionally observed later at an
`Async<T>` boundary.

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
| `UNI20_DEBUG_DAG_DUMP_ON_EXCEPTION` | Enable a live registry report and DOT snapshot when an exception originates in an async coroutine; defaults to off |
| `UNI20_DEBUG_DAG_REQUEST_FILE` | Control file consumed by `start_diagnostics_service()` |
| `UNI20_DEBUG_DAG_SIGNAL` | Signal for the service, e.g. `SIGUSR1`, `USR2`, or a number; setting it starts the service during program initialization |
| `UNI20_DEBUG_DAG_POLL_MS` | Diagnostics-service poll interval in milliseconds |
| `UNI20_DEBUG_DAG_STACKTRACE_FRAMES` | Maximum stack frames retained in snapshots and DOT tooltips; `0` hides trace text, `all`/`full`/`unlimited`/`max` disables the cap |
| `UNI20_DEBUG_DAG_STACKTRACE_INTERNAL_FRAMES` | Whether stacktrace text includes internal Uni20/libstdc++ frames; defaults to `true` |

`TaskRegistry::default_graphviz_dump_path()` produces paths of the form
`<dir>/<prefix>.<pid>.<sequence>.dot`.

## Output Conventions

- timestamps are local time with timezone offset
- tasks and epochs are numbered to improve human scanability
- terminal reports use the common presentation layer and compact source locations
- complete captured stacktraces are available in Graphviz tooltips; when stacktrace support is unavailable, output is explicitly marked degraded
- Graphviz source provenance is optional; missing provenance does not mean missing dependency data
- task, epoch, and edge tooltips escape Pango markup characters and encode real
  line breaks so `xdot` can display wrapped C++ templates and references correctly

Presentation output queries the actual destination stream. A terminal uses its
detected width; redirected output uses `UNI20_FALLBACK_TERMINAL_WIDTH` (132 by
default) unless `COLUMNS` overrides it. `UNI20_COLOR`, `NO_COLOR`,
`UNI20_GLYPHS`, and `UNI20_CHARSET` provide the normal presentation controls.

## APIs

| API | Purpose |
|---|---|
| `TaskRegistry::dump()` | full global dump (deadlock triage path) |
| `TaskRegistry::dump_epoch_context(epoch, reason)` | focused dump for one epoch |
| `TaskRegistry::snapshot()` | captures the current task/epoch/value graph as structured records |
| `TaskRegistry::snapshot_best_effort()` | captures a structured snapshot without indefinite lock waits |
| `TaskRegistry::diagnose_snapshot(snapshot)` | derives blocked-task, missing-writer, and cycle annotations |
| `task_registry_report(snapshot, diagnostics)` | builds a presentation-native report from a captured snapshot |
| `TaskRegistry::graphviz_dot()` | returns current task/epoch/value DAG as Graphviz DOT |
| `TaskRegistry::graphviz_dot_best_effort()` | returns a partial DOT snapshot without indefinite lock waits |
| `TaskRegistry::graphviz_dot(snapshot, diagnostics)` | renders a captured snapshot and annotations as DOT |
| `TaskRegistry::dump_graphviz(stream)` | writes DOT to a C stream, stderr by default |
| `TaskRegistry::dump_graphviz_file(path)` | writes DOT to a file |
| `TaskRegistry::dump_graphviz_file_best_effort(path)` | writes a best-effort DOT snapshot to a file |
| `TaskRegistry::coroutine_exception_diagnostics_options()` | returns the active process-wide exception diagnostics policy |
| `TaskRegistry::set_coroutine_exception_diagnostics_options(options)` | enables, disables, or redirects automatic exception diagnostics |
| `TaskRegistry::reset_coroutine_exception_diagnostics_options()` | restores the environment-derived exception diagnostics policy |
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
5. hover task, epoch, and `co_await` elements in `xdot` to inspect full captured provenance

If output volume is too large during exception handling paths, use focused epoch dumps
(`dump_epoch_context`) at throw sites and reserve full dump for deadlock endpoints.

## Related References

- DAG debug example: `dag_debug_examples.md`
- Runtime semantics: `runtime_model.md`
- Exception routing: `exceptions_and_cancellation.md`
- Scheduler deadlock behavior: `schedulers.md`
- Fast lookup: `quick_reference.md`
