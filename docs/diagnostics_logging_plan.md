# Diagnostics and Logging Plan

Status: design note, not current API behavior. This note records the intended
direction for ordinary Uni20 diagnostics and logging beyond the current
`trace.hpp` macros.

This note predates the screen-display design in
[Screen Display Layer Plan](display_layer_plan.md). The display layer is the
primary plan for human-facing terminal/Python progress output. This document
remains useful for durable logging, structured events, async queueing, and sink
routing concerns that overlap with display but should not make ordinary screen
output feel like a heavyweight logging framework.

## Motivation

The existing trace/check/precondition macros are useful for local debugging,
assertions, and emergency diagnostics, but they are not the right abstraction
for normal warnings, progress messages, solver statistics, or long-running task
logs.

As Uni20 becomes more asynchronous, direct writes to `std::cout` / `std::cerr`
from library code will become increasingly fragile:

- messages from worker threads can interleave unpredictably,
- Python `sys.stdout` / `sys.stderr` capture will not reliably order with C++ stdio,
- pytest/Jupyter/logging integrations need Python-aware output paths,
- users may want diagnostics in files, terminal reports, or completely disabled,
- tests should be able to assert on diagnostics without parsing text.

The core rule should be: ordinary library code does not print. It emits diagnostics into a context.

## Separation from `trace.hpp`

Keep `trace.hpp` focused on correctness boundaries and emergency output:

- `PANIC`, `CHECK`, `PRECONDITION`: abort-path diagnostics.
- `ERROR` / `ERROR_IF`: exception/error boundary diagnostics.
- Local debug traces where immediate source-location output is desired.

Add a separate diagnostics/logging layer for:

- warnings,
- progress messages,
- solver iteration summaries,
- convergence/stagnation notices,
- async task debug logs,
- structured performance/statistics events.

The diagnostics layer should use the presentation layer only for rendering. It should not decide terminal glyphs, colors, wrapping, or ASCII fallback itself.

## Separation from Async DAG Debugging

This is also separate from the async task-registry and Graphviz DAG debugging
path. The DAG machinery answers structural runtime questions: which tasks exist,
which buffers they await, where a dependency cycle or missing writer appears, and
how to render that dependency graph for debugging.

The diagnostics/logging layer proposed here is for ordinary events emitted by
library algorithms, including algorithms that happen to run inside async tasks.
For example, a Krylov solver running asynchronously should be able to report
iteration progress, convergence notices, stagnation warnings, and final
statistics without writing directly to `std::cout` and without depending on the
DAG debug renderer. Async task metadata can be attached to those events when it
is useful, but the event is still solver/logging data rather than graph topology.

## Core Model

Introduce structured diagnostic events:

```cpp
struct diagnostic_event {
  diagnostic_severity severity;
  std::string context;
  std::string message;
  source_location where;
  std::thread::id thread_id;
  std::optional<std::uint64_t> task_id;
  std::optional<std::string> task_name;
  // Optional structured key/value payload for solver stats, residuals, etc.
};
```

The exact payload representation can be decided later. The important point is that the event is structured at emission time and rendered only at the sink boundary.

## Contexts

Use named diagnostic contexts:

```cpp
auto& log = diagnostics::context("krylov");
log.debug("restart", iteration, residual);
log.warning("residual stagnated", residual);
```

Likely built-in contexts:

- default/global,
- krylov,
- async,
- tensor,
- linalg,
- python-bindings,
- cuda/accelerator, when that layer exists.

Most contexts should be quiet by default. Warnings and important notices can go to the default terminal/Python sink; verbose progress/debug contexts should normally be disabled unless explicitly enabled.

## Sinks

Diagnostics should be routed through pluggable sinks:

- null sink: discard events,
- terminal sink: render with the presentation layer,
- file sink: deterministic log output,
- vector sink: collect events for tests,
- Python sink: buffer and replay through Python `sys.stdout` / `sys.stderr` or warnings machinery,
- composite sink: fan out to multiple sinks.

For Python, avoid direct C++ writes to `stdout` / `stderr` except for abort-path emergency diagnostics. A Python binding should install a Python-aware sink or drain events at safe points.

## Async Behavior

Async tasks should not generally render directly. They should emit events into a context-owned queue/buffer:

```text
worker task -> diagnostic event queue -> owner/main-thread drain/render
```

This gives deterministic ordering opportunities and avoids mixed C++/Python buffering problems. The first implementation can be simple: a mutex-protected queue plus explicit drain points. More sophisticated lock-free or per-task buffers can come later if needed.

Events emitted from async tasks should include task metadata early, even if the first renderer ignores it. A warning with a task id/name is much more useful than a loose line from an unknown worker thread.

## Configuration

Start with programmatic configuration:

```cpp
diagnostics::context("krylov").set_level(diagnostic_level::debug);
diagnostics::context("krylov").set_sink(file_sink("krylov.log"));
```

Environment-based configuration can come later, for example:

```bash
UNI20_LOG=krylov:info,async:warn,*:error
UNI20_LOG_FILE=uni20.log
```

The default should remain conservative: important warnings are visible, debug/progress spam is off.

## Solver Diagnostics

Krylov and other iterative solvers should use this layer for optional progress and convergence reporting:

- iteration number,
- matvec count,
- Ritz value,
- residual norm,
- restart information,
- stagnation/loss-of-orthogonality warnings,
- final solver statistics.

The same structured events can support terminal progress, test assertions, benchmark metadata, and Python callbacks without changing solver internals.

## Open Design Points

- Exact representation of structured key/value payloads.
- Whether contexts are global singletons, attached to an execution context, or both.
- How async child tasks inherit parent diagnostic contexts.
- How Python bindings drain diagnostics during long-running calls.
- Whether warnings should also integrate with Python's `warnings` module.
- How much of the presentation layer should be bound to Python.
