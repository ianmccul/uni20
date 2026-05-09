# Async DAG Debug Examples

`examples/async_dag_debug_example.cpp` demonstrates the Graphviz diagnostics path.
It builds a small async graph, writes snapshots before execution, while a task is
suspended, after partial progress, and after completion.

The examples label selected nodes with `Async<T>::debug_name(...)` and
`AsyncTask::debug_name(...)`. These calls are opt-in diagnostics only; they make the
DOT easier to read and compile down to no useful runtime output in dummy registry
builds.

`examples/async_dag_gallery_example.cpp` generates several graph shapes inspired by
the existing async examples:

- branching expression graphs from `async_ops_example.cpp`
- compact expression versus explicit-kernel graphs from `async_kernel_shapes_example.cpp`
- a small map/reduce tree from `async_tbb_reduction_example.cpp`

`examples/async_dag_deadlock_tbb_example.cpp` builds a deliberate two-task TBB
dataflow cycle, lets the worker threads block, then has the main thread sleep and
capture a best-effort DOT snapshot of the deadlocked DAG.

Build with DAG instrumentation:

```bash
cmake -S . -B ./build_codex/build_gcc13_debug_dag \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUNI20_DEBUG_DAG=ON
cmake --build ./build_codex/build_gcc13_debug_dag --target async_dag_debug_example
cmake --build ./build_codex/build_gcc13_debug_dag --target async_dag_gallery_example
cmake --build ./build_codex/build_gcc13_debug_dag --target async_dag_deadlock_tbb_example
```

`UNI20_DEBUG_DAG=ON` implies `UNI20_DEBUG_ASYNC_TASKS=ON`. New `Debug` build
trees default to DAG instrumentation, but an existing CMake cache can still hold
older or explicitly configured `OFF` values; pass `-DUNI20_DEBUG_DAG=ON` when
refreshing such a tree.

Run the example:

```bash
./build_codex/build_gcc13_debug_dag/examples/async_dag_debug_example /tmp/uni20-dag-example
./build_codex/build_gcc13_debug_dag/examples/async_dag_gallery_example /tmp/uni20-dag-gallery
./build_codex/build_gcc13_debug_dag/examples/async_dag_deadlock_tbb_example /tmp/uni20-dag-deadlock 2
```

The example writes DOT files similar to:

```text
/tmp/uni20-dag-example/async-dag-01-constructed.dot
/tmp/uni20-dag-example/async-dag-02-suspended.dot
/tmp/uni20-dag-example/async-dag-03-partial.dot
/tmp/uni20-dag-example/async-dag-04-complete.dot
/tmp/uni20-dag-example/async-dag-request.<pid>.<seq>.dot
/tmp/uni20-dag-example/async-dag-service.<pid>.<seq>.dot
```

The gallery example writes DOT files similar to:

```text
/tmp/uni20-dag-gallery/async-dag-gallery-branch-01-constructed.dot
/tmp/uni20-dag-gallery/async-dag-gallery-kernel-shapes-01-constructed.dot
/tmp/uni20-dag-gallery/async-dag-gallery-reduction-01-constructed.dot
/tmp/uni20-dag-gallery/async-dag-gallery-reduction-02-after-one-run.dot
/tmp/uni20-dag-gallery/async-dag-gallery-reduction-03-complete.dot
```

The deadlock example writes DOT files similar to:

```text
/tmp/uni20-dag-deadlock/async-dag-deadlock-tbb-01-scheduled.dot
/tmp/uni20-dag-deadlock/async-dag-deadlock-tbb-02-after-sleep.dot
```

Render a snapshot with Graphviz:

```bash
dot -Tsvg /tmp/uni20-dag-example/async-dag-02-suspended.dot \
  -o /tmp/uni20-dag-example/async-dag-02-suspended.svg
```

## Interpreting the Graph

The graph is a point-in-time scheduler and dependency snapshot:

| Graph item | Meaning |
|---|---|
| `data_N` | `Async<T>` value node. Labels come from `Async<T>::debug_name(...)` when set and include `storage=...`, `state=...`, and optional `value=...` metadata. |
| `task_N` | Coroutine task. The label shows the optional task name, task id, lifecycle state, transition count, and coroutine pointer. |
| `epoch_N` | Read/write ordering generation for one async value. Epoch nodes explain why a task is currently allowed to run or blocked. |
| `arg read` | Coarse constructor-time dependency from a `ReadBuffer` parameter. |
| `arg write` | Coarse constructor-time dependency from a `WriteBuffer` parameter. |
| `co_await read` | Fine-grained read dependency observed when the coroutine actually awaited the buffer. |
| `co_await write` | Fine-grained write dependency observed when the coroutine awaited/acquired a writer. |
| `created_at` | Best-effort source line where a task or epoch was created, shown only when stacktrace support is available. |
| `scheduled_at` | Best-effort source line where a task was first submitted to a scheduler. |
| `awaiting_at` | Best-effort source line for the most recent concrete `co_await` site on a suspended task. |
| `epochs` | Association between a value node and an epoch context. |
| `next` | Link from one epoch generation to the next generation of the same async value. |
| `await read` | A reader task is currently queued on that epoch. |
| `await write` | A writer task is currently queued on that epoch. |

Color conventions:

- blue cylinders are async value nodes
- orange boxes are coroutine tasks
- gray ovals are epoch contexts
- pale yellow tasks are blocked
- red or pink nodes/edges are diagnostic highlights, such as missing writers or
  likely dependency cycles

Completed tasks and some epoch contexts are destroyed as execution progresses, so
later snapshots can be sparse. Prefer pre-execution or suspended snapshots when
you want to inspect the shape of a computation.

For data nodes, `storage=0x...` is the `shared_storage<T>` control-block identity.
`state=constructed` means a `T` object exists at snapshot time; `state=unconstructed`
means the storage exists but does not currently hold a constructed `T` object.
`state=invalid` means no storage control block was available. Shape-like objects
that expose `mdspan().extents()` or `extents()` may also show
`value=shape=(...)`; built-in scalar values show their value. Other constructed
objects omit the `value=...` line by default. Custom types can provide a one-line
`uni20_async_debug_value(T const&)` overload in their own namespace to override
the default.

The `02-suspended` graph is the most useful one to inspect: it contains the
scheduled tasks, async value nodes when `UNI20_DEBUG_DAG=ON`, constructor-captured
`arg read`/`arg write` edges, concrete `co_await read` edges, and the blocked
reader waiting for `late_input`. The diagnostic note and highlighted nodes mark
this as a missing-writer case until the later writer is scheduled.

When built with `UNI20_ENABLE_STACKTRACE=ON` and compiler/library support, visible
labels include compact source locations. Full stacktraces are emitted as Graphviz
tooltips on task, epoch, and concrete `co_await` edges. Use
`UNI20_DEBUG_DAG_STACKTRACE_FRAMES` or `TaskRegistry::set_stacktrace_options(...)`
to cap the tooltip trace length; `0` keeps visible source labels but hides full
trace text, and `all`/`full`/`unlimited`/`max` removes the cap. Without stacktrace
support, the same graph structure is emitted without those provenance labels.

In the gallery output, `reduction-01-constructed` is the best first graph to view.
It shows the full binary tree before execution. `kernel-shapes-01-constructed`
is useful for comparing how expression helpers and explicit coroutine kernels
appear in the same task registry snapshot. The labels in these graphs are examples
of how call sites can name meaningful values and kernels without changing the DAG
semantics.

In the TBB deadlock output, `02-after-sleep` is the useful snapshot. It should show
two suspended tasks: `left = right + 1` waits to read `right`, while
`right = left + 1` waits to read `left`. The diagnostic note should report blocked
tasks and a dependency cycle, and the cycle participants are highlighted in red.
The example intentionally exits after writing the DOT instead of calling
`get_wait()` or `run_all()`, since the dataflow cycle cannot complete.

The same example also demonstrates:

- direct file dumps with `TaskRegistry::dump_graphviz_file(...)`
- best-effort file dumps with `TaskRegistry::dump_graphviz_file_best_effort(...)`
- queued programmatic requests with `request_graphviz_dump()` plus
  `service_debug_requests(...)`
- control-file requests through `start_diagnostics_service(...)`

If either example is built without `UNI20_DEBUG_ASYNC_TASKS`, it still compiles,
but the registry is a dummy. The executable exits before writing DOT files and
prints the DAG-instrumented configure/build command to use instead.
