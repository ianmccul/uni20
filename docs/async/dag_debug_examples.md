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

Build with DAG instrumentation:

```bash
cmake -S . -B ./build_codex/build_gcc13_debug_dag \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUNI20_DEBUG_DAG=ON \
  -DUNI20_DEBUG_ASYNC_TASKS=ON
cmake --build ./build_codex/build_gcc13_debug_dag --target async_dag_debug_example
cmake --build ./build_codex/build_gcc13_debug_dag --target async_dag_gallery_example
```

Run the example:

```bash
./build_codex/build_gcc13_debug_dag/examples/async_dag_debug_example /tmp/uni20-dag-example
./build_codex/build_gcc13_debug_dag/examples/async_dag_gallery_example /tmp/uni20-dag-gallery
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

Render a snapshot with Graphviz:

```bash
dot -Tsvg /tmp/uni20-dag-example/async-dag-02-suspended.dot \
  -o /tmp/uni20-dag-example/async-dag-02-suspended.svg
```

The `02-suspended` graph is the most useful one to inspect: it contains the
scheduled tasks, async value nodes when `UNI20_DEBUG_DAG=ON`, constructor-captured
`arg read`/`arg write` edges, concrete `co_await read` edges, and the blocked
reader waiting for `late_input`. The diagnostic note and highlighted nodes mark
this as a missing-writer case until the later writer is scheduled.

In the gallery output, `reduction-01-constructed` is the best first graph to view.
It shows the full binary tree before execution. `kernel-shapes-01-constructed`
is useful for comparing how expression helpers and explicit coroutine kernels
appear in the same task registry snapshot. The labels in these graphs are examples
of how call sites can name meaningful values and kernels without changing the DAG
semantics.

The same example also demonstrates:

- direct file dumps with `TaskRegistry::dump_graphviz_file(...)`
- best-effort file dumps with `TaskRegistry::dump_graphviz_file_best_effort(...)`
- queued programmatic requests with `request_graphviz_dump()` plus
  `service_debug_requests(...)`
- control-file requests through `start_diagnostics_service(...)`

If the example is built without `UNI20_DEBUG_ASYNC_TASKS`, it still compiles, but
the registry is a dummy and the DOT files contain only an empty graph.
