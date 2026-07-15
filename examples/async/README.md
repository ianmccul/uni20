# Async Examples

These programs demonstrate Uni20's epoch-ordered async runtime from basic
`Async<T>` expressions through TBB execution, Tensor kernels, failures, and DAG
diagnostics.

## Fundamentals

- `async_example.cpp`: basic reads, writes, `try_await`, and scheduler progress.
- `async_example2.cpp`: interactive expression graphs, assignment/rebinding,
  and stepping `DebugScheduler` one task at a time.
- `async_ops_example.cpp`: expression-DAG composition and `all(...)`.
- `async_buffer_semantics_example.cpp`: read/write ownership, release ordering,
  cancellation, and exception routing.
- `async_buffer_await_paths_example.cpp`: borrowed and owning value, storage,
  and consuming await paths.
- `future_example.cpp`: a graph blocked on `FutureValue` until late fulfillment.

## Kernel And Scheduler Shapes

- `async_kernel_shapes_example.cpp`: equivalent compact-expression and explicit
  captureless-coroutine implementations.
- `async_tensor_matrix_product_example.cpp`: async Tensor matrix-product
  overwrite and update through normal kernel dispatch.
- `async_tbb_matrix_product_batch_example.cpp`: configurable parallel Tensor
  GEMM batch with scheduler concurrency, backend, and precision selection.
- `async_tbb_reduction_example.cpp`: paused construction and parallel TBB
  execution of a map/reduction tree.
- `async_fib_example.cpp`: recursively scheduled Fibonacci DAG on TBB.
- `async_ad_stress_example.cpp`: configurable AD/dataflow stress workload with
  dead and failing branches.

## DAG And Failure Diagnostics

- `async_dag_debug_example.cpp`: basic task-registry Graphviz capture.
- `async_dag_gallery_example.cpp`: branching, kernel-shape, and lifecycle DAG
  snapshots.
- `async_dag_deadlock_tbb_example.cpp`: deliberate TBB dataflow deadlock and
  diagnostic snapshot.
- `async_diagnostics_guide_example.cpp`: presentation-layer guide to registry
  reports, stacktrace capability, Graphviz output, and runtime controls.
- `async_coroutine_failure_example.cpp`: deliberate coroutine failure propagated
  through writer boundaries, with optional automatic Graphviz capture.

## Retained Regression Program

- `bug.cpp`: minimal TBB assignment-chain/get-wait regression program. The
  filename is historical; use the canonical examples above for new code.

The deadlock and coroutine-failure examples intentionally exercise failure
paths. See the [examples index](../), [Async Documentation](../../docs/async/),
[Async Kernel Authoring](../../docs/async/kernel_authoring.md), and
[DAG Debug Examples](../../docs/async/dag_debug_examples.md).
