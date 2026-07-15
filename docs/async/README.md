# Async Documentation

This directory documents Uni20 async runtime behavior and dataflow reverse-mode AD.

It is designed for two use cases:

- onboarding new contributors
- giving experienced contributors (and AI agents) a fast reference for exact behavior

## Status and Scope

The docs are split into two groups:

| Group | Documents | Use this when |
|---|---|---|
| Primary (canonical) | `getting_started.md`, `coroutines_primer.md`, `runtime_model.md`, `buffers_and_awaiters.md`, `storage.md`, `kernel_authoring.md`, `cookbook.md`, `exceptions_and_cancellation.md`, `schedulers.md`, `tbb_execution_primer.md`, `reverse_mode_ad.md`, `task_registry_debug.md`, `dag_debug_examples.md`, `quick_reference.md` | you want current behavior and API usage |
| Supplemental (design/history) | `tensor_lifetime_and_dispatch_draft.md`, `audit_legacy_docs.md` | you need design rationale or migration context |

## Read Path by Audience

### New to Uni20 async

1. [`getting_started.md`](getting_started.md)
2. [`coroutines_primer.md`](coroutines_primer.md)
3. [`runtime_model.md`](runtime_model.md)
4. [`buffers_and_awaiters.md`](buffers_and_awaiters.md)
5. [`cookbook.md`](cookbook.md)
6. [`exceptions_and_cancellation.md`](exceptions_and_cancellation.md)
7. [`schedulers.md`](schedulers.md)
8. [`tbb_execution_primer.md`](tbb_execution_primer.md)
9. [`reverse_mode_ad.md`](reverse_mode_ad.md)
10. [`task_registry_debug.md`](task_registry_debug.md)

### Experienced contributor / AI agent quick lookup

1. [`quick_reference.md`](quick_reference.md)
2. [`buffers_and_awaiters.md`](buffers_and_awaiters.md)
3. [`kernel_authoring.md`](kernel_authoring.md)
4. [`exceptions_and_cancellation.md`](exceptions_and_cancellation.md)
5. [`task_registry_debug.md`](task_registry_debug.md)

## Document Map

| Document | Purpose | Best For |
|---|---|---|
| [`getting_started.md`](getting_started.md) | First runnable mental model and code patterns | New contributors |
| [`coroutines_primer.md`](coroutines_primer.md) | C++ coroutine background for this runtime | New contributors |
| [`runtime_model.md`](runtime_model.md) | Core semantics of `Async<T>`, epochs, ownership, ordering | Everyone |
| [`buffers_and_awaiters.md`](buffers_and_awaiters.md) | Two-capability buffer model, await paths, proxies, and exact await behavior | Runtime and kernel authors |
| [`storage.md`](storage.md) | Async value/alias kinds, shared ownership, assignment, and timeline rebinding | Tensor and runtime authors |
| [`kernel_authoring.md`](kernel_authoring.md) | All-async Tensor operation wrappers, lifetimes, alias checks, output preparation, and failures | Tensor operation authors |
| [`cookbook.md`](cookbook.md) | Common kernel patterns and debugging recipes | New and experienced contributors |
| [`exceptions_and_cancellation.md`](exceptions_and_cancellation.md) | Exception hierarchy, sink routing, cancellation details | Runtime contributors |
| [`schedulers.md`](schedulers.md) | `DebugScheduler`, `TbbScheduler`, `TbbNumaScheduler` behavior | Performance and integration work |
| [`tbb_execution_primer.md`](tbb_execution_primer.md) | oneTBB threads, arenas, concurrency, constraints, task groups, and resumable waits | TBB scheduler contributors |
| [`reverse_mode_ad.md`](reverse_mode_ad.md) | Dataflow reverse-mode concepts and `Var<T>` behavior | AD contributors |
| [`task_registry_debug.md`](task_registry_debug.md) | Debug dumps, stacktraces, runtime controls | Debugging and test triage |
| [`dag_debug_examples.md`](dag_debug_examples.md) | Graphviz DAG examples, including deadlock snapshots | Runtime debugging |
| [`quick_reference.md`](quick_reference.md) | Condensed API/error/env reference | Experienced developers and AI agents |
| [`tensor_lifetime_and_dispatch_draft.md`](tensor_lifetime_and_dispatch_draft.md) | Background reasoning for owner-retaining Tensor aliases and async dispatch | Design history |
| [`audit_legacy_docs.md`](audit_legacy_docs.md) | Divergence report vs older docs | Migration and cleanup |

## Example Programs

Runnable examples in `examples/` that pair well with this docs set:

- `examples/async/async_buffer_semantics_example.cpp`: read/write ownership, release ordering, cancellation, and exception routing
- `examples/async/async_buffer_await_paths_example.cpp`: borrowed and owning value, storage, and consuming await paths
- `examples/async/async_example.cpp`: basic read/write and `try_await(...)`
- `examples/async/async_ops_example.cpp`: expression DAG composition and `all(...)`
- `examples/async/async_tensor_matrix_product_example.cpp`: all-async Tensor matrix-product overwrite and update
- `examples/async/async_tbb_matrix_product_batch_example.cpp`: configurable parallel batch of async Tensor matrix
  products using `TbbScheduler`, with `fp32`, `fp64`, and configured `fp128` precision plus presentation-layer
  reporting
- `examples/async/async_tbb_reduction_example.cpp`: parallel scheduling with `TbbScheduler`
- `examples/async/async_dag_deadlock_tbb_example.cpp`: Graphviz snapshot of a deliberate TBB dataflow deadlock
- `examples/async/async_diagnostics_guide_example.cpp`: presentation-layer guide to task-registry reports,
  Graphviz/xdot provenance, stacktrace capability, and output customization
- `examples/async/async_coroutine_failure_example.cpp`: structured exception propagation through two async
  writer boundaries, with automatic live task-registry reporting and optional Graphviz capture

## Related Docs

- [Tensor operations, semantics, and Async support](../tensor/operations.md):
  current operation contracts and support matrix
- [Async Storage](storage.md): storage, async value kinds, timeline rebinding,
  and write-through assignment
- [Roadmap](../roadmap.md): broader architecture and roadmap context

## Ground Truth and Drift Policy

Canonical async documents record the intended runtime contract, tests encode
selected behavior, and source records the current implementation. If they
disagree, report and resolve the conflict rather than silently treating one as
authoritative. This docs set is kept aligned to:

- [Async runtime source map](../../src/uni20/async/README.md)
- [Async linalg wrappers](../../src/uni20/linalg/async/README.md)
- [Async tensor aliases](../../src/uni20/tensor/README.md)
- [Async tests](../../tests/async/)
- [Async examples](../../examples/async/)

If semantics change, update docs in this folder in the same PR.

## Legacy Local Drafts

The following local drafts may still exist in some working trees but are intentionally not
part of the tracked canonical docs set:

- `docs/async.md`
- `docs/async_api.md`
- `docs/async_design.md`
- `docs/async_new.md`
- `docs/Epoch.md`
