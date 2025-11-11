# Tensor Design and Expression Template Roadmap

## 1. Purpose and Scope
This document captures the current state of Uni20's tensor stack, critiques architectural pain points, and outlines the roadmap toward a `TensorExpression`-centric system. The goal is to ensure that upcoming expression templates, asynchronous execution, and heterogeneous backends integrate cleanly with the existing kernel ecosystem.

## 2. Existing Tensor Container Architecture
Uni20 exposes tensors primarily through the `Tensor` alias, with `BasicTensor` providing ownership semantics while inheriting `TensorView` for mdspan compatibility. These components rely on trait-based storage, layout, and accessor policies that determine data residency and iteration behavior. Today, tensors default to `VectorStorage`, a CPU-resident storage policy linked directly to the CPU kernel path. Layout helpers and accessor factories are mdspan-inspired and must evolve to remain compatible with expression-aware evaluation.

## 3. Current Kernel and Algorithm Surface
Level-1 tensor routines—such as `assign`, `sum_view`, and `apply_unary_inplace`—construct merged iteration plans that traverse arbitrary strided tensors. Tensor contractions lower to `kernel::contract`, which recursively leverages CPU backends like `GemmLoop`. These kernels define the baseline execution model that `TensorExpression` nodes must eventually target, ensuring continuity for existing workloads.

## 4. Backend Ecosystem Overview
Backends are selected through build-time tracing infrastructure that chooses BLAS, MKL, or future accelerators based on availability. Backend hooks expose CPU or GPU pathways while emitting telemetry via `UNI20_API_CALL` macros. Expression evaluation must surface backend tags derived from storage policies so that execution routes seamlessly through the established backend layer.

## 5. Identified Gaps and Weaknesses
The current tensor stack eagerly materializes intermediate results, preventing fusion and hindering scheduler-friendly execution. Storage policies are CPU-centric, lacking explicit async or GPU handles. Views and owning tensors lack a shared expression tree, creating friction for composition. Kernels execute synchronously, forcing async callers to wrap entire operations rather than subexpressions. Backend selection remains manual, requiring callers to supply explicit tags that could instead be inferred from context.

## 6. Design Goals for `TensorExpression`
The expression system must unify owning tensors, views, and async nodes under a single template hierarchy. Expression nodes should represent unary/binary element-wise operations, contractions, and linear-algebra factorizations while preserving shape metadata for validation and backend selection. Iteration-plan machinery must remain usable so that expression evaluation can reuse existing kernels without redundant implementations.

## 7. Expression Template Building Blocks
Expression templates will introduce node categories for tensor leaves, scalar literals, unary/binary operations, contraction nodes, and linear-algebra nodes (e.g., QR, SVD). Trait detection and compile-time introspection should infer result extents, value categories, and preferred backend tags. Lowering rules include:

- Binary add nodes collapse into fused iteration plans analogous to `sum_view`, materializing only when required.
- Contraction nodes delegate to `kernel::contract` while propagating operand-derived backend preferences.
- Linear algebra nodes (QR, SVD) map to BLAS, MKL, or future GPU libraries, reusing existing façade APIs.

## 8. Evaluation Strategies
Evaluation strategies balance eager and lazy materialization. Simple element-wise expressions can remain lazy until materialization is demanded, enabling loop fusion across chained additions or scalar operations. Contraction nodes dispatch directly to `kernel::contract` to leverage tuned recursion, whereas linear algebra nodes hand off to BLAS-like backends. Illustrative flows include:

- **Tensor addition:** `auto C = A + B;` builds a binary node whose evaluation merges iteration plans and issues a fused kernel when `C` materializes or is assigned.
- **Tensor contraction:** `auto T = contract(A, B, modes);` constructs a contraction node that lowers to `kernel::contract` with backend inference from operands.
- **Linear algebra:** `auto [Q, R] = qr(A);` creates an expression node that defers to the backend-specific QR routine, supporting CPU (MKL) and future GPU solvers.

## 9. Async-Aware Tensor Execution
`Async<T>` orchestrates epoch-based reads, writes, and coroutines. To embrace async evaluation, the design should provide `AsyncTensor` wrappers or expression-aware awaitables that schedule evaluation lazily. Expression nodes should participate in scheduler semantics, propagating cancellation and exceptions under `DebugScheduler` and `TbbScheduler`. Example patterns include:

- **Async addition:** `AsyncTensor C = async_add(A, B);` returns an awaitable expression that schedules evaluation on first `co_await`.
- **Async contraction:** `AsyncTensor T = async_contract(A, B);` queues backend work on the global scheduler, returning when `kernel::contract` completes.
- **Async QR/SVD:** Awaitable nodes wrap backend linear algebra tasks, integrating with coroutine lifetimes while respecting storage ownership.

## 10. GPU and Heterogeneous Considerations
Uni20's CUDA integration currently centers on `CudaTask`, hinting at future GPU schedulers. Expression templates must understand GPU-capable storage policies, emitting device kernels when operands reside on GPU memory. Requirements include asynchronous launches, stream/queue management, and explicit data movement when mixing CPU and GPU tensors. Examples encompass:

- **GPU tensor addition:** GPU-backed storage triggers a GPU kernel launch with async completion propagated through the scheduler.
- **Mixed-device contraction:** Expression lowering coordinates explicit transfers or unified-memory semantics before calling device or host kernels.
- **GPU QR/SVD:** Once cuSOLVER bindings exist, linear-algebra nodes dispatch to GPU routines, mirroring CPU fallbacks when hardware is unavailable.

## 11. Interaction with Backend Selection and Tracing
Expression evaluation must emit backend events via `UNI20_API_CALL` to preserve tracing fidelity. Backend preferences emerge from storage tags, explicit overrides, or runtime heuristics. Tracing metadata distinguishes CPU and GPU execution, ensuring that expression-driven evaluations remain observable and debuggable across heterogeneous devices.

## 12. Migration Strategy and Open Questions
A phased rollout mitigates risk:

1. Introduce `TensorExpression` traits and wrap existing level-1 operations without altering semantics.
2. Extend coverage to contraction nodes, ensuring `kernel::contract` interop remains intact.
3. Integrate async-aware evaluation, enabling awaitable tensors and scheduler-friendly execution.
4. Add GPU storage policies and backend bridges, culminating in GPU-aware linear algebra nodes.

Open questions include expression lifetime semantics, error propagation across async boundaries, scheduler integration granularity, and multi-device coherence policies. Testing strategy should evolve from unit tests for expression nodes to async integration suites and backend regression testing as GPU features emerge.

## Appendices
- **Glossary:** tensor, mdspan, expression node, scheduler, backend.
- **Key Headers:** `src/tensor/tensor.hpp`, `src/tensor/contract.hpp`, `src/async/async.hpp`, `src/backend/blas_backend.hpp`.

## Testing
Tests were not run for this documentation-only change.
