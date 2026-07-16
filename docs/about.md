# About Uni20

**Status:** current capabilities overview, updated 2026-07.

Uni20 is a C++23 tensor-network research library built around three ideas:

1. tensor operations should lower through explicit, inspectable backend kernels;
2. asynchronous execution should preserve data lifetime and dependency order;
3. symmetry and storage metadata are part of the mathematical object and must
   survive every lowering step.

The project is still in active design and does not promise a stable public API.
It now has several complete vertical slices, however, rather than only isolated
prototypes. Dense tensor operations can run through ordered CPU, BLAS, and
LAPACK backends; the same operations can be scheduled over `Async<Tensor>`
values; and runnable examples exercise those paths with oneTBB and Uni20's
presentation and diagnostic layers.

## Current Capabilities

The table distinguishes implemented facilities from foundations and planned
work. Backend and scalar coverage varies by operation; the linked subsystem
guides define the exact contracts.

| Area | Current status |
|---|---|
| Dense tensors | Implemented owning `Tensor` types with compile-time rank, runtime extents, column-major, row-major, and strided layouts, generated tensors, lazy conjugation, explicit materialization, reshape, and variadic elementwise overwrite/update operations. |
| Dense backend dispatch | Implemented operation-value dispatch with compile-time type probing, structured runtime decline reasons, ordered fallback, callable-carrying elementwise operations, and optional dispatch diagnostics. CPU reference, BLAS, and initial LAPACK paths are active. |
| Dense linear algebra | Implemented tensor/mdspan front ends include accessor-aware elementwise transforms and copy, GEMM, GEMV, matrix initialization, matrix exponential, self-adjoint and nonsymmetric eigensystems, Schur operations, and tridiagonal eigensystems. Backend coverage is operation-specific. |
| Async runtime | Implemented `Async<T>`, epoch-ordered read/write buffers, exception and cancellation propagation, `DebugScheduler`, `TbbScheduler`, `TbbNumaScheduler`, scheduler-aware waits, task-registry diagnostics, stacktraces where available, and Graphviz DAG snapshots. |
| Async tensor operations | Implemented lifetime-safe conjugating and reshape aliases, variadic elementwise overwrite/update operations, matrix products with immediate or async scalar parameters, and preserving or storage-consuming self-adjoint `eigh` with independent async outputs. |
| Krylov algorithms | Implemented matrix-free symmetric/Hermitian Lanczos, nonsymmetric Arnoldi, generalized problems, Krylov exponential action, and an independent Taylor exponential-action reference. Projected dense work lowers through Uni20 linalg dispatch. |
| Scalar support | `float32`, `float64`, real and complex paths are first-class. Configured MPLAPACK builds add binary128 probes and selected dense/Krylov paths. |
| Presentation and diagnostics | Implemented semantic reports, terminal/plain/ASCII rendering, width-aware tables, mdspan previews, structured kernel errors, source locations, and optional stacktrace formatting. |
| Reverse-mode AD | Async value-level `Var<T>` and `ReverseValue<T>` foundations are implemented and tested. Tensor linalg differentiation is not yet wired through the operation layer. |
| Symmetry and block sparsity | Quantum-number, U(1), block-space, local-space, and selection-rule foundations exist. A complete symmetry-aware `BlockTensor` and its lowering pipeline remain design work. |
| Python | Nanobind smoke bindings and build metadata are implemented. Tensor operations, async values, packaging, and notebook display are future work. |
| CUDA and distributed execution | CUDA/cuSOLVER target structure and design work exist, but there is no complete device scheduler or distributed tensor execution path yet. |

## The Working Vertical Slice

A synchronous dense tensor operation follows this path:

```text
Tensor operation and output policy
  -> storage-derived backend selector
  -> resolved mdspan operands
  -> operation-value dispatch walk
  -> BLAS, LAPACK, or CPU-reference kernel
```

An asynchronous operation reuses that path rather than defining a second
backend system:

```text
Async<Tensor> operands
  -> epoch enrollment and scheduled coroutine
  -> await tensor values and async scalar parameters
  -> the same synchronous Tensor operation
  -> the same mdspan/backend dispatch walk
```

For example, the implemented async matrix-product API can schedule an overwrite
followed by an update on the same output timeline:

```cpp
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/tensor.hpp>

using matrix_type = uni20::DenseMatrix<double>;

uni20::async::DebugScheduler scheduler;
uni20::async::ScopedScheduler use_scheduler(&scheduler);

uni20::async::Async<matrix_type> lhs = make_lhs();
uni20::async::Async<matrix_type> rhs = make_rhs();
uni20::async::Async<matrix_type> output;
uni20::async::Async<double> update_scale = 0.5;

uni20::linalg::assign_product(output, lhs, rhs);
uni20::linalg::add_product(output, lhs, rhs, update_scale);

auto const& result = output.get_wait(scheduler);
```

The operation wrapper owns output construction and alias checks. The scheduler
owns dependency ordering. Tensor storage selects the default backend list. Leaf
kernels see resolved mdspans rather than Tensor or Async objects.

## Examples That Exercise Real Paths

These programs are built from `examples/CMakeLists.txt` and are also used as
testable integration slices:

- `async_tensor_matrix_product_example`: small end-to-end
  `Async<DenseMatrix>` overwrite and update through backend dispatch.
- `async_tbb_matrix_product_batch_example`: configurable parallel matrix
  products with matrix size, product count, thread count, backend choice,
  `fp32`, `fp64`, optional `fp128`, validation, and presentation-layer timing
  reports.
- `gemm_dispatch_example`: matrix inputs/results plus the ordered backend walk
  and runtime decline diagnostics.
- `kernel_dispatch_error_example`: structured failure reporting for exhausted
  runtime candidates and type-level rejection at a dynamic boundary.
- `krylov_matrix_market_example` and
  `krylov_nonsymmetric_matrix_market_example`: native Lanczos/Arnoldi solves on
  Matrix Market fixtures.
- `krylov_exponential_matrix_market_probe_example`: adaptive Krylov exponential
  action and estimator diagnostics.
- `async_diagnostics_guide_example` and
  `async_coroutine_failure_example`: task provenance, exception propagation,
  terminal reports, stacktrace capability, and optional Graphviz output.

## Design Boundaries

Uni20 is a C++-first development library, not a released NumPy replacement or
a complete tensor-network application suite. In particular:

- APIs may be renamed or reshaped when a clearer design emerges.
- Current tensor rank is compile-time because the implementation is mdspan-based;
  dynamic-rank tensors require a distinct descriptor design.
- Async wrappers are added operation by operation because output construction,
  mutation, consumption, and multi-output failure routing have different
  contracts.
- There is no implicit host fallback for future device tensors and no implicit
  dense fallback for symmetry-aware tensors.
- Python validates only the extension and build-information boundary today.
- Existing MPS, model, symmetry, CUDA, MPI, and contraction prototypes should
  not be mistaken for one complete end-user tensor-network workflow.

## Where to Continue

- [Getting Started](getting_started.md): configure, build, and test Uni20.
- [Architecture Overview](architecture/overview.md): implemented layers and
  planned extensions.
- [Tensor Operations](tensor/operations.md): canonical ownership, output,
  layout, and async support matrix.
- [Kernel Dispatch](architecture/kernel_dispatch.md): backend capability and decline model.
- [Async Documentation](async/): runtime semantics and kernel-authoring
  guides.
- [Krylov Algorithms](krylov/algorithms.md): solver behavior and supported
  scalar types.
- [Roadmap](roadmap.md): current priorities and work that remains.
