# About Uni20

**Status:** current capabilities overview, updated 2026-09.

Uni20 is a C++23 tensor-network research library built around three ideas:

1. an algorithm written over `T` should lift naturally to `Async<T>`, with
   epochs and coroutines managing lifetime, dependency order, and suspension;
2. each numerical backend should either perform an operation or cleanly
   decline, leaving ordered kernel dispatch to select the next implementation;
3. symmetry and storage metadata are part of the mathematical object and must
   survive every lowering step.

The project is still in active design and does not promise a stable public API.
It now has several connected vertical slices, however, rather than only
isolated prototypes. Dense tensor operations run through ordered CPU, BLAS,
LAPACK, CUDA, cuBLAS, and cuSOLVER backends; the same operation surface can be
scheduled over `Async<Tensor>` values; and the first finite U(1) two-site DMRG
calculation runs on CPU or with its working state resident on one CUDA device.

## Current Capabilities

The table distinguishes implemented facilities from foundations and planned
work. Backend and scalar coverage varies by operation; the linked subsystem
guides define the exact contracts.

| Area | Current status |
|---|---|
| Dense tensors | Implemented owning `Tensor` types with compile-time rank, runtime extents, column-major, row-major, and strided layouts, generated tensors, lazy conjugation, explicit materialization, reshape, and variadic elementwise overwrite/update operations. |
| Dense backend dispatch | Implemented operation-value dispatch with compile-time type probing, structured runtime decline reasons, ordered fallback, callable-carrying elementwise operations, and optional dispatch diagnostics. CPU reference, BLAS, and initial LAPACK paths are active. |
| Dense linear algebra | Implemented tensor/mdspan front ends include accessor-aware elementwise transforms and copy, reductions and matrix norms, contraction, GEMM, GEMV, matrix initialization and exponential, dense solve, QR/LQ, exact and truncating SVD, self-adjoint and nonsymmetric eigensystems, Schur operations, and tridiagonal eigensystems. Backend coverage is operation-specific. |
| Async runtime | Implemented `Async<T>`, epoch-ordered read/write buffers, exception and cancellation propagation, host `DebugScheduler`/`TbbScheduler`/`TbbNumaScheduler`, unified host/multi-device `DebugCudaScheduler` and `TbbCudaScheduler`, scheduler-aware waits, task-registry diagnostics, stacktraces where available, and Graphviz DAG snapshots. |
| Async tensor operations | Implemented lifetime-safe aliases; preserving and consuming reshape/materialization; copy, transform, contraction, reduction, matrix product, matrix initialization, matrix exponential, matrix norm, and dense solve wrappers; and preserving or storage-consuming QR, LQ, self-adjoint `eigh`, exact SVD, and truncating SVD with independent async outputs. |
| Krylov algorithms | Implemented matrix-free symmetric/Hermitian Lanczos, nonsymmetric Arnoldi, generalized problems, Krylov exponential action, and an independent Taylor exponential-action reference. Projected dense work lowers through Uni20 linalg dispatch. |
| Scalar support | `float32`, `float64`, real and complex paths are first-class. Configured MPLAPACK builds add binary128 probes and selected dense/Krylov paths. |
| Presentation and diagnostics | Implemented semantic reports, terminal/plain/ASCII rendering, width-aware tables, mdspan previews, structured kernel errors, source locations, and optional stacktrace formatting. |
| Reverse-mode AD | Async value-level `Var<T>` and `ReverseValue<T>` foundations are implemented and tested. Tensor linalg differentiation is not yet wired through the operation layer. |
| Symmetry and block sparsity | Implemented bosonic Abelian U(1) `BlockTensor` with typed domain/codomain spaces, sparse and complete block patterns, separate and packed host storage, packed CUDA storage, mapped permutation/repartition views, structure-preserving linear operations, generalized adjacent contraction, diagonal-block storage, staged per-charge block SVD and truncation, and matrix-free Krylov adaptation. MPI placement and broader symmetry categories remain future work. |
| Python | Nanobind smoke bindings and build metadata are implemented. Tensor operations, async values, packaging, and notebook display are future work. |
| CUDA and distributed execution | Scoped process-wide CUDA runtime ownership, canonical per-device resources, typed buffers, completion ledgers, stream and provider-handle pools, unified debug/oneTBB host/multi-device task schedulers, `CudaTensor`, reference elementwise kernels, cuBLAS lowering, and cuSOLVER SVD are implemented. A single-device resident U(1) DMRG path exercises the stack. Multi-GPU placement, MPI communication, and broader CUDA kernel coverage remain future work. |

## Working Vertical Slices

A synchronous dense tensor operation follows this path:

```text
Tensor operation and output policy
  -> storage-derived backend selector
  -> fixed operands normalized to mdspecs
  -> operation-value dispatch walk
  -> selected backend acquires execution-domain access
  -> BLAS, LAPACK, or CPU-reference kernel
```

An asynchronous operation reuses that path rather than defining a second
backend system:

```text
Async<Tensor> operands
  -> epoch enrollment and scheduled coroutine
  -> await tensor values and async scalar parameters
  -> operation-specific output preparation
  -> fixed operands normalized to mdspecs
  -> the same operation-tag backend dispatch walk
  -> selected backend acquires execution-domain access
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
owns dependency ordering. Tensor storage selects the default backend list.
Backend operation entry points see normalized mdspecs, and their leaf kernels
see mdspans resolved under the appropriate execution-domain leases rather than
Tensor or Async objects.

The finite DMRG path composes those lower layers without flattening its U(1)
structure:

```text
finite MPS, MPO, and directional environment cache
  -> legal BlockTensor contraction worklists
  -> matrix-free effective-Hamiltonian applications
  -> fixed-work local Lanczos solve
  -> staged per-charge SVD and global state selection
  -> directional two-site replacement and environment refresh
```

The host path uses block-level oneTBB scheduling around single-threaded dense
providers. The CUDA path retains MPS sites, environments, centers, Krylov
vectors, and SVD factors in packed device storage while lowering dense block
work through CUDA, cuBLAS, and cuSOLVER. Compact MPO coefficients and singular
value selection metadata remain on the host.

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
- `cuda_hello_world_example`: CUDA build/runtime discovery, visible-device
  capabilities, scoped process-wide initialization, and per-device resource
  smoke checks.
- `block_tensor_example`, `block_tensor_product_state_example`, and
  `block_tensor_aklt_example`: construction and contraction of symmetry-aware
  tensors without dense projection.
- `block_tensor_svd_truncation_example` and
  `block_tensor_svd_nullspace_example`: staged sector-preserving decomposition,
  selection, and independent materialization of kept or null-space factors.
- `spin_half_heisenberg_dmrg_example`: finite U(1) two-site DMRG on CPU or a
  single resident CUDA device, including exact small-chain checks and larger
  reproducible benchmark controls.

## Design Boundaries

Uni20 is a C++-first development library, not a released NumPy replacement or
a complete tensor-network application suite. In particular:

- APIs may be renamed or reshaped when a clearer design emerges.
- Current tensor rank is compile-time because the implementation is mdspan-based;
  dynamic-rank tensors require a distinct descriptor design.
- Core execution interfaces target scheduled, multi-process HPC systems with
  x86-64 or AArch64 hosts and potentially many GPUs per node; workstation
  topology and device numbering are not architectural limits.
- Async wrappers are added operation by operation because output construction,
  mutation, consumption, and multi-output failure routing have different
  contracts.
- There is no implicit host fallback for future device tensors and no implicit
  dense fallback for symmetry-aware tensors.
- Python validates only the extension and build-information boundary today.
- The pure-Uni20 U(1) finite two-site DMRG path is implemented for host and
  single-device resident-CUDA execution. Multi-GPU placement, MPI-distributed
  block ownership, broader symmetry categories, general initial-state
  canonicalization, and later DMRG algorithm families remain research work.

## Where to Continue

- [Getting Started](getting_started.md): configure, build, and test Uni20.
- [Architecture Overview](architecture/overview.md): implemented layers and
  planned extensions.
- [Deployment Environment](architecture/deployment_environment.md): target HPC
  system shape and the resulting portability constraints.
- [Tensor Operations](tensor/operations.md): canonical ownership, output,
  layout, and async support matrix.
- [Kernel Dispatch](architecture/kernel_dispatch.md): backend capability and decline model.
- [Async Documentation](async/): runtime semantics and kernel-authoring
  guides.
- [Krylov Algorithms](krylov/algorithms.md): solver behavior and supported
  scalar types.
- [Roadmap](roadmap.md): current priorities and work that remains.
