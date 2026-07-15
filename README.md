# Uni20

Uni20 is a C++23 tensor-network research library combining dense tensor
operations, ordered backend dispatch, matrix-free Krylov algorithms, and an
epoch-based asynchronous runtime.

The project is still in active design and does not promise a stable public API,
but it now has working end-to-end paths: `Tensor` operations lower through
CPU/BLAS/LAPACK kernels, `Async<Tensor>` matrix products and eigensystems reuse
the same dispatch layer, oneTBB executes independent work, and the presentation
and diagnostic layers report results, backend decisions, stacktraces, and async
DAG state.

Current limitations are explicit. A complete symmetry-aware `BlockTensor`,
CUDA/MPI execution, tensor-level automatic differentiation, and useful Python
tensor bindings remain future work.

Start with:

- [About Uni20](docs/about.md) for implemented capabilities and runnable
  vertical slices.
- [Getting Started](docs/getting_started.md) to configure, build, and test.
- [Documentation Index](docs/) for subsystem guides and design notes.
- [Roadmap](docs/roadmap.md) for current priorities and remaining work.
- [Contributing](docs/CONTRIBUTING.md) for development and review expectations.
