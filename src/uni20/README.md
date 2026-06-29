# `src/uni20`

This directory contains the C++ source for the Uni20 library. Public and
developer-facing headers live under module subdirectories; implementation
details should stay within the owning module rather than leaking through the
top-level namespace.

## Directory Map

- `async/`: epoch-ordered async runtime, scheduler integration, and buffer
  lifetime machinery.
- `backend/`: low-level BLAS/LAPACK/CUDA backend bindings and dispatch support.
- `common/`, `core/`, `tags/`: scalar aliases, concepts, diagnostics, common
  utilities, and lightweight type tags.
- `kernel/`, `level1/`, `linalg/`: kernel entry points and prototype dense
  linear-algebra front ends.
- `krylov/`: native matrix-free Krylov eigensolvers and exponential-action
  algorithms.
- `mdspan/`, `storage/`, `tensor/`: mdspan adapters, storage abstractions, and
  tensor views/types.
- `symmetry/`: quantum-number and block-sparse symmetry infrastructure.
- `expokit/`: legacy transition area for dense exponential code being replaced
  or moved into current modules.

## References

- Contributor and style rules: [`/AGENTS.md`](../../AGENTS.md)
- Project documentation index: [`/docs/README.md`](../../docs/README.md)
- Architecture overview: [`/docs/architecture_diagram.md`](../../docs/architecture_diagram.md)
- Backend dispatch notes: [`/docs/backend_dispatch.md`](../../docs/backend_dispatch.md)
- Kernel dispatch notes: [`/docs/kernel_dispatch.md`](../../docs/kernel_dispatch.md)
