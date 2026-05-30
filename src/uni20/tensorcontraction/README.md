# TensorContraction Bridge

This directory contains a temporary vendored snapshot of the TensorContraction
runtime used to prototype a U(1) DMRG path in uni20.

The copied code is intentionally quarantined here. It is not intended to define
the long-term uni20 backend architecture or coding style. The goal is to get a
working effective-Hamiltonian apply path quickly, then replace useful pieces
with native uni20 abstractions and kernels over time.

Snapshot source:

- Repository: `Uni20-dev/TensorContraction`
- Branch: `origin/iterative-execution`
- Commit: `be0739a`

Build with:

```bash
cmake -S . -B build_codex/tensorcontraction -DUNI20_ENABLE_TENSORCONTRACTION=ON
cmake --build build_codex/tensorcontraction
```

The vendored runtime requires CUDA Toolkit, cuBLAS, MPI, and NCCL.

`EffectiveHamiltonianOperator` is the first DMRG-facing boundary over this
runtime.  It owns fixed TensorContraction `A` and `B` block families and exposes
`apply(x, y)` over `MatrixFamily` block vectors, treating `x` as the
TensorContraction `C` family and `y` as the `R` family.  This mirrors the local
effective-Hamiltonian matvec needed by Krylov solvers without committing the
main uni20 tensor or MPS APIs to the temporary TensorContraction layout.
