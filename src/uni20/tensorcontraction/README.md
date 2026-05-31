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

By default the bridge uses CUDA's default memory pool.  The original
TensorContraction custom-pool path is still available with
`USE_DEFAULT_POOL=OFF`, in which case `NCCL_HEADROOM` reserves GPU memory for
NCCL before the custom pool preallocates the remaining free memory.  Set
`TENSORCONTRACTION_MEMORY_LOG=1` to print the selected pool mode and NCCL
headroom diagnostics.

Some Open MPI builds print `hwloc` PCI-domain warnings before program startup on
systems with large PCI domain identifiers.  These warnings are harmless for the
current DMRG examples; run with `HWLOC_HIDE_ERRORS=2` to suppress them.

`EffectiveHamiltonianOperator` is the first DMRG-facing boundary over this
runtime.  It owns fixed TensorContraction `A` and `B` block families and exposes
`apply(x, y)` over `MatrixFamily` block vectors, treating `x` as the
TensorContraction `C` family and `y` as the `R` family.  This mirrors the local
effective-Hamiltonian matvec needed by Krylov solvers without committing the
main uni20 tensor or MPS APIs to the temporary TensorContraction layout.

`vector_algebra.hpp` provides the host-side block-vector operations needed by
the first Lanczos prototype: `dot`, `norm`, `scale`, `axpy`, `copy`, `zero`, and
`normalize`.

`lanczos.hpp` ports the small-iteration Lanczos shape used by MPTK's DMRG path
onto `MatrixFamily` block vectors.  The implementation intentionally avoids
restart and full reorthogonalization so it remains comparable to MPTK for local
DMRG benchmarking.

`svd.hpp` adds the first two-site split primitive: a single-block host SVD with
max-rank and singular-value cutoff truncation.  It is intentionally narrow and
self-contained for DMRG prototyping; the long-term replacement should be a
native block-sparse SVD implementation that can distribute independent sectors
over the available CUDA/MPI resources.
