# `src/uni20/krylov`

This directory contains native matrix-free Krylov algorithms for Uni20. The
solvers operate through a small vector-operation interface rather than through
ARPACK-style work arrays or reverse communication. Vectors are opaque to the
solver, so callers can keep them on host memory, device memory, or distributed
storage as long as the matrix-free operations provide host scalar results for
norms and inner products.

## Contents

- `matrix_free.hpp`: matrix-free operator concepts, parameter types, result
  types, diagnostics, and dimension validation helpers.
- `symmetric_lanczos.hpp`: real symmetric and complex Hermitian Lanczos
  eigensolvers, including restart and generalized-problem support.
- `nonsymmetric_arnoldi.hpp`: real and complex nonsymmetric Arnoldi solvers.
- `krylov_exponential.hpp`: Lanczos/Arnoldi exponential-action projection
  algorithms.
- `taylor_exponential.hpp`: independent Taylor exponential-action reference
  algorithm used for validation and fallback experiments.
- `dense_linalg.hpp`, `dense_subspace.hpp`, `dense_subspace_unused.hpp`,
  `tridiagonal.hpp`: small dense projected-problem infrastructure used by the
  Krylov solvers. These are prototype helpers and are expected to migrate toward
  the final Uni20 rank-2 tensor/mdspan linear-algebra entry points.
- `dense_host_vector.hpp`: simple host-vector adapter for tests, examples, and
  prototype callers.

## Design Notes

- Core Uni20 does not vendor ARPACK. The native solvers intentionally follow
  useful ARPACK/IRAM/IRLM ideas, but ARPACK comparisons and larger benchmark
  dashboards belong in a separate validation repository.
- Public behavior, supported scalar types, defaults, and internal tuning are
  documented in [`/docs/krylov_algorithms.md`](../../../docs/krylov_algorithms.md).
- Default `ncv`/`nkeep` policy notes live in
  [`/docs/krylov_solver_defaults.md`](../../../docs/krylov_solver_defaults.md).
- Precision validation status lives in
  [`/docs/krylov_precision_validation.md`](../../../docs/krylov_precision_validation.md).
- Test matrix provenance and stress-test intent live in
  [`/docs/krylov_test_matrices.md`](../../../docs/krylov_test_matrices.md).

## Tests And Examples

Unit tests live under [`/tests/krylov`](../../../tests/krylov), with Matrix
Market fixtures under `/tests/krylov/matrix_market`. Example drivers live under
[`/examples/krylov`](../../../examples/krylov).
