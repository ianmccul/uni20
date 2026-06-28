# Krylov Solver Defaults

This note records the current native Krylov default policy. Explicit user
parameters always win; these defaults are only used when `krylov_dimension` or
`retained_ritz_count` is zero.

For the full algorithm inventory, scalar coverage, parameter list, and internal
tuning values, see [krylov_algorithms.md](krylov_algorithms.md).

## Dimensions

The public parameters are:

| parameter | meaning |
| --- | --- |
| `eigenvalue_count` / `nev` | Number of requested converged eigenpairs. |
| `krylov_dimension` / `ncv` | Maximum Lanczos or Arnoldi subspace dimension. |
| `retained_ritz_count` / `nkeep` | Native restart dimension retained between cycles. |

The required restarted relation is:

```text
nev <= nkeep < ncv <= problem_dimension
```

For non-restarted solves, `nkeep` is ignored.

## Symmetric and Hermitian Problems

Symmetric/Hermitian Lanczos uses:

```text
ncv = min(problem_dimension, max(20, 2*nev + 1))
nkeep = nev
```

This keeps the ARPACK-style implicit restart behavior by default: the solver
retains the requested invariant subspace and uses the rest of the Krylov space
for shifts. It is usually a good default for extremal Hermitian eigenvalues.

For clustered spectra, increase either or both of:

```text
ncv
nkeep
```

Increasing `nkeep` gives thick-restart guard vectors, but weakens each restart
filter because the number of discarded directions is `ncv - nkeep`.

## Nonsymmetric Problems

Real and complex nonsymmetric Arnoldi use roomier defaults:

```text
ncv = min(problem_dimension, max(20, 6*nev + 8))
nkeep = min(ncv - 1, max(2*nev, nev + 4))
```

Nonsymmetric restarts are more fragile than Hermitian Lanczos restarts because
the projected problem can be nonnormal and, in real arithmetic, complex
conjugate pairs occupy real Schur two-planes. The default therefore keeps guard
Schur dimensions and gives the restart a larger search space.

This is intentionally more conservative than the minimal ARPACK guideline
`ncv > 2*nev`. It costs more vector storage and local projected dense work, but
has a better chance of converging without the user already knowing the matrix.

## Iteration Limits

`max_iterations` is still an iteration budget, not a cure for a cramped
subspace. Raising it can eventually converge many hard cases, but a too-small
`ncv` may waste many restart cycles or stagnate. For difficult nonsymmetric
problems, first try a larger `ncv`, then increase `max_iterations`.

## External Comparisons

ARPACK comparison adapters can use the same default `ncv` formulas when
`krylov_dimension == 0`. ARPACK does not expose a separate `nkeep`, so
`retained_ritz_count` is necessarily a native-only control. Those adapters are
benchmark/oracle infrastructure and should live outside core Uni20.
