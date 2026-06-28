# Krylov Test Matrices

This document records the provenance and oracle status of generated matrices
used by the Krylov eigensolver tests. These matrices are test fixtures only:
they are not runtime dependencies of Uni20.

## Selection Policy

Prefer matrices with explicit formulas, stable provenance, and known spectra.
Each fixture should state whether it is an exact eigenvalue oracle, a condition
number oracle, an inertia oracle, or only a convergence stress case.

Matrix Market fixtures remain useful for sparse real-world behavior, but their
text files generally do not encode solver precision. Precision coverage is
therefore handled by instantiating the same generated matrix or Matrix Market
fixture through `float` and `double` solver paths with precision-appropriate
tolerances.

## Current Fixtures

| ID | Source | License | Oracle | Purpose |
|---|---|---|---|---|
| `prescribed_diagonal_spectrum` | Uni20 generated | Uni20 project license | exact eigenvalues and 2-norm condition number when nonsingular | Baseline with controlled gaps, clusters, multiplicities, and conditioning. |
| `symmetric_tridiagonal_toeplitz` | Classical formula | Public mathematical formula | exact eigenvalues and 2-norm condition number when nonsingular | Sparse symmetric baseline. Eigenvalues are `a + 2 b cos(j pi/(n+1))`. |
| `path_laplacian` | Uni20 generated graph Laplacian | Public mathematical formula | exact eigenvalues | Sparse positive-semidefinite graph test. Eigenvalues are `2 - 2 cos(j pi/n)`. |
| `shifted_path_laplacian` | Uni20 generated graph Laplacian | Public mathematical formula | exact eigenvalues and 2-norm condition number when nonsingular | Large identity-shift stress case for residual scaling and breakdown thresholds. |
| `diagonal_clustered_extremes` | Uni20 generated stress matrix | Uni20 project license | exact eigenvalues and 2-norm condition number | Lanczos ghost/duplicate-Ritz stress case with clustered extremal eigenvalues and dense interior spectrum. |
| `symmetric_interior_gap` | Uni20 generated stress matrix | Uni20 project license | exact eigenvalues and 2-norm condition number | Interior-target trap for ordinary Ritz extraction; should require shift-invert or fail cleanly. |
| `diagonal_clustered_wanted_end` | Uni20 generated stress matrix | Uni20 project license | exact eigenvalues and 2-norm condition number | Restart torture case where a cramped search space fails on a tightly clustered wanted end, while a larger/thicker space converges. |
| `anymatrix_core_symmstoch` | Anymatrix `core/symmstoch` and `core/soules` | BSD-2-Clause | exact eigenvalues and 2-norm condition number when nonsingular | Dense symmetric stochastic matrix with prescribed spectrum. |
| `anymatrix_core_blockhouse_coordinate_reflector` | Anymatrix `core/blockhouse` special case | BSD-2-Clause | exact eigenvalues and 2-norm condition number | Symmetric orthogonal/involutory matrix with repeated eigenvalues. |

## Stress Assertions

Stress tests should avoid exact iteration-count requirements. Prefer assertions
that accepted eigenpairs satisfy scaled residual tolerances, failure cases return
a nonzero status without invalid values, duplicate Ritz values are not accepted
as distinct simple eigenvalues, and restart diagnostics remain internally
consistent.

Current native stress tests cover:

- clustered extremal diagonal spectra, checking for duplicate accepted Ritz
  values and honest residuals;
- large shifted path Laplacians, checking convergence without false breakdown;
- ordinary smallest-magnitude extraction on an interior-gap spectrum, checking
  that the solver fails cleanly rather than returning converged garbage.
- restart sensitivity on a tightly clustered wanted end, checking that a
  cramped `nev=6, ncv=8` configuration fails cleanly while a wider search space
  with retained guard Ritz vectors converges to the exact clustered eigenvalues.

## Anymatrix Attribution

Anymatrix is distributed under the BSD 2-Clause License:

> Copyright (c) 2021, Nicholas J. Higham and Mantas Mikaitis

Relevant references:

- Nicholas J. Higham and Mantas Mikaitis, *Anymatrix: An Extendable MATLAB
  Matrix Collection*, Numer. Algorithms, 90:3, 1175-1196, 2021.
- Nicholas J. Higham and Mantas Mikaitis, *Anymatrix: An Extendable MATLAB
  Matrix Collection, Users' Guide*, MIMS EPrint 2021.15, 2021.

The current C++ fixtures reimplement small formulas directly and record
Anymatrix provenance in metadata. The full MATLAB toolbox is not vendored into
Uni20.
