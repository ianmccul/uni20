# src/uni20/linalg/backends/lapack

This directory contains LAPACK operation-tag backend adapters.

## Contents

- `common.hpp`: checked LAPACK workspace-query conversion helpers.
- `matrix_norm.hpp`: general dense matrix norms through `lange`.
- `linear_solve.hpp`: destructive dense general solves through `gesv`.
- `tridiagonal_eigen.hpp`: symmetric tridiagonal eigenvalues and eigenvectors
  through `sterf` and `steqr`.
- `nonsymmetric_eigen.hpp`: real and complex nonsymmetric eigensystems through
  `geev`, including real conjugate-pair unpacking.
- `self_adjoint_eigh.hpp`: real symmetric and complex Hermitian eigensystems
  through `syev`/`heev`.
- `svd.hpp`: exact real and complex singular value decompositions through
  `gesvd`, including values-only, one-sided, complete, and reduced-factor
  input-overwrite jobs.
- `schur.hpp`: real and complex Schur decomposition, real Hessenberg Schur
  reduction, and Schur block/entry reordering through `gees`, `hseqr`, and
  `trexc`.

## Notes

- Each operation should provide `kernel_accepts_types(LapackBackend, ...)` and
  `try_kernel(LapackBackend, ...)` over normalized `MdspecLike` operands with
  the mutability its provider call requires. The backend acquires host leases,
  then calls an ordinary operation-specific mdspan leaf and the raw wrappers
  under [the LAPACK provider layer](../../../backend/lapack/). Resolved mdspans
  are not redispatched through an operation tag.
- Return a non-success `KernelAttempt` only for clean preflight decline. LAPACK
  `INFO` failures after a provider call are terminal and must not trigger
  fallback.
- Current update-matrix paths require a directly addressable column-major
  mdspan. Unsupported layouts decline before any operand is modified.
- Read-only matrix norms additionally accept row-major storage by exchanging
  one and infinity norms under the provider transpose. Complex non-Frobenius
  forms decline because LAPACK's component-sum convention differs from
  Uni20's mathematical complex magnitude.
- Keep copy/materialization behavior explicit when adapting tensor views to
  LAPACK-compatible storage.

## Related Documentation

- [Linalg backend source map](../)
- [BLAS/LAPACK mdspan wrappers](../../../../../docs/linalg/blas_lapack_wrappers.md)
- [Mdspan linear algebra dispatch](../../../../../docs/linalg/mdspan_dispatch.md)
