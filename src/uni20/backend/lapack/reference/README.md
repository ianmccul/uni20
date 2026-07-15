# src/uni20/backend/lapack/reference

This directory contains declarations for the ordinary configured LAPACK
provider.

## Contents

- `band.hpp`: band-matrix routines.
- `general.hpp`: general dense matrix routines.
- `norms.hpp`: LAPACK norm helpers.
- `tridiagonal.hpp`: tridiagonal routines.

## Notes

- Declarations should stay close to the provider ABI.
- Checked wrappers and cross-provider selection live in `../lapack.hpp`.

## Related Documentation

- [LAPACK provider source layer](../)
- [BLAS/LAPACK mdspan wrappers](../../../../../docs/linalg/blas_lapack_wrappers.md)
