# src/uni20/backend/blas/openblas

This directory contains OpenBLAS-specific backend wiring. It is enabled only
when the configured BLAS vendor is detected as OpenBLAS.

## Contents

- `backend_openblas.hpp`: OpenBLAS backend include point.
- `CMakeLists.txt`: OpenBLAS extension interface target.

## Notes

- OpenBLAS-specific APIs should stay here. Generic BLAS wrappers belong in
  `../reference` or the parent `blas/` layer.

## Related Documentation

- [BLAS provider source layer](../README.md)
- [BLAS/LAPACK mdspan wrappers](../../../../../docs/linalg/blas_lapack_wrappers.md)
