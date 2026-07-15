# src/uni20/backend/blas/reference/detail

This directory contains low-level implementation details for the reference BLAS
declaration layer.

## Contents

- `blasproto.hpp`: raw BLAS routine prototypes.

## Notes

- Keep this directory ABI-focused. Higher-level checks, shape handling, and
  tensor/linalg policy belong above the reference declaration layer.

## Related Documentation

- [Reference BLAS source layer](../README.md)
- [BLAS/LAPACK mdspan wrappers](../../../../../../docs/linalg/blas_lapack_wrappers.md)
