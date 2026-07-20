# src/uni20/linalg/backends

This directory contains dense linalg backend front ends. These files translate
Uni20 dense matrix operations into a concrete backend family while staying above
the [raw external-library wrappers](../../backend/).

## Contents

- [`cpu/`](cpu/): CPU dense matrix helpers and CPU matrix exponential implementation.
- [`blas/`](blas/): operation-tag backend adapters that delegate to mdspan BLAS helpers.
- [`cublas/`](cublas/): provider-ready CUDA operation adapters using acquired cuBLAS execution leases.
- [`lapack/`](lapack/): LAPACK-backed matrix operation entry points.
- [`cusolver/`](cusolver/): cuSOLVER-backed matrix operation entry points.

## Notes

- Backend front ends should own linalg-level policy such as result allocation,
  view preparation, and operation selection.
- Raw provider signatures and ABI details belong in the
  [backend source layer](../../backend/).

## Related Documentation

- [Linalg source map](../)
- [Linear algebra documentation](../../../../docs/linalg/)
- [Kernel dispatch](../../../../docs/architecture/kernel_dispatch.md)
