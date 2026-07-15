# src/uni20/linalg/ops

This directory contains public dense-matrix linalg operation wrappers.
Operation values and diagnostic names are defined centrally in
`../operation_tags.hpp`.

## Contents

- `gemm.hpp`: fixed-output Tensor GEMM front end.
- `gemv.hpp`: fixed-output Tensor GEMV front end.
- `matrix_exponential.hpp`: fixed-output matrix exponential dispatch.
- `matrix_product.hpp`: fixed-update and resizable-overwrite Tensor matrix
  products over GEMM dispatch.
- `matrix_set.hpp`: structured matrix initialization over accessor-respecting
  CPU kernels.
- `nonsymmetric_eigen.hpp`: fixed-output nonsymmetric eigensystem dispatch.
- `schur.hpp`: fixed-output Schur decomposition and block-reordering dispatch.
- `self_adjoint_eigh.hpp`: destructive in-place, preserving value, and
  allocation-reusing consuming forms for dense symmetric/Hermitian
  eigensystems.
- `tridiagonal_eigen.hpp`: symmetric tridiagonal eigensystem dispatch.

## Notes

- Operation wrappers should validate linalg-level shape requirements before
  dispatching to a backend.
- Value operations should distinguish a preserving `TensorView const&` form
  from consuming overloads constrained to mutable `OwningTensor` rvalues.
  Consuming compatible storage is an optimization permission, not part of the
  result contract; incompatible owners may still require materialization.
- Consuming `eigh` reuses directly addressable column-major and row-major host
  storage. A row-major square matrix is exposed to LAPACK through a
  column-major `{1, LDA}` mapping of the same allocation: the selected triangle
  is exchanged, and complex eigenvectors are conjugated in place after LAPACK.
  The returned mapping preserves a padded `LDA` and the storage container's
  unused tail. Mappings without a unit-stride axis still materialize.
- Bare mdspans call `dispatch_kernel(selector, operation, operands...)`
  directly; do not add operation-specific aliases for generic dispatch.
- Keep backend-specific implementation details in `../backends`.

## Related Documentation

- [Linalg source map](../)
- [Tensor operations](../../../../docs/tensor/operations.md)
- [Mdspan linear algebra dispatch](../../../../docs/linalg/mdspan_dispatch.md)
- [Kernel dispatch](../../../../docs/architecture/kernel_dispatch.md)
