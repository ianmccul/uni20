# src/uni20/linalg/ops

This directory contains public dense-matrix linalg operation wrappers.
Operation values and diagnostic names are defined centrally in
`../operation_tags.hpp`.

## Contents

- `contract.hpp`: fixed-output pairwise Tensor contraction dispatch.
- `gemm.hpp`: fixed-output Tensor GEMM front end.
- `gemv.hpp`: fixed-output Tensor GEMV front end.
- `linear_solve.hpp`: destructive-workspace and preserving-value dense general
  linear solves.
- `lq.hpp`: destructive-workspace and preserving-value reduced real LQ
  factorizations.
- `matrix_exponential.hpp`: fixed-output matrix exponential dispatch.
- `matrix_norm.hpp`: dense maximum-entry, one, infinity, and Frobenius norms.
- `matrix_product.hpp`: fixed-update and resizable-overwrite Tensor matrix
  products over GEMM dispatch.
- `matrix_set.hpp`: structured matrix initialization over accessor-respecting
  CPU kernels.
- `nonsymmetric_eigen.hpp`: fixed-output nonsymmetric eigensystem dispatch.
- `qr.hpp`: destructive-workspace and preserving-value reduced real QR
  factorizations.
- `schur.hpp`: fixed-output Schur decomposition and block-reordering dispatch.
- `self_adjoint_eigh.hpp`: destructive in-place, preserving value, and
  allocation-reusing consuming forms for dense symmetric/Hermitian
  eigensystems.
- `svd.hpp`: destructive, preserving, and storage-consuming exact dense SVD
  forms for singular values only, either factor separately, or both factors.
  Left and right singular-vector extents are independently reduced or full.
- `truncated_svd.hpp`: preserving and consuming reduced SVD with rank, cutoff,
  and discarded-weight policy plus stable truncation statistics.
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
- Exact SVD dispatches through `gesvd`. `singular_values`, `svd_left`,
  `svd_right`, and `svd` request only the provider outputs they return. A
  preserving call materializes column-major work storage. A consuming call
  uses compatible input storage as destructive work and may adopt it as one
  reduced factor through `JOBU='O'` or `JOBVT='O'`; full factors allocate
  separately.
- Reduced real QR and LQ dispatch through `geqrf`/`orgqr` and
  `gelqf`/`orglq`. Preserving `qr` and `lq` calls materialize column-major work;
  `qr_factorization` and `lq_factorization` expose the destructive workspace
  contract directly.
- Truncating SVD remains a Tensor policy layer over reduced exact SVD rather
  than a second provider operation. It returns right-sized factors, permits
  rank zero, and reports same-precision scaled squared-norm statistics.
- Bare mdspans call `dispatch_kernel(selector, operation, operands...)`
  directly; do not add operation-specific aliases for generic dispatch.
- Keep backend-specific implementation details in `../backends`.

## Related Documentation

- [Linalg source map](../)
- [Tensor operations](../../../../docs/tensor/operations.md)
- [Mdspan linear algebra dispatch](../../../../docs/linalg/mdspan_dispatch.md)
- [Kernel dispatch](../../../../docs/architecture/kernel_dispatch.md)
