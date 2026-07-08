# `src/uni20/linalg/blas`

This directory contains the BLAS-compatible dense linalg adapter layer. It sits
above the raw provider facades in `src/uni20/backend/blas` and
`src/uni20/backend/lapack`, and below tensor-level linalg front ends.

## Contents

- `blas_matrix.hpp`: provider-ready readable and writable BLAS matrix operands
  plus backend-independent transform algebra.
- `mdspan_matrix.hpp`: mdspan-axis staging descriptor construction and lowering
  to provider-ready operands.
- `gemm.hpp`: direct no-copy GEMM wrappers over the configured BLAS provider.
- `blas.hpp`: include point for this adapter layer.

## Notes

- Keep raw provider ABI details in `src/uni20/backend/blas` and
  `src/uni20/backend/lapack`.
- Keep tensor allocation, symmetry policy, and high-level fallback policy above
  this layer.
- `try_*` direct wrappers decline layouts and ABI dimensions that cannot be
  represented without copies. They also decline complex conjugate-only GEMM
  operands because the generic provider path is currently gated to the portable
  `N/T/C` transpose opcodes. The transform helper can spell conjugate-only as
  `R`; OpenBLAS exposes this through `CblasConjNoTrans` and its develop-branch
  Fortran GEMM dispatcher, while MKL Fortran GEMM rejected `R` in local
  testing. Checked wrappers treat inconsistent GEMM dimensions or unsupported
  direct transforms as logic errors and abort through Uni20 checks. Prepared
  wrappers that allocate temporaries, or explicit provider extensions such as
  OpenBLAS conjugate-no-transpose, should make that contract explicit in their
  names and documentation. Prepared wrappers may also use internal
  output-storage conjugation as a non-user-visible workaround for unsupported
  readable transform combinations.
- Direct mdspan wrappers must not treat pointer-like `data_handle_type` as
  enough for raw BLAS access. Writable outputs require default-accessor views.
  Readable inputs require default accessors or explicitly lowerable accessors,
  currently Uni20's `conjugated_accessor` over a default accessor.
