# `src/uni20/linalg/blas`

This directory contains the BLAS-compatible dense linalg adapter layer. It sits
above the raw provider facades in `src/uni20/backend/blas` and
`src/uni20/backend/lapack`, and below tensor-level linalg front ends.

## Contents

- `matrix_transform.hpp`: backend-independent matrix transform algebra.
- `matrix_operand.hpp`: provider-ready readable and writable BLAS matrix
  operands.
- `mdspan_matrix_stage.hpp`: mdspan-axis staging descriptor construction.
- `mdspan_matrix_operand.hpp`: lowering from mdspan staging descriptors to
  provider-ready operands.
- `gemm.hpp`: direct no-copy GEMM wrappers over the configured BLAS provider.
- `blas.hpp`: include point for this adapter layer.

## Notes

- Keep raw provider ABI details in `src/uni20/backend/blas` and
  `src/uni20/backend/lapack`.
- Keep tensor allocation, symmetry policy, and high-level fallback policy above
  this layer.
- Direct wrappers decline unsupported layouts or transform combinations before
  side effects. Prepared wrappers that allocate temporaries should make that
  contract explicit in their names and documentation.
