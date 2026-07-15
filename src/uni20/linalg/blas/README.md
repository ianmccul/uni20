# src/uni20/linalg/blas

This directory contains the BLAS-compatible dense linalg adapter layer. It sits
above the raw [BLAS](../../backend/blas/README.md) and
[LAPACK](../../backend/lapack/README.md) provider facades, and below
tensor-level linalg front ends.

## Contents

- `blas_matrix.hpp`: provider-ready readable and writable BLAS matrix operands
  plus backend-independent transform algebra.
- `blas_vector.hpp`: provider-ready readable and writable BLAS vector operands.
- `mdspan_access.hpp`: shared direct-accessor eligibility for ranked BLAS
  mdspan operands.
- `mdspan_matrix.hpp`: mdspan-axis staging descriptor construction and lowering
  to provider-ready operands.
- `mdspan_vector.hpp`: rank-one stride and accessor staging for BLAS increments.
- `gemm.hpp`: direct no-copy GEMM wrappers over the configured BLAS provider.
- `gemv.hpp`: direct no-copy GEMV wrappers over matrix and vector operands.
- `blas.hpp`: include point for this adapter layer.

## Notes

- Keep raw provider ABI details in the [BLAS](../../backend/blas/README.md) and
  [LAPACK](../../backend/lapack/README.md) source layers.
- Keep tensor allocation, symmetry policy, and high-level fallback policy above
  this layer.
- `try_*` direct wrappers decline layouts and ABI dimensions that cannot be
  represented without copies. They also decline complex conjugate-only GEMM
  matrix operands and conjugating GEMV input vectors because the generic
  provider path is currently gated to the portable `N/T/C` matrix transpose
  opcodes and GEMV has no vector-conjugation opcode. The transform helper can
  spell conjugate-only as `R`; OpenBLAS exposes this through
  `CblasConjNoTrans` and its develop-branch Fortran GEMM dispatcher, while MKL
  Fortran GEMM rejected `R` in local testing. Checked wrappers treat
  inconsistent operation dimensions or unsupported direct transforms as logic
  errors and abort through Uni20 checks. Prepared
  wrappers that allocate temporaries, or explicit provider extensions such as
  OpenBLAS conjugate-no-transpose, should make that contract explicit in their
  names and documentation. Prepared wrappers may also use internal
  output-storage conjugation as a non-user-visible workaround for unsupported
  readable transform combinations.
- Direct mdspan wrappers must not treat pointer-like `data_handle_type` as
  enough for raw BLAS access. Writable outputs require default-accessor views.
  Readable inputs require default accessors or explicitly lowerable accessors,
  currently Uni20's `conjugated_accessor` over a default accessor.
- GEMV explicitly scales its output when `alpha == 0` or the logical input is
  empty. This preserves `y = alpha*A*x + beta*y` semantics without depending on
  provider-specific quick-return behavior, and `beta == 0` does not read the
  previous output values.
- Direct rank-one lowering accepts positive mdspan strides and canonicalizes an
  unobserved empty or singleton stride to increment `1`. A negative stride on a
  multi-element view declines: the Fortran BLAS negative-increment convention
  adjusts its starting element and cannot use the mdspan logical-origin handle
  unchanged.

## Related Documentation

- [Linalg source map](../README.md)
- [BLAS/LAPACK mdspan wrappers](../../../../docs/linalg/blas_lapack_wrappers.md)
- [Mdspan linear algebra dispatch](../../../../docs/linalg/mdspan_dispatch.md)
- [Mdspan source utilities](../../mdspan/README.md)
