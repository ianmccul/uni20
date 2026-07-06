# `src/uni20/backend/blas/reference`

This directory contains raw BLAS declarations and reference interface plumbing
for the configured BLAS provider.

## Contents

- `reference_blas.hpp`: reference BLAS include point.
- `detail/blasproto.hpp`: low-level BLAS routine prototypes.
- `CMakeLists.txt`: interface target linked to the discovered BLAS/LAPACK
  libraries.

## Notes

- Keep declarations close to the external ABI and avoid adding linalg policy at
  this layer.
- Checked, shape-aware, or tensor-aware operations belong in higher modules.
