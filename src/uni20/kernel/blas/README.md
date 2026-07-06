# `src/uni20/kernel/blas`

This directory contains BLAS-backed kernel entry points. It bridges Uni20
operation tags and resolved dense views to BLAS backend calls.

## Contents

- `blas.hpp`: BLAS kernel include point.
- `backend_blas.hpp`: BLAS backend integration for kernels.
- `contract.hpp`: BLAS-backed contraction entry points.
- `CMakeLists.txt`: interface target for BLAS kernels.

## Notes

- Keep this layer below tensor allocation and symmetry policy.
- Prefer falling back through the generic kernel dispatch path when BLAS cannot
  support a scalar, layout, or operation shape.
