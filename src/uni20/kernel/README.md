# `src/uni20/kernel`

This directory contains low-level kernel entry points over resolved views and
operation tags. Kernels are below tensor ownership, symmetry metadata, and async
scheduling policy; callers should lower to explicit views before reaching this
layer.

## Contents

- `operations.hpp`: operation tags used by dispatch and kernel customization
  points.
- `contract.hpp`: generic contraction entry point declarations.
- `cpu/`: always-available CPU kernels and CPU backend tag integration.
- `blas/`: BLAS-backed kernel implementations enabled when BLAS support is
  configured.
- `mkl/`: placeholder directory for MKL-specific kernel work.

## Notes

- Dispatch is based on operation tags and resolved operand/view types, not on
  tensor objects carrying unresolved high-level policy.
- Keep kernel code free of block-sparse or quantum-number decisions. Symmetry
  code should resolve legal dense block work before calling kernels.
