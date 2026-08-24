# src/uni20/linalg/backends/blas

This directory contains operation-tag backend adapters that route dense linalg
operations to BLAS. These adapters sit above the
[BLAS descriptor layer](../../blas/), which owns mdspan-to-BLAS
descriptor construction and direct provider calls.

The implemented backend adapters are contraction, GEMM, and GEMV. They delegate
`try_kernel(BlasBackend, operation, ...)` to the corresponding direct
`uni20::linalg::blas::try_*` wrapper. Direct representability failures return
structured `KernelAttempt` decline reasons; provider failures remain terminal
errors.

## Related Documentation

- [Linalg backend source map](../)
- [BLAS/LAPACK mdspan wrappers](../../../../../docs/linalg/blas_lapack_wrappers.md)
- [Mdspan linear algebra dispatch](../../../../../docs/linalg/mdspan_dispatch.md)
