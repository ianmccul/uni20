# `src/uni20/linalg/backends/blas`

This directory contains operation-tag backend adapters that route dense linalg
operations to BLAS. These adapters sit above `src/uni20/linalg/blas`, which owns
mdspan-to-BLAS descriptor construction and direct provider calls.

The implemented backend adapters are GEMM and GEMV. They delegate
`try_kernel(BlasBackend, operation, ...)` to the corresponding direct
`uni20::linalg::blas::try_*` wrapper. Direct representability failures return
structured `KernelAttempt` decline reasons; provider failures remain terminal
errors.
