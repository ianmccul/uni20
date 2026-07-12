# `src/uni20/linalg/backends/blas`

This directory contains operation-tag backend adapters that route dense linalg
operations to BLAS. These adapters sit above `src/uni20/linalg/blas`, which owns
mdspan-to-BLAS descriptor construction and direct provider calls.

The first backend adapter is GEMM: it implements `try_kernel(BlasBackend,
gemm_op, ...)` by delegating to `uni20::linalg::blas::try_gemm(...)`. Direct
representability failures return structured `KernelAttempt` decline reasons;
provider failures remain terminal errors.
