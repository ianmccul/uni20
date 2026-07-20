# src/uni20/linalg/backends/cublas

This directory adapts provider-ready dense linalg operands to cuBLAS operation
tags. Resource acquisition remains outside the backend walk: callers first
obtain a `cublas::ExecutionLease`, lower CUDA Tensor storage to scoped buffer
access and BLAS matrix descriptors, then dispatch to `CublasBackend`.

The current first leaf is GEMM. CUDA Tensor/mdspan lowering is intentionally a
separate layer because CUDA storage and accessor semantics are not implemented
yet.
