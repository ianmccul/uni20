# src/uni20/linalg/backends/cublas

This directory adapts opaque CUDA mdspan operands to cuBLAS operation tags.
`CublasBackend` validates and stages the mdspans before acquiring resources,
then obtains the context-owned cuBLAS execution pool, leases a handle and idle
stream, opens synchronized CUDA buffer access, and calls the provider-ready
leaf.

The current first operation is GEMM. Ordinary Tensor code reaches it through
`linalg::gemm(output, alpha, lhs, rhs, beta)`; the Tensor front end resolves
mdspans and dispatches the storage-selected `CublasBackend`. The direct path
uses blocking resource admission. Future async CUDA lowering will await the
same execution resources before entering the non-suspending provider leaf.
