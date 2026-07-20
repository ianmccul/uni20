# src/uni20/linalg/backends/cublas

This directory adapts opaque CUDA mdspan operands to cuBLAS operation tags.
`CublasBackend` validates and stages the mdspans before acquiring resources,
then obtains the cuBLAS execution pool owned by `DeviceResources`, leases a
handle and idle stream, opens synchronized CUDA buffer access, and calls the
provider-ready leaf.

The current first operation is GEMM. Ordinary Tensor code reaches it through
`linalg::gemm(output, alpha, lhs, rhs, beta)`; the Tensor front end resolves
mdspans and dispatches the storage-selected `CublasBackend`. The direct path
uses blocking resource admission. Future async CUDA lowering will await the
same execution resources before entering the non-suspending provider leaf.

An empty GEMM output succeeds before operand staging. A zero inner extent is a
clean `unsupported_instance` decline before staging, resource acquisition, or
buffer-access publication. Complete CUDA-domain handling will come from a
future `CudaReferenceBackend` after `CublasBackend` in the storage-selected
backend list; reusable CUDA fill and scale kernels will implement the degenerate
`C = beta*C` operation.

## Related Documentation

- [Linalg backend source map](../)
- [Provider-ready cuBLAS operations](../../cublas/)
- [CUDA kernel dispatch and provider scheduling](../../../../../docs/backends/cuda/kernel_dispatch.md)
- [Dense BLAS/LAPACK wrapper coverage](../../../../../docs/linalg/dense_blas_lapack_coverage.md)
