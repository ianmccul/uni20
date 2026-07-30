# src/uni20/linalg/backends/cublas

This directory adapts normalized CUDA device-mdspan operands to cuBLAS
operation tags. Tensor frontends select the backend while storage policy is
available, then lower fixed operands before dispatch. `CublasBackend` validates
their device-mdspan metadata, blocks for an execution lease, opens synchronized
CUDA buffer access, and calls the provider-ready leaf.

The current first operation is GEMM. Direct Tensor dispatch uses the blocking
`try_kernel` entry point. Coroutine dispatch detects the backend's
`try_make_kernel_task` customization, which prepares the same operands, binds a CUDA
child to their device, awaits an execution lease, and invokes the prepared
provider leaf without redispatching.

The ordinary adapter is `gemm.hpp`; it has no dependency on coroutine runtime
support. Async Tensor lowering additionally includes `gemm_task.hpp`, keeping
task creation and resource awaiters out of synchronous-only targets.

CUDA-mdspan recognition, buffer bounds and device validation, completion-ledger
access, and the shared blocking/coroutine preparation path are private
implementation details under `detail/`. The adjacent [`linalg/cublas/`](../../cublas/)
layer accepts only provider-ready matrix descriptors and an acquired execution
lease.

An empty GEMM output succeeds before operand staging. cuBLAS accepts a zero
inner extent with null zero-sized input buffers and applies the degenerate
`C = beta*C` operation, so canonical CUDA matrices retain provider execution for
that case. `CudaReferenceBackend` follows `CublasBackend` in the storage-selected
backend list. It currently handles contiguous Tensor transfer; additional
fallback kernels remain operation-specific.

`assign_product_op` may provisionally construct, resize, or relocate its
replaceable output before mdspan layout and transform acceptance completes. A
decline submits no CUDA work and leaves the prepared output available to a later
backend, which may reuse or replace it. Fixed-output `gemm_op` retains the
stronger unchanged-on-decline contract. The adapter receives the normalized
readable descriptors, prepares the output, normalizes that output once after
preparation, and then calls the private cuBLAS mdspan implementation directly.
It does not redispatch the resolved operands through a `gemm_op` customization.

The Tensor conformance tests share their scalar and canonical-layout cases with
the host GEMM backends. cuBLAS-specific tests cover deferred buffer offsets,
transpose and conjugate-transpose subviews, padded leading dimensions, clean
layout decline before resource acquisition, structured incompatible-device
decline, and hard alias contract violations. Unknown accessor semantics are
rejected at type probing rather than bypassed through the descriptor.

## Related Documentation

- [Linalg backend source map](../)
- [Private CUDA operand lowering](detail/)
- [Provider-ready cuBLAS operations](../../cublas/)
- [CUDA kernel dispatch and provider scheduling](../../../../../docs/backends/cuda/kernel_dispatch.md)
- [Dense BLAS/LAPACK wrapper coverage](../../../../../docs/linalg/dense_blas_lapack_coverage.md)
