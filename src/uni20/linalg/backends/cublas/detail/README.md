# src/uni20/linalg/backends/cublas/detail

This directory contains private CUDA-storage lowering for `CublasBackend`.

`gemm.hpp` recognizes supported CUDA mdspan accessors and layouts, validates
buffer bounds, aliasing, and device placement, and creates the provider-ready
GEMM plan shared by direct and coroutine dispatch. Direct dispatch acquires an
execution lease with the blocking pool interface. The backend's optional
`try_make_kernel_task` hook returns a `CudaTask` that awaits the same resource.

Provider-ready matrix calls remain in [`linalg/cublas/`](../../../cublas/), and
raw cuBLAS API wrappers remain in [`backend/cublas/`](../../../../backend/cublas/).

See [CUDA kernel dispatch and provider scheduling](../../../../../../docs/backends/cuda/kernel_dispatch.md)
for the direct and coroutine execution paths.

Return to the [cuBLAS backend adapter](../).
