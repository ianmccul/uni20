# src/uni20/linalg/backends/direct_gemm

This directory contains the operation-specific backend that lowers a tensor
contraction to one rank-two GEMM dispatch.

`DirectGemmContractionBackend` retains an execution backend selector. It plans
M/N/K grouping over mdspec metadata, projects each fixed operand without
acquiring a data handle, and dispatches `gemm_op` through that retained
selector. The selected GEMM backend therefore owns host or CUDA access
acquisition and provider execution.

The backend cleanly declines before nested dispatch when the contraction does
not admit one direct rank-two projection. A nested GEMM decline is also clean;
once its single GEMM succeeds, the contraction is complete.

## Related Documentation

- [Linalg backend source map](../)
- [Tensor contraction](../../../../../docs/linalg/tensor_contraction.md)
- [Kernel dispatch](../../../../../docs/architecture/kernel_dispatch.md)
