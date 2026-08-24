# src/uni20/linalg/backends/looped_gemm

This directory contains the operation-specific backend that lowers a tensor
contraction to a loop of rank-two GEMM dispatches.

`LoopedGemmContractionBackend` currently accepts exactly one residual M or N
stride group after joint stride merging. Each loop iteration advances copied
mdspec metadata to one disjoint output slice, then dispatches `gemm_op` through
the backend's retained execution selector. K remains a single GEMM dimension,
so every slice uses the contraction's original `beta`.

Immediate mdspans are advanced through their accessor offset policy. Deferred
mdspecs must provide a descriptor `offset_by` operation; this preserves the
storage and resource identity needed by host, CUDA, and future execution-domain
acquisition.

Every projected matrix must still expose a unit-stride axis. Slicing does not
materialize or pack a tensor that lacks one; such instances cleanly decline to
the next contraction backend.

## Related Documentation

- [Linalg backend source map](../)
- [Tensor contraction](../../../../../docs/linalg/tensor_contraction.md)
- [Kernel dispatch](../../../../../docs/architecture/kernel_dispatch.md)
