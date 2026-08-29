# src/uni20/backend/cusolver

This directory owns cuSOLVER provider state and checked error handling. The
Tensor-facing exact SVD lowering lives in the linalg backend layer.

## Contents

- `cusolver_error.hpp/.cpp`: checked cuSOLVER status handling.
- `execution.hpp/.cpp`: device-local handle pools and stream-paired execution
  leases.
- `CMakeLists.txt`: cuSOLVER backend target setup.

## Notes

- `UNI20_BACKEND_CUSOLVER` is an optional CUDA-dependent build feature.
- Each `DeviceResources` lazily owns one cuSOLVER execution pool. The pool
  defaults to two exclusive handles, capped by the CUDA stream count.
- `cusolver::execution_pool(resources, count)` configures another count before
  first provider use. Later explicit configuration must match, and the count
  cannot exceed stream capacity.
- A leased handle is rebound to the operation stream and returned through a
  callback at that stream's completion boundary.
- The first operation is blocking real `float`/`double` exact SVD through
  `cusolverDnSgesvd`/`cusolverDnDgesvd` for supported column-major CUDA-buffer
  descriptors.
- Higher-level entry points belong in the
  [linalg backend layer](../../linalg/backends/).

## Related Documentation

- [Backend source layer](../)
- [cuSOLVER architecture](../../../../docs/backends/cuda/cusolver.md)
- [CUDA runtime resolution](../../../../docs/backends/cuda/runtime_resolution.md)
