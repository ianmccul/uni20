# CUDA Backend Documentation

Uni20 has a tested low-level CUDA runtime foundation on the current main branch:
structured errors, device guards, reference-counted stream-pool leases,
immutable completion tokens, an actually-idle stream pool, and typed move-only
device buffers with scoped read/write guards. Deterministic and oneTBB
device-bound CUDA task schedulers are implemented. Uni20 does not yet have CUDA
Tensor storage, Tensor kernels, CUDA resource awaiters, or device-context
scheduler selection.

## Active Design

- [Runtime Model](runtime.md)
- [CUDA Kernel Dispatch and Device Scheduling](kernel_dispatch.md)
- [Runtime Resolution](runtime_resolution.md)
- [CUDA Buffer Completion Lowering](epoch_design_draft.md)
- [cuSOLVER Architecture](cusolver.md)
- [Memory Allocation](memory_allocation.md)

## Background

- [Backend Library Compatibility](libraries.md) records provider/version
  constraints.
- [GPU Landscape](landscape.md) surveys tensor-network GPU libraries and
  implementation choices.

## Source Navigation

- [CUDA runtime foundation](../../../src/uni20/backend/cuda/)
- [CUDA task schedulers](../../../src/uni20/async/)
- [cuSOLVER provider scaffolding](../../../src/uni20/backend/cusolver/)
- [cuSOLVER operation-tag backend](../../../src/uni20/linalg/backends/cusolver/)
- [CUDA hello-world example](../../../examples/cuda/)
