# CUDA Backend Documentation

Uni20 has a tested low-level CUDA runtime foundation on the current main branch:
structured errors, device guards, reference-counted stream-pool leases,
immutable completion tokens, an actually-idle stream pool, and typed move-only
device buffers with scoped `ReadAccess<T>`/`WriteAccess<T>` objects.
Deterministic and oneTBB device-bound CUDA task schedulers are implemented.
Uni20 does not yet have CUDA Tensor storage, Tensor kernels, CUDA resource
awaiters, or storage-driven device-context scheduler selection.

## Start Here

- [CUDA Buffers](buffers.md) introduces the implemented low-level allocation
  and stream-synchronized access API for kernel and provider authors.
- [Runtime Model](runtime.md) explains device selection, stream ownership,
  completion tokens, and structured errors.
- [CUDA hello-world example](../../../examples/cuda/) reports the configured
  runtime and exercises the stream/completion foundation.

## Detailed Reference

- [CUDA Buffer Completion Lowering](epoch_design_draft.md) defines the exact
  access, completion-ledger, lifetime, and failure contract.
- [Memory Allocation](memory_allocation.md) records the current allocator path
  and future allocation-policy work.

## Active Design

- [CUDA Kernel Dispatch and Device Scheduling](kernel_dispatch.md)
- [Runtime Resolution](runtime_resolution.md)
- [cuSOLVER Architecture](cusolver.md)

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
