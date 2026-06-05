# TensorContraction Matrix Storage Prototype

This note describes the temporary TensorContraction matrix-storage refactor used
while Uni20's final CUDA storage API is still being designed.

## Prototype Model

TensorContraction now distinguishes three concepts:

- `tensor::MatrixHandle`: logical id, shape, and MPI owner rank metadata.
- `tensor::HostMatrixView`: a host pointer plus a `MatrixHandle` and host memory
  kind (`Pageable` or `Pinned`).
- `tensor::DeviceMatrixView`: a device pointer plus a `MatrixHandle`, CUDA
  device id, device context, and content-valid flag.

The existing `tensor::Matrix` remains as a compatibility wrapper during the
migration.  It exposes `handle()` and `hostView()` so transfer code can stop
using null/non-null host pointers as implicit storage-state decisions.

## Transfer Semantics

Host/device transfers should be explicit:

- H2D upload is a device write epoch.
- D2H download is a device read epoch plus a host-visible result.
- Device kernels acquire read/write access to device buffers and publish CUDA
  completion tokens.

The prototype keeps using `GpuBuffer`, `Swapper::GpuAccessPlan`, and
`CudaDeviceContext` for CUDA stream/event management.  These are the current
single-threaded equivalent of the planned `GpuEpochQueue` reader/writer model.

## Host Memory

Pinned host storage is optional.  `MatrixAllocator` marks its `cudaMallocHost`
chunks as `HostMemoryKind::Pinned`; ordinary `MatrixFamily` storage remains
`HostMemoryKind::Pageable`.

This lets benchmark paths request transfer-friendly host buffers without
forcing every host matrix allocation to be pinned.

## Coalesced Sub-Blocks

`MatrixFamily` stores all host blocks in one contiguous slab, while each
`tensor::Matrix` remains the logical sub-block descriptor.  The CUDA prototype
mirrors that shape: `Swapper` can allocate one device slab for a whole
`MatrixFamily`, then register one `GpuBuffer` per sub-block with a pointer into
the parent slab.

The parent slab is tracked by `GpuBuffer::AllocationGroup`:

- `basePtr` and `bytes` describe the full device allocation.
- `liveBuffers` counts sub-block buffers that still reference the allocation.
- `dependencies` accumulates outstanding read/write completion events from each
  sub-block before the allocation is released.

Each sub-block still owns its own reader/writer generations.  This is important:
coalescing should reduce allocation and memcpy overhead, but it must not collapse
the dependency model into one global MatrixFamily epoch.  Different blocks may
later be updated by different kernels, devices, or MPI ranks.

The current fast path is deliberately conservative.  It is used only when the
operation sees the complete MatrixFamily, the matrices exactly cover the host
slab, and the scheduler chooses a single CUDA device.  Partial, already mixed,
or multi-device layouts fall back to the existing per-block path.

## Slab Operations

Avoid calling these operations "collective" operations.  In this codebase
"collective" should remain available for MPI/NCCL-style communication.  The
preferred term is **slab operation**: one operation whose memory scope is a whole
coalesced allocation slab rather than one logical sub-block.

`Swapper::SlabAccessPlan` is the current prototype synchronization primitive for
these operations.  It:

- verifies that the requested matrices cover one complete coalesced pre-store
  slab;
- waits on every participating sub-block reader/writer generation before
  returning a stream;
- exposes the slab base pointer and byte count for bulk operations such as H2D,
  D2H, or future device reductions;
- publishes one completion token back to every sub-block when the operation
  scope exits.

This gives slab transfers and reductions one explicit scheduling boundary
without collapsing normal block-granular dependency tracking.  If later kernels
modify only one sub-block, only that sub-block should advance.  If a slab
operation touches every byte, every sub-block receives the same completion token.

## Future Uni20 Shape

The same design should generalize to Uni20 block-sparse tensors:

- a tensor storage object owns one or more coalesced slabs;
- block descriptors carry shape, quantum-number sector, offset, device, and MPI
  rank metadata;
- operations acquire reader/writer access at block granularity;
- transfer code can coalesce adjacent blocks when their layout and dependencies
  make that safe.

This keeps the transfer and lifetime policy separate from the mathematical
block structure.  Removing U(1) symmetry should mean constructing a different
storage/block layout, not silently falling back to dense operations inside the
U(1) path.

## Uni20 Differences

This prototype is not the final Uni20 CUDA API.  In Uni20 proper:

- storage should live in `Tensor<T, GpuStorage>` or an equivalent backend-aware
  tensor object, not in TensorContraction-specific matrix families;
- CPU async tasks should own logical causality, while GPU storage owns
  stream/event memory hazards;
- the CUDA scheduler may add thread-safe acquisition, idle-aware stream pools,
  and dedicated cuBLAS/cuSOLVER lanes;
- MPI or remote storage should be modeled as a storage/backend property rather
  than as mutable fields inside a matrix descriptor.

Open design questions remain for block-sparse storage layout, remote GPU
storage, and how much of the TensorContraction prototype should survive once
the real Uni20 CUDA runtime exists.
