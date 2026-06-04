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
