# src/uni20/backend/cuda

This directory contains the CUDA runtime foundation and is the CUDA
backend-library wiring point. CUDA Tensor storage descriptors are implemented;
Tensor CUDA kernel coverage is not yet complete, but GEMM now lowers end to end
from the ordinary Tensor API through `CublasBackend`.
Deterministic and oneTBB unified host/multi-device coroutine schedulers are
implemented under `src/uni20/async/`.

## Contents

- `cuda_error.hpp`: structured CUDA runtime failures and checked-call helpers.
- `cuda_error_presentation.hpp`: presentation-layer rendering for CUDA failures.
- `device.hpp`: validated device identities and process-wide immutable hardware
  capability caching.
- `buffer.hpp`: typed move-only device allocations and scoped read/write access
  guards. Each buffer owns its completion ledger and briefly locks only that
  ledger when publishing access completions.
- `runtime.hpp`: scoped process-wide CUDA initialization, canonical per-device
  resources, device guards, reference-counted stream-pool leases, immutable
  completion tokens, and device-local idle-stream pools. Device resources also
  retain lazily constructed provider-resource pools until shutdown.
- `resource_pool.hpp`: fixed-capacity provider-resource pools and move-only leases.
- `task_awaiters.hpp`: CUDA-task device selection plus non-blocking stream and
  generic provider-resource acquisition. Concrete awaiters derive from the
  runtime-neutral `async::CudaTaskAwaiterTag`; generic async headers do not name
  individual CUDA awaiter types.
- `CMakeLists.txt`: CUDA backend target setup.

The Tensor-facing `CudaStorage` policy and opaque `CudaBufferView` mdspan
handle live in [`storage/cuda_storage.hpp`](../../storage/cuda_storage.hpp).

## Notes

- Keep CUDA toolkit discovery and target properties local to backend targets.
- `cuda::Device` is a cheap ordinal value. Device discovery validates the
  ordinal and initializes one immutable capability snapshot per visible device;
  it does not create schedulers, streams, provider handles, or allocation pools.
- `cuda::initialize(...)` installs one process-wide runtime lifetime. It owns
  one canonical `DeviceResources` per enrolled device and must outlive all CUDA
  Tensors, buffers, streams, provider leases, and tasks. Ordinary code obtains
  resources through `cuda::device_resources(device)` rather than passing the
  runtime through operations. Direct resource construction is reserved for
  isolated tests and low-level bring-up.
- Runtime capability checks should remain operation-specific; do not assume one
  CUDA build option makes every CUDA library feature available.
- A stream-pool slot is available only after all work previously queued to that
  stream has completed. Destroying the final `cuda::Stream` handle enqueues a
  lightweight host function that marks the slot idle.
- Provider resources use FIFO `ResourcePool<T>` admission for queued async
  waiters. Provider-specific awaiters may impose a documented resource order,
  such as cuBLAS handle before stream, and pass the completed execution lease to
  a non-suspending backend leaf.
- `Stream::record_completion()` creates and records the private event on the
  producer stream's device. Consumers install same- or cross-device dependencies
  with `stream.wait_on(completion)`.
- Buffer dependencies use completion events and `cudaStreamWaitEvent`; the pool
  does not attempt dependent-task stream affinity.
- `cuda::CudaBuffer<T>` retains the latest exclusive-writer completion and reader
  completions since that writer as a completion ledger. The existing async
  `EpochQueue` or synchronous program order remains the causal model; CUDA
  buffer access does not build another DAG or wait for future host publication.
- Submitters use `buffer.read_synchronized_with(stream)` and
  `buffer.write_synchronized_with(stream)` to construct scoped
  `ReadAccess<T>` and `WriteAccess<T>` guards. Raw device pointers are exposed
  only through those guards. Compatible readers overlap; a writer waits for all
  unfinished prior device accesses. Live guard tokens reject host-side
  read/write overlap that would be invalid for an ordinary mutable value. They
  diagnose incorrect ordering rather than queueing or suspending the caller.
- Pageable host transfers use `buffer.blocking_read_access()` and
  `buffer.blocking_write_access()`. These guards host-wait for the current
  completion ledger, expose the device pointer only for a synchronous CUDA
  runtime call, and ordinarily publish no event when released. Synchronous
  `cudaMemcpy` uses CUDA's default-stream path and may synchronize beyond one
  buffer's ledger, so this is a broad blocking boundary and not a stream-capture
  path. Pageable host-to-device transfer records the remaining default-stream
  DMA through `BlockingWriteAccess::release_with_completion()` because host
  staging may finish before the device copy. These guards are not the path for
  non-blocking `Async<CudaTensor>` operations.
- A buffer's state mutex protects only its own completion snapshots and
  publication. It is not held while a backend or provider call executes, and
  independent buffers do not contend on it.
- CUDA runtime failures use `CudaRuntimeError` and Uni20's presentation layer.
  Cleanup failures and invalid stream-pool state remain fail-fast logic errors.

## Related Documentation

- [Backend source layer](../)
- [CUDA backend documentation](../../../../docs/backends/cuda/)
- [CUDA buffer guide](../../../../docs/backends/cuda/buffers.md)
- [CUDA kernel dispatch and provider scheduling](../../../../docs/backends/cuda/kernel_dispatch.md)
- [Execution architecture](../../../../docs/architecture/execution.md)
