# src/uni20/backend/cuda

This directory contains the CUDA runtime foundation and is the CUDA
backend-library wiring point. It does not yet provide Tensor CUDA kernels.
Deterministic and oneTBB unified host/multi-device coroutine schedulers are
implemented under `src/uni20/async/`.

## Contents

- `cuda_error.hpp`: structured CUDA runtime failures and checked-call helpers.
- `cuda_error_presentation.hpp`: presentation-layer rendering for CUDA failures.
- `device.hpp`: validated device identities and process-wide immutable hardware
  capability caching.
- `buffer.hpp`: typed move-only device allocations, device contexts, and scoped
  read/write access guards.
- `runtime.hpp`: device guards, reference-counted stream-pool leases, immutable
  completion tokens, and the device-local idle-stream pool.
- `CMakeLists.txt`: CUDA backend target setup.

## Notes

- Keep CUDA toolkit discovery and target properties local to backend targets.
- `cuda::Device` is a cheap ordinal value. Device discovery validates the
  ordinal and initializes one immutable capability snapshot per visible device;
  it does not create schedulers, streams, provider handles, or allocation pools.
- Runtime capability checks should remain operation-specific; do not assume one
  CUDA build option makes every CUDA library feature available.
- A stream-pool slot is available only after all work previously queued to that
  stream has completed. Destroying the final `cuda::Stream` handle enqueues a
  lightweight host function that marks the slot idle.
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
- The context mutex protects only completion snapshots and publication. It is
  not held while a backend or provider call executes.
- CUDA runtime failures use `CudaRuntimeError` and Uni20's presentation layer.
  Cleanup failures and invalid stream-pool state remain fail-fast logic errors.

## Related Documentation

- [Backend source layer](../)
- [CUDA backend documentation](../../../../docs/backends/cuda/)
- [CUDA buffer guide](../../../../docs/backends/cuda/buffers.md)
- [CUDA kernel dispatch and provider scheduling](../../../../docs/backends/cuda/kernel_dispatch.md)
- [Execution architecture](../../../../docs/architecture/execution.md)
