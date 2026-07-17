# src/uni20/backend/cuda

This directory contains the CUDA runtime foundation and is the CUDA
backend-library wiring point. It does not yet provide Tensor CUDA kernels or a
CUDA coroutine scheduler.

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
- `cuda::Buffer<T>` retains the latest exclusive-writer completion and reader
  completions since that writer. The existing async `EpochQueue` or synchronous
  program order remains the causal model.
- Submitters use `buffer.read(stream)` and `buffer.write(stream)` scoped guards.
  Raw device pointers are exposed only through those guards. Compatible readers
  overlap; a writer waits for all unfinished prior accesses.
- The context mutex protects only completion snapshots and publication. It is
  not held while a backend or provider call executes.
- CUDA runtime failures use `CudaRuntimeError` and Uni20's presentation layer.
  Cleanup failures and invalid stream-pool state remain fail-fast logic errors.

## Related Documentation

- [Backend source layer](../)
- [CUDA backend documentation](../../../../docs/backends/cuda/)
- [CUDA kernel dispatch and provider scheduling](../../../../docs/backends/cuda/kernel_dispatch.md)
- [Execution architecture](../../../../docs/architecture/execution.md)
