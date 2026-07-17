# src/uni20/backend/cuda

This directory contains the CUDA runtime foundation and is the CUDA
backend-library wiring point. It does not yet provide Tensor CUDA kernels or a
CUDA coroutine scheduler.

## Contents

- `cuda_error.hpp`: structured CUDA runtime failures and checked-call helpers.
- `cuda_error_presentation.hpp`: presentation-layer rendering for CUDA failures.
- `device.hpp`: validated device identities and process-wide immutable hardware
  capability caching.
- `runtime.hpp`: device guards, move-only streams/events, completion tokens, and
  the device-local idle-stream pool.
- `CMakeLists.txt`: CUDA backend target setup.

## Notes

- Keep CUDA toolkit discovery and target properties local to backend targets.
- `cuda::Device` is a cheap ordinal value. Device discovery validates the
  ordinal and initializes one immutable capability snapshot per visible device;
  it does not create schedulers, streams, provider handles, or allocation pools.
- Runtime capability checks should remain operation-specific; do not assume one
  CUDA build option makes every CUDA library feature available.
- A stream-pool slot is available only after all work previously submitted to
  that stream has completed. Returning a lease records a non-timing completion
  event and enqueues a lightweight host function that marks the slot idle.
- Buffer dependencies use completion events and `cudaStreamWaitEvent`; the pool
  does not attempt dependent-task stream affinity.
- CUDA runtime failures use `CudaRuntimeError` and Uni20's presentation layer.
  Cleanup failures and invalid lease state remain fail-fast logic errors.

## Related Documentation

- [Backend source layer](../)
- [CUDA backend documentation](../../../../docs/backends/cuda/)
- [CUDA kernel dispatch and provider scheduling](../../../../docs/backends/cuda/kernel_dispatch.md)
- [Execution architecture](../../../../docs/architecture/execution.md)
