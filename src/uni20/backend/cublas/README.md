# src/uni20/backend/cublas

This directory contains low-level cuBLAS ownership and checked provider calls.
It sits above the CUDA stream and completion primitives and below linalg Tensor
policy.

## Contents

- `cublas_error.hpp`: structured cuBLAS failures and checked-call helpers.
- `cublas_error_presentation.hpp`: presentation-layer rendering for cuBLAS failures.
- `execution.hpp`: reusable handle slots and dynamic handle-plus-stream leases.
- `task_awaiters.hpp`: non-blocking CUDA-task acquisition of an execution lease.
- `gemm.hpp`: checked column-major `S/D/C/ZGEMM` wrappers.

## Ownership

An `ExecutionPool` owns expensive cuBLAS handles but does not permanently pair
them with streams. Each operation acquires a handle first, then an actually-idle
stream, and receives one move-only `ExecutionLease`. The handle is conservatively
returned at the submitted stream tail. The stream independently returns to its
pool when its final reference is released and the stream becomes idle.

CUDA tasks use `co_await cublas::acquire_execution(pool)`. Blocking bring-up code
may call `pool.acquire()`. Provider wrappers do not acquire resources internally.
