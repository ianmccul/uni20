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

`cublas::execution_pool(context)` lazily constructs one pool owned by the CUDA
device context. Direct Tensor dispatch uses its blocking `acquire()` path.
CUDA tasks can use `co_await cublas::acquire_execution(pool)` so exhausted
resource admission suspends rather than occupying a scheduler participant.
The checked provider wrappers consume an already acquired lease and never
perform resource admission themselves.
