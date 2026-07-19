# CUDA/cuSOLVER Architecture Notes

**Status:** active design note. A complete cuSOLVER Tensor backend is not
implemented on the current main branch.

These notes describe the intended direction for the future native uni20
CUDA/cuSOLVER runtime.  They are not a statement of current implementation
status.  The temporary TensorContraction bridge currently has a narrower
host-facing cuSOLVER SVD path.

The generic host-execution and dispatch contract is defined in
[CUDA Kernel Dispatch and Device Scheduling](kernel_dispatch.md). This note
keeps the cuSOLVER-specific consequences.

## Primary References

- NVIDIA cuSOLVER documentation:
  [Thread Safety](https://docs.nvidia.com/cuda/archive/12.9.0/cusolver/index.html#thread-safety),
  [Parallelism with Streams](https://docs.nvidia.com/cuda/archive/12.9.0/cusolver/index.html#parallelism-with-streams),
  and
  [cusolverDnSetStream](https://docs.nvidia.com/cuda/archive/12.9.0/cusolver/index.html#cusolverdnsetstream).
- NVIDIA CUDA Programming Guide:
  [CUDA Streams](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)
  and the surrounding asynchronous execution section.

The key interpretation for uni20 is:

- CUDA streams are ordered work queues.  Work enqueued into one stream executes
  in stream order; multiple streams can expose independent work to the runtime.
- cuSOLVER is thread-safe, so independent host scheduler threads may use
  independent cuSOLVER handles.
- A cuSOLVER handle has an associated stream.  That stream association is part
  of the execution context for the solver call.
- If a uni20 operation immediately copies results to host and synchronizes, the
  operation is effectively blocking at the API boundary even if cuSOLVER queued
  device work asynchronously internally.
- Synchronous host/device `cudaMemcpy` calls are blocking; asynchronous copies
  still become blocking if the API boundary immediately waits for the stream.

## Resource Model

The preferred native Uni20 model is one execution context and scheduler per
CUDA device. The context owns bounded pools of cuSOLVER handles, actually-idle
streams, workspaces, and completion resources:

```text
CUDA device context
  -> device scheduler arena
  -> actually-idle stream pool
  -> exclusive cuSOLVER handle pool
  -> workspace and completion resources
```

A solver request begins only when its complete resource set is available. The
coroutine may suspend while waiting for that composite acquisition. Once
admitted, setting the handle stream, invoking cuSOLVER, and recording submission
completion form one non-suspending operation.

Do not create and destroy cuSOLVER handles per operation. Handle creation is
runtime state setup, not useful numerical work. Handles are device-local leased
resources, not permanent properties of particular oneTBB worker threads.

An exclusive handle lease may move between physical workers in the same device
arena because the arena observer establishes the same CUDA device on every
participant. Different operations must not mutate one handle concurrently.
Provider-specific state that remains live after host-call return stays retained
until the relevant completion boundary.

The operation may rotate through the bounded stream pool. Workspace and
completion tracking follow the submitted operation and selected stream, not the
handle alone.

Multiple handle leases allow independent solver work with host-side concurrency.
A cuSOLVER call such as SVD may occupy one device-scheduler participant while
launching and coordinating many device kernels, even though device work may
remain pending after the API returns. Scheduler concurrency and handle-pool
capacity therefore provide separate bounds.

Multiple cuSOLVER leases become useful when the operation boundary is resident
and asynchronous. For example, block-sparse SVD can submit independent symmetry
sectors, publish GPU completion tokens, and let dependent work resume later.

Raw cuSOLVER handles remain internal to RAII leases and backend adapters. An
internal coroutine may await composite resource admission, but the actual
backend walk and cuSOLVER call are ordinary non-coroutine code. No further
`co_await` occurs while the provider call is in progress.

The first implementation runs cuSOLVER directly on the device scheduler. A
separate provider scheduler or lane is justified only if profiling shows that
host-intensive calls starve lightweight submission or unrelated device
continuations.

## Host Blocking Boundaries

The main numerical cuSOLVER calls should be treated as potentially asynchronous
device-work submissions.  They can still perform nontrivial CPU-side setup, and
the amount of host work is implementation- and routine-dependent, so profiling
with Nsight Systems remains the authority for performance decisions.

Known host-side boundaries are:

- handle creation and destruction;
- workspace-size queries and other metadata/setup calls;
- `cusolverDnSetStream`, which mutates the handle state for the next call;
- synchronous `cudaMemcpy` between host and device;
- explicit synchronization such as `cudaStreamSynchronize`,
  `cudaDeviceSynchronize`, or `cudaEventSynchronize`;
- copying device outputs such as `devInfo`, singular values, or factor data back
  to host and then waiting for those copies.

For example, `cusolverDnDgesvd` writes `devInfo` on the device.  A wrapper that
needs `devInfo` on the CPU before returning must enqueue a D2H copy and wait, so
the wrapper is host-blocking even if the solver call itself was submitted
asynchronously.

## SVD Boundary

The original TensorContraction bridge used this host-facing shape:

```text
host MatrixFamily center
  -> copy dense matrix to cuSOLVER
  -> run single-block SVD
  -> copy U, S, Vt back to host
  -> split MPS tensors on CPU
```

The first CUDA DMRG bridge keeps this path as a fallback, but the dense
placeholder-symmetry path can pack resident two-site vector blocks directly into
cuSOLVER input memory:

```text
resident two-site blocks
  -> pack one dense placeholder-symmetry SVD input on GPU
  -> run single-block cuSOLVER SVD
  -> copy singular values and split site blocks to host-owned FiniteMPS
```

The U(1) prototype has a stricter active-data boundary:

```text
resident two-site blocks
  -> assemble one dense SVD input per charge sector on GPU
  -> run independent cuSOLVER SVDs by symmetry sector
  -> copy singular values and devInfo scalars to host for truncation decisions
  -> scatter U/Vt directly into resident block-sparse MPS tensors
  -> materialize those tensors to host only at an explicit cold-storage boundary
```

The current implementation centralizes those sector SVDs on one target GPU.
That is enough to enforce the active-tensor residency model and remove U/Vt host
copies.  Distributing sectors across CUDA devices and MPI workers is a placement
policy problem and should be layered on top of the same resident block-sparse
API rather than reintroducing implicit host tensors.

## Scheduler Boundary

The CUDA/cuSOLVER layer should not become a second dependency DAG. CPU-side
epochs establish causal readiness: a GPU operation is submitted only once its
metadata and object lifetime are valid. The current GPU runtime lowers this
through scoped `buffer.read_synchronized_with(stream)` /
`buffer.write_synchronized_with(stream)` guards and
retained writer/reader completions, without reconstructing an independent epoch
graph.

This keeps the split clear:

- async epoch model: task ordering, object lifetime, and high-level dependency
  causality;
- per-device scheduler: execution of `CudaTask` activations routed from global
  scheduling or heterogeneous nested `co_await`, with bounded host
  participation;
- GPU storage/runtime: buffer access completions, stream/event synchronization,
  resource leasing, and device work submission;
- cuSOLVER backend: a non-suspending solver call that consumes leased resources
  and publishes submission/completion tokens.

The `CudaTask` is first routed to its device scheduler, then suspends until its
composite stream/handle/workspace request is available. The entire relevant
backend walk executes without suspension. Once the solver API returns and the
completion event is recorded, the scheduler participant is free; operation
resources remain retained until their provider-specific release boundary.

## Open Questions

- Whether profiling eventually justifies a separate bounded cuSOLVER execution
  lane rather than direct execution on the device scheduler.
- Which routine families permit a cuSOLVER handle to be reused immediately
  after the host API returns, rather than after device completion.
- Whether block-sparse SVD should use one cuSOLVER call per sector, batched
  Jacobi SVD for many small sectors, or a hybrid policy.
- How much cuSOLVER workspace should be cached per handle or operation class,
  and how workspace
  pressure should interact with the broader CUDA allocator.
- How MPI client-server scheduling should assign symmetry-sector SVD blocks to
  remote CUDA workers.
