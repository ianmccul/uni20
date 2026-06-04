# CUDA/cuSOLVER Architecture Notes

These notes describe the intended direction for the future native uni20
CUDA/cuSOLVER runtime.  They are not a statement of current implementation
status.  The temporary TensorContraction bridge currently has a narrower
host-facing cuSOLVER SVD path.

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

The preferred native uni20 model is one solver execution lane per scheduler
thread and CUDA device:

```text
host scheduler thread
  -> CUDA device
    -> cuSOLVER handle
    -> small pool of CUDA streams
    -> optional per-stream or per-lane workspace cache
```

Do not create and destroy cuSOLVER handles per operation.  Handle creation is
runtime state setup, not useful numerical work.  The current TensorContraction
SVD prototype already follows the first approximation of this rule by caching a
cuSOLVER handle and stream per host thread per CUDA device.

Do not create many cuSOLVER handles in one host thread just to call a
host-synchronous wrapper.  If the wrapper synchronizes before returning
host-visible data, extra same-thread handles only add resource pressure.

Using one cuSOLVER handle with multiple streams can be useful only for a narrow
single-threaded enqueue pattern: set the handle's stream just before each solver
call, submit independent work, and do not synchronize immediately.  This can
allow the GPU to overlap the queued stream work when the solver calls return
quickly enough and the individual tasks do not fill the device.

That pattern is not the preferred uni20 scheduler model when it is exposed as
ad-hoc mutable handle state.  A solver lane may still rotate through a small
bounded stream pool, provided the stream selection happens immediately before
the cuSOLVER call and is part of a single submission operation.  This loses
nothing relative to a fixed stream when the operation later synchronizes, and
can expose useful overlap when independent solver blocks remain resident on the
GPU.  Workspace and completion tracking must follow the selected stream, not
the handle alone.

Multiple lanes, typically from multiple scheduler threads or a dedicated
cuSOLVER thread pool, are the robust way to submit independent solver work with
host-side concurrency.  Within one lane, a small stream pool is useful for
device-side overlap; it is not a substitute for multiple host submitters.

Multiple cuSOLVER lanes become useful when the operation boundary is resident
and asynchronous.  For example, block-sparse SVD can submit independent symmetry
sectors to different solver lanes, publish GPU completion tokens, and let the
CPU scheduler resume dependent work later.

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

The current bridge keeps this path as a fallback, but the CUDA DMRG path now
packs resident two-site vector blocks directly into cuSOLVER input memory:

```text
resident two-site blocks
  -> pack one dense placeholder-symmetry SVD input on GPU
  -> run single-block cuSOLVER SVD
  -> copy singular values and split site blocks to host-owned FiniteMPS
```

This removes the final Lanczos-vector host materialization, but it is still not
the final uni20 target.  The native path should assemble SVD inputs from
resident GPU tensor blocks and keep later consumers resident where possible:

```text
resident two-site blocks
  -> assemble one or more block-diagonal SVD inputs on GPU
  -> run independent cuSOLVER SVDs by symmetry sector
  -> keep U/S/Vt resident when the next operation can consume them on GPU
  -> materialize to host only at explicit API boundaries
```

For the dense placeholder-symmetry case this reduces to one block.  For U(1)
and later symmetry support, each allowed charge sector is an independent SVD
block.  Those blocks are the natural unit of distribution across CUDA devices
and MPI workers.

## Scheduler Boundary

The CUDA/cuSOLVER layer should not become a general CPU-side DAG executor.
CPU-side async tasks should establish causal readiness: a GPU operation should
only be submitted once the CPU metadata and object lifetime are valid.  The GPU
storage layer then handles device-local read/write dependencies with CUDA
events and streams.

This keeps the split clear:

- CPU scheduler: task ordering, coroutine suspension/resumption, object
  lifetime, and high-level dependency causality.
- GPU storage/runtime: buffer read/write epochs, stream/event synchronization,
  library handle selection, and device work submission.
- cuSOLVER lane: a concrete handle plus a bounded stream/workspace context used
  to submit solver work for one device on one scheduler thread.

## Open Questions

- Whether solver lanes are owned by general CUDA scheduler threads or by a
  dedicated cuSOLVER scheduler pool.
- Whether block-sparse SVD should use one cuSOLVER call per sector, batched
  Jacobi SVD for many small sectors, or a hybrid policy.
- How much cuSOLVER workspace should be cached per lane, and how workspace
  pressure should interact with the broader CUDA allocator.
- How MPI client-server scheduling should assign symmetry-sector SVD blocks to
  remote CUDA workers.
