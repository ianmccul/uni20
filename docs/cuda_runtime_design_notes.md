# CUDA Runtime Design Notes

These notes record longer-term uni20 CUDA runtime design ideas that are broader
than the temporary TensorContraction bridge.  They are design direction, not
current implemented behavior.

## Stream Ownership

CUDA stream handles should be wrapped in move-only C++ ownership types.

The basic stream ownership invariant is:

- a `tensor::cuda::Stream` owns one concrete CUDA stream slot;
- copying is disabled;
- moving transfers ownership;
- destruction returns the stream slot to its owning `CudaDeviceContext`;
- returning a stream to the pool means it can accept more ordered work, not that
  the GPU has finished all previously enqueued work in that stream.

This is analogous to Rust-style affine ownership: there is one live owner of a
borrowed stream slot, and the slot is returned automatically on all normal exit
paths.  Correctness should not depend on users manually returning streams.

The `GpuEpochQueue` design in `gpu_epoch_design_draft.md` uses
`tensor::cuda::Stream` only as a transient enqueue resource.  Durable memory
dependencies are opaque `tensor::cuda::Completion` tokens recorded at
publication time.

## Idle-Aware Stream Pool

The first TensorContraction implementation should not require idle streams.  It
can safely reuse a non-idle stream because CUDA stream order appends new work
after existing work in that stream.  Memory correctness is handled by epoch
events and `cudaStreamWaitEvent`.

For the full uni20 async runtime, an idle-aware stream pool may still be useful
as a scheduling and backpressure mechanism:

```text
tensor::cuda::Stream stream = co_await device.acquire_idle_stream();
enqueue kernels/copies into stream.stream();
publish GpuEpochQueue events;
arrange for the stream slot to be marked idle when queued work completes;
```

This would let the CPU scheduler limit the amount of queued device work, improve
fairness between tasks, and expose resource pressure through normal coroutine
suspension instead of unbounded enqueueing.

One possible implementation is to enqueue a lightweight host callback with
`cudaLaunchHostFunc` after the submitted GPU work.  The callback would mark the
stream slot idle and wake any coroutine waiting for an idle stream.

Important constraints:

- CUDA host callbacks must stay lightweight and must not call CUDA APIs.
- Idle status is a resource-management property, not a memory-correctness
  property.
- `GpuEpochQueue` should still publish events for read/write dependencies,
  because a different stream may need to wait on the data before the producer
  stream becomes idle.
- `co_await acquire_idle_stream()` belongs in the uni20 async scheduler or
  `CudaDeviceContext` resource layer, not inside `GpuEpochQueue`.

## Current Prototype Boundary

The TensorContraction bridge should implement the simpler model first:

- move-only stream leases;
- event-based read/write epochs;
- no stream-affinity cache;
- no idle-stream waiting;
- no CUDA host callbacks.

This keeps the prototype robust and leaves idle-aware scheduling as a future
optimization for the real uni20 CUDA runtime.
