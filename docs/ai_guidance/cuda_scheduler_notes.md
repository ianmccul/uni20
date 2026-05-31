# CUDA Scheduler Design Notes

These notes record design direction only.  They are not implemented uni20 GPU
runtime behavior.

The intended uni20 CUDA runtime should model GPU resources explicitly.  A CUDA
device context is expected to own the device-local execution resources: streams,
cuBLAS handles, memory-pool state, and eventually event pools and library
handles for cuSOLVER/cuTensorNet-style kernels.

The scheduler should submit GPU tasks through a small number of host worker
threads.  A cuBLAS handle is device-local and has a mutable associated stream,
so the safe resource model is one handle per concurrently used stream slot.  In
practice this likely means either:

- fixed worker/device affinity, with each device context owning the handles used
  by its workers; or
- a handle pool keyed by `(worker, device)` or by explicit borrowed stream slots.

The first design is simpler and should be preferred until workload measurements
justify fully flexible cross-device worker scheduling.  The second design is
more general, but requires stricter borrowing rules to avoid racing
`cublasSetStream` on a shared handle.

The temporary TensorContraction bridge now prototypes only the resource shape:
one `CudaDeviceContext` per active device, with a fixed pool of work-stream
slots and one cuBLAS handle per slot.  It deliberately does not prototype the
full uni20 scheduler, MPI client/server model, or final CUDA allocator design.

## Memory-Block Epochs

The async `EpochContext` mechanism provides the right conceptual model for
GPU-memory dependencies.  In async uni20, one logical value owns a chain of
epochs: readers attach to the current epoch, a writer creates or completes the
next generation, and RAII reader/writer handles advance the epoch when access
finishes.  The CUDA analogue should be a per-memory-block epoch state rather
than work-item-local synchronization.

For GPU buffers, the object being synchronized is the memory block.  Each
device-resident block should own its access state:

- the stream/event that completed the current written generation;
- outstanding read generations, coalesced by stream when possible;
- host/GPU/remote validity state for explicit materialization boundaries;
- debug counters/snapshots that can explain which stream owns the current
  generation and why a wait is required.

The core protocol should be:

- a read access waits only for the latest writer generation, unless it is
  already on the same stream;
- a write access waits for the latest writer and all outstanding readers;
- a write publishes a new generation and invalidates or supersedes previous
  read generations;
- multiple reads may share the same generation and should not force a new CUDA
  event unless a later writer must wait for a reader on a different stream;
- same-stream dependencies are represented by CUDA stream order and should not
  allocate or record events.

This is intentionally similar to `EpochContext`, but the GPU variant should not
copy its coroutine scheduling machinery.  CUDA events are only the concrete
cross-stream readiness tokens for memory epochs.  A future scheduler can replace
blocking waits with suspend/resume semantics while preserving the same
per-block access-state protocol.

This model also gives a criterion for TensorContraction cleanup.  Any operation
that touches matrix memory should express synchronization through the owning
`GpuBuffer` access state.  Separate raw event maps or batch-local sync events
are acceptable only as temporary bridge code, or for non-matrix resources such
as reusable scratch buffers.
