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
