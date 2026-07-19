# Backend Documentation

This directory groups provider- and platform-specific execution design.

- [CUDA](cuda/) contains the current low-level buffer/runtime guides, active
  scheduler and provider designs, and GPU ecosystem surveys.
- [MPI](mpi/) contains the distributed persistent-object and dispatch
  design.

These directories mix current low-level foundations with planned
heterogeneous execution. Each document identifies its status. The current
working backend path is summarized by [Kernel
Dispatch](../architecture/kernel_dispatch.md) and [Linear Algebra](../linalg/).

## Source Navigation

- [Provider wrapper source map](../../src/uni20/backend/)
- [Operation-tag backend source map](../../src/uni20/linalg/backends/)

There is no MPI source subtree yet; the MPI documentation is forward design.
