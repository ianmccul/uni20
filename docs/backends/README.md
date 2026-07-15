# Backend Documentation

This directory groups provider- and platform-specific execution design.

- [CUDA](cuda/README.md) contains GPU ecosystem surveys and active CUDA runtime,
  memory, epoch, and cuSOLVER design notes.
- [MPI](mpi/README.md) contains the distributed persistent-object and dispatch
  design.

These directories describe planned heterogeneous execution unless a document
explicitly identifies an implemented CPU/provider adapter. The current working
backend path is summarized by [Kernel Dispatch](../architecture/kernel_dispatch.md)
and [Linear Algebra](../linalg/README.md).
