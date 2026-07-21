# MPI Backend Documentation

[Persistent Object Store and Kernel Dispatch](persistent_dispatch.md) is the
current distributed-execution design note. It covers immutable persistent
objects, rank placement, operation lowering, and interaction with backend
dispatch.

[Distributed Kernel Dispatch](../../architecture/distributed_kernel_dispatch.md)
records the general planning, commitment, collective-consistency, and local
composition constraints shared by possible distributed execution models.

MPI Tensor execution is not implemented on the current main branch. The note is
forward architecture and must not be read as current API behavior.
