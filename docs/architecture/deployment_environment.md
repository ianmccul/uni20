# Deployment Environment

**Status:** guiding architectural constraints, updated 2026-07.

Uni20 is intended to run on research workstations and on scheduled HPC
clusters. The cluster environment is the stronger design constraint: core
Tensor, storage, dispatch, async, CUDA, and distributed interfaces must not
assume a single process, one host architecture, or a workstation-sized GPU
topology.

This document describes the shape of the intended deployment environment. It
is not a provider-library support matrix or a promise that every operation is
already implemented on every listed accelerator.

## Target Shape

The architecture should accommodate:

- Slurm-managed batch jobs;
- one or more MPI processes per node;
- x86-64 and AArch64 application threads;
- CPU-only and accelerator-equipped nodes;
- one to many NVIDIA GPUs visible to each process;
- multiple GPU generations in the supported deployment population;
- high-speed node interconnects used by MPI and GPU communication libraries;
- large shared parallel file systems;
- site-provided compilers, MPI implementations, CUDA installations, modules,
  and containers.

The application remains responsible for requesting a valid collection of
nodes, processes, CPUs, GPUs, and memory from the site scheduler. Uni20 must
interpret the resources visible inside that allocation without assuming their
physical numbering or whole-system topology.

## Architectural Principles

### Memory kind and location are different

Memory kind is a compile-time storage property. Runtime location identifies a
particular CUDA device, MPI rank, NUMA domain, or future placement target.
Encoding a CUDA ordinal or MPI rank in a Tensor type would make ordinary
multi-device and distributed placement impractical.

See [Storage Kind and Location](storage_kind_and_location.md) for the canonical
contract.

### Visible resources define the process environment

CUDA code must use runtime-visible device ordinals. It must not assume that
visible device zero is physical device zero, that all node devices are visible,
or that every MPI process sees the same device set. Slurm, containers, and
other launch mechanisms may remap or restrict device visibility.

A process-wide Uni20 CUDA runtime is process-local under MPI. It owns canonical
resources for the devices enrolled in that process; it is not a machine-wide
singleton shared between ranks.

### Per-device resources must scale beyond a workstation

Schedulers, stream pools, provider handles, device constants, diagnostics, and
completion tracking must be organized per enrolled device. Their cost should
scale with devices actually used by a process rather than with every physical
device in the cluster.

The design must remain practical for an eight-GPU node and must not encode a
two-GPU upper bound. Larger tightly connected systems make this requirement
more visible, but do not require every process to enroll every available GPU.

### Host architecture is not an execution-policy tag

Generic host code must remain portable between x86-64 and AArch64. ISA-specific
vectorization and tuning belong in compiler-generated code or explicitly
selected backends, not in Tensor or async semantics.

### Transfers and communication are explicit operations

Host/device, peer-device, and cross-rank movement must not appear as an
ordinary backend fallback. Placement planning emits explicit transfer or
communication operations whose storage, synchronization, failure, and cost
remain visible to the async and diagnostic layers.

### Capability is established, not inferred

Finding CUDA or recognizing a GPU architecture does not prove that a particular
cuBLAS, cuSOLVER, cuTENSOR, cuQuantum, NCCL, or other provider path is usable.
Compile-time configuration and side-effect-free runtime capability checks must
gate provider selection. Unsupported providers may decline before commitment;
they must not fail after mutating operands and then fall through to another
backend.

### Compatibility spans old and new accelerators

The deployment population includes Volta-class V100 hardware as well as Hopper
H100/H200 and Blackwell systems. CUDA 12.9 compatibility is therefore an
important baseline for Uni20 CUDA code. APIs introduced after that baseline may
be used only behind a versioned optional path with a correct alternative.

Provider releases may support a narrower accelerator range than the CUDA
runtime. Uni20 should retain capability-based backend selection rather than
raising the minimum accelerator generation globally to match one provider.

### Cluster operating systems do not lower the language baseline

Uni20 requires C++23 and the compiler versions documented in the contributor
guide. Older cluster operating-system toolchains may require a site module,
toolchain installation, or container. Supporting an older base operating
system does not justify adding a second, older C++ language mode.

### Throughput is the default optimization target

Large calculations normally favor sustained throughput over minimizing the
latency of one task. Async scheduling, CUDA resource admission, MPI progress,
and placement policy should permit independent useful work while an operation
waits for a device or communication dependency. Correctness must not depend on
fortunate scheduler timing or a particular rank/device assignment.

## Motivating NCHC Systems

The National Center for High-performance Computing systems are a primary
deployment target. The following examples are a dated motivation for the
principles above, not a frozen inventory or an exhaustive certification list.

| System | Relevant published shape |
|---|---|
| NANO 4 H200 nodes | Dual Intel Xeon hosts, eight NVIDIA H200 GPUs, and 2 TB host memory per node. |
| NANO 4 GB200 NVL72 | 72 NVIDIA Grace processors, 72 Blackwell GPUs, and 13.5 TB memory per system. |
| NANO 5 | Dual Intel Xeon hosts, eight H100 or H200 GPUs, 2 TB host memory, and multiple 400 Gb/s InfiniBand ports per node. |
| Forerunner 1 | Large dual-socket Intel CPU partition, 200 Gb/s InfiniBand, and an additional ARM node population. |
| TAIWANIA 2 | 252 nodes with eight NVIDIA V100 32 GB GPUs per node and 100 Gb/s InfiniBand. |
| TAIWANIA 3 | Large CPU-only partition with dual-socket Intel nodes and a shared parallel file system. |

Source: [NCHC supercomputer specifications](https://www.nchc.org.tw/Page?itemid=58&mid=109),
accessed 2026-07-22. The live NCHC page is authoritative for current hardware
and service details.

## Consequences for Subsystem Design

- Tensor descriptors carry storage kind and runtime placement without owning
  execution resources such as streams.
- CUDA access resolution chooses resources for the descriptor's enrolled
  device rather than consulting an implicit global device zero.
- A future `BlockTensor` may place different logical blocks on different
  devices or MPI ranks while preserving symmetry metadata.
- Async causality remains independent of the CUDA and MPI mechanisms used to
  prove backend completion.
- MPI communicators, CUDA streams, and provider handles remain execution
  context or backend resources rather than intrinsic mathematical Tensor
  properties.
- Tests should include single-device, multi-device, CPU-only, and restricted
  visibility configurations where the relevant hardware is available.
- Local workstation tests validate mechanisms, but they do not define the
  maximum topology supported by an interface.

## Related Documentation

- [Architecture Overview](overview.md)
- [Execution Architecture](execution.md)
- [Storage Kind and Location](storage_kind_and_location.md)
- [Ordering and Backend Lowering](ordering_and_backend_lowering.md)
- [Distributed Kernel Dispatch](distributed_kernel_dispatch.md)
- [CUDA Backend Documentation](../backends/cuda/)

