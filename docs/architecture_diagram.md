# Uni20 Architecture Overview

This diagram summarizes the current repository architecture.

- **Solid nodes/edges**: implemented and used today
- **Dashed nodes/edges**: present as stubs or planned extensions

```mermaid
graph TD
    subgraph Interfaces
        CppApps[C++ Applications]
        PyBind[Python Bindings (nanobind)]
    end

    subgraph PublicAPI[Public API Surface]
        Uni20Lib[libuni20]
    end

    subgraph Core[Core Modules under src/uni20]
        Tensor[Tensor / TensorView]
        Mdspan[mdspan utilities]
        Level1[Level1 tensor ops]
        Linalg[Linalg ops + backends]
        Kernel[Kernel dispatch]
        Async[Async runtime]
        Common[Common/core utilities]
        Expokit[Expokit integration]
    end

    subgraph AsyncRuntime[Async Runtime Details]
        ReadWrite[ReadBuffer / WriteBuffer]
        Epoch[EpochQueue / EpochContext]
        VarAD[Var + ReverseValue]
        DebugSched[DebugScheduler]
        TbbSched[TbbScheduler]
        TbbNuma[TbbNumaScheduler]
        CudaTask[CudaTask only]
        GpuSched[GPU Scheduler (planned)]
        DistSched[Distributed Scheduler (planned)]
    end

    subgraph Backend[Backend Layer]
        BlasBackend[BLAS backend]
        CpuLinalg[CPU linalg backend]
        CudaBackend[CUDA backend stubs]
        CuSolver[cuSOLVER backend stubs]
    end

    CppApps --> Uni20Lib
    PyBind --> Uni20Lib

    Uni20Lib --> Tensor
    Uni20Lib --> Mdspan
    Uni20Lib --> Level1
    Uni20Lib --> Linalg
    Uni20Lib --> Kernel
    Uni20Lib --> Async
    Uni20Lib --> Common
    Uni20Lib --> Expokit

    Tensor --> Mdspan
    Level1 --> Kernel
    Linalg --> CpuLinalg
    Linalg --> BlasBackend
    Kernel --> BlasBackend
    Kernel --> CpuLinalg
    Kernel -.-> CudaBackend
    Linalg -.-> CuSolver

    Async --> ReadWrite
    Async --> Epoch
    Async --> VarAD
    Async --> DebugSched
    Async --> TbbSched
    Async --> TbbNuma
    Async --> CudaTask
    Async -.-> GpuSched
    Async -.-> DistSched

    style GpuSched stroke-dasharray: 5 5
    style DistSched stroke-dasharray: 5 5
    style CudaBackend stroke-dasharray: 5 5
    style CuSolver stroke-dasharray: 5 5
```

## Notes

- Python bindings are implemented with `nanobind` in `bindings/python/`.
- Async execution is scheduler-driven; the primary schedulers are `DebugScheduler`, `TbbScheduler`, and `TbbNumaScheduler`.
- `CudaTask` exists as a coroutine type, but a full CUDA scheduler/runtime path is still planned.
- BLAS and CPU linalg paths are active; CUDA/cuSOLVER integration remains partial.
