# Uni20

Uni20 is a C++23 tensor-network research library combining dense Tensor
operations, ordered CPU/BLAS/LAPACK backend dispatch, matrix-free Krylov
algorithms, and an epoch-based asynchronous runtime. The API is still in active
design, but working `Async<Tensor>` examples now exercise the complete path from
scheduling through backend kernels and structured presentation.

<table class="uni20-home-grid">
<tr>
<td>
<a class="uni20-home-card-title" href="md_docs_2about.html">About Uni20</a>
<span>See implemented capabilities, runnable vertical slices, and current boundaries.</span>
</td>
<td>
<a class="uni20-home-card-title" href="md_docs_2getting__started.html">Get Started</a>
<span>Configure, build, test, and run the current CMake workflow.</span>
</td>
<td>
<a class="uni20-home-card-title" href="md_docs_2async_2runtime__model.html">Async Runtime</a>
<span>Understand epochs, schedulers, tensor aliases, and operation lowering.</span>
</td>
<td>
<a class="uni20-home-card-title" href="md_docs_2krylov__algorithms.html">Krylov Solvers</a>
<span>Review Lanczos, Arnoldi, exponential, and restart behavior.</span>
</td>
</tr>
</table>

## Core Documentation

- [About Uni20](about.md)
- [Contributor guide](CONTRIBUTING.md)
- [Getting started](getting_started.md)
- [Architecture overview](architecture_diagram.md)
- [Roadmap](roadmap.md)
- [Testing](testing.md)
- [Scalar policy](scalar_policy.md)
- [Build information](buildinfo.md)
- [Python bindings](Python.md)

## Implemented Vertical Slices

- [Tensor operations and Async support](tensor_operations.md)
- [Kernel dispatch and runtime diagnostics](kernel_dispatch.md)
- [Async Tensor kernel authoring](async/kernel_authoring.md)
- [Generated tensors and reshape](tensor_creation_and_reshape.md)
- [Presentation and structured reports](presentation.md)
- [Async task and DAG diagnostics](async/task_registry_debug.md)

## Runtime And Tensor Work

- [Async runtime model](async/runtime_model.md)
- [Async buffers and awaiters](async/buffers_and_awaiters.md)
- [Async schedulers](async/schedulers.md)
- [Tensor operations, semantics, and Async support](tensor_operations.md)
- [Generated tensors and reshape](tensor_creation_and_reshape.md)
- [Storage kind and location](storage_kind_and_location.md)
- [Backend dispatch](backend_dispatch.md)

## Linear Algebra And Models

- [Krylov algorithms](krylov_algorithms.md)
- [Krylov precision validation](krylov_precision_validation.md)
- [Krylov test matrices](krylov_test_matrices.md)
- [MPS](mps.md)
- [MPO operators](operators.md)
- [QNum](qnum.md)

## Development Notes

- [Agent-assisted development](agent_assisted_development.md)
- [Reviewing changes](code_review.md)
- [AI guidance](ai_guidance/README.md)
- [Trace macros](trace_macros.md)
- [Doxygen policy](Doxygen.md)
