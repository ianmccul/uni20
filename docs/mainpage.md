# Uni20

Uni20 is an early-stage C++23 tensor-network library focused on asynchronous
execution, tensor layout machinery, dense and block-sparse building blocks, and
Krylov/LAPACK solver infrastructure.

<table class="uni20-home-grid">
<tr>
<td>
<a class="uni20-home-card-title" href="md_docs_2getting__started.html">Get Started</a>
<span>Configure, build, test, and run the current CMake workflow.</span>
</td>
<td>
<a class="uni20-home-card-title" href="md_docs_2architecture__diagram.html">Architecture</a>
<span>Orient around the major runtime, tensor, backend, and model components.</span>
</td>
<td>
<a class="uni20-home-card-title" href="md_docs_2async_2runtime__model.html">Async Runtime</a>
<span>Understand tasks, schedulers, buffers, and dependency ordering.</span>
</td>
<td>
<a class="uni20-home-card-title" href="md_docs_2krylov__algorithms.html">Krylov Solvers</a>
<span>Review Lanczos, Arnoldi, exponential, and restart behavior.</span>
</td>
</tr>
</table>

## Core Documentation

- [Contributor guide](CONTRIBUTING.md)
- [Getting started](getting_started.md)
- [Testing](testing.md)
- [Scalar policy](scalar_policy.md)
- [Build information](buildinfo.md)
- [Python bindings](Python.md)

## Runtime And Tensor Work

- [Async runtime model](async/runtime_model.md)
- [Async buffers and awaiters](async/buffers_and_awaiters.md)
- [Async schedulers](async/schedulers.md)
- [Tensor operations, semantics, and Async support](tensor_operations.md)
- [Tensor dispatch and view semantics (background draft)](tensor_dispatch_and_view_semantics_draft.md)
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
- [Roadmap](roadmap.md)
- [Trace macros](trace_macros.md)
- [Doxygen policy](Doxygen.md)
