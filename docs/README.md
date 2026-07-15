# Uni20 Documentation

This index routes readers to subsystem-owned documentation. Each subsystem
`README.md` identifies its canonical guides, active designs, and background
material.

## Start Here

- [About Uni20](about.md): implemented capabilities, runnable vertical slices,
  and current boundaries.
- [Getting Started](getting_started.md): prerequisites, configuration, build,
  tests, benchmarks, and Python smoke bindings.
- [Architecture Overview](architecture/overview.md): implemented layers and
  planned extensions.
- [Roadmap](roadmap.md): completed foundations, current priorities, and later
  work.
- [Contributor Guide](CONTRIBUTING.md): development workflow, evidence, and
  review expectations.
- [Source Tree Map](../src/uni20/README.md): module ownership and local source
  navigation.
- [Examples Index](../examples/README.md): runnable demonstrations, probes,
  diagnostics, and retained experiments.

## Document Status

- **Current guide:** defines intended behavior for an implemented area and must
  change with the code.
- **Design note:** records active intended design and may include interfaces
  that are not implemented.
- **Background:** preserves surveys, experiments, integration findings, or
  historical reasoning and does not define current behavior.

Tests encode selected contracts and source records the current implementation.
If either conflicts with a canonical guide, report and resolve the discrepancy
rather than silently choosing one source.

## Subsystems

| Area | Entry point | Scope |
|---|---|---|
| Architecture | [Architecture docs](architecture/README.md) | Layering, backend selection, kernel dispatch, ordering, storage location, and heterogeneous execution design |
| Tensor | [Tensor docs](tensor/README.md) | Dense Tensor values and views, scalar policy, creation, reshape, materialization, and operation semantics |
| Linear algebra | [Linear algebra docs](linalg/README.md) | Mdspan lowering, BLAS/LAPACK adapters, dense operation dispatch, and MPLAPACK |
| Async and AD | [Async docs](async/README.md) | `Async<T>`, epochs, buffers, schedulers, exceptions, DAG diagnostics, Tensor kernels, and value-level reverse AD |
| Krylov | [Krylov docs](krylov/README.md) | Solvers, defaults, convergence, exponential estimators, precision validation, and fixtures |
| Symmetry | [Symmetry docs](symmetry/README.md) | Quantum numbers and future symmetry-aware block-sparse Tensor design |
| Tensor networks | [Tensor-network docs](tensor_network/README.md) | Sparse operators, models, finite MPS foundations, and TensorContraction integration findings |
| Backends | [Backend docs](backends/README.md) | CUDA and MPI architecture, platform constraints, runtime design, and background surveys |
| Diagnostics | [Diagnostics docs](diagnostics/README.md) | Presentation, display, trace diagnostics, logging plans, and Graphviz |
| Python | [Python docs](python/README.md) | Current smoke bindings and future Tensor dtype/presentation policy |
| Development | [Developer docs](development/README.md) | Testing, review, Doxygen, build information, and agent-assisted workflow |
| AI guidance | [AI guidance](ai_guidance/README.md) | Non-normative retrieval summaries for automated assistants |
| Design papers | [Long-form papers](latex/README.md) | Historical LaTeX sources and rendered background material |

## Current Vertical Slices

- [Tensor Operations](tensor/operations.md) and
  [Generated Tensors and Reshape](tensor/creation_and_reshape.md)
- [Kernel Dispatch](architecture/kernel_dispatch.md) and
  [Mdspan Linear Algebra Dispatch](linalg/mdspan_dispatch.md)
- [Async Tensor Kernel Authoring](async/kernel_authoring.md) and
  [Async Storage](async/storage.md)
- [Krylov Algorithms](krylov/algorithms.md) and
  [Precision Validation](krylov/precision_validation.md)
- [Presentation Formatting](diagnostics/presentation.md) and
  [Task Registry Diagnostics](async/task_registry_debug.md)

## Generated API Documentation

[Doxygen policy and build instructions](development/doxygen.md) describe the
generated API site. `mainpage.md`, `Doxyfile.in`, and `doxygen-custom.css` stay
at the documentation root because they are build inputs rather than subsystem
guides.
