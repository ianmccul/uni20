# Uni20 Documentation

This index separates current implementation guides from forward design and
background material. A document's own status statement is authoritative when
it is more specific than this index.

## Start Here

- [About Uni20](about.md): what works now, runnable vertical slices, and current
  boundaries.
- [Getting Started](getting_started.md): prerequisites, configuration, build,
  tests, benchmarks, and Python smoke bindings.
- [Architecture Overview](architecture_diagram.md): implemented layers and
  planned extensions.
- [Roadmap](roadmap.md): completed foundations, current priorities, and later
  work.
- [Contributor Guide](CONTRIBUTING.md): development workflow, evidence, and
  commit/review expectations.
- [Testing Infrastructure](testing.md): test modules, CMake options, and CTest
  usage.

## Document Status

- **Current guide:** describes implemented behavior and should change with the
  code.
- **Design note:** records intended or active design. It may contain interfaces
  that are not implemented.
- **Background:** preserves evidence, surveys, experiments, or historical
  reasoning; it does not define current behavior.

Plans that contain both implemented checkpoints and future work say so near
their title. For exact operation coverage, prefer current guides and tests over
schematic code in a design document.

## Current Implementation Guides

### Tensor, Scalars, and Presentation

- [Tensor Operations, Semantics, and Async Support](tensor_operations.md)
- [Generated Tensors and Reshape](tensor_creation_and_reshape.md)
- [Async Storage, Identity, and Assignment](async_storage.md)
- [Scalar Type Policy](scalar_policy.md)
- [Presentation Formatting](presentation.md)
- [Build Information](buildinfo.md)
- [Trace Macros](trace_macros.md)

### Async Runtime and AD

The [Async Documentation Index](async/README.md) is the primary entry point.
Its current guides include:

- [Async Getting Started](async/getting_started.md)
- [Runtime Model](async/runtime_model.md)
- [Buffers and Awaiters](async/buffers_and_awaiters.md)
- [Async Tensor Kernel Authoring](async/kernel_authoring.md)
- [Exceptions and Cancellation](async/exceptions_and_cancellation.md)
- [Schedulers](async/schedulers.md)
- [oneTBB Execution Primer](async/tbb_execution_primer.md)
- [Reverse-Mode AD](async/reverse_mode_ad.md)
- [Task Registry and DAG Diagnostics](async/task_registry_debug.md)
- [Async Cookbook](async/cookbook.md)
- [Async Quick Reference](async/quick_reference.md)

The [Coroutine Primer](async/coroutines_primer.md) supplies C++ background,
[DAG Debug Examples](async/dag_debug_examples.md) documents the runnable
diagnostic examples, and [Legacy Async Docs Audit](async/audit_legacy_docs.md)
records cleanup history.

### Dense Linear Algebra and Dispatch

- [Kernel Dispatch Design](kernel_dispatch.md): implemented dispatch contract
  and forward backend model.
- [Backend Dispatch Design](backend_dispatch.md): lower-level capability and
  fallback background.
- [Mdspan BLAS/LAPACK Wrapper Plan](mdspan_blas_lapack_wrapper_plan.md):
  implemented BLAS/initial LAPACK checkpoints plus remaining wrapper work.
- [Mdspan Linear Algebra Dispatch Plan](mdspan_linalg_dispatch_plan.md):
  implemented vertical slices and forward plan.
- [MPLAPACK Binary128 Setup](mplapack_binary128_setup.md)

### Krylov Algorithms

- [Krylov Algorithms](krylov_algorithms.md)
- [Krylov Solver Defaults](krylov_solver_defaults.md)
- [Krylov Precision Validation](krylov_precision_validation.md)
- [Krylov Test Matrices](krylov_test_matrices.md)
- [Krylov Exponential Error Bounds](krylov_exponential_error_bounds.md)
- [Krylov Exponential Estimators](krylov_exponential_estimators.md)
- [Real Nonsymmetric Arnoldi Policy for iDMRG](real_nonsymmetric_idmrg_policy.md)

### Symmetry and Existing Tensor-Network Foundations

- [Quantum Numbers and Symmetry](qnum.md)
- [Sparse Matrices](matrix.md)
- [Local Operators](operators.md)
- [Models](models.md)
- [Finite MPS Prototype](mps.md)

These facilities are not yet one complete symmetry-aware `BlockTensor` or DMRG
application path. The design documents below define the intended integration.

## Architecture and Active Design

### Execution and Lowering

- [Execution Architecture](execution_architecture.md)
- [Storage Memory Kind vs Location](storage_kind_and_location.md)
- [Ordering Ownership and Backend Lowering](ordering_and_backend_lowering.md)
- [Raw Primitives and Symmetric Lowering](raw_primitives_and_symmetric_lowering.md)
- [Trace as a Dense Reduction Primitive](trace_as_reduction_primitive.md)
- [Axis Labels, Contraction, and Braiding](axis_labels_and_braiding.md)

### Block-Sparse and Contraction Work

- [Block-Sparse Tensors and Layout](block_sparse_tensor.md)
- [BlockTensor Design](block_tensor.md)
- [Block Coalescing and GEMM Grouping](block_coalescing.md)
- [TensorContraction Integration Findings](tensorcontraction_integration_findings.md)
- [R/A/B/C Contraction Scheduling](rabc_contraction_scheduling.md)
- [R/A/B/C Lanczos Replay Fixtures](rabc_lanczos_fixtures.md)

### CUDA and Distributed Execution

- [GPU Landscape for Tensor Networks](gpu_landscape.md) (background survey)
- [CUDA Backend Library Compatibility](cuda_backend_libraries.md)
- [CUDA Runtime Resolution Strategy](cuda_runtime_resolution_strategy.md)
- [CUDA/cuSOLVER Architecture Notes](cuda_cusolver_architecture.md)
- [CUDA Memory Allocation](cuda_memory_allocation.md)
- [CUDA Runtime Design Notes](cuda_runtime_design_notes.md)
- [GPU Epoch Design Draft](gpu_epoch_design_draft.md)
- [MPI Persistent Object Store and Kernel Dispatch](uni20_mpi_persistent_dispatch_design.md)

### Background Drafts and Surveys

- [Tensor Dispatch and View Semantics Draft](tensor_dispatch_and_view_semantics_draft.md)
- [Async Tensor Lifetime and Dispatch Draft](async_tensor_lifetime_and_dispatch_draft.md)
- [Diagnostics and Logging Plan](diagnostics_logging_plan.md)
- [Screen Display Layer Plan](display_layer_plan.md)
- [Python Dtype Promotion Policy](python_dtype_promotion.md)
- [Tensor-Network Linear Algebra API Survey](tensor_network_linalg_survey.md)

These pages preserve design reasoning. Current tensor and async contracts live
in [Tensor Operations](tensor_operations.md), [Async Storage](async_storage.md),
and the [Async Documentation Index](async/README.md).

## Language Bindings and Developer Tools

- [Python Bindings](Python.md): current smoke bindings and future presentation
  policy.
- [Doxygen Documentation](Doxygen.md)
- [Graphviz Basics](graphviz.md)
- [Using Git and VS Code](使用Git與VSCode指南.md)

## Development Process

- [Agent-Assisted Development](agent_assisted_development.md)
- [Reviewing Uni20 Changes](code_review.md)

## AI Guidance

Files under `ai_guidance/` are non-normative retrieval summaries. Start with
the [AI Guidance Index](ai_guidance/README.md).

- [Remote ChatGPT Profile](ai_guidance/chatgpt.md)
- [Async Runtime Guidance](ai_guidance/async_runtime.md)
- [Reverse-Mode AD Guidance](ai_guidance/reverse_mode_ad.md)
- [Presentation and Python Guidance](ai_guidance/presentation_and_python.md)
- [Architecture Status](ai_guidance/architecture_status.md)
- [Tensor Dispatch Design](ai_guidance/tensor_dispatch_design.md)
- [CUDA Scheduler Notes](ai_guidance/cuda_scheduler_notes.md)
- [Glossary](ai_guidance/glossary.md)
