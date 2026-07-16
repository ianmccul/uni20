# Tensor-Network Documentation

This directory combines current tensor-network design constraints with the
functional reference implementation on the `tensorcontraction-integration`
branch. Its MPS, MPO, DMRG, CUDA, MPI, model, and sparse-matrix code is not
present on `main`, but it is not obsolete: it defines behavior, benchmarks, and
capabilities that Uni20 intends to reproduce through its current Tensor,
symmetry, dispatch, Async, CUDA, and distributed-execution architecture.

## Current Main-Branch Foundations

- [Symmetry](../symmetry/) documents the quantum-number and block-space
  foundations that are present on `main`.
- [Tensor Operations](../tensor/operations.md) and
  [Tensor-Network Linear Algebra API Survey](../linalg/tensor_network_api_survey.md)
  document the dense kernels, including truncating SVD, needed by the first
  pure-Uni20 dense CPU DMRG path.

## TensorContraction Integration Reference

- [Sparse Matrices](sparse_matrix.md), [Local Operators](operators.md),
  [Models](models.md), and [Finite MPS and DMRG](mps.md) document the functional
  branch implementation. They are capability and migration references, not a
  source inventory for `main`.
- [TensorContraction Integration Findings](contraction_integration_findings.md)
  records measured conclusions from the working CUDA/MPI integration.
- [R/A/B/C Contraction Scheduling](rabc_contraction_scheduling.md) develops the
  intended pure-Uni20 replacement for the effective-Hamiltonian scheduling and
  cost model.
- [R/A/B/C Lanczos Fixtures](rabc_lanczos_fixtures.md) records the capture and
  replay workflow used to preserve numerical and performance evidence.

The R/A/B/C and DMRG executables are runnable on the integration branch, not on
the current `main` branch. The goal is to implement their dense and
symmetry-aware functionality in pure Uni20, including U(1) block structure,
resident CUDA execution, MPI placement, matrix-free Krylov solves, and
truncating SVD, without retaining the external TensorContraction code lineage.
Their measured behavior feeds the [architecture](../architecture/) and
[symmetry](../symmetry/) designs and supplies regression targets for that work.

Relevant current source foundations are the
[dense Tensor layer](../../src/uni20/tensor/),
[symmetry metadata](../../src/uni20/symmetry/), and
[low-level tensor kernels](../../src/uni20/kernel/).
