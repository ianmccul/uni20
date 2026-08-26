# Tensor-Network Documentation

This directory combines current tensor-network design constraints, the first
pure-Uni20 BlockTensor DMRG-shaped path, and the functional reference
implementation on the historical `tensorcontraction-integration` branch. The
reference branch still defines later finite-sweep, CUDA, MPI, and performance
targets; new implementation work uses the current Tensor, symmetry, dispatch,
Async, CUDA, and distributed-execution architecture.

## Current Main-Branch Foundations

- [Symmetry](../symmetry/) documents the quantum-number and block-space
  foundations that are present on `main`.
- [Tensor Operations](../tensor/operations.md) and
  [Tensor-Network Linear Algebra API Survey](../linalg/tensor_network_api_survey.md)
  document the dense kernels used below BlockTensor operations.
- [First Pure-Uni20 Two-Site DMRG Slice](two_site_dmrg_vertical_slice.md)
  documents the implemented U(1) local and MPO/environment effective-
  Hamiltonian, Lanczos, and staged block-SVD integration checkpoints.

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

The complete R/A/B/C and DMRG executables remain runnable only on the
integration branch. Pure Uni20 now implements a fixed-center immediate-host
R/A/B/C term compiler and matrix-free Krylov checkpoint, but not yet finite-
chain environment updates or sweeps. The remaining goal includes resident CUDA
execution, MPI placement, and full sweep integration without retaining the
external TensorContraction code lineage. The integration branch's measured
behavior feeds the [architecture](../architecture/) and [symmetry](../symmetry/)
designs and supplies regression targets for that work.

Relevant current source foundations are the
[dense Tensor layer](../../src/uni20/tensor/),
[symmetry metadata](../../src/uni20/symmetry/), and
[low-level tensor kernels](../../src/uni20/kernel/).
