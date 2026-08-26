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
- [MPO Environment Updates](environment_updates.md) specifies identity
  boundaries and the implemented left/right symmetry-preserving updates.
- [Finite Chains And Environment Caches](finite_chains.md) specifies validated
  MPS/MPO ownership, revision-tracked replacement, lazy directional cache
  construction, and exact invalidation ranges.
- [Directional Two-Site MPS Splitting](two_site_splitting.md) specifies staged
  state selection, sweep-direction singular-value absorption, canonical site
  materialization, and finite-chain replacement.
- [Directional Two-Site DMRG Sweeps](two_site_dmrg_sweeps.md) specifies local
  ground-state solving, mutation boundaries, traversal order, and incremental
  environment refresh.

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
integration branch. Pure Uni20 now implements immediate-host directional
environment updates, validated finite MPS/MPO owners, directional environment
caches, a fixed-center R/A/B/C term compiler, a matrix-free Krylov solve,
selected block-SVD pair replacement, and directional two-site traversal. The
remaining goal includes sweep-level convergence and measurement, resident CUDA
execution, MPI placement, and full benchmark integration without retaining the
external TensorContraction code lineage. The integration branch's measured
behavior feeds the
[architecture](../architecture/) and [symmetry](../symmetry/) designs and
supplies regression targets for that work.

Relevant current source foundations are the
[dense Tensor layer](../../src/uni20/tensor/),
[symmetry metadata](../../src/uni20/symmetry/), and
[low-level tensor kernels](../../src/uni20/kernel/).
