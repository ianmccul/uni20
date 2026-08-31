# Tensor-Network Documentation

This directory combines current tensor-network design constraints, the
pure-Uni20 finite BlockTensor DMRG path, and retained findings from the
historical `tensorcontraction-integration` branch. Host and single-device
resident-CUDA U(1) sweeps now use the current Tensor, symmetry, dispatch,
Async, and CUDA architecture. The reference branch remains useful for MPI and
later distributed-execution targets rather than defining current capability.

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
- [DMRG Performance Baselines](dmrg_performance_baselines.md) records the first
  CPU and resident-CUDA scaling measurements, external orientation points,
  benchmarking rules, block scheduling, provider-resource conclusions, and
  implemented per-sector SVD parallelism.
- [Spin-Half Model Builders](models.md) specifies the implemented U(1) local
  space, normalized Néel product MPS, and reduced-boundary open Heisenberg MPO.

## TensorContraction Integration Reference

- [Sparse Matrices](sparse_matrix.md), [Local Operators](operators.md), and
  [Finite MPS and DMRG](mps.md) document the functional branch implementation.
  They are capability and migration references, not a source inventory for
  `main`. The model guide above now distinguishes its implemented spin-half
  surface from later integration-branch model targets.
- [TensorContraction Integration Findings](contraction_integration_findings.md)
  records measured conclusions from the working CUDA/MPI integration.
- [R/A/B/C Contraction Scheduling](rabc_contraction_scheduling.md) develops the
  intended pure-Uni20 replacement for the effective-Hamiltonian scheduling and
  cost model.
- [R/A/B/C Lanczos Fixtures](rabc_lanczos_fixtures.md) records the capture and
  replay workflow used to preserve numerical and performance evidence.

The MPI R/A/B/C fixture executors remain runnable only on the integration
branch. Pure Uni20 now implements validated finite MPS/MPO owners, directional
environment caches, a sparse R/A/B/C compiler and dispatched host/CUDA
executor, matrix-free local Krylov solves, selected block-SVD pair replacement,
directional two-site traversal, alternating terminal-energy convergence, and
resident single-device CUDA storage throughout the working state. The
remaining program includes post-truncation measurements, general initial-state
canonicalization, broader models and symmetries, multi-GPU placement, MPI
distribution, and continued performance work without retaining the external
TensorContraction code lineage. The integration branch's measured behavior
continues to supply design constraints and regression targets.

Relevant current source foundations are the
[dense Tensor layer](../../src/uni20/tensor/),
[symmetry metadata](../../src/uni20/symmetry/), and
[low-level tensor kernels](../../src/uni20/kernel/). Current physical-model
construction lives in the [model layer](../../src/uni20/models/).
