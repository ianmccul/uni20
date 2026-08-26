# Tensor-Network Layer

This module owns tensor-network vocabulary and algorithms built from the
symmetry, dense linalg, and Krylov layers. It does not define a second tensor
container: MPS sites, MPO sites, local operators, and optimization centers are
aliases over `BlockTensor` with canonical morphism boundaries.

Current entry points:

- `site_types.hpp`: canonical `MpsSite`, `MpoSite`, `MpoEnvironment`,
  `TwoSiteCenter`, `TwoSiteLocalOperator`, and `ScalarEnvironment` aliases.
- `environment.hpp`: identity boundary environments and immediate-host sparse
  left/right updates with distinct or shared bra/ket MPS sites.
- `finite_chain.hpp`: validated homogeneous finite MPS and MPO owners with
  revision-tracked site and adjacent-pair replacement.
- `environment_cache.hpp`: lazy or complete directional environment caches
  with exact revision-based invalidation.
- `two_site_split.hpp`: staged two-site block-SVD, directional singular-value
  absorption, canonical site materialization, and finite-MPS replacement.
- `two_site_dmrg.hpp`: converged local ground-state updates and directional
  finite-chain traversal with incremental environment refresh.
- `two_site_effective_hamiltonian.hpp`: immediate-host output-first local and
  MPO/environment two-site apply objects.

Concrete physical-model constructors live one layer above this module in
`models/`; they return these ordinary finite-chain owners rather than defining
parallel tensor-network containers.

MPO compilation and sweep policy belong here. Generic symmetry selection and
BlockTensor operations remain in `symmetry/`;
tensor-network connectivity is implemented here; dense numerical kernels remain
in `linalg/` and `kernel/`; Krylov solvers continue to treat BlockTensor vectors
as opaque.

The immediate-host `TwoSiteEffectiveHamiltonian` compiles environment and MPO
stored keys into a fixed-center R/A/B/C term plan. Its first execution policy
is left-first and allocates a dense matrix temporary per term. Reuse-aware
planning, sweep-level convergence and measurement, CUDA placement, and MPI
distribution remain separate extensions.
