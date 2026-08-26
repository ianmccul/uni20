# Tensor-Network Layer

This module owns tensor-network vocabulary and algorithms built from the
symmetry, dense linalg, and Krylov layers. It does not define a second tensor
container: MPS sites, MPO sites, local operators, and optimization centers are
aliases over `BlockTensor` with canonical morphism boundaries.

Current entry points:

- `site_types.hpp`: canonical `MpsSite`, `MpoSite`, `MpoEnvironment`,
  `TwoSiteCenter`, `TwoSiteLocalOperator`, and `ScalarEnvironment` aliases.
- `two_site_effective_hamiltonian.hpp`: immediate-host output-first local and
  MPO/environment two-site apply objects.

Chain ownership, MPO compilation, environment construction, and sweep policy
belong here when implemented. Symmetry selection and sparse worklist planning
remain in `symmetry/`; dense numerical kernels remain in `linalg/` and
`kernel/`; Krylov solvers continue to treat BlockTensor vectors as opaque.

The immediate-host `TwoSiteEffectiveHamiltonian` compiles environment and MPO
stored keys into a fixed-center R/A/B/C term plan. Its first execution policy
is left-first and allocates a dense matrix temporary per term. Reuse-aware
planning, chain/environment construction, CUDA placement, and MPI distribution
remain separate extensions.
