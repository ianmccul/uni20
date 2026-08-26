# Tensor-Network Layer

This module owns tensor-network vocabulary and algorithms built from the
symmetry, dense linalg, and Krylov layers. It does not define a second tensor
container: MPS sites, MPO sites, local operators, and optimization centers are
aliases over `BlockTensor` with canonical morphism boundaries.

Current entry points:

- `site_types.hpp`: canonical `MpsSite`, `MpoSite`, `TwoSiteCenter`,
  `TwoSiteLocalOperator`, and `ScalarEnvironment` aliases.
- `two_site_effective_hamiltonian.hpp`: the first immediate-host output-first
  two-site apply object. It uses mapped repartition/permutation views and
  adjacent grouped BlockTensor contraction.

Chain ownership, MPO compilation, environment construction, and sweep policy
belong here when implemented. Symmetry selection and sparse worklist planning
remain in `symmetry/`; dense numerical kernels remain in `linalg/` and
`kernel/`; Krylov solvers continue to treat BlockTensor vectors as opaque.

The current two-site apply is a local-operator slice with scalar boundary
environments. It is not yet the general left-environment/MPO-pair/right-
environment effective Hamiltonian needed by a finite-chain sweep.
