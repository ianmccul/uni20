# Model Layer

This module constructs physical model data over the symmetry-aware
tensor-network layer. Model builders return ordinary `FiniteMps` and
`FiniteMpo` owners; they do not define alternate tensor or dispatch types.

Current entry point:

- `spin_half_heisenberg.hpp`: the U(1) spin-half local space, normalized Néel
  product MPS, and reduced-boundary open Heisenberg MPO.

The Néel builder produces a rank-one MPS that is canonical in both directions,
so it can directly initialize the alternating two-site DMRG run controller in
`tensor_network/two_site_dmrg.hpp`.

Generic spaces and block-sparse operations remain in `symmetry/`. Finite-chain
ownership, environments, splitting, and sweeps remain in `tensor_network/`.
The canonical model contract is documented in
[`docs/tensor_network/models.md`](../../../docs/tensor_network/models.md).
