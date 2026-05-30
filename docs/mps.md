# Finite MPS Prototype

Uni20 now has a minimal in-memory finite MPS layer for the first DMRG prototype.
It is deliberately not a persistence format and does not try to copy MPTK's
wavefunction storage infrastructure.

## Scope

- `MpsSiteTensor` stores one site as dense row-major matrices, one matrix per
  physical state.
- `FiniteMPS` stores a finite sequence of site tensors and validates adjacent
  bond-space compatibility.
- `TwoSiteWavefunction` packs two adjacent sites into one TensorContraction
  `MatrixFamily` block with rows `(left bond, left physical)` and columns
  `(right physical, right bond)`.
- `MpoEnvironment` builds first-pass left/right environments as one dense
  bond-bond matrix per MPO virtual index, while iterating sparse
  `OperatorComponent` and `LocalOperator` entries over the explicit local
  physical space.
- `make_two_site_effective_hamiltonian` compiles two adjacent MPO components
  plus their left/right environments into a vectorized single-block
  TensorContraction `EffectiveHamiltonianOperator`.
- `solve_two_site` packs the current two-site MPS center, compiles the local
  effective Hamiltonian, runs the TensorContraction Lanczos wrapper, and returns
  both vectorized and matrix-shaped optimized center data.
- `split_two_site_solution` runs the current single-block SVD split and absorbs
  singular values into the right tensor for a left-to-right move, or into the
  left tensor for a right-to-left move. `FiniteMPS::replace_adjacent` installs
  the resulting pair back into the in-memory chain.
- `sweep_two_site_left_to_right` and `sweep_two_site_right_to_left` perform the
  first directional dense DMRG sweep pass by rebuilding CPU environment chains,
  solving each two-site problem, splitting the optimized center, replacing the
  MPS tensors, and updating the environment on the swept side.

This gives the DMRG prototype an in-memory two-site center vector that can be
passed to the temporary TensorContraction effective-Hamiltonian matvec boundary.

## Deferred

Wavefunction save/load is intentionally deferred.  If checkpointing becomes
necessary before the MPTK front-end merge, prefer a simple neutral dump format
that MPTK can import with a separate conversion utility, rather than making
uni20 depend on MPTK persistence details.
