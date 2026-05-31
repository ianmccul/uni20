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
- `run_two_site_dmrg` is the first front-end wrapper. It alternates
  left-to-right and right-to-left passes for a fixed number of full sweeps and
  returns the per-bond sweep diagnostics without adding persistence or global
  energy bookkeeping.
- `examples/spin_half_heisenberg_dmrg.cpp` is the current manual executable
  example. It checks a length-4 open spin-1/2 Heisenberg chain against an
  internal dense exact-diagonalization reference, then runs a length-20 dense
  placeholder-symmetry chain for several sweeps and reports the global MPS
  energy after each sweep.

This gives the DMRG prototype an in-memory two-site center vector that can be
passed to the temporary TensorContraction effective-Hamiltonian matvec boundary.

## Observed sweep behaviour

When comparing the length-20 open spin-1/2 Heisenberg example against an
already-converged MPTK two-site DMRG run at bond dimension 16, the lowest local
energy observed partway through a uni20 sweep can be slightly below the final
converged sweep energy before the state has settled.  This is not a violation
of the variational principle: each two-site local solve is variational in the
current mixed-canonical environment, while the subsequent SVD truncation and
canonical-center shift changes the state used by the next local problem.

In the observed run, continuing for a few more sweeps brought the reported
edge/global energy back into agreement with the MPTK converged value at roughly
roundoff scale.  For benchmarking, compare converged sweeps or matching sweep
positions rather than the minimum local energy seen during an unconverged sweep.

## Deferred

Wavefunction save/load is intentionally deferred.  If checkpointing becomes
necessary before the MPTK front-end merge, prefer a simple neutral dump format
that MPTK can import with a separate conversion utility, rather than making
uni20 depend on MPTK persistence details.
