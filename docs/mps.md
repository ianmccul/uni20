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
  plus their left/right environments into a matrix-free TensorContraction
  `EffectiveHamiltonianOperator`.  The two-site center is represented as one
  block per local two-site physical basis state, and TensorContraction applies
  each term as `left_environment * center_block * right_environment` rather
  than materializing the full effective Hamiltonian.
- `solve_two_site` packs the current two-site MPS center, compiles the local
  effective Hamiltonian, runs the TensorContraction Lanczos wrapper, and keeps
  the optimized center resident when the CUDA backend is active.  The old dense
  host matrix can still be materialized explicitly for fallback paths and tests.
- `split_two_site_solution` runs the current single-block SVD split and absorbs
  singular values into the right tensor for a left-to-right move, or into the
  left tensor for a right-to-left move.  On CUDA builds it first tries the
  resident cuSOLVER split path, which packs the two-site vector blocks on the
  GPU and avoids copying the Lanczos vector to the host. `FiniteMPS::replace_adjacent`
  installs the resulting pair back into the in-memory chain.
- `sweep_two_site_left_to_right` and `sweep_two_site_right_to_left` perform the
  first directional dense DMRG sweep pass by rebuilding CPU environment chains,
  solving each two-site problem, splitting the optimized center, replacing the
  MPS tensors, and updating the environment on the swept side.
- `run_two_site_dmrg` is the first front-end wrapper. It alternates
  left-to-right and right-to-left passes for a fixed number of full sweeps and
  returns the per-bond sweep diagnostics without adding persistence or global
  energy bookkeeping.
- `examples/spin_half_heisenberg_dmrg.cpp` is the shared source for the manual
  Heisenberg DMRG executables. `spin_half_heisenberg_dmrg` uses the dense
  placeholder symmetry where both local states carry the identity charge.
  `spin_half_heisenberg_u1_dmrg` uses the U(1) spin labels from
  `make_spin_half_u1_site()`. Both executables check a length-4 open spin-1/2
  Heisenberg chain against an internal dense exact-diagonalization reference,
  then run a configurable chain for several sweeps and report the edge local
  solve energy after each sweep.  The U(1) executable currently validates the
  symmetry-labelled model path; the MPS storage and SVD split still use dense
  single-sector bond spaces until the block-sparse MPS path is implemented.
  The long-chain examples accept `UNI20_HEISENBERG_LENGTH`,
  `UNI20_HEISENBERG_SWEEPS`, and `UNI20_HEISENBERG_MAX_RANK`.

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

In the observed length-20, bond-dimension-16 run, continuing for a few more
sweeps brought the reported edge/global energy back into agreement with the MPTK
converged final-sweep value at roughly roundoff scale: uni20 reported
`-8.682468365409264`, while MPTK reported `-8.682468365409258` at the same
sweep-end position.  A lower value, `-8.682468366629129`, was observed as a
mid-sweep local variational minimum in MPTK.  That number is useful for tracing
the sweep trajectory, but it is not the same benchmark point as the canonical
state at the end of the sweep.

For benchmarking, compare converged sweeps or matching sweep positions rather
than the minimum local energy seen during an unconverged sweep.

## Deferred

Wavefunction save/load is intentionally deferred.  If checkpointing becomes
necessary before the MPTK front-end merge, prefer a simple neutral dump format
that MPTK can import with a separate conversion utility, rather than making
uni20 depend on MPTK persistence details.
