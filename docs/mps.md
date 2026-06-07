# Finite MPS Prototype

Uni20 now has a minimal in-memory finite MPS layer for the first DMRG prototype.
It is deliberately not a persistence format and does not try to copy MPTK's
wavefunction storage infrastructure.

## Scope

- `MpsSiteTensor` stores one site as dense row-major matrices, one matrix per
  physical state.
- `FiniteMPS` stores a finite sequence of site tensors and validates adjacent
  bond-space compatibility.
- `ThreeLegBlockMatrix` is the first generic block-sparse storage primitive for
  tensors with two `BlockSpace` legs and one `LocalSpace` leg. It stores dense
  row-major blocks keyed by `(row sector, local state, column sector)` and
  enforces the zero-flux rule `q_column = q_row + q_local`. This covers both
  U(1) MPS site tensors and MPS-like environment tensors.
- `SparseMpoSite` compiles an `OperatorComponent` into scalar four-leg entries
  keyed by `(left virtual, bra physical, ket physical, right virtual)`. It
  validates both the local-operator transform rule `q_bra = q_ket + q_operator`
  and the MPO zero-flux rule `q_left_virtual + q_ket = q_right_virtual + q_bra`.
- `BlockSparseFiniteMPS` and `make_block_sparse_product_state()` provide the
  first U(1)-aware product-state representation. The helper builds cumulative
  one-dimensional bond sectors, so an alternating spin-half product state
  carries the expected running `S^z` charge on each bond.
- `BlockSparseMpoChain` stores compiled `SparseMpoSite` objects and validates
  adjacent MPO virtual spaces. It is the strict U(1) MPO representation used by
  the block-sparse Heisenberg front-end.
- `BlockSparseEnvironment` is a `ThreeLegBlockMatrix` with convention
  `(bra bond, MPO virtual, ket bond)`. The block-sparse left/right environment
  updates iterate only legal MPS blocks and sparse MPO scalar entries.
- `BlockSparseTwoSiteLayout` represents a two-site center as legal dense blocks
  keyed by `(left bond sector, fused physical-pair state, right bond sector)`.
  The fused physical-pair space carries charge `q_left_physical +
  q_right_physical`, so Lanczos vectors remain block sparse.
- `make_two_site_svd_sectors()` builds the fused row/column sectors needed for
  a block SVD of a two-site center. Rows are `(left bond, left physical)` and
  columns are `(right physical, right bond)`, grouped by the shared-bond charge.
- The strict block-sparse `solve_two_site`, `split_two_site_center`,
  directional sweep helpers, and `run_two_site_dmrg` overloads operate on
  `BlockSparseFiniteMPS` and `BlockSparseMpoChain`. They use
  TensorContraction `MatrixFamily` as the dense-block container, but every block
  carries a U(1) logical key and the code never constructs the no-symmetry dense
  two-site center.
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
  host matrix can still be materialized explicitly only in the dense prototype
  or in terminal diagnostic/reference helpers.
- The strict U(1) `solve_two_site` path is resident CUDA/MPI-only. It builds a
  TensorContraction `EffectiveHamiltonianOperator` from the legal U(1)
  environment, MPO, input, and output block keys, then calls
  `lanczos_lowest_with_engine` with `VectorAlgebraEngine`. Setting
  `UNI20_TENSORCONTRACTION_BACKEND=host` is a test/debug mode for reference
  helpers, not a valid U(1) DMRG solve backend.
- `split_two_site_solution` runs the current single-block SVD split and absorbs
  singular values into the right tensor for a left-to-right move, or into the
  left tensor for a right-to-left move.  On CUDA builds it first tries the
  resident cuSOLVER split path, which packs the two-site vector blocks on the
  GPU and avoids copying the Lanczos vector to the host. `FiniteMPS::replace_adjacent`
  installs the resulting pair back into the in-memory chain.
- The strict U(1) split currently synchronizes the optimized center at the
  replacement boundary, assembles one host-visible dense matrix per charge
  sector, and requires the device cuSOLVER SVD path for each sector. It does not
  fall back to LAPACK or the reference SVD. The remaining host materialization is
  for writing the current host-owned `BlockSparseFiniteMPS` replacement tensors;
  the linear algebra kernel is still cuSOLVER.
- `sweep_two_site_left_to_right` and `sweep_two_site_right_to_left` perform the
  first directional dense DMRG sweep pass by rebuilding CPU environment chains,
  solving each two-site problem, splitting the optimized center, replacing the
  MPS tensors, and updating the environment on the swept side.
- `run_two_site_dmrg` is the first front-end wrapper. It alternates
  left-to-right and right-to-left passes for a fixed number of full sweeps and
  returns the per-bond sweep diagnostics without adding persistence or global
  energy bookkeeping.
- `examples/spin_half_heisenberg_dmrg.cpp` is the dense/no-symmetry manual
  Heisenberg DMRG executable. It uses the placeholder symmetry where both local
  states carry the identity charge.
- `examples/spin_half_heisenberg_u1_dmrg.cpp` is the strict U(1) block-sparse
  Heisenberg executable. It builds `BlockSparseFiniteMPS` product states,
  compiles a `BlockSparseMpoChain`, checks a length-4 open spin-1/2 chain
  against the known exact energy, then runs a configurable chain for several
  sweeps and reports the edge local solve energy after each sweep.
  The long-chain examples accept `UNI20_HEISENBERG_LENGTH`,
  `UNI20_HEISENBERG_SWEEPS`, and `UNI20_HEISENBERG_MAX_RANK`.
- `examples/fermi_hubbard_u1u1_dmrg.cpp` is the strict U(1)xU(1) block-sparse
  Fermi-Hubbard executable. It starts from an alternating half-filled product
  state, compiles the nearest-neighbor Hubbard MPO, and emits the same
  `MP_BENCHFILE` timing columns as the U(1) Heisenberg executable. The example
  accepts `UNI20_HUBBARD_LENGTH`, `UNI20_HUBBARD_SWEEPS`,
  `UNI20_HUBBARD_MAX_RANK`, `UNI20_HUBBARD_T`, and `UNI20_HUBBARD_U`.
  By default it treats the edge local solve energy as a benchmark diagnostic;
  set `UNI20_HUBBARD_CHECK_GLOBAL_ENERGY=1` for the slower global-energy
  monotonicity check.
  The block-sparse examples include `RabcOutputBlocks` and `RabcOutputShape` in
  `MP_BENCHFILE`; these identify the effective-Hamiltonian output shape used by
  TensorContraction placement and coefficient-bundle guards.

This gives the DMRG prototype an in-memory two-site center vector that can be
passed to the temporary TensorContraction effective-Hamiltonian matvec boundary.
The U(1) local solve now generates CUDA/MPI TensorContraction worklists from the
legal block layouts.  Environment construction and final MPS tensor replacement
remain host-owned prototype boundaries; those are explicit storage boundaries,
not silent dense fallbacks.

## Symmetry Invariants

Quantum-number metadata is part of the tensor state, not auxiliary annotation.
The U(1) DMRG path must keep `LocalSpace`, `BlockSpace`, `QNum`, and leg
orientation information attached through MPS/MPO construction, environment
updates, two-site solves, SVD splits, TensorContraction worklist generation, and
eventual CUDA/MPI placement.

Implicit dense materialization is not allowed in symmetry-aware code because it
erases the block coordinates that define the legal tensor state. There is no
dense fallback for a U(1)-typed path. A dense calculation is a distinct
no-symmetry path, for example the dense Heisenberg executable whose local states
both carry the identity charge, or an explicit conversion that changes the
symmetry group and exits the U(1) execution path.

Dense debug/reference projections may exist only as terminal diagnostics. They
must be explicitly named, documented as leaving the symmetry-aware path, and must
not feed back into U(1) MPS/MPO/DMRG state. If an operation cannot yet preserve
block structure, the U(1) path should reject it rather than silently using a
dense implementation.

The current selection-rule conventions are:

- `ThreeLegBlockMatrix`: `q_column = q_row + q_local`.
- `LocalOperator` coefficients: `q_bra = q_ket + q_operator`.
- `SparseMpoSite`: `q_left_virtual + q_ket = q_right_virtual + q_bra`.

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
