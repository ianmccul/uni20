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

This gives the DMRG prototype an in-memory two-site center vector that can be
passed to the temporary TensorContraction effective-Hamiltonian matvec boundary.

## Deferred

Wavefunction save/load is intentionally deferred.  If checkpointing becomes
necessary before the MPTK front-end merge, prefer a simple neutral dump format
that MPTK can import with a separate conversion utility, rather than making
uni20 depend on MPTK persistence details.
