# First Pure-Uni20 Two-Site DMRG Slice

**Status:** implemented immediate-host and single-device CUDA-resident U(1) sweeps.

## Scope

The first pure-Uni20 DMRG-shaped path solves a length-two spin-half Heisenberg
problem in the total-charge-one sector. It combines existing general
components rather than introducing a dense DMRG-specific vector:

```text
TwoSiteCenter BlockTensor
    -> output-first local effective-Hamiltonian apply
    -> mapped physical-leg bends and grouped BlockTensor contraction
    -> BlockTensorMatrixFreeOps
    -> symmetric_lanczos_standard
    -> zero-copy 3/1 to 2/2 repartition
    -> block_svd decomposition
    -> state selection
    -> left / diagonal singular-value / right-adjoint materialization
```

Every Krylov vector and SVD factor retains its U(1) boundary spaces and stored
block keys. The path never projects the center or Hamiltonian into a dense
symmetry-free matrix.

## Canonical Types

`src/uni20/tensor_network/site_types.hpp` defines aliases over `BlockTensor`:

```cpp
MpsSite<Scalar, LeftBond, Physical, RightBond, Storage>
MpoSite<Scalar, LeftAuxiliary, InputPhysical,
        RightAuxiliary, OutputPhysical, Storage>
MpoEnvironment<Scalar, BraBond, Auxiliary, KetBond, Storage>
TwoSiteCenter<Scalar, LeftBond, LeftPhysical,
              RightPhysical, RightBond, Storage>
TwoSiteLocalOperator<Scalar, LeftPhysical, RightPhysical, Storage>
ScalarEnvironment<Scalar, Storage>
```

The two-site center uses:

```text
Domain<left bond, left physical, right physical>
    -> Codomain<right bond>
```

The local operator uses explicit dual physical factors on both sides. Its
input factors therefore compare exactly with physical factors bent from the
center codomain.

## Effective-Hamiltonian Apply

`LocalTwoSiteEffectiveHamiltonian` owns an immutable local Hamiltonian and is
callable as:

```cpp
operation(output, input);
```

It performs these structural operations:

1. Bend both center physical factors from domain to codomain.
2. Permute the bent factors into left-then-right physical order.
3. Call `contract_adjacent<2>` to contract both physical factors with the
   operator in one sparse worklist.
4. Permute and bend the output factors back to the canonical center boundary.
5. Copy into the fixed output structure required by Krylov.

Permutation and repartition are borrowed views. Only the owning grouped
contraction result allocates new blocks. The operation does not alter the input
or Hamiltonian and does not change the output's block structure.

## Verified Result

The integration test uses two legal center blocks, corresponding to
`|up,down>` and `|down,up>`. Lanczos finds the singlet energy `-3/4` and a
normalized antisymmetric vector. Repartitioning that vector to a matrix gives
two charge sectors with singular value `1/sqrt(2)` in each. Contracting the
materialized right-adjoint, diagonal singular-value, and left factors
reconstructs the original center to numerical tolerance.

## MPO And Environment Planner

`TwoSiteEffectiveHamiltonian` compiles the general operands:

```text
left environment + two MPO sites + center + right environment
```

The environment convention is:

```text
Domain<bra bond, MPO auxiliary> -> Codomain<ket bond>
key = (bra bond, MPO auxiliary, ket bond)
```

Together with the MPO key order `(left auxiliary, ket, right auxiliary,
bra)`, this lets construction join stored logical keys into terms:

```text
R(output center block) += alpha * A(left environment block)
                                * B(input center block)
                                * transpose(C(right environment block))
```

The prototype freezes the center boundary and stored-key pattern at
construction. A reachable output block omitted from that pattern is rejected:
Krylov application may not silently change vector spaces. Coordinate-indexed
joins snapshot the MPO coefficients into a canonical sparse
`f(r,a,b,c)` plan, coalescing duplicate coordinates.

`make_two_site_effective_hamiltonian(...)` constructs the operation used by the
DMRG sweep. It retains identity mapped views of the environments by value, so
compiling each bond copies descriptor metadata but not environment payload.
The environment-cache owners must outlive the local solve. MPO payloads are not
retained after their coefficients have been compiled into `f`.

MPO coefficient blocks remain immediately host-readable because construction
snapshots their scalar values into the sparse `f` plan. Center and environment
blocks need only model `BlockTensorView`; they may expose immediate mdspans or
descriptor-backed CUDA mdspecs.

The current backend is right-first. Effective-Hamiltonian construction
prepares each distinct `(B_b,C_c)` group, the output order, and reusable
intermediate storage before the local Krylov loop. Every application computes
each `B_b * transpose(C_c)` value once, then accumulates independent output
groups through ordinary dense contraction dispatch. The canonical `f` plan
remains neutral so future backends can choose left-first, right-first, or mixed
execution from the same hypergraph. The prepared intermediate leaf storage
matches the fixed center storage, so a packed CUDA center and CUDA environments
dispatch their dense contractions to cuBLAS without host materialization.
Multi-device placement, communication, and hybrid planning remain deferred.

## CUDA-Resident Sweep Checkpoint

`PackedCompleteBlockStorage<CudaStorage>` supplies descriptor-backed dense
blocks with stable offsets into one CUDA allocation. Each block is also an
independent logical CUDA buffer with its own completion ledger. The fixed
BlockTensor vector primitives dispatch fill, copy, scale, and AXPY through CUDA
kernels. Selecting `ParallelPackedCompleteBlockStorage<CudaStorage>` submits
independent blocks from the scheduler batch without giving up packed allocation.
Full block inner products and Euclidean norms use cuBLAS dot and norm routines;
only the scalar result is synchronized back to the host. The small projected
tridiagonal eigensystem remains a host LAPACK operation.

The CUDA regressions cover both a manually resident local eigensolver and
end-to-end resident bond and directional-sweep updates. The MPS and environment
cache use packed CUDA storage. Center contraction, the right-first R/A/B/C
workspace, fixed local Krylov solve, selected site factors, and completed-side
environment updates remain in that memory domain. MPO scalar coefficient tables
remain on the host and are no longer retained after plan construction. Boundary
and derived environments, widened local tensors, SVD decompositions, selected
factors, and installed sites preserve an explicit leaf allocation context. The
CUDA test exercises this on a non-default enrolled device when two devices are
available.

The native cuSOLVER Tensor SVD backend supports blocking real tall-matrix
factorizations. Its CUDA BlockTensor bridge assembles charge-sector matrices on
the device and launches independent sector factorizations from scheduler
participants. Wide sectors are transposed and their factors restored on-device.
U, S, and Vh remain resident; only compact singular-value arrays are copied to
the host for global selection. Selected U/Vh columns are gathered directly into
resident site storage, with the singular values applied by a CUDA transform on
the factor selected by the sweep direction.

The U(1) test factors the Heisenberg interaction into neutral `Sz` and two
charge-changing MPO channels. The compiled planner produces only the four terms
relevant to the fixed total-charge-one center and reproduces the same `-3/4`
Lanczos ground state as the local-operator path. A separate two-dimensional
bond test verifies the exact `A * B * transpose(C)` dense geometry.

## Environment Update Checkpoint

Identity boundaries plus left and right `MpoEnvironment` updates are now
implemented as symmetry-preserving sparse BlockTensor operations. Each update
accepts distinct bra and ket MPS sites, while a convenience overload uses the
same site for expectation values. See
[MPO Environment Updates](environment_updates.md) for the exact key joins and
dense formulas.

## Finite-Chain Cache Checkpoint

`FiniteMps`, `FiniteMpo`, and `MpoEnvironmentCache` now provide validated chain
ownership, complete or lazy left/right cache construction, and exact
revision-based invalidation. Adjacent MPS sites can be replaced together while
changing only their shared internal bond. See
[Finite Chains And Environment Caches](finite_chains.md).

## Directional Split Checkpoint

`decompose_two_site_center()` applies the staged block-SVD to the center cut.
After explicit state selection, `materialize_two_site_mps_split()` absorbs the
singular values into the site in the sweep direction and constructs a canonical
MPS pair. `replace_two_site_from_svd()` installs that pair through the finite
chain's revision-tracked replacement boundary while returning the diagonal
bond tensor and truncation statistics. See
[Directional Two-Site MPS Splitting](two_site_splitting.md).

## Directional Sweep Checkpoint

The directional sweep forms each center, solves the cached MPO effective
Hamiltonian in the selected local center storage, installs its selected split,
and refreshes the completed-side environment in the same leaf memory domain
before visiting the next bond. See
[Directional Two-Site DMRG Sweeps](two_site_dmrg_sweeps.md).

## Next Boundary

Alternating directional traversal and zero-discard terminal-energy convergence
are implemented for a host or single-device CUDA-resident MPS and environment
cache. MPO coefficients and the compact truncation spectrum remain host-side.
Post-truncation measurement, general initial-state canonicalization,
multi-device placement, asynchronous provider completion, and MPI execution
remain independent extensions.
