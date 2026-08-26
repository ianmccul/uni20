# First Pure-Uni20 Two-Site DMRG Slice

**Status:** implemented immediate-host U(1) integration checkpoint.

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

`TwoSiteEffectiveHamiltonian` compiles the general immediate-host operands:

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
Krylov application may not silently change vector spaces. Runtime application
groups terms by output block, uses beta zero then one, and may execute distinct
output groups through the storage-selected synchronous scheduler batch.

The current leaf plan is deliberately left-first. It allocates one rank-two
temporary for `A * B`, then accumulates the second contraction through ordinary
tensor dispatch. It does not yet compare left-first and right-first cost,
coalesce reusable intermediates, or retain device-resident scratch.

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
revision-based invalidation. Adjacent MPS sites can be replaced atomically while
changing only their shared internal bond. See
[Finite Chains And Environment Caches](finite_chains.md).

## Next Boundary

The next finite-DMRG layer must absorb selected SVD factors according to sweep
direction, construct the two replacement sites, and install them through the
adjacent-pair mutation boundary. A first sweep driver can then reuse the left
and right cache entries outside the active pair. Term and environment plans can
be optimized independently with reusable intermediates, placement, CUDA, and
MPI execution.
