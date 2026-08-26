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

`TwoSiteEffectiveHamiltonian` owns an immutable local Hamiltonian and is
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

## Next Boundary

This checkpoint has scalar boundary environments and one local two-site
operator. The next DMRG step is not another dense primitive. It is the general
effective-Hamiltonian planner over:

```text
left environment + two MPO sites + center + right environment
```

That planner must preserve logical block keys, accumulate all legal paths into
fixed output blocks, and lower each dense leaf through ordinary Uni20 dispatch.
After that, chain ownership, environment updates, directional SVD absorption,
and site replacement can form the first finite sweep.
