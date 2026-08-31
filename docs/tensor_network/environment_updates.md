# MPO Environment Updates

**Status:** implemented immediate-host BlockTensor reference path.

An MPO environment is a symmetry-preserving three-leg tensor:

```cpp
MpoEnvironment<Scalar, BraBond, Auxiliary, KetBond, Storage>
```

with boundary and key order:

```text
Domain<bra bond, MPO auxiliary> -> Codomain<ket bond>
key = (bra bond, MPO auxiliary, ket bond)
```

Its stored dense block is a matrix whose axes are `(bra bond multiplicity,
ket bond multiplicity)`. A legal but unstored block is zero. Environment
updates join stored MPS, MPO, and environment keys and allocate only the
reachable result blocks.

## Identity Boundaries

`make_identity_mpo_environment<Scalar>(bond, auxiliary, index)` constructs one
identity matrix block for every sector of `bond`. The selected auxiliary state
must carry the identity charge. This selection is explicit because the left and
right boundary states of a triangular MPO need not have the same auxiliary
index.

The first implementation accepts `BlockSpace` bonds, a `LocalSpace` auxiliary,
and immediate sparse host output storage. `MpoEnvironmentCache` selects the two
finite-chain boundary states explicitly and calls this primitive.

## Directional Updates

An MPS site uses:

```text
Domain<left bond, physical> -> Codomain<right bond>
key = (left bond, physical, right bond)
```

An MPO site uses:

```text
Domain<left auxiliary, ket physical>
  -> Codomain<right auxiliary, bra physical>
key = (left auxiliary, ket, right auxiliary, bra)
```

The primary update overloads accept distinct bra and ket sites. The
three-argument convenience overload uses one site for both and therefore
constructs the ordinary expectation-value environment.

For dense blocks `A` (bra site), `B` (ket site), and `E` (incoming
environment), the left update evaluates:

```text
E_right += w * conj(A)^T * E_left * B
```

The right update evaluates:

```text
E_left += w * conj(A) * E_right * B^T
```

Here `w` is one scalar stored MPO block. Transpose notation describes the
contracted dense axes; conjugation is supplied by Uni20's lazy accessor, so
complex bra semantics are not bypassed by a raw provider call.

## Sparse Planning

The left planner joins stored keys as follows:

| Operand | Required coordinates |
| --- | --- |
| Incoming environment | `(bra-left, aux-left, ket-left)` |
| Bra site | `(bra-left, bra-physical, bra-right)` |
| MPO site | `(aux-left, ket-physical, aux-right, bra-physical)` |
| Ket site | `(ket-left, ket-physical, ket-right)` |
| Output environment | `(bra-right, aux-right, ket-right)` |

The right planner performs the reverse join, matching the incoming environment
to the sites' right bonds and the MPO's right auxiliary, and produces
`(bra-left, aux-left, ket-left)`.

Only stored input combinations participate. Reachable output keys are sorted
and deduplicated before the result is allocated. Terms are then grouped by
output block. The first contribution overwrites with beta zero; later
contributions accumulate with beta one. Distinct output blocks may execute as
synchronous lightweight scheduler-batch items when selected by the output
storage policy.

The current planner uses a direct stored-block join and allocates one rank-two
dense temporary per contribution. This is the correctness reference. Indexed
joins, intermediate reuse, alternative multiplication order, async block
epochs, CUDA placement, and MPI-distributed environments remain later
execution-policy work.

Finite-chain ownership, lazy or complete directional construction, and
revision-based invalidation are specified in
[Finite Chains And Environment Caches](finite_chains.md).

## Validation

The update requires exact equality of every connected space, including labels,
and one shared symmetry and scalar type. Tests cover:

- identity boundaries with multiple U(1) bond sectors;
- left and right updates of a two-site U(1) Heisenberg product state;
- distinct complex bra and ket sites, including bra conjugation;
- multiple physical paths accumulating into one output block;
- nontrivial dense bond dimensions in both directions.
