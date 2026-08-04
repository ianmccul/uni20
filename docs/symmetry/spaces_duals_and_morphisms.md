# Spaces, Duals, and Tensor Morphisms

**Status:** canonical design for the future symmetry-aware `BlockTensor`;
not current implemented API behavior.

This note defines how Uni20 represents the mathematical boundary of a
symmetry-aware tensor. It refines the categorical model in
[BlockTensor](block_tensor.md) and
[Axis Labels, Contraction, and Braiding](axis_labels_and_braiding.md). The
long-form companion paper,
[Spaces, Duals, Morphisms, and BlockTensor](../latex/block-tensor-spaces-and-morphisms.tex),
builds on the earlier planar-network design paper.

The central decision is:

> `Space` versus `DualSpace` and `Domain` versus `Codomain` are independent
> distinctions.

`CoSpace` and the proposed `CoBlockSpace`/`CoLocalSpace`/`CoQNumSpace` family
conflated these distinctions. They are not part of the intended design.

## 1. Scope

This model applies to symmetry-aware tensor-network objects such as
`BlockTensor`. An ordinary dense `Tensor` remains an array object and does not
need identity-bearing spaces or a categorical boundary merely to store and
operate on dense data.

The full model is used for bosonic abelian symmetries as well as non-abelian,
fermionic, and braided categories. The U(1) implementation is a cheap
specialization of the same contract, not a separate algebra that must later be
replaced.

Four structures must remain distinct:

| Structure | Meaning | Example |
|---|---|---|
| Symmetry or category | Fusion, duality, and eventually braid/recoupling rules | bosonic U(1) |
| Space | An immutable identity-bearing object with a sector decomposition | the MPS bond at cut 7 |
| Object duality | `Space<S>` versus `DualSpace<S>` | a bond object and its categorical dual |
| Morphism side | membership in `Domain<...>` or `Codomain<...>` | input and output boundary factors |

A fifth structure, the **leg occurrence**, identifies one ordered use of a
space in a particular tensor boundary. Multiple occurrences may refer to the
same space. Display labels belong to spaces or occurrences for diagnostics;
they do not define identity or contraction semantics.

## 2. Structural Decompositions and Semantic Spaces

The implemented `BlockSpace`, `QNumList`, and `QNum` types describe sector
structure. They do not yet provide the semantic identity required by the tensor
network model.

The future space layer should be an immutable identity-bearing value whose
decomposition may be one of:

- a coalesced collection of `(QNum, degeneracy)` sectors, currently represented
  by `BlockSpace`;
- an ordered list of explicit local-state charges, currently represented by
  `QNumList`;
- one irrep label, currently represented by `QNum`;
- a symmetry-free dense extent.

The exact C++ representation remains open. A useful conceptual spelling is:

```cpp
Space<BlockSpace>
Space<LocalSpace>
Space<QNumSpace>
Space<DenseSpace>
```

The durable requirements are more important than these type names:

- Copying a space preserves its `SpaceId`.
- Constructing an isomorphic space does not silently reuse an existing
  `SpaceId`.
- `same_space(a, b)` tests semantic identity.
- `isomorphic_space(a, b)` tests compatible structure.
- Contracting different but isomorphic spaces requires an explicit
  isomorphism or an operation whose contract introduces one.
- A label is diagnostic metadata and changing it cannot change either result.

`SpaceId` cannot be a process-local address. Distributed construction,
truncation, and decomposition must produce identities that all participating
MPI ranks can reproduce or serialize consistently.

This distinction catches errors that dimensions and charges alone cannot. Two
MPS bonds may both contain the same U(1) sectors with the same degeneracies but
belong to different cuts; they are isomorphic, not interchangeable.

## 3. Object Duality and Morphism Side

A symmetry-aware tensor is a morphism

```text
A : X_0 tensor ... tensor X_(m-1)
      ->
    Y_0 tensor ... tensor Y_(n-1)
```

with an ordered domain and ordered codomain. Conceptually:

```cpp
using ASpec = MorphismSpec<
    Domain<X0, ..., Xm>,
    Codomain<Y0, ..., Yn>>;
```

Each `Xi` or `Yi` may independently be a `Space<S>` or `DualSpace<S>`.
`Domain` and `Codomain` say which side of the map contains an object;
`DualSpace` says which categorical object it is.

`DualSpace` retains the identity of its primal space. It is not a newly
allocated space with coincidentally negated charges:

```text
dual(dual(X)) = X
primal_space_id(dual(X)) = space_id(X)
```

The category owns how sectors transform under duality. For bosonic U(1),
`dual(q) = -q`. A general category may also require nontrivial evaluation,
coevaluation, pivotal, or normalization data.

### All-out Boundary Form

Selection rules are easiest to state after orienting every boundary leg
outward:

- `Codomain<X>` contributes `X`;
- `Domain<X>` contributes `DualSpace<X>`.

If `X` is already a `DualSpace`, duality is involutive. Equivalently, effective
outward duality is the exclusive-or of morphism side and the explicit dual
flag.

Dualizing a tensor product reverses its order. One canonical all-out ordering
for `A : X_0 tensor ... tensor X_(m-1) -> Y_0 tensor ... tensor Y_(n-1)` is:

```text
DualSpace<X_(m-1)> tensor ... tensor DualSpace<X_0>
    tensor Y_0 tensor ... tensor Y_(n-1).
```

The implementation may choose an equivalent cyclic starting point, but it must
document one convention and use it consistently for block keys, fusion trees,
and contraction plans.

For bosonic U(1), a block is legal when

```text
sum(codomain charges) - sum(domain charges) = 0.
```

For an MPS site represented as

```text
A : left_bond tensor physical -> right_bond
```

this gives the familiar rule

```text
q_right = q_left + q_physical.
```

The generic implementation must not hardcode charge addition. It asks the
category for invariant channels of the oriented sectors:

```cpp
auto channels = symmetry.invariant_channels(oriented_sectors);
```

For multiplicity-free U(1), the result is empty or contains one trivial
channel. Non-abelian implementations may return several channels and a
non-empty `CouplingDescriptor`.

## 4. Repartition Is Wire Bending

Moving a leg across the domain/codomain boundary is not a metadata edit. It is
an explicit categorical operation, provisionally named `repartition`, that
uses the category's evaluation or coevaluation maps.

The canonical rigid-category isomorphisms include

```text
Hom(A tensor X, B)  ~= Hom(A, B tensor DualSpace<X>)
Hom(A, B tensor X)  ~= Hom(A tensor DualSpace<X>, B).
```

Therefore moving a leg across the boundary toggles whether that factor is a
`Space` or `DualSpace`. This preserves the all-out boundary object; merely
changing `Domain` to `Codomain` would not.

Moving several adjacent factors may reverse their order:

```text
DualSpace<X tensor Y> ~= DualSpace<Y> tensor DualSpace<X>.
```

The operation must consequently specify both the new boundary split and the
resulting order. In a bosonic U(1) specialization, bending is normally only a
checked metadata and axis-order operation. Other categories may attach pivotal
maps, twists, phases, or recoupling transformations.

`repartition` is distinct from:

- `relabel`, which changes diagnostics only;
- `conj`, which changes numerical value semantics but does not move legs;
- `adjoint`, which reverses the morphism and applies the category's adjoint
  rules;
- `braid`, which exchanges strands;
- an arbitrary axis permutation, which may not be legal without recoupling or
  braiding.

## 5. Composition and Partial Contraction

Full contraction is morphism composition. If

```text
A : X -> Y
B : Y -> Z
```

then

```text
compose(B, A) : X -> Z.
```

Partial contraction is tensor product with identities followed by composition.
For example, let

```text
A : L -> M tensor X
B : X tensor R -> N.
```

Then the contraction over `X` is

```text
A tensor id_R : L tensor R -> M tensor X tensor R
id_M tensor B : M tensor X tensor R -> M tensor N

compose(id_M tensor B, A tensor id_R)
    : L tensor R -> M tensor N.
```

The implementation does not materialize identity tensors. A checked
contraction planner proves this composition, preserves the external boundary,
and lowers it directly to block worklists and dense kernels.

More general networks use the same primitives:

- associating and recoupling tensor products;
- explicit `repartition` operations;
- permutations in a symmetric category or explicit braids otherwise;
- tensor products with implicit identities;
- composition;
- categorical trace for closed boundaries.

A planar network can be sliced into tensor-product layers followed by
composition. The global planner owns the embedding and crossing decisions;
one `BlockTensor` stores only its local ordered domain/codomain boundary.

## 6. Bosonic Abelian Specialization

Bosonic U(1) still uses the full morphism model because it provides:

- exact bond-space identity checks;
- explicit input/output boundaries;
- one contraction vocabulary that extends to richer categories;
- correct space creation and sharing across SVD, QR, and truncation;
- a direct route from the U(1) selection rule to the general invariant-channel
  query.

The abstraction should compile away in the common case:

- fusion is multiplicity-free;
- associators and braiding are trivial;
- a charge-`q` sector of `DualSpace<X>` carries charge `-q`;
- `CouplingDescriptor` is empty;
- canonical bends and symmetric permutations need no numerical kernel;
- block legality reduces to a signed charge sum.

A convenience `contract(A, B, pairs)` may infer the unique canonical bend and
permutation in a bosonic symmetric category. It must still construct the same
checked plan. In a braided category, a request with an unspecified crossing is
ambiguous and must be rejected.

This is not overhead added for hypothetical anyons. It is how the initial U(1)
DMRG path prevents contraction of the wrong equal-shaped bond and ensures that
a decomposition creates one new bond space shared by all factors.

## 7. Consequences for BlockTensor

`BlockTensor` should be parameterized by a morphism boundary specification, not
a flat list of co-prefixed leg kinds:

```cpp
template <
    class T,
    class MorphismSpec,
    class Storage = HostOnly,
    class Coupling = Trivial>
class BlockTensor;
```

Illustrative U(1) spellings are:

```cpp
using MpsSiteSpec = MorphismSpec<
    Domain<BlockSpace, LocalSpace>,
    Codomain<BlockSpace>>;

using MpoSiteSpec = MorphismSpec<
    Domain<BlockSpace, LocalSpace>,
    Codomain<BlockSpace, LocalSpace>>;

BlockTensor<double, MpsSiteSpec> mps_site;
BlockTensor<double, MpoSiteSpec> mpo_site;
```

These aliases name decomposition kinds. Each tensor value also carries the
identity-bearing space values for its ordered domain and codomain occurrences.
The same static spec can therefore describe every site while the runtime
`SpaceId`s distinguish individual bonds.

The block map remains keyed by an opaque structure:

```text
BlockKey = (sector index per boundary occurrence, CouplingDescriptor)
```

It must never assume one block per flat sector tuple. The selection rule is
evaluated from the all-out oriented boundary, and the coupling descriptor
records which invariant channel or fusion basis the block uses.

Storage is orthogonal to this model. A `BlockTensorStorage` policy may allocate
one host buffer, several CUDA buffers, or MPI-distributed buffers without
changing spaces, duality, boundary order, block keys, or coupling metadata.

## 8. Decompositions Create Spaces

Tensor-network decompositions have categorical output contracts. For an SVD of

```text
A : X -> Y,
```

Uni20 creates one new bond space `B` and returns factors whose boundaries share
that exact identity. One possible convention is:

```text
V : X -> B
S : B -> B
U : B -> Y

A = compose(U, compose(S, V)).
```

The precise factor naming and multiplication order may follow the linalg API,
but these invariants do not change:

- `B` is a new identity-bearing space;
- every occurrence of the decomposition bond refers to the same `SpaceId`;
- sector truncation defines the decomposition of `B`;
- another isomorphic space is not silently accepted in its place.

The same rule applies to QR, LQ, polar decomposition, and DMRG bond truncation.

## 9. Operation Vocabulary

The intended categorical operation layer is:

| Operation | Contract |
|---|---|
| `tensor_product(A, B)` | Form the ordered monoidal product |
| `compose(B, A)` | Compose exactly matching domain/codomain boundaries |
| `repartition(A, split)` | Bend boundary legs, toggling explicit duality |
| `contract(A, B, pairs)` | Checked planner front end for partial contraction |
| `trace(A, pair)` | Form a categorical trace |
| `braid(A, ...)` | Apply explicit strand exchanges and category data |
| `relabel(A, ...)` | Change diagnostic metadata only |

Lower-level dense kernels operate on the selected reduced blocks. They do not
decide space identity, leg orientation, fusion legality, or planarity.

## 10. Initial Implementation Boundary

The first host-only U(1) `BlockTensor` does not need general F- or R-symbol
machinery, but it must establish the contracts that later implementations rely
on:

1. Introduce immutable identity-bearing spaces over the implemented structural
   decompositions.
2. Represent ordered `Domain` and `Codomain` explicitly.
3. Derive U(1) block legality through the all-out boundary convention.
4. Keep `BlockKey` extensible with an empty initial `CouplingDescriptor`.
5. Implement exact space-identity checks for composition and contraction.
6. Implement the cheap U(1) form of `repartition`.
7. Ensure decomposition operations create and share one new bond space.
8. Preserve every space identity and boundary occurrence in contraction
   worklists, async storage, CUDA lowering, and future MPI placement.

Dense materialization is not a fallback for this path. Any diagnostic
projection must be explicitly named as leaving the symmetry-aware execution
model and must not feed data back into U(1) DMRG state.

## 11. Durable Invariants

The following statements are design requirements:

- A tensor boundary is an ordered `Domain` and ordered `Codomain`.
- `Space`/`DualSpace` is independent of `Domain`/`Codomain`.
- `DualSpace` preserves the primal `SpaceId`.
- Repartitioning toggles explicit duality and may reverse factor order.
- Contractions require compatible dual objects and matching space identities,
  unless an explicit isomorphism is part of the operation.
- Display labels cannot alter legality.
- Block selection is a category fusion query over the oriented boundary.
- Global planarity belongs to a network or contraction planner.
- Crossings, recouplings, and categorical traces are never silently replaced
  by dense axis operations.
- Storage placement never erases boundary, space, sector, or coupling metadata.
