# Spaces, Duals, and Tensor Morphisms

**Status:** canonical design for symmetry-aware tensor spaces and
`BlockTensor`. `BlockSpace`, `IrregularSpace`, `LocalSpace`, `QNumSpace`,
`DenseSpace`, `Space`, `SymmetrySpace`, `Dual<S>`, `Domain`, and `Codomain`
are implemented. Bosonic U(1) edge repartition is the first categorical
boundary transformation.

This note defines how Uni20 represents the mathematical boundary of a
symmetry-aware tensor. It refines the categorical model in
[BlockTensor](block_tensor.md) and
[Axis Labels, Contraction, and Braiding](axis_labels_and_braiding.md). The
long-form companion paper,
[Spaces, Duals, Morphisms, and BlockTensor](../latex/block-tensor-spaces-and-morphisms.tex),
builds on the earlier planar-network design paper.

The central decisions are:

> Concrete space values model the `Space` concept directly.

and:

> `Space` versus `Dual<Space>` and `Domain` versus `Codomain` are independent
> distinctions.

`Space<LocalSpace>` wrappers and the proposed
`CoBlockSpace`/`CoLocalSpace`/`CoQNumSpace` family are not part of the intended
design.

## 1. Scope

The full morphism model applies to symmetry-aware tensor-network objects such
as `BlockTensor`. The concrete space vocabulary is also useful outside that
layer: `DenseSpace` describes a symmetry-neutral bundled axis, while the other
implemented spaces retain explicit symmetry metadata.

Bosonic abelian symmetries use the same boundary model as non-abelian,
fermionic, and braided categories. U(1) is a cheap specialization of the
contract, not a separate algebra that must later be replaced.

Four structures remain distinct:

| Structure | Meaning | Example |
|---|---|---|
| Symmetry or category | Fusion, duality, and eventually braid/recoupling rules | bosonic U(1) |
| Space value | Immutable axis structure plus a mutable semantic label | the MPS bond at cut 7 |
| Object duality | `X` versus `Dual<X>` | a bond object and its categorical dual |
| Morphism side | membership in `Domain<...>` or `Codomain<...>` | input and output boundary factors |

A **leg occurrence** is one ordered use of a space in a tensor boundary.
Several occurrences may carry equal space values. Axis positions determine
ordering; labels express intended compatibility between legs.

## 2. Implemented Space Values

`Space` is a concept, not a wrapper or base class. The implemented models are:

| Type | Structure |
|---|---|
| `BlockSpace` | canonical unique `(QNum, dimension)` sectors |
| `IrregularSpace` | ordered `(QNum, dimension)` blocks, including repeats |
| `LocalSpace` | ordered explicit local-state `QNum`s, including repeats |
| `QNumSpace` | one irrep label |
| `DenseSpace` | symmetry-neutral dense extent |

`BlockSpace`, `IrregularSpace`, `LocalSpace`, and `QNumSpace` also model
`SymmetrySpace` because they expose an explicit `Symmetry`. `DenseSpace`
deliberately does not: it is a plain dense axis which contributes no quantum
number to a selection rule.

Examples:

```cpp
Symmetry const u1{"N:U(1)"};

BlockSpace bond(
    u1,
    {
        {make_qnum(u1, {{"N", -1}}), 4},
        {make_qnum(u1, {{"N", 0}}), 8},
        {make_qnum(u1, {{"N", 1}}), 4},
    },
    "bond-7");

IrregularSpace projected_bond(
    u1,
    {
        {make_qnum(u1, {{"N", 1}}), 2},
        {make_qnum(u1, {{"N", 0}}), 8},
        {make_qnum(u1, {{"N", 1}}), 2},
    },
    "projected-bond");

LocalSpace site(
    u1,
    {
        make_qnum(u1, {{"N", 0}}),
        make_qnum(u1, {{"N", 1}}),
    },
    "fermion");

QNumSpace total_charge(make_qnum(u1, {{"N", 0}}), "total");
DenseSpace batch(32, "sample");
```

The structural part of every space is immutable after construction. This
ensures that a tensor leg cannot silently change dimension, sectors, or local
state ordering while storage and block maps still depend on it.

### Labels and equality

Every space has a `std::string` label. The empty string is a valid unlabelled
value. Labels need not be globally unique, generated, interned, or stable
across processes.

Exact equality includes both structure and label:

```text
same symmetry/structure + same label       -> equal spaces
same symmetry/structure + different label  -> unequal spaces
```

The label may be replaced with `set_label(...)` without changing structural
metadata. This supports workflows which construct an MPS and subsequently
assign bond labels across adjacent tensors. A network-level operation that
labels a bond must update the two endpoint leg occurrences together.

Structural compatibility is a separate question. A future helper such as
`isomorphic_space(a, b)` may deliberately ignore labels while comparing
symmetry, sectors, multiplicities, local-state order, or dense extent as
appropriate. An operation must choose exact equality or structural
compatibility explicitly; it must not silently discard labels.

There is no `SpaceId`. A process-global or distributed unique identifier would
add construction, serialization, and MPI-coordination requirements without
providing semantics beyond the labels and structure already needed by tensor
operations.

## 3. Space-Specific Invariants

### BlockSpace

`BlockSpace` requires an explicit valid `Symmetry`, including when it contains
no sectors. Construction accepts an initializer list or input range of
`BlockSector` values. Every dimension is positive, every `QNum` belongs to the
declared symmetry, and each `QNum` occurs once. Sectors are stored in canonical
packed-quantum-number order.

### IrregularSpace

`IrregularSpace` also requires an explicit valid `Symmetry`, including when
empty, but preserves its supplied block order and permits repeated `QNum`
sectors. Every block still has positive dimension.

This representation is useful after symmetry projection. Distinct sectors in a
larger symmetry may collapse to the same `QNum` when one component is removed.
The tensor can retain its existing block-sparse segmentation without
immediately repacking data.

`regularize(IrregularSpace)` explicitly produces a canonical `BlockSpace`.
For every original block it records the destination block index and half-open
degeneracy range. The label is preserved.

### LocalSpace

`LocalSpace` requires an explicit valid `Symmetry` and preserves the exact
ordered list of local-state `QNum`s. Repeated quantum numbers are meaningful:
for a physical site, two different basis states may transform in the same
sector.

`regularize(LocalSpace)` explicitly coalesces equal charges into a
`BlockSpace` and returns the block index and within-block offset of every local
state. The label is preserved.

### QNumSpace

`QNumSpace` carries exactly one valid `QNum`; its symmetry follows from that
value. It is useful for one-irrep legs and fixed total-charge boundaries.

### DenseSpace

`DenseSpace` carries an extent and label only. It is useful for batching or
bundling structurally identical symmetric tensors without introducing a
fictitious symmetry charge. A zero extent is valid.

## 4. Object Duality and Morphism Side

A symmetry-aware tensor is a morphism

```text
A : X_0 tensor ... tensor X_(m-1)
      ->
    Y_0 tensor ... tensor Y_(n-1)
```

with an ordered domain and ordered codomain:

```cpp
auto domain = Domain{left_bond, physical};
auto codomain = Codomain{right_bond};
```

`Domain<Spaces...>` and `Codomain<Spaces...>` are distinct heterogeneous value
types. Each stores its concrete spaces in boundary order. The factor count and
factor types are compile-time properties; the space values, including sector
structure and labels, are runtime metadata. `space<I>()` and `spaces()` expose
that structure read-only. A leg label may be changed explicitly:

```cpp
domain.set_label<0>("bond-7");
codomain.set_label<0>("bond-8");
```

This is the only boundary mutation currently provided. It cannot replace a
space or change its structural metadata. Exact boundary equality compares the
ordered space values and therefore includes labels. Empty `Domain<>` and
`Codomain<>` values represent the tensor unit.

`Dual<S>` is a generic value adaptor over any concrete `Space`. `Domain` and
`Codomain` identify which side of the map contains an object; `Dual<S>` records
object duality independently. The `DualSpace` concept makes that status
discoverable without parallel type families such as `DualBlockSpace`,
`DualLocalSpace`, and `DualQNumSpace`.

The adaptor preserves the underlying basis occurrence order, dimensions, and
label. An operation which observes a quantum number through `Dual<S>` obtains
`dual(q)`. In particular, dualizing a canonical `BlockSpace` does not rebuild
or re-sort it: sector coordinate `i` still denotes the same basis occurrence
and degeneracy range, but its observed charge is dual. This is required for
zero-copy boundary transformations because storage bindings remain attached
to basis occurrences rather than to a newly canonicalized charge list.

Duality is involutive:

```text
dual(dual(X)) = X.
```

The category owns how sectors transform. For bosonic U(1), `dual(q) = -q`.
A general category may also require evaluation, coevaluation, pivotal,
normalization, or braid data.

### All-out boundary form

Selection rules are easiest to state after orienting every boundary leg
outward:

- `Codomain<X>` contributes `X`;
- `Domain<X>` contributes `Dual<X>`.

If `X` is already dual, duality is involutive. Effective outward duality is the
exclusive-or of morphism side and explicit duality.

Dualizing a tensor product reverses order. One canonical all-out ordering for
`A : X_0 tensor ... tensor X_(m-1) -> Y_0 tensor ... tensor Y_(n-1)` is:

```text
Dual<X_(m-1)> tensor ... tensor Dual<X_0>
    tensor Y_0 tensor ... tensor Y_(n-1).
```

For bosonic U(1), a block is legal when:

```text
sum(codomain charges) - sum(domain charges) = 0.
```

For `A : left_bond tensor physical -> right_bond`, this gives:

```text
q_right = q_left + q_physical.
```

The generic implementation asks the category for invariant channels rather
than hardcoding charge addition.

## 5. Repartition Is Wire Bending

Moving a leg across the domain/codomain boundary is an explicit categorical
operation named `repartition`. It toggles object duality. The first operation
supports the two planar boundary ends and their inverses:

```text
Hom(X tensor A, B)  ~= Hom(A, Dual<X> tensor B)
Hom(A tensor X, B)  ~= Hom(A, B tensor Dual<X>)
Hom(A, X tensor B)  ~= Hom(Dual<X> tensor A, B)
Hom(A, B tensor X)  ~= Hom(A tensor Dual<X>, B).
```

Moving several adjacent factors may reverse order:

```text
Dual<X tensor Y> ~= Dual<Y> tensor Dual<X>.
```

In bosonic U(1), bending is a checked metadata, key-order, and dense-axis-order
operation with unit numerical factor. The numerical payload does not move.
Canonical transformed keys may require a new sorted logical index which maps
to unchanged physical block bindings. A moved dense axis is exposed through a
permuted strided mdspan over the same allocation.

Other categories may attach pivotal maps, twists, phases, `1j` factors, or
recoupling transformations. Those belong in logical block metadata and kernel
lowering. They do not justify rewriting the stored tensor elements. Braiding
uses the same zero-copy key/axis-permutation mechanism plus its category-defined
exchange factor.

`repartition` is distinct from:

- `set_label`, which changes semantic leg compatibility but not structure;
- `conj`, which changes numerical value semantics but does not move legs;
- `adjoint`, which reverses the morphism;
- `braid`, which exchanges strands;
- an arbitrary axis permutation, which may require recoupling or braiding.

## 6. Composition and Partial Contraction

Full contraction is morphism composition. If:

```text
A : X -> Y
B : Y -> Z
```

then:

```text
compose(B, A) : X -> Z.
```

Partial contraction is tensor product with identities followed by composition.
The implementation need not materialize identity tensors: a checked planner
proves the composition, preserves the external boundary, and lowers it to
block worklists and dense kernels.

The implemented first host operation is the adjacent pair
`contract<left_axis, right_axis>(left, right)`. It requires the rightmost
codomain space of `left` and leftmost domain space of `right` to be exactly
equal. It retains `left.domain`, then the uncontracted part of `right.domain`,
and retains the uncontracted part of `left.codomain`, then `right.codomain`.
Stored blocks are paired by the contracted basis occurrence before dense block
kernels run. Missing sparse blocks remain exact zero; no symmetry metadata is
flattened or reconstructed from dense indices.

Before contracting two leg occurrences, the planner validates the relationship
required by the operation. Ordinary composition normally requires exact space
equality, including labels. A structural conversion or explicit isomorphism
may instead request structural compatibility. Axis positions and explicit
contraction pairs determine which occurrences are joined; labels never replace
that ordering information.

More general networks use the same primitives:

- associating and recoupling tensor products;
- explicit `repartition` operations;
- the implemented zero-copy `permute<Axis...>` within each side in the bosonic
  symmetric category, or explicit braids otherwise;
- tensor products with implicit identities;
- composition;
- categorical trace for closed boundaries.

## 7. Consequences for BlockTensor

`BlockTensor` should be parameterized directly by its domain and codomain
types, not by a flat list of co-prefixed leg kinds or another boundary wrapper:

```cpp
template <
    class T,
    class DomainType,
    class CodomainType,
    BlockTensorStorage Storage>
class BlockTensor;
```

Illustrative U(1) spellings are:

```cpp
template<class T, CompleteBlockStorage Storage>
using AMatrix = BlockTensor<
    T,
    Domain<BlockSpace, LocalSpace>,
    Codomain<BlockSpace>,
    Storage>;

template<class T, SparseBlockStorage Storage>
using MpoSite = BlockTensor<
    T,
    Domain<LocalSpace, LocalSpace>,
    Codomain<LocalSpace, LocalSpace>,
    Storage>;
```

These aliases name space kinds and storage requirements. Each tensor value
carries concrete space values for its ordered domain and codomain leg
occurrences. The same static type can describe every site while runtime labels
distinguish intended bonds, physical legs, impurities, or alternating site
types.

The block map remains keyed by an opaque structure:

```text
BlockKey = (selection index per coordinate-bearing boundary occurrence,
            CouplingDescriptor)
```

Boundary coordinates occur in domain-left-to-right order followed by
codomain-left-to-right order. `BlockSpace`, `IrregularSpace`, and `LocalSpace`
contribute coordinates; fixed `QNumSpace` and `DenseSpace` factors do not.
Their deterministic lexicographic order is part of logical block identity; a
storage layout may arrange the corresponding data differently. The coupling
descriptor compares after all stored boundary coordinates.

It must not assume one block per flat sector tuple. The selection rule is
evaluated from the all-out oriented boundary, and the coupling descriptor
records the invariant channel or fusion basis.

Storage is orthogonal. A `BlockTensorStorage` policy may allocate one host
buffer, several CUDA buffers, or MPI-distributed buffers without changing
spaces, duality, boundary order, block keys, or coupling metadata.

## 8. Decompositions Create Bond Spaces

For an SVD of `A : X -> Y`, Uni20 creates a bond space `B` and returns factors
whose corresponding bond occurrences carry equal copies of `B`:

```text
V : X -> B
S : B -> B
U : B -> Y
```

The decomposition chooses `B`'s sector structure and label. Truncation changes
that newly produced structure before the result is exposed. The same rule
applies to QR, LQ, polar decomposition, and DMRG bond truncation.

Unlike a global identity token, the value contract is naturally copyable,
serializable, and reconstructible on MPI ranks. A network may subsequently
rename the bond by updating all of its endpoint occurrences.

## 9. Initial Implementation Boundary

The first host-only U(1) `BlockTensor` should:

1. Use the concrete implemented space values directly.
2. Use the implemented ordered `Domain` and `Codomain` values.
3. Represent explicit object duality with the generic `Dual<S>` adaptor.
4. Derive U(1) block legality through the all-out boundary convention.
5. Keep `BlockKey` extensible with an empty initial `CouplingDescriptor`.
6. Require exact compatible leg values for ordinary composition.
7. Implement the cheap U(1) form of `repartition`.
8. Preserve every space value and boundary occurrence in contraction
   worklists, async storage, CUDA lowering, and future MPI placement.

Dense materialization is not a fallback for this path. A diagnostic projection
must be explicitly named as leaving the symmetry-aware execution model and
must not feed data back into U(1) DMRG state.

## 10. Durable Invariants

- Concrete space types model `Space`; there is no `Space<...>` wrapper.
- Structural space metadata is immutable; the string label is mutable.
- Exact space equality includes the label.
- `BlockSpace` is canonical; `IrregularSpace` preserves ordered segmentation.
- `DenseSpace` is symmetry-neutral.
- A tensor boundary is an ordered `Domain` and ordered `Codomain`.
- `Space`/`Dual<Space>` is independent of `Domain`/`Codomain`.
- `Dual<S>` preserves basis occurrence indices while dualizing observed charges.
- Repartitioning toggles explicit duality, moves only an edge factor, and may
  reverse factor order under repeated bends.
- Labels validate intended leg compatibility but never determine axis order.
- Block selection is a category fusion query over the oriented boundary.
- Global planarity belongs to a network or contraction planner.
- Storage placement never erases boundary, space, sector, label, or coupling
  metadata.
- Logical key and dense-axis order may change without changing physical payload
  placement.
