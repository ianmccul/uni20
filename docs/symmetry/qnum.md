# Quantum Numbers and Symmetry

This document describes the currently implemented `uni20` quantum-number API in
`src/uni20/symmetry/`, together with the near-term extension points that will
matter for tensor-network and DMRG work.

The implemented layer contains:

- `Symmetry` is a canonicalized runtime handle for a direct-product symmetry
  specification such as `N:U(1),Sz:U(1)`.
- `QNum` is a packed irrep label together with the `Symmetry` needed to
  interpret it.
- `QNumList` is a mutable ordered quantum-number container used while building
  or regularizing spaces.
- `BlockSpace` is an immutable coalesced `(QNum, dim)` tensor space.
- `IrregularSpace` is an immutable ordered sequence of `(QNum, dim)` blocks
  whose quantum numbers may repeat.
- `LocalSpace` is an immutable ordered local-state space whose charges may
  repeat.
- `QNumSpace` is an immutable one-irrep space.
- `DenseSpace` is an immutable symmetry-neutral dense extent.
- `Space` and `SymmetrySpace` describe their common contracts.
- `Domain<...>` and `Codomain<...>` are ordered morphism-boundary values.
- `U1` is the first value-level quantum-number type and currently the only
  implemented symmetry factor.

The relevant headers are:

- `src/uni20/symmetry/block_space.hpp`
- `src/uni20/symmetry/block_sector.hpp`
- `src/uni20/symmetry/dense_space.hpp`
- `src/uni20/symmetry/irregular_space.hpp`
- `src/uni20/symmetry/local_space.hpp`
- `src/uni20/symmetry/morphism_boundary.hpp`
- `src/uni20/symmetry/qnum_space.hpp`
- `src/uni20/symmetry/space.hpp`
- `src/uni20/symmetry/symmetry.hpp`
- `src/uni20/symmetry/qnum.hpp`
- `src/uni20/symmetry/u1.hpp`

## Implemented API

### Symmetry

`Symmetry` is constructed primarily from a string specification:

```cpp
using namespace uni20;

Symmetry particle{"N:U(1)"};
Symmetry full{"N:U(1),Sz:U(1)"};
```

Current properties:

- Parsing is whitespace-tolerant and canonicalizes to a shared internal
  representation.
- Equality compares canonical identity.
- Component names such as `N` and `Sz` are semantic, not decorative.
- The current implementation interns canonical symmetry instances for process
  lifetime.

`Symmetry` also exposes:

- `valid()`
- `factor_count()`
- `factors()`
- `to_string()`
- `Symmetry::parse(...)`

### U1

`U1` is the value-level irrep type for one U(1) factor:

```cpp
U1 q0;
U1 q1{1};
U1 q2{half_int{2.5}};
```

Implemented operations:

- `dual(U1)`
- `qdim(U1)` returning `1.0`
- `degree(U1)` returning `1`
- `operator+` and `operator-`
- `to_string(U1)` using decimal half-integer form such as `2.5`
- `to_string_fraction(U1)` using fractional form such as `5/2`
- stream output, `std::format`, and `std::hash`

U(1) values are stored as `half_int`, so both integer and half-integer charges
are supported.

### QNum

`QNum` is the packed irrep label used by the tensor code:

```cpp
Symmetry sym{"N:U(1),Sz:U(1)"};
QNum q = make_qnum(sym, {{"N", U1{1}}, {"Sz", U1{half_int{0.5}}}});
```

Implemented operations:

- `QNum::identity(sym)`
- `valid()`
- `symmetry()`
- `raw_code()`
- `dual(q)`
- `is_identity(q)`
- `qdim(q)`
- `degree(q)`
- `to_string(q)` producing comma-separated named components such as
  `N=1,Sz=0.5`
- `u1_component(q, "N")`
- `coerce(q, target_symmetry)` for named-component coercion
- `operator+` as the unique-fusion shortcut for currently implemented abelian
  factors

For the current U(1) implementation, missing component assignments in
`make_qnum(...)` default to the identity.

### QNumList

`QNumList` is a mutable ordered quantum-number container with one important
invariant: every entry must have the same `Symmetry`.

Implemented operations:

- construction from `Symmetry`
- construction from `Symmetry` plus an initializer list
- `symmetry()`
- `size()`
- `empty()`
- `push_back(...)`
- `clear()`
- `contains(...)`
- indexed access
- iteration
- `sort()`
- `normalize()` which sorts and removes duplicates

Its construction-time semantics are:

- duplicates are allowed
- order is meaningful
- automatic coalescing is not allowed
- an empty list still has a definite `Symmetry`

This makes it useful while building sparse physical legs, sparse MPO bond
spaces, and regularization metadata. Persistent tensor boundaries should use
the immutable `LocalSpace` or `BlockSpace` values.

### Space concepts

`Space` is a concept satisfied directly by concrete space values. It requires
copyable value semantics, exact equality, and a mutable `std::string` label:

```cpp
static_assert(Space<BlockSpace>);
static_assert(Space<IrregularSpace>);
static_assert(Space<LocalSpace>);
static_assert(Space<QNumSpace>);
static_assert(Space<DenseSpace>);
```

`SymmetrySpace` additionally requires an explicit `Symmetry`:

```cpp
static_assert(SymmetrySpace<BlockSpace>);
static_assert(SymmetrySpace<IrregularSpace>);
static_assert(SymmetrySpace<LocalSpace>);
static_assert(SymmetrySpace<QNumSpace>);
static_assert(!SymmetrySpace<DenseSpace>);
```

The structural portion of a space is immutable. `set_label(...)` is the only
mutation supplied by these values. Exact equality includes the label.

### BlockSpace

`BlockSpace` is the immutable coalesced tensor-space representation. It stores
canonical unique blocks `(QNum, dim)`.

Implemented operations:

- construction from `Symmetry`
- construction from `Symmetry` plus an initializer list or input range of
  blocks
- `symmetry()`
- `size()`
- `empty()`
- `total_dim()`
- `contains(...)`
- indexed access
- iteration
- `sectors()`
- `label()` and `set_label(...)`
- equality comparison

Construction requires a valid explicit `Symmetry`, including for an empty
space. Each block dimension must be positive and every `QNum` must belong to
that symmetry. Construction sorts blocks canonically and rejects repeated
`QNum` sectors.

### IrregularSpace

`IrregularSpace` is the valid non-regular block-space representation. It stores
an ordered sequence of `BlockSector` values:

- construction order is preserved
- repeated `QNum` sectors are preserved
- every dimension remains positive
- every block belongs to the explicit `Symmetry`
- an empty space retains its symmetry
- `label()` and `set_label(...)` manage its semantic leg label

It is useful when projecting out or forgetting part of a symmetry. Sectors
which were distinct under the original symmetry may acquire the same projected
`QNum`, while their existing block segmentation and order remain useful.
Representing that result as an `IrregularSpace` avoids immediate data movement.

### LocalSpace

`LocalSpace` is the immutable ordered local-state representation:

- construction from an explicit `Symmetry` and an initializer list, input
  range, or `QNumList`
- repeated `QNum` values are preserved
- state ordering is preserved
- an empty space retains its explicit `Symmetry`
- `qnums()` exposes a read-only span
- `label()` and `set_label(...)` manage the semantic leg label

It is suitable for physical legs where different basis states may carry the
same charge.

### QNumSpace

`QNumSpace` contains one valid `QNum`, with the symmetry obtained from that
irrep. It is useful for fixed-charge boundaries and one-irrep legs. Its only
mutable field is the string label.

### DenseSpace

`DenseSpace` contains a dense extent and string label. It deliberately has no
`Symmetry`: a dense leg may batch or bundle symmetric tensors without
contributing a charge to their selection rules. Zero extent is permitted.

### Regularization Helpers

Three explicit regularization helpers are implemented:

- `regularize(QNumList)` coalesces the sparse space into a `BlockSpace` and
  records, for each original entry, the destination block and offset inside that
  block
- `regularize(LocalSpace)` performs the same coalescing for an immutable local
  space and preserves its label
- `regularize(IrregularSpace)` coalesces repeated blocks and records the
  destination canonical block plus half-open degeneracy range for every
  original block

Convenience conversions are:

- `to_block_space(QNumList)`
- `to_block_space(LocalSpace)`
- `to_block_space(IrregularSpace)`

These helpers are intended to support explicit sparse-to-coalesced transitions
without making coalescing automatic. `BlockSpace` itself is always regular, so
there is no `regularize(BlockSpace)` operation.

## Encoding and Ordering

Each symmetry factor is encoded to a factor-local `uint64_t`, and `QNum`
combines those factor-local codes into one tagged `uint64_t` payload. Inline
codes use the low 63 bits. Bit 63 is reserved for a future process-local,
interned out-of-line representation; construction currently throws
`std::overflow_error` when a combined code does not fit in the inline range.

For U(1), the local encoding is chosen so that numerical order matches the
natural display order:

- `0`
- `+1/2`
- `-1/2`
- `+1`
- `-1`
- `+3/2`
- `-3/2`
- `...`

This ordering is useful both for debugging output and for canonical sorting in
`QNumList` and canonical block ordering in `BlockSpace`.

## Coercion by Named Components

Component names are used to align related symmetries.

Examples:

- coercing `N:U(1)` into `N:U(1),Sz:U(1)` fills the missing `Sz` component with
  the identity
- coercing `N:U(1),Sz:U(1)` into `N:U(1)` is allowed only when `Sz` is the
  identity

This is intended to make block bookkeeping practical when related spaces carry
slightly different direct-product labels.

## Near-Term Extension Points

The current code is deliberately U(1)-first. The intended growth path is:

### Additional factor types

New symmetry factors should be addable without editing `Symmetry` itself. The
current internal adapter layer already points in that direction:

- value-level factor type, analogous to `U1`
- internal `SymmetryFactor<T>` adapter
- string construction through the factor registry

### General fusion output

`operator+` is only the unique-fusion shortcut. A future non-abelian API will
need an explicit fusion result type that can represent:

- multiple output irreps
- multiplicities
- possibly explicit vertex or coupling-channel labels

That design is intentionally deferred until there is a concrete non-abelian
implementation to drive it.

### Braiding and anyons

The long-term target is broader than ordinary group irreps. Uni20 will need to
support braided fusion tensor categories closely enough to model:

- fermionic sign structure
- anyonic MPOs
- braided tensor manipulations

That means the eventual symmetry context must carry more than just fusion and
duality data. It will also need a place for braid data such as `R`-matrices.

### Coupling data

The current code does not yet implement F-moves or recoupling coefficients.
Future APIs are expected to cover at least:

- F-symbols
- 6j symbols
- 9j symbols
- potentially more general coupling data for selected subalgebras or
  subcategories

This layer will matter once Uni20 moves beyond abelian block bookkeeping and
starts doing genuine non-abelian or braided tensor algebra.

Coupling caches will be partitioned by normalized symmetry context, rather than
stored in one global cross-symmetry cache. Cache keys will therefore contain
only ordered irrep codes meaningful within the owning context and will not
repeat the `Symmetry` pointer. If an operation combines compatible symmetry
descriptions, it must first normalize every `QNum` to one common target—for
example by matching named components, inserting identities, and reordering
components—then query that target context's cache. Normalization must reject
non-identity component loss or incompatible factor types.

## Morphism Boundaries

The implemented `Domain<Spaces...>` and `Codomain<Spaces...>` value templates
provide the morphism-side axis needed by the next tensor-network layer:

- a concrete space versus `Dual<S>`;
- membership in an ordered `Domain<...>` versus `Codomain<...>`.

The first axis remains deferred; the second is represented explicitly today.
Each boundary retains concrete space types and ordered values. Space structure
is exposed read-only, while `set_label<I>(...)` changes one leg label without
replacing its space. `Domain<>` and `Codomain<>` represent the tensor unit.

The earlier `conjugate<T>` / `co<T>` direction-wrapper proposal conflated these
axes and is no longer the intended design. Moving a leg between domain and
codomain is an explicit wire-bending operation that toggles object duality.
See [Spaces, Duals, and Tensor Morphisms](spaces_duals_and_morphisms.md).

The first U(1) three-leg tensor can therefore be modeled as a morphism such as:

```text
left_bond tensor local_space -> right_bond
```

This does not require direction variants of `BlockSpace` or `QNumList`.

## Deferred Next Steps

These items are intentionally deferred rather than part of the current public
API, but they are likely to be the next symmetry-side tasks once the first U(1)
DMRG prototype is underway.

### Space-level helpers

Likely next convenience helpers on spaces:

- `find_block(q)` / `index_of(q)` on `BlockSpace`
- `dim(q)` on `BlockSpace`
- space-level coercion for `QNumList` and `BlockSpace`
- space-level shift helpers in the old `delta_shift` spirit

These are expected to become useful as soon as the MPO and AMatrix compiler code
is written.

### Space dualization

The morphism layer should define a generic `Dual<S>` using the symmetry's
value-level `dual(QNum)` operation. It should not require parallel concrete
families such as `DualBlockSpace` and `DualLocalSpace`. Dualization is a
categorical view of the space, not a mutation of its stored sector structure.

### Inverse regularization metadata

The current regularization helpers provide the forward block/index mapping
needed to coalesce sparse spaces explicitly. Later work may also want:

- explicit inverse mapping helpers
- tensor-reshape metadata derived from the regularization
- eventually actual transform operators where needed

The first U(1) DMRG path does not require those operators yet.

## Keeping Code and Docs in Sync

The symmetry headers reference this document directly so that API changes are
more likely to prompt a doc update in the same patch. If you change the public
surface of `Symmetry`, `QNum`, `QNumList`, `BlockSpace`,
`IrregularSpace`, `LocalSpace`, `QNumSpace`, `DenseSpace`, `Domain`,
`Codomain`, or `U1`, update both:

- this file
- the relevant header comments in `src/uni20/symmetry/`
