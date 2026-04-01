# Quantum Numbers and Symmetry

This document describes the currently implemented `uni20` quantum-number API in
`src/uni20/symmetry/`, together with the near-term extension points that will
matter for tensor-network and DMRG work.

The code is intentionally small today:

- `Symmetry` is a canonicalized runtime handle for a direct-product symmetry
  specification such as `N:U(1),Sz:U(1)`.
- `QNum` is a packed irrep label together with the `Symmetry` needed to
  interpret it.
- `QNumList` is the explicit sparse tensor-space representation.
- `BlockSpace` is the coalesced `(QNum, dim)` tensor-space representation.
- `U1` is the first value-level quantum-number type and currently the only
  implemented symmetry factor.

The relevant headers are:

- `src/uni20/symmetry/block_space.hpp`
- `src/uni20/symmetry/symmetry.hpp`
- `src/uni20/symmetry/qnum.hpp`
- `src/uni20/symmetry/u1.hpp`

## Implemented API

### `Symmetry`

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

### `U1`

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

### `QNum`

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
- `is_scalar(q)`
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

### `QNumList`

`QNumList` is the explicit sparse tensor-space representation. It is also a
small container with one important invariant: every entry must have the same
`Symmetry`.

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

As a tensor space, `QNumList` means:

- duplicates are allowed
- order is meaningful
- automatic coalescing is not allowed
- an empty list still has a definite `Symmetry`

This makes it suitable for sparse physical legs, sparse MPO bond spaces, and
other cases where the explicit outer-loop structure should remain visible to the
contraction engine.

### `BlockSpace`

`BlockSpace` is the coalesced tensor-space representation. It stores ordered
blocks `(QNum, dim)`.

Implemented operations:

- construction from `Symmetry`
- construction from `Symmetry` plus an initializer list of blocks
- `symmetry()`
- `size()`
- `empty()`
- `total_dim()`
- `is_regular()`
- `push_back(...)`
- `clear()`
- `contains(...)`
- indexed access
- iteration
- equality comparison

`BlockSpace` allows repeated `QNum` sectors, but that is treated as an
irregular state that should be normalized explicitly before dense block-space
operations.

### Regularization Helpers

Two explicit regularization helpers are now implemented:

- `regularize(QNumList)` coalesces the sparse space into a `BlockSpace` and
  records, for each original entry, the destination block and offset inside that
  block
- `regularize(BlockSpace)` coalesces repeated blocks with the same `QNum` and
  records the destination block plus the covered range inside that block

There is also a convenience conversion:

- `to_block_space(QNumList)`

These helpers are intended to support explicit sparse-to-coalesced transitions
without making coalescing automatic.

## Encoding and Ordering

Each symmetry factor is encoded to a factor-local `uint64_t`, and `QNum`
combines those factor-local codes into one packed `uint64_t`.

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

## Keeping Code and Docs in Sync

The symmetry headers reference this document directly so that API changes are
more likely to prompt a doc update in the same patch. If you change the public
surface of `Symmetry`, `QNum`, `QNumList`, or `U1`, update both:

- this file
- the relevant header comments in `src/uni20/symmetry/`
