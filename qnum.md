# QNum Design Sketch

This is a temporary design note for the Uni20 quantum number and symmetry layer.

The immediate goal is to support block-sparse DMRG bookkeeping with U(1) symmetry,
while keeping the interface suitable for later non-Abelian and braided categories.

## Scope

This note describes:

- the `QNum` value type
- the `QNumList` container type
- the `Symmetry` / `SymmetryImpl` runtime model
- how future fusion support constrains the design
- the role of braiding
- the minimum functionality needed for the first DMRG implementation

This note does not define tensor storage or tensor algorithms directly.

## Design Goals

- Very fast hashing, equality, and copying of quantum numbers.
- Avoid quantum-number bookkeeping becoming a runtime bottleneck.
- Keep the U(1) implementation simple.
- Support future non-Abelian symmetries and braided categories without redesign.
- Remove `Projection` from the core API completely.
- Allow a `QNum` to be used as a small standalone value object.

## Terminology

- A `QNum` is an irreducible representation label.
- A `Symmetry` is a direct product of braided fusion tensor categories.
- A multiplicity label will eventually be needed once the fusion API is
  generalized beyond the U(1) first pass.

## Core Types

## `QNum`

`QNum` is the fundamental value type.

Conceptually it contains:

```cpp
class QNum {
  SymmetryImpl const* sym_;
  std::uint64_t code_;
};
```

Properties:

- cheap to copy
- cheap to hash
- equality is pointer-plus-code equality
- can be passed by value
- carries enough context to interpret the irrep label

The packed integer `code_` is a canonical encoded representation of the irrep.
Each concrete symmetry factor chooses its own encoding, subject to the invariants below.

## `Symmetry`

`Symmetry` is a lightweight handle to canonical shared runtime state.

Conceptually:

```cpp
class Symmetry {
  SymmetryImpl const* impl_;
};
```

`SymmetryImpl` describes a direct product of symmetry/category factors and
defines the interpretation of packed `QNum` codes.

Only a small number of `SymmetryImpl` objects are expected to exist in a process.
They should be canonicalized globally and may be intentionally leaked for the
lifetime of the process.

Concrete factor behavior should be routed through a virtual interface rather than
hardcoded directly into `Symmetry`, `SymmetryImpl`, or `QNum`.

That keeps the public `Symmetry` handle independent of specific factor types and
allows new symmetry factors to be added without modifying the `Symmetry` class.

## `QNumList`

`QNumList` is a container of `QNum` values that all share the same symmetry.

Conceptually it behaves like a `std::vector<QNum>`, but with the invariant that
all entries have identical `SymmetryImpl*`.

This is useful for:

- lists of allowed quantum numbers for an index or basis
- block-label bookkeeping
- APIs that naturally return several `QNum`s but do not need multiplicity labels

The intended invariant is:

- an empty `QNumList` still carries a definite `Symmetry`
- appending a `QNum` with a different symmetry is an error

This is distinct from future fusion output. Once multiplicities matter, the
fusion API will need more structure than a plain list of `QNum`.

Likely first-pass operations:

- `size()`, `empty()`, iteration
- `symmetry() -> Symmetry`
- `push_back(q)`
- indexed access
- `contains(q) -> bool`
- sorting in canonical encoded order
- duplicate removal / normalization helpers

The encoded `uint64_t` order is intended to be the canonical display and sorting
order, so `QNumList` should preserve or recover that ordering cheaply.

## Future Fusion API

The concrete fusion result type is intentionally left undefined for now.

The current U(1) first pass only needs:

- `operator+` as the unique-fusion shortcut
- no general `operator*`
- no public multiplicity-bearing result type yet

One plausible future formulation is to represent fusion data in terms of
explicit vertices. A vertex would carry something like:

```cpp
struct Vertex {
  QNum left;
  QNum right;
  QNum out;
  std::uint32_t multiplicity;
};
```

This may prove to be the better abstraction once nontrivial multiplicities and
coupling data are implemented, but the best representation remains open.

## Representation Encoding

Each symmetry factor maps its native representation label to and from a `uint64_t`.

Requirements:

- encoded `0` must always represent the identity / scalar irrep
- encoded order is the default display order
- encoding must be canonical and stable
- encode/decode must be cheap

For U(1), the intended order is:

- `0`
- `+1`
- `-1`
- `+2`
- `-2`
- ...

That means U(1) should first apply a signed-to-unsigned bijection before any
pairing or direct-product packing.

For direct products:

- use recursive pairing or mixed-radix packing as appropriate
- the exact method is an implementation detail of `SymmetryImpl`

Overflow policy for now:

- if encoding cannot fit in the `uint64_t` payload, fail immediately
- an explicit hard failure is acceptable for the first implementation

Possible future extension:

- reserve one high bit as an escape hatch and use out-of-line storage when
  required

That future path should remain internal to the implementation.

## Fundamental Operations

These are the core operations that `SymmetryImpl` must support.

### Representation structure

- `dual(q) -> QNum`
- `is_scalar(q) -> bool`
- `qdim(q) -> real`
- `degree(q) -> int` for integral cases
- `to_string(q) -> std::string`
- `parse_qnum(...) -> QNum`

`qdim` is fundamental.

`degree` is a convenience for integral categories. It may throw if the
underlying symmetry is not integral.

### Fusion

Operator shorthand:

- `q1 + q2` is a convenience shortcut for the unique-fusion case
- `q1 * q2` is intentionally deferred until the general fusion result type is settled

`operator+` is valid only when fusion yields exactly one channel:

- one output irrep
- no unresolved multiplicity ambiguity

Otherwise it throws.

This is intended as a useful shortcut for abelian or otherwise trivial-fusion code.

For the first implementation, the full fusion API remains unimplemented.
Since the immediate target is U(1), `operator+` is the important shorthand and
`operator*` can wait until multiplicities and general fusion output are designed properly.

## Braiding

Braiding is part of the symmetry/category context, not a separate property of the
packed integer by itself.

That means:

- `QNum` carries a `SymmetryImpl*`
- the `R` matrix is determined by the symmetry context plus the representation labels

In particular, categories with the same fusion rules but different braid solutions
must correspond to different `SymmetryImpl` instances.

Examples:

- semion
- anti-semion

Operations that do not depend on braiding do not need to care about this distinction.
Operations that do depend on braiding dispatch through the same `SymmetryImpl`.

## Why `Projection` Is Gone

`Projection` is removed from the core design.

Reason:

- it is not a stable primitive across different symmetry settings
- for non-Abelian symmetries there are many possible projection choices
- future work needs more general coupling data associated with subalgebras or
  subcategories rather than one fixed notion of projection

Instead, the core value layer handles irreps only.

Basis-dependent data such as Clebsch-Gordan coefficients belongs to a separate coupling layer.

## Coupling Layer

The coupling layer is separate from `QNum` storage.

Future examples include:

- `U1 in SU2`
- `D_inf in SU2`
- `Z3 in SU2`

This layer will provide:

- Clebsch-Gordan coefficients for a chosen subalgebra or subcategory
- recoupling coefficients such as 6j and 9j symbols
- possibly additional basis- or gauge-dependent structure

The first U(1) implementation does not need this machinery, but the design must leave room for it.

## Runtime Dispatch

The old MPT design relied heavily on erased runtime structures and legacy storage tricks.
Uni20 should keep the useful flexibility while simplifying the machinery.

Tentative model:

- concrete factor implementations such as `U1`, later `SU2`, etc.
- a runtime `SymmetryImpl` for direct products and canonicalization
- generic free functions that dispatch through `SymmetryImpl`

Virtual dispatch is acceptable if needed. The important point is to keep the
public interface small and the `QNum` value type cheap.

## Symmetry Construction

The existing MPT-style string construction is still a good front-end API and
should be kept in some form.

Example:

```cpp
Symmetry sym{"N:U(1),Sz:U(1)"};
```

Here `N` and `Sz` are not just decorative aliases. They are the actual component
names of the direct-product symmetry, intended to carry physical meaning such as
particle number and spin-z.

This is useful because it is:

- compact
- readable in debugging and configuration files
- convenient for tests and small driver programs

The parsed result should still canonicalize to shared `SymmetryImpl` objects.

So the string form should be treated as a convenient construction and
serialization layer, not as the primary internal representation.

## Symmetry Coercion And Normalization

Named symmetry components also make it possible to normalize or coerce related
symmetry specifications.

Examples:

- a missing component may be treated as the identity value
- a `QNum` may be converted to a larger symmetry by filling missing named
  components with the identity
- a `QNum` may be converted to a smaller symmetry only if every dropped
  component is the identity

So conversions between related symmetries can be defined by component name,
rather than only by positional layout.

Tentative rules:

- missing named components default to the identity when extending a symmetry
- dropping a named component is only valid when that component is the identity
- incompatible component names or incompatible factor types are errors

This coercion behavior should be explicit in the API, but the names carried by
`Symmetry` are an important part of what makes it possible.

## Minimal First Implementation

The first Uni20 version should implement:

- `QNum`
- `QNumList`
- `Symmetry`
- one concrete factor: `U1`
- direct-product support sufficient for multiple U(1)-like factors
- `dual`
- `is_scalar`
- `qdim`
- `degree`
- string formatting/parsing
- hashing

This is enough for:

- DMRG block bookkeeping
- compiling TensorContraction term lists
- future host-side sparse MPO bookkeeping

## Not In Scope For First Pass

- SU(2) implementation
- general multiplicity-bearing fusion API
- explicit 6j / 9j recoupling
- anyonic categories
- out-of-line `QNum` storage for oversized encodings
- persistent serialization design

## Open Questions

- Whether `degree(q)` should throw at runtime or be unavailable at compile time
  for non-integral categories.
- Whether direct-product packing should use recursive pairing everywhere or
  mixed-radix packing when bounds are known.
- Exact canonicalization API for creating and interning `SymmetryImpl` objects.
- What the eventual multiplicity-bearing fusion result type should be.
