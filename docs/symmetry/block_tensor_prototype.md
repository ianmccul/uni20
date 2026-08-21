# Bosonic Abelian BlockTensor Prototype

**Status:** active design contract with the first sparse host slice
implemented. It defines the remaining host-only bosonic abelian vertical
slice. The broader [BlockTensor design](block_tensor.md) records later CUDA,
MPI, recoupling, and view requirements.

## 1. Purpose

The prototype provides the symmetry-preserving block container needed by a
U(1) two-site DMRG implementation. It must represent:

- MPS A-matrices with every symmetry-legal block allocated;
- sparse four-leg MPO site tensors;
- possibly sparse three-leg environment tensors;
- irreducible operators with an explicit transform irrep;
- additional dense bundle axes without erasing symmetry metadata.

It is a general fixed-order tensor morphism, not an MPS-specific container.
The MPS and MPO names are aliases over the same `BlockTensor` mechanism.

The prototype supports direct products of bosonic abelian symmetry factors.
The current concrete target is one or more U(1) factors.

### Implemented first slice

The current code establishes the storage and boundary seam needed by later
operations:

- tensor orders zero through four;
- `LocalSpace`, `QNumSpace`, `BlockSpace`, and their `Dual<S>` adaptors in arbitrary
  domain/codomain positions;
- independent compile-time classification of block-key coordinates and dense
  numerical axes for all five concrete space kinds;
- explicit sparse key construction with legality, ordering, and duplicate
  validation;
- `SeparateSparseBlockStorage`, with one independently owning column-major
  dense `Tensor` per block;
- `PackedSparseBlockStorage`, with canonical offsets into one contiguous host
  buffer;
- mutable and const mdspan access to stored blocks;
- delegation of the dense leaf backend list through each block storage policy;
- a generic `Dual<S>` value adaptor whose quantum-number observations are dual;
- zero-copy bosonic compile-time permutations within each morphism side;
- zero-copy bosonic left/right `repartition` views with transformed canonical
  keys and strided dense-block axis permutations.

This slice is synchronous and host-resident. Mapped permutation and repartition
views retain an lvalue source reference and therefore cannot outlive their
source tensor. Payload element writes remain valid, but assigning or
structurally modifying a source invalidates every view transitively built from
it. The slice does not yet provide complete storage, builders, the general
numerical block-operation surface, async buffer ownership, or the remaining
space kinds described below.

## 2. Initial Type Shape

The initial owning type is:

```cpp
template<
    class T,
    class DomainType,
    class CodomainType,
    BlockTensorStorage Storage>
class BlockTensor;
```

There is no additional `MorphismSpec` wrapper. `DomainType` and
`CodomainType` are already distinct types and values. A tensor stores concrete
instances of both.

The complete host prototype will allow each boundary to contain any fixed
number and ordering of:

- `LocalSpace`;
- `BlockSpace`;
- `IrregularSpace`;
- `QNumSpace`;
- `DenseSpace`.

The tensor order is known at compile time:

```cpp
DomainType::size() + CodomainType::size()
```

Concrete sector dimensions, local states, dense extents, and labels remain
runtime values.

The implemented slice currently accepts `LocalSpace`, `QNumSpace`,
`BlockSpace`, and their explicit duals, and requires total tensor order from
zero through four.

## 3. Explicit Symmetry Context

A `BlockTensor` stores an explicit `Symmetry`, including when its boundaries
are empty or contain only `DenseSpace` factors. This follows the same rule as
an empty `BlockSpace`: absence of sectors must not erase the symmetry context.

Construction validates that every `LocalSpace`, `BlockSpace`,
`IrregularSpace`, and `QNumSpace` factor carries exactly that symmetry.
`DenseSpace` carries no symmetry and contributes the identity to the selection
rule.

The prototype rejects a symmetry containing a factor outside its supported
bosonic abelian set. Supporting a new symmetry is an explicit extension of the
block-legality implementation, not an unchecked interpretation of existing
blocks.

## 4. Domain, Codomain, and Duality

Domain versus codomain determines the first sign with which a charge
contributes:

```text
sum(domain charges) = sum(codomain charges).
```

This gives the required rules directly:

```text
A-matrix:
    q_left + q_physical = q_right

MPO site:
    q_left_auxiliary + q_ket = q_right_auxiliary + q_bra

environment:
    q_bra_bond + q_mpo_auxiliary = q_ket_bond
```

The prototype does not encode these signs into block keys. Orientation is
derived from boundary membership, while `Dual<S>` toggles the charge
contribution independently. `Dual<S>` preserves the underlying basis
occurrence order and dimensions; operations which observe a charge obtain
`dual(q)`. The `DualSpace` concept makes explicit duality discoverable.

`repartition<Side, End>(tensor)` bends one leftmost or rightmost factor across
the morphism boundary and toggles its explicit duality. The current bosonic
operation returns an lvalue view: it does not copy, reorder, or rewrite
numerical payload. It may sort a transformed logical key index and may expose a
moved dense axis through `layout_stride`. Rvalues are rejected so the view
cannot retain a dangling storage reference.

`permute<Axis...>(tensor)` uses flattened domain-then-codomain positions and
gives the source factor at every output position. The first bosonic operation
permutes within domain and codomain only. Moving a factor across the boundary
is not a permutation: use `repartition`. A non-edge factor can therefore be
moved to an edge with `permute` and then bent explicitly with `repartition`.

For bosonic U(1), the per-block bend factor is one. Non-abelian, fermionic, or
braided extensions may add pivotal, `1j`, recoupling, or exchange factors to
the transformed block metadata and kernel lowering. They must not silently
fold those factors into the stored tensor elements.

## 5. Block Coordinates and Dense Axes

Logical boundary legs, block-selection coordinates, and numerical dense axes
are distinct. A space contributes a key coordinate only when a block must
select among several structural choices. It contributes a dense axis only when
the selected choice contains numerical multiplicity:

| Space | Block-key coordinate | Dense-block axis | Charge contribution |
|---|---|---|---|
| `BlockSpace` | canonical sector index | sector dimension | sector `QNum` |
| `IrregularSpace` | stored block index | block dimension | block `QNum` |
| `LocalSpace` | ordered state index | none | selected state `QNum` |
| `QNumSpace` | none | none | fixed stored `QNum` |
| `DenseSpace` | none | full dense extent | identity |

An `IrregularSpace` key uses the block index, not only its `QNum`, because
repeated and out-of-order blocks are distinct structural occurrences.
Likewise, repeated charges in a `LocalSpace` remain distinct state indices.

`BlockKey` is an opaque value with deterministic lexicographic ordering. Its
initial representation may contain a fixed-size array of the selected factor
indices, but callers must not depend on that representation. A later coupling
or fusion descriptor must be addable without changing operation-level code.

Coordinates are ordered first by the domain factors from left to right, then
by the codomain factors from left to right, skipping factors that do not have a
block coordinate. Lexicographic key ordering compares those coordinates in that
order. A future coupling descriptor follows the factor coordinates and
participates only after they compare equal. This order is part of the logical
key contract even when storage uses a different lookup or memory arrangement.

Dense-block axes follow the same boundary order while independently skipping
factors without numerical multiplicity. Thus an MPS block for
`BlockSpace × LocalSpace -> BlockSpace` is a matrix, while a four-`LocalSpace`
MPO block is a rank-zero mdspan backed by one scalar. The dense mdspan extents
are derived entirely from the key and boundary spaces. Block records do not
independently own a second, potentially inconsistent copy of those extents.

## 6. Legal, Stored, and Resident Blocks

These are separate states:

- **Legal:** the key is in range and satisfies the symmetry selection rule.
- **Stored:** the tensor has a numerical block for that legal key.
- **Resident:** the stored block is available at a particular execution
  location.

Using an illegal key in an operation which requires legality is a contract
error. A legal but unstored block has the exact mathematical value zero. A
stored but non-resident block is not zero; later distributed storage must
obtain it from its recorded location.

The query operations are total and non-throwing with respect to key legality:

- `is_legal(key)` returns `false` for an out-of-range or charge-forbidden key;
- `contains(key)` returns `false` for either an illegal key or a legal unstored
  key;
- `find_block(key)` returns no block in either case.

Operations which require a legal or stored key retain explicit preconditions.
In particular, sparse-builder insertion rejects an illegal key and
`block(key)` requires that the key is stored. This separation lets callers
probe untrusted or generated keys without weakening construction and direct
access contracts.

This distinction permits the same mathematical boundary to support both MPS
and environment representations:

- an A-matrix storage normally stores every legal block;
- an MPO site stores a sparse subset of legal scalar blocks;
- an environment stores the subset produced by its MPO and MPS connectivity.

Completeness is a guarantee of the storage implementation, not another
top-level `BlockTensor` template parameter.

## 7. Storage Concepts

Every storage implementation models `BlockTensorStorage`. A positive
refinement advertises complete global coverage:

```cpp
template<class Storage>
concept CompleteBlockStorage =
    BlockTensorStorage<Storage> &&
    Storage::stores_all_legal_blocks;

template<class Storage>
concept SparseBlockStorage =
    BlockTensorStorage<Storage> &&
    !Storage::stores_all_legal_blocks;
```

Ordinary `BlockTensorStorage` permits legal blocks to be absent. A separate
public pattern tag such as `SparseBlocks` or `CompleteBlocks` is unnecessary.
Concrete storage types expose the trait as part of their contract.

The base concept requires the semantic capabilities used by `BlockTensor`, not
one prescribed container representation. A conforming storage provides:

- its scalar value type and an exact match with the owning tensor's `T`;
- `stores_all_legal_blocks` as a compile-time boolean;
- ownership of every buffer referenced by its stored-block records;
- canonical iteration over stable `(BlockKey, block binding)` records;
- lookup by `BlockKey` without inserting or changing structure;
- const and mutable dense-block descriptors with the correct value semantics;
- retainable read and write access to the buffers needed by submitted work.

Construction supplies an already validated canonical key sequence and the
boundary-derived extent of every block. Storage must preserve those keys and
extents exactly. The concrete record, lookup container, mapping type, buffer
partition, and access-token types are associated implementation details and may
differ between host, CUDA, and distributed policies. The first implementation
must express these requirements as checked concepts or equivalent compile-time
diagnostics; it must not rely only on prose or naming convention.

`stores_all_legal_blocks` always describes the global mathematical block set.
A future distributed complete storage may keep only some blocks on each MPI
process while its replicated global layout accounts for every legal block.

The storage implementation owns three related mechanisms:

```text
stored key set
    canonical complete enumeration or canonical sparse subset

layout
    block key -> buffer, offset, mapping, location

buffers
    allocations, ownership, access state, and lifetime
```

They are constructed together, but their meanings remain distinct.
Completeness may enable one packed allocation; it does not require one.

## 8. Initial Host Storage

The first implemented slice provides two sparse host policies. Both use:

- canonical block-key ordering;
- zero initialization for newly allocated numerical values;
- stable block bindings after construction.

`SeparateSparseBlockStorage` owns one column-major dense `Tensor` for each
stored key. `PackedSparseBlockStorage` owns one contiguous allocation and one
canonical offset per stored key. Both accept an explicit key set, reject
illegal keys, sort it canonically, and reject duplicate keys before allocation.
The packed policy is the likely default for sparse scalar-block MPO sites; the
separate policy is the simplest baseline for early block operations and
comparison tests.

The next storage step is a distinct complete host policy which enumerates every
legal key and models `CompleteBlockStorage`. A sparse value may happen to store
all legal keys, but that runtime fact does not give its policy the compile-time
complete-storage guarantee.

A zero extent does not remove a structurally legal key. For example,
`DenseSpace(0)` contributes no key coordinate but produces a zero-element
dense axis in each otherwise legal block. Complete storage retains the block
record and a valid zero-size dense descriptor, although several such records
may share the same packed offset and consume no scalar storage.

By contrast, an empty `BlockSpace`, `IrregularSpace`, or `LocalSpace` has no
coordinates. A boundary containing such a factor has an empty Cartesian key
space and therefore no legal or stored blocks. Empty domain and codomain values
represent the tensor unit and produce one coordinate-free key; its block is a
rank-zero scalar.

The complete and sparse forms may share implementation helpers. They remain
distinct concrete storage contracts so `CompleteBlockStorage` can provide a
compile-time guarantee to algorithms and to the `AMatrix` alias.

The layout must not assume that a future complete storage has only one buffer.
A CPU/GPU placement may partition all legal blocks among a host allocation and
one or more allocations per device.

## 9. Construction and Structural Mutation

Block structure is fixed after construction:

- boundary factor types and order do not change;
- space sectors, states, and extents do not change;
- the stored key set and buffer bindings do not change;
- numerical block values remain mutable;
- leg labels may be changed explicitly without changing structure.

A later sparse builder will collect keys and optional initial values before
allocating:

```cpp
auto builder = make_block_tensor_builder<T>(symmetry, domain, codomain);
builder.add_block(key);
auto tensor = std::move(builder).build(storage_configuration);
```

Complete construction derives the key set from the boundary and selection
rule. Operations which change sectors, truncate bonds, or change the stored
pattern construct or replace a tensor rather than inserting blocks into a live
layout.

For the implemented slice, the sparse constructor accepts the complete stored
key list directly and performs the same validation before allocating.

This rule keeps block descriptors stable and avoids structural mutation while
async work may still retain buffer access.

## 10. Block Access

The owning tensor exposes boundary and stored-block structure without
pretending to be one dense tensor:

```cpp
tensor.symmetry();
tensor.domain();
tensor.codomain();
tensor.order();
tensor.stored_block_count();

tensor.is_legal(key);
tensor.contains(key);
tensor.find_block(key);
tensor.block(key);       // precondition: stored
tensor.blocks();         // canonical stored-block order
```

Const access yields read-only block value semantics. Mutable access is
available only from a mutable tensor.

`BlockTensor` does not model dense `TensorView` or `MdspecLike`. Each returned
block does. A host block may resolve immediately to an mdspan; CUDA and other
deferred storage will return an mdspec carrying the appropriate data
descriptor.

Generic algorithms iterate stored blocks or use `find_block`. Algorithms
constrained to `CompleteBlockStorage` may use legal-key iteration and direct
block access without presence probes.

### Implemented morphism operations

The first `permute<Axis...>` and `repartition<Side, End>` operations return
zero-copy lvalue views. Both transform boundary values, canonical logical keys,
and dense mdspan axis order while retaining every source payload address.
Permutation is currently the bosonic symmetric-category operation with unit
exchange factor. Non-bosonic exchange remains an explicit future braid.

## 11. Async and Kernel Lowering

`Async<BlockTensor<...>>` is the owner-level asynchronous value. Async-ness is
not a block storage kind.

Storage buffers use retainable ownership and access state so block descriptors
remain valid for submitted work. The initial packed host implementation may
hazard the whole allocation conservatively. Later multi-buffer layouts provide
finer independent scheduling without changing the tensor's logical boundary.

Numerical lowering follows:

```text
BlockTensor operation
    -> validated block worklist
    -> fixed block mdspec operands
    -> backend-domain leases
    -> mdspan dense kernels
```

BlockTensor-level operations select legal block combinations. Dense kernels do
not decide quantum-number compatibility and must never receive a silent dense
projection of the entire symmetry-aware tensor.

## 12. Initial Tensor-Network Aliases

The intended aliases are conceptually:

```cpp
template<class T, CompleteBlockStorage Storage = /* packed host complete */>
using AMatrix = BlockTensor<
    T,
    Domain<BlockSpace, LocalSpace>,
    Codomain<BlockSpace>,
    Storage>;

template<class T, SparseBlockStorage Storage = /* packed host sparse */>
using MpoSite = BlockTensor<
    T,
    Domain<LocalSpace, LocalSpace>,
    Codomain<LocalSpace, LocalSpace>,
    Storage>;
```

The `MpoSite` domain order is `(left auxiliary, ket)` and its codomain order is
`(right auxiliary, bra)`. All four factors are `LocalSpace`, so each stored MPO
block is scalar and the stored block pattern is inherently sparse.

An `MPO` is a chain of `MpoSite` values. It validates exact equality of each
site's right auxiliary space with the next site's left auxiliary space. The
chain is a separate owner and is not another spelling of one `BlockTensor`.

Environment aliases use the same three-factor kinds as an A-matrix but select
a storage implementation which permits missing legal blocks.

## 13. Deferred Extensions

The first prototype deliberately defers:

- non-abelian fusion multiplicities and coupling descriptors;
- recoupling, braiding, and category-dependent block factors;
- lazy transpose, conjugate, adjoint, scaled, and non-bosonic repartition views;
- CUDA, mixed CPU/GPU, and MPI storage;
- replicated immutable block tensors;
- coalescing and placement policies beyond canonical packed host storage;
- block SVD, sector-aware singular-value selection, and factor materialization;
- runtime-order Python-facing tensors.

The opaque block key, explicit boundary orientation, storage concepts, and
mdspec block interface are required now specifically so these features remain
additive.

Although SVD is deferred, its structural contract is already fixed. It follows
the staged design in
[Block SVD: decomposition, selection, and materialization](block_tensor.md#9-block-svd-decomposition-selection-and-materialization):
factorize the required logical blocks, select metadata-bearing singular
triplets, and only then allocate the final output tensors. Convenience helpers
may compose those stages, but must not construct a full `BlockTensor` and
subsequently mutate its block structure to truncate it.

## 14. Required Prototype Tests

The implemented tests cover both sparse policies at tensor orders zero through
four. They exercise the tensor-unit scalar, rank-one identity-sector
selection, `BlockSpace` block extents, `LocalSpace` coordinate-only states,
coordinate-free charged `QNumSpace` factors, explicit dual spaces, MPS matrix
blocks, rank-zero MPO blocks, canonical sparse keys, repeated local charges,
legal-but-unstored blocks, mutable and const access, empty boundary factors,
packed offsets, dense-block extent validation, and construction failures.
Repartition tests additionally prove transformed-key sorting, direct
transformed selection rules, unchanged payload addresses, dense-axis stride
permutations, const propagation, and left/right bend involution. Permutation
tests cover boundary types and labels, key resorting, dense strides, inverse
permutations, and the tensor unit.

The completed host prototype must additionally test:

- arbitrary mixtures and orderings of all five supported space types;
- explicit symmetry preservation for empty and dense-only boundaries;
- an empty block, irregular, or local factor producing no legal keys;
- a zero-extent dense factor retaining legal zero-size block records without
  allocating scalar elements;
- rejection of mismatched or unsupported symmetries;
- U(1) and direct-product-U(1) legality from domain/codomain orientation;
- repeated `LocalSpace` charges and repeated `IrregularSpace` blocks remaining
  distinct keys;
- `DenseSpace` contributing no key coordinate and one neutral dense axis with
  its full extent;
- deterministic block-key ordering;
- complete storage allocating every legal block;
- sparse storage treating missing legal blocks as zero;
- probes returning false or no block for illegal and legal-unstored keys;
- illegal sparse-builder keys and direct access to unstored keys being rejected;
- block extents matching boundary-derived dimensions;
- const and mutable block access;
- label changes preserving storage and changing exact boundary equality;
- A-matrix, MPO-site, and environment aliases satisfying their storage
  requirements.

No test may validate the prototype by flattening symmetry-aware state and then
feeding that dense representation back into the tensor-network path.

## 15. Durable Prototype Invariants

- Domain and codomain orientation determines bosonic abelian charge signs.
- Boundary order, block-key coordinate count, and dense-block order are
  independent compile-time properties.
- All symmetry-bearing spaces agree with the tensor's explicit `Symmetry`.
- Dense axes never erase or invent a symmetry charge.
- Illegal, unstored, and non-resident blocks are distinct conditions.
- Missing legal blocks mean exact zero only for storage which permits them.
- Complete storage is complete globally, not necessarily at each location.
- Block structure and bindings are stable after construction.
- Individual blocks lower through mdspec and execution-domain leases.
- Whole-tensor dense materialization is never an implicit symmetry fallback.
- `Dual<S>` preserves basis occurrences and dimensions while dualizing observed
  charges.
- Bosonic repartition changes boundary, key, and mdspan mapping metadata without
  changing numerical payload storage.
- Bosonic permutation is zero-copy within each morphism side; crossing sides is
  an explicit bend.
