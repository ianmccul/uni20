# Bosonic Abelian BlockTensor Prototype

**Status:** active design contract with immediate host storage, packed CUDA
storage, and the first async sparse host slice implemented. It defines the
remaining bosonic abelian vertical slice. The broader
[BlockTensor design](block_tensor.md) records later CUDA, MPI, recoupling, and
view requirements.

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
- `ParallelSeparateSparseBlockStorage`, with the same separate ownership and
  synchronous scheduler-batch execution across independent output blocks;
- `PackedSparseBlockStorage`, with canonical offsets into one contiguous leaf
  allocation;
- `ParallelPackedSparseBlockStorage`, with the same packed representation and
  synchronous scheduler-batch execution across disjoint output blocks;
- `PackedCompleteBlockStorage` and `ParallelPackedCompleteBlockStorage`, which
  canonically allocate every symmetry-legal block in one packed allocation;
- `PackedDiagonalBlockStorage`, with one contiguous host buffer containing only
  the generalized diagonal of each explicitly stored logical block;
- `AsyncSeparateSparseBlockStorage`, with one independently scheduled
  `Async<Tensor>` per stored block;
- mutable and const `TensorView` access to stored blocks, using mdspan for
  immediate storage and descriptor-backed mdspec for async storage;
- delegation of the dense leaf backend list through each block storage policy;
- a generic `Dual<S>` value adaptor whose quantum-number observations are dual;
- zero-copy bosonic compile-time permutations within each morphism side;
- zero-copy bosonic left/right `repartition` views with transformed canonical
  keys and strided dense-block axis permutations; and
- synchronous adjacent pairwise contraction through ordinary tensor-level
  kernel dispatch into selectable local sparse output storage;
- synchronous adjacent grouped contraction over one or more paired boundary
  factors, including simultaneous dense-axis lowering;
- scheduler-batch contraction grouped by independently writable output block;
- structure-preserving copy, zero, scaling, addition, AXPY, inner-product, and
  norm operations over immediate, mapped, and per-block async values; and
- the first storage-selected async lowering for rank-two dense block GEMMs.

Most structural operations have a local-storage path. Fixed packed CUDA
BlockTensors support resident vector operations, dense-kernel lowering,
pairwise contraction, per-charge SVD, and selected-factor materialization.
Mapped permutation and repartition views retain an lvalue source reference and
therefore cannot outlive their source tensor. They preserve immediate data
handles or async block epoch identity while changing only logical and mapping
metadata. Payload element writes remain valid, but assigning or structurally
modifying a source invalidates every view transitively built from it. The slice
does not yet provide the full numerical block-operation surface, packed async
host hazards, or multi-device factorization.

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
operation returns a borrowed view: it does not copy, reorder, or rewrite
numerical payload. It may sort a transformed logical key index and may expose a
moved dense axis through `layout_stride`. A mapped view owns its boundary, key,
and dense-block descriptor metadata, so temporary mapped views can be composed
without retaining intermediate view objects. It also retains the ultimate
source leaf-allocation context independently of the block descriptors. An empty
mapped CUDA tensor therefore still identifies its device resources for result
allocation and provider planning. Owning tensor rvalues remain rejected because
destroying the owner would invalidate the numerical payload and allocation
context.

`permute<Axis...>(tensor)` uses flattened domain-then-codomain positions and
gives the source factor at every output position. The first bosonic operation
permutes within domain and codomain only. Moving a factor across the boundary
is not a permutation: use `repartition`. A non-edge factor can therefore be
moved to an edge with `permute` and then bent explicitly with `repartition`.
`as_block_tensor_view(tensor)` is the identity form: it copies the source
boundary, key, and block-descriptor metadata into a value-retained borrowed
view without changing axis order or copying numerical payload.

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

`BlockTensorStorageFor<Storage, T, KeyCount, DenseOrder>` requires mutable and
const block values satisfying `MutableRankedTensorView` and `RankedTensorView`.
Each view exposes its immediate or descriptor-backed mdspec without itself
modeling `MdspecLike`. The positive execution and placement refinements are:

```cpp
LocalBlockStorageFor<Storage, T, KeyCount, DenseOrder>
ImmediateLocalBlockStorageFor<Storage, T, KeyCount, DenseOrder>
AsyncLocalBlockStorageFor<Storage, T, KeyCount, DenseOrder>
DistributedBlockStorageFor<Storage, T, KeyCount, DenseOrder>
```

The local refinement excludes distributed ownership without constraining the
leaf memory domain or acquisition model. The immediate refinement additionally
requires actual mdspans. The async refinement requires one
`Async<Tensor>`-like value per block in this first implementation. The
distributed refinement is the compile-time seam for a later placement-aware
policy. Completeness, sparse layout, execution mode, leaf memory kind, and
distribution remain separate properties even when one concrete `Storage` type
combines them.

The base concept requires the semantic capabilities used by `BlockTensor`, not
one prescribed container representation. A conforming storage provides:

- its scalar value type and an exact match with the owning tensor's `T`;
- `stores_all_legal_blocks` as a compile-time boolean;
- a block execution policy selecting serial or scheduler-batch work grouping;
- ownership of every buffer referenced by its stored-block records;
- canonical iteration over stable `(BlockKey, block binding)` records;
- lookup by `BlockKey` without inserting or changing structure;
- const and mutable dense-block TensorViews with the correct value semantics;
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

The implemented slice provides six sparse host policies and two packed-complete
host policies. All use:

- canonical block-key ordering;
- unspecified numerical values after allocation; every operation must overwrite
  or explicitly initialize a block before reading it;
- stable block bindings after construction.

`SeparateSparseBlockStorage` owns one column-major dense `Tensor` for each
stored key. `PackedSparseBlockStorage` owns one contiguous allocation and one
canonical offset per stored key. Both accept an explicit key set, reject
illegal keys, sort it canonically, and reject duplicate keys before allocation.
The packed policy is the likely default for sparse scalar-block MPO sites; the
separate policy is the simplest baseline for early block operations and
comparison tests.

`PackedDiagonalBlockStorage` also accepts an explicit sparse key set, but it
stores only entries for which every dense index within a block is equal. A
rank-N block with extents `(d0, ..., dN-1)` therefore owns
`min(d0, ..., dN-1)` values; rank-zero blocks own one value. The accessor
presents the full logical block, returning exact zero for structural
off-diagonal entries. This representation covers ordinary rectangular
diagonal morphisms and higher-rank copy tensors without allocating structural
zeros. It does not infer which logical blocks exist: the BlockTensor key set
continues to carry that symmetry structure explicitly. `diagonal_values(key)`
provides direct access to the compressed values. Each full block view models
`DiagonalMdspecLike`: it remains an ordinary logical rank-N `MdspecLike`, while
`diagonal_components(block.mdspec())` exposes the rank-one strided physical
values to specialized dense kernels. Through the full mutable block view,
assigning zero to an off-diagonal element is permitted, while assigning a
nonzero value violates the storage precondition.

`ParallelSeparateSparseBlockStorage` uses the same independently owning block
representation as `SeparateSparseBlockStorage`, but selects synchronous
scheduler-batch execution for block operations. Pairwise contraction groups
the worklist by result ordinal and executes one batch item per result block.
Every contribution to that block stays in the same item and retains canonical
accumulation order, so no two items write the same dense block. The active
scheduler controls serial debug ordering or TBB parallelism, and the result is
fully computed when `contract` returns.

`ParallelPackedSparseBlockStorage` applies the same synchronous execution
contract to `PackedSparseBlockStorage`. The packed allocation and its offsets
remain fixed during a batch, and distinct output-block items write disjoint
element ranges. It is useful for temporary vectors that need both canonical
packed allocation and block-level scheduling. Host storage exposes ordinary
offset mdspans. `CudaStorage` creates one physical allocation partitioned into
one logical `CudaBuffer` per block. Those logical buffers share allocation
lifetime but retain independent completion ledgers, so different scheduler
items may submit work for disjoint blocks on different CUDA streams.

`PackedCompleteBlockStorage` uses the same contiguous representation but does
not accept an explicit stored-key set. Its owning `BlockTensor` enumerates every
symmetry-legal key in canonical order and constructs the packed offsets once.
`ParallelPackedCompleteBlockStorage` adds the same synchronous disjoint-block
batch execution policy. For CUDA leaf storage it uses the same independently
synchronized logical block partition as the parallel sparse policy. These
policies provide the compile-time
`CompleteBlockStorage` guarantee used by transient DMRG centers and their
Krylov vectors; sparse MPO and environment storage remains independent.

Aligned packed storage records both block-start offsets and the exclusive end
of each numerical payload. Elements between a payload end and the next block
start are storage padding. Construction initializes that padding to numerical
zero while leaving block payloads unspecified. Allocation-wide zeroing,
scaling, copying between identical layouts, AXPY, inner products, and norms may
therefore process the complete physical allocation: each operation preserves
zero padding. A nonzero fill is different and must skip the gaps or restore
them to zero before an allocation-wide reduction. Multiplication by a finite
factor preserves zero padding and may use the allocation-wide path. A non-finite
factor would turn a padded zero into a NaN, so scale, assign-scale, and AXPY use
the ordinary blockwise path for that rare case.

For packed CUDA storage, compatible fixed linear operations use one aggregate
buffer access and one elementwise kernel rather than dispatching one kernel per
block. Exact-layout inner products and norms use one cuBLAS level-one call.
Inputs with different key patterns or physical offsets, mapped BlockTensor
views, unsupported scalar operations, and allocations larger than the cuBLAS
level-one integer ABI retain the ordinary blockwise path.
The aggregate access publishes one shared completion to every logical block
ledger, so subsequent independent block scheduling remains valid.

`AsyncSeparateSparseBlockStorage` owns one `Async<ColumnMajorTensor<...>>` per
stored key. Its `block(key)` result is a TensorView whose borrowed data
descriptor identifies that exact async value. The view does not keep the owning
`BlockTensor` alive. Submitted operations obtain `ReadBuffer` and `WriteBuffer`
capabilities from the async value; those capabilities retain the storage,
selected epoch, and failure propagation state. This literal per-block
`Async<Tensor>` representation is useful when individual numerical values must
remain pending. It is distinct from the lower-overhead synchronous batch policy.

A sparse value may happen to store all legal keys, but that runtime fact does
not give its policy the compile-time complete-storage guarantee.

`legal_block_keys()` returns those keys in canonical order for structural
planning or explicit sparse materialization. Like
`legal_block_count()`, it scans the full Cartesian product of key-bearing
boundary factors and is not intended as a repeated hot-path query. A
charge-indexed implementation may replace this scan without changing the
observable key order.

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
tensor.block_by_ordinal(ordinal);

tensor.async_block(key); // only AsyncLocalBlockStorageFor
tensor.async_block_by_ordinal(ordinal);
```

`BlockTensorView` is the common algorithm contract for this structure. It
requires immutable boundary metadata, a canonical sorted stored-key sequence,
stable ordinal lookup, and a dense `TensorView` for each stored block. Owning
`BlockTensor` and mapped permutation or repartition views model the same
concept; mutability, immediate access, and per-block async access are separate
refinements. `BorrowedBlockTensorView` separately states that block descriptors
materialized from a temporary view remain valid after that view is destroyed;
the ultimate numerical payload owner must still outlive them.

Const access yields read-only block value semantics. Mutable access is
available only from a mutable tensor.

`BlockTensor` does not model dense `TensorView` or `MdspecLike`. Each returned
block models `TensorView`, but not `MdspecLike`; its `.mdspec()` exposes the
normalized dense descriptor. A host block's mdspec may already be an mdspan.
CUDA and other deferred storage use descriptor-backed mdspec metadata.

Canonical block ordinals are stable for the lifetime of the tensor because
block structure is immutable. They are execution bindings, not logical block
identity: worklists retain `BlockKey` values and acquire ordinals only during
storage lowering.

Generic algorithms iterate stored blocks or use `find_block`. Algorithms
constrained to `CompleteBlockStorage` may use legal-key iteration and direct
block access without presence probes.

### Implemented linear operations

The first structure-preserving numerical surface is:

```cpp
set_zero(tensor);
scale(tensor, factor);

copy(output, input);
assign_scale(output, factor, input);
add(output, lhs, rhs);
add_inplace(output, input);
axpy(output, factor, input);

auto sum = add(lhs, rhs);
auto value = inner_product(lhs, rhs);
auto magnitude = norm(tensor);
```

`set_zero` is an overwrite operation implemented with constant fill. It does
not multiply the existing values by zero, because newly allocated storage may
be uninitialized and IEEE NaNs would survive multiplication by zero.

These operations accept the `BlockTensorView` concept and refine it only for
the access they perform: immediate host blocks, mutable blocks, or independent
per-block async timelines. Owning `BlockTensor` values and zero-copy mapped
views therefore use the same numerical surface. Operations which allocate a
new tensor retain an explicit storage-policy choice for the owning result. That
policy must preserve the operands' immediate or per-block async execution mode;
cross-mode materialization is an explicit operation rather than an implicit
effect of a linear operation.

All operands must have exactly equal symmetry, domain, and codomain values.
Boundary labels and explicit duality therefore participate in compatibility.
Legal but unstored blocks represent exact zero, which fixes the structural
rules:

- zero and in-place scaling retain the existing stored-key pattern;
- a returned sum stores the union of the input key sets;
- a fixed output must already contain every key needed by the operation;
- extra blocks in an overwritten fixed output are set to zero;
- in-place addition and AXPY leave output-only blocks unchanged; and
- inner products use the intersection of the two stored-key sets.

`DiagonalBlockTensorView` identifies values whose stored numerical blocks are
generalized diagonals. A diagonal fixed or selected output accepts only
diagonal inputs; dense outputs may consume diagonal inputs. Accepted diagonal
updates operate on `diagonal_components(...)` directly rather than iterating
or assigning structural off-diagonal zeros. Norms likewise reduce only the
stored components. When a dense output consumes a diagonal input, non-finite
scaling preserves exact structural off-diagonal zeros through a component-only
slow path; ordinary finite scaling retains the single-kernel dense path.

Fixed-output structural requirements are checked before any numerical block is
modified. Block structure remains immutable. An unrestricted elementwise
transform is intentionally absent because a function for which `f(0) != 0`
would require materializing every legal but unstored block.

Immediate parallel storage executes independent block operations through the
scheduler batch interface. Per-block async storage schedules mutations on the
corresponding block epoch. The current `inner_product_host` and `norm_host`
reductions are explicitly blocking synchronization points; their rank-zero
Tensor wrappers are `inner_product` and `norm`.

`krylov::BlockTensorVectorOps<Tensor>` uses this surface to define a Krylov
vector space from one owning prototype. Membership requires the exact frozen
symmetry, boundary values, and stored-key pattern. Consequently,
`allocate_like` preserves block metadata: sparse policies reproduce the frozen
key set, while complete policies rederive the same canonical legal-key set from
the frozen boundaries. A matrix-free `matvec` cannot silently widen the vector
space or flatten it into a dense tensor.
`krylov::BlockTensorMatrixFreeOps<Tensor, Operator>` adds an owned output-first
operation callable. It validates input and output membership before every
apply, while the callable supplies the actual Hamiltonian or environment
contraction.

### Implemented morphism operations

The first `permute<Axis...>` and `repartition<Side, End>` operations return
zero-copy borrowed views. Both transform boundary values, canonical logical
keys, and dense mdspec axis order while retaining every immediate payload
address or async block epoch identity. Their owned descriptor metadata permits
direct composition through temporary mapped views; the ultimate tensor owner
must still outlive the result.
Permutation is currently the bosonic symmetric-category operation with unit
exchange factor. Non-bosonic exchange remains an explicit future braid.

The first pairwise contraction is:

```cpp
auto result = contract<left_axis, right_axis>(left, right);
contract<left_axis, right_axis>(output, left, right);
```

It contracts only the rightmost codomain factor of `left` with the leftmost
domain factor of `right`; the explicit axes must name those adjacent factors.
The two space values must be exactly equal, including labels and explicit
duality. The result boundary is:

```text
domain   = left.domain + right.domain_without_first
codomain = left.codomain_without_last + right.codomain
```

For a coordinate-bearing contracted space, only blocks with the same basis or
sector coordinate are paired. A `BlockSpace` degeneracy axis is contracted by
the dense strided kernel. `LocalSpace` and `QNumSpace` add no artificial dense
axis; their contributing scalar or external-axis blocks are multiplied and
accumulated directly. With sparse output storage, the operation builds blocks
only from stored contributing pairs. With complete output storage, it allocates
every legal result block and sets legal blocks without contributions to exact
zero. Neither path materializes a whole-tensor dense bridge.
`PackedSparseBlockStorage<>` is the default host output; callers may select a
different immediate host output policy as the third template argument. The
packed-complete policies use this path for transient complete centers. Parallel
policies group all contributions to one output block into one synchronous batch
item and return only after every output block is complete.

The fixed-output form preserves the output's existing structure, which is the
required convention for Krylov `matvec`. It validates the exact result symmetry
and boundaries and verifies that the output stores every worklist result key
before modifying numerical data. The first contribution overwrites each
produced block, later contributions accumulate, and output-only blocks become
zero. Therefore an exact-pattern output incurs no preliminary whole-vector
zero pass. Output storage must not overlap either input; direct same-object
aliases are rejected.

An immediate-host grouped form contracts a planar adjacent factor group:

```cpp
auto result = contract_adjacent<count>(left, right);
contract_adjacent<count>(output, left, right);
```

It contracts the last `count` codomain factors of `left` with the first
`count` domain factors of `right`. Every paired space value must be exactly
equal. All paired block coordinates are matched together and all paired dense
degeneracy axes are passed to one ordinary dense tensor contraction. The
single-factor `contract<left_axis, right_axis>` interface remains the explicit
axis form and has the same sparse semantics.

An owning synchronous contraction result uses packed sparse storage over the
left input's leaf storage by default. If that leaf exposes an allocation
context, the result is constructed in the same context. For CUDA this preserves
the selected device and resource installation rather than allocating from the
process-wide default device. Explicit `OutputStorage` selection remains
available and uses the source context when that storage accepts it.

When both inputs use `AsyncSeparateSparseBlockStorage`, the same `contract`
front end retains the left input's async storage policy by default and returns
its `BlockTensor` structure immediately. The first async numerical lowering
accepts rank-two dense blocks with a `BlockSpace` degeneracy contraction and
natural GEMM output axis order. It schedules `assign_product` for the first
contribution to an output block and `add_product` for later contributions.
Different output blocks have independent epoch queues; contributions to one
output block serialize on that block's writer timeline. The selected output
storage supplies the backend list, and submission requires an active async
scheduler. Other dense orders, coordinate-only contracted legs, mapped input
views, nontrivial output permutations, packed async storage, device-only blocks,
and custom accessors remain outside this first async overload.

## 11. Async and Kernel Lowering

`Async<BlockTensor<...>>` and storage-level asynchronous blocks solve different
problems. The outer wrapper is optional and is needed only when tensor structure
itself is pending, for example after dynamic truncation or async loading. CPU
block parallelism, CUDA submission, and MPI distribution do not require an
outer async wrapper when the boundary and stored key set are already known.

Storage controls physical layout, complete-versus-sparse guarantees, dense leaf
storage, execution context, and placement. The planned controlled families are:

- serial CPU storage with immediate host mdspans and reliable inline kernels;
- parallel CPU storage with immediate blocks and synchronous scheduler batches
  grouped by independently writable output;
- per-block async storage for values that need independent epoch timelines;
- CUDA storage with packed block data, CUDA mdspec lowering, one shared physical
  allocation, and device completion tracked independently for each logical
  block buffer; and
- `MpiBlockStorage<LocalStorage>`, with globally replicated logical placement
  metadata and each mutable dense block wholly owned by one MPI process.

MPI storage delegates local work to its node-local serial, parallel CPU, or
CUDA storage. A dense leaf kernel sees only local resolved blocks; it never sees
a remote or distributed dense operand. MPI therefore does not imply
`Async<BlockTensor>`.

Storage buffers use retainable ownership and access state so scheduled work
outlives borrowed block descriptors safely. Packed CUDA storage validates its
disjoint static partition at construction. Child buffer descriptors retain the
shared allocation while preventing a whole-allocation access ledger from
bypassing their per-block completion domains. A future packed async host policy
may use an analogous subrange or coalesced-group hazard representation without
changing logical boundaries.

Numerical lowering follows:

```text
BlockTensor operation
    -> validated block worklist
    -> storage binding and placement records
    -> block TensorViews carrying dense backend policy
    -> fixed block mdspec operands plus retained owner/epoch capabilities
    -> backend-domain leases
    -> mdspan dense kernels
```

BlockTensor-level operations select legal block combinations. Dense kernels do
not decide quantum-number compatibility and must never receive a silent dense
projection of the entire symmetry-aware tensor.

The mathematical planner is storage-independent. Its worklist carries logical
input and output keys. Storage lowering binds those keys to stable local
ordinals now and later to location-aware records. The first implementation
records whether a binding initializes or accumulates an output block before
submitting any numerical work.

## 12. Initial Tensor-Network Aliases

The implemented aliases in `src/uni20/tensor_network/site_types.hpp` are:

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

The `MpoSite` domain order is `(left auxiliary, ket)` and its codomain order is
`(right auxiliary, bra)`. All four factors are `LocalSpace`, so each stored MPO
block is scalar and the stored block pattern is inherently sparse.

An `MPO` is a chain of `MpoSite` values. It validates exact equality of each
site's right auxiliary space with the next site's left auxiliary space. The
chain is a separate owner and is not another spelling of one `BlockTensor`.

The first `LocalTwoSiteEffectiveHamiltonian` applies a local operator to a fixed
`TwoSiteCenter` through mapped physical-leg bends and
`contract_adjacent<2>`. It supplies the output-first callable required by
`krylov::BlockTensorMatrixFreeOps`. The general `TwoSiteEffectiveHamiltonian`
joins two `MpoSite` values and two `MpoEnvironment` values into a fixed-center
R/A/B/C coefficient plan. It snapshots and coalesces the sparse
`f(r,a,b,c)` entries. The current backend prepares its right-first `(B,C)`
grouping, output order, and intermediate workspace once when the effective
Hamiltonian is constructed. Immediate host and packed CUDA centers retain
their respective leaf storage in that workspace. Repeated Krylov applications
reuse that state and batch independent output blocks through dispatched dense
contractions; the path neither flattens the center nor constructs a high-rank
BlockTensor intermediate. The logical plan retains canonical block keys rather
than storage
ordinals; backend preparation binds those keys to the concrete operands. Each
application overwrites every stored output block, filling blocks without a
contributing term with zero to preserve the implicit-zero sector semantics.

`make_identity_mpo_environment`, `extend_left_environment`, and
`extend_right_environment` provide the first environment-construction
primitives. They join only stored environment, MPS, and MPO keys, allocate the
reachable output pattern, and lower each dense contribution through ordinary
tensor contraction dispatch. The bra MPS site is conjugated lazily. Primary
overloads permit distinct bra and ket sites; convenience overloads use one MPS
site for both. An explicit-context identity constructor supports
descriptor-backed storage. Derived environments inherit the prior environment's
allocation context whenever the result storage accepts it, so an enrolled
non-default CUDA device is not replaced by the runtime default.

## 13. Deferred Extensions

The first prototype deliberately defers:

- non-abelian fusion multiplicities and coupling descriptors;
- recoupling, braiding, and category-dependent block factors;
- lazy transpose, conjugate, adjoint, scaled, and non-bosonic repartition views;
- mixed CPU/GPU, packed async, multi-device CUDA, and MPI storage;
- replicated immutable block tensors;
- coalescing and placement policies beyond canonical packed host storage;
- asynchronous, generalized CUDA, and distributed block-SVD factorization and
  materialization;
- runtime-order Python-facing tensors.

The opaque block key, explicit boundary orientation, storage concepts, and
mdspec block interface are required now specifically so these features remain
additive.

The initial immediate-host block SVD accepts any `ImmediateBlockTensorView` and
follows the staged design in
[Block SVD: decomposition, selection, and materialization](block_tensor.md#9-block-svd-decomposition-selection-and-materialization):
assemble and factorize one matrix per conserved charge, select
metadata-bearing singular triplets, and only then allocate the final output
tensors. One decomposition can independently materialize kept, discarded, or
paired-null triplets, together with side-specific full null spaces. Missing
stored blocks remain explicit zero submatrices in the sector assembly metadata.
Convenience helpers may compose those stages, but must not construct a full
`BlockTensor` and subsequently mutate its block structure to truncate it.
In particular, a two-site tensor whose natural contraction result has a 3/1
boundary may be zero-copy repartitioned to 2/2 and passed directly to
`block_svd`; the fallback does not require a new owning intermediate.

## 14. Required Prototype Tests

The implemented immediate tests cover all three immediate sparse policies at tensor
orders zero through four. They exercise the tensor-unit scalar, rank-one identity-sector
selection, `BlockSpace` block extents, `LocalSpace` coordinate-only states,
coordinate-free charged `QNumSpace` factors, explicit dual spaces, MPS matrix
blocks, rank-zero MPO blocks, canonical sparse keys, repeated local charges,
legal-but-unstored blocks, mutable and const access, empty boundary factors,
packed offsets, dense-block extent validation, and construction failures.
Repartition tests additionally prove transformed-key sorting, direct
transformed selection rules, unchanged payload addresses, dense-axis stride
permutations, const propagation, and left/right bend involution. Permutation
tests cover boundary types and labels, key resorting, dense strides, inverse
permutations, and the tensor unit. Pairwise contraction tests cover both input
storage policies, packed and separate output, dense matrix multiplication,
adjacent two-factor contraction with two dense degeneracy axes,
cross-sector sparsity, repeated local-state accumulation into a rank-zero
result, exact-space rejection, planar external-boundary order, and synchronous
scheduler batching grouped by output block under both a recording debug
scheduler and a TBB scheduler.
Linear-operation tests cover sparse zero semantics, union-pattern result
construction, fixed-output containment checks before mutation, exact boundary
compatibility, complex conjugate-linear inner products, stable multi-block
norms, mapped views, all immediate storage policies, and per-block async
updates and reductions.
Block-SVD tests cover global selection across charge sectors, repeatable kept
and complement materialization, paired and side-specific null spaces,
reconstruction for real and complex scalars, repeated-charge boundary
fragments, implicit zero blocks, and empty selections.
The Krylov adapter tests freeze a two-sector U(1) vector structure, reject
boundary or stored-pattern changes, and run that BlockTensor through the native
symmetric Lanczos solver without dense projection. Its matrix-free operation
uses fixed-output BlockTensor contraction with a block-diagonal U(1) operator.
The first DMRG-shaped integration test applies a U(1) two-site Heisenberg
operator, obtains the singlet through native BlockTensor Lanczos, repartitions
the center to 2/2, materializes its staged block SVD, and reconstructs the
center without losing either charge sector.
The general effective-Hamiltonian test factors the same interaction into
neutral and charge-changing MPO channels, compiles environment/MPO paths into
the fixed center pattern, rejects a non-closed pattern, and verifies
nontrivial dense environment multiplication with scheduler-batched output.
Environment tests construct multi-sector identity boundaries, update a U(1)
Heisenberg product state from both directions, verify complex bra conjugation,
accumulate repeated physical paths, and exercise nontrivial dense bond sizes.
Contraction tests also cover fixed-output structure preflight, output-only zero
blocks, direct alias rejection, async output epoch ordering, and complete
packed output allocation with zero-valued legal blocks that have no contribution.
Async-storage tests additionally cover mdspec const/mutable semantics, stable
epoch identity through permutation, numerical block GEMM, one blocked sector
not preventing an independent sector from completing, and failure propagation
only to a dependent output block.

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
- Individual blocks lower through mdspec, retained epoch/owner capabilities,
  and execution-domain leases.
- Whole-tensor dense materialization is never an implicit symmetry fallback.
- `Dual<S>` preserves basis occurrences and dimensions while dualizing observed
  charges.
- Bosonic repartition changes boundary, key, and mdspec mapping metadata without
  changing numerical payload storage or async block epoch identity.
- Bosonic permutation is zero-copy within each morphism side; crossing sides is
  an explicit bend.
- Pairwise contraction matches logical block coordinates before invoking dense
  kernels and preserves every uncontracted boundary occurrence.
