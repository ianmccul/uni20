# BlockTensor: Symmetric Block-Sparse Tensor Design

**Status:** central active design for the future symmetry-aware `BlockTensor`;
not current implemented API behavior.

The first implementation is governed by the narrower
[Bosonic Abelian BlockTensor Prototype](block_tensor_prototype.md). Where the
illustrative future interfaces in this note differ, the prototype document is
authoritative for the initial API.

This is a draft design note. It records the agreed design of the uni20
`BlockTensor` — the symmetry-typed, block-sparse tensor that is the central data
model the execution stack operates on — together with its dense-block leaf, its
storage/layout foundation, its view algebra (transpose / conjugate / scale), and
the forward-compatibility seams that keep multiplicity-free abelian symmetries now
extensible to outer multiplicities, fusion trees, and the fully general braided
case later. It is design direction, not a description of current implemented
behavior.

Related notes:

- `docs/symmetry/block_tensor_prototype.md` -- the scoped host-only bosonic
  abelian implementation contract.
- `docs/symmetry/spaces_duals_and_morphisms.md` — the canonical space,
  duality, domain/codomain, and contraction model.
- `docs/symmetry/block_sparse_tensor.md` — the two-level tensor + layout linchpin (this note refines it).
- `docs/symmetry/axis_labels_and_braiding.md` — user-facing axis-label policy and explicit braid semantics.
- `docs/architecture/storage_kind_and_location.md` — memory kind (compile-time) vs location (runtime).
- `docs/architecture/ordering_and_backend_lowering.md` — Async owns ordering; CUDA/MPI as lowerings; two-clocks lifetime.
- `docs/symmetry/block_coalescing.md` — single-axis GEMM grouping (the LocalSpace coalescing axis).
- `docs/architecture/execution.md` — mechanism vs policy; the planner owns placement.
- `docs/architecture/backend_dispatch.md` — rationale for type eligibility,
  runtime attempts, and clean decline.
- `docs/symmetry/qnum.md` — `Symmetry`, `QNum`, `QNumList`, `BlockSpace` (the symmetry foundation this builds on).
- `docs/architecture/kernel_dispatch.md` — the `backend_list` walk and the async/scheduler seam this tensor feeds.
- `docs/backends/mpi/persistent_dispatch.md` — persistent immutable objects; the replication model behind `ReplicatedBlockTensor`.

## Summary

`BlockTensor` is a general tensor of arbitrary order (number of legs) whose legs
are **typed** by symmetry role, whose populated sectors are **dense blocks** that are `layout_stride` views
into a small set of shared **buffers**, and whose every block carries a **scalar
coupling factor** in addition to its `(buffer, offset, strides, location)`. The
tensor is **policy-free mechanism**: a block lives fully on one device but may be
placed on any device / MPI rank, and *which* device is a runtime value in the layout.
Storage is split across **two policies** (§1): a compile-time leaf `TensorStorage`
(memory kind, `Cpu`/`Cuda`) on the dense blocks, and the container's
`BlockTensorStorage` (the `Storage` parameter), which fixes the form of the
per-block metadata record — including, for `Mpi<…>` policies, distribution.
**Location** values and the per-block scalar are runtime. Boundary-preserving
transforms are **views** when their categorical lowering permits it: a
compile-time morphism-boundary transform plus lazy per-block op-state and
scalars that lower directly to the BLAS `op` (`N`/`T`/`C`/`R`) and `alpha`.
The per-block scalar is the
multiplicity-free abelian shadow of fusion-tree recoupling, so the same slot that
carries a category-defined bend or adjoint phase today carries a recoupling
transform later. The
mechanism is designed for CPU + CUDA + MPI from day one; the first implementation
target is `HostOnly`, with immutable host replication (MPO/environment tensors) as
the first MPI use, ahead of device residence.

## 1. Scope and the two-policy storage split

Storage is governed by **two policies**, one per level of the two-level model (§2).
(This replaces an earlier conception that held distribution entirely outside the
type — that conception conflated the storage of an individual dense tensor with
the storage of the block-sparse container.)

- **`TensorStorage`** — the storage policy of a dense block (and of the ordinary
  dense `Tensor`): which memory the data lives in, `Cpu` / `Cuda`. A
  **compile-time** memory kind; it selects legal kernels through
  `kernel_accepts_types`. MPI is
  *not* a `TensorStorage`: a dense tensor striped across MPI ranks is conceivable
  but off-roadmap — dense blocks are sized to be resident on a single node.
- **`BlockTensorStorage`** — the storage policy of the container (the `Storage`
  parameter, §4–§5). It specifies the **form of the per-block metadata record**
  (§6) and, through it, the blocks' `TensorStorage`. `HostOnly` records
  `(buffer, offset, strides)`; `Mpi<Host>` adds the owning MPI rank; `Mpi<Cuda>`
  adds `(MPI rank, device ordinal)` and stores its blocks with
  `TensorStorage == Cuda`.

So **distribution is part of the container's type**, as a compile-time
*capability* of the `BlockTensorStorage` policy, while **placement is runtime**:
under `Mpi<Cuda>` every block lives at some `(MPI rank, device)`, and those are
runtime values in the per-block record. Dispatch keys on the capability
(`is_distributed_v` and friends are compile-time traits of the policy); the
planner and the MPI layer consume the values.

Symmetry safety does **not** depend on the memory kind being compile-time: it lives
in the **leg kinds** (§3), which are always compile-time, so a block moving
CPU↔GPU can never drop its `QNum`. This is what allows a policy like
`HostOrDevice` (§5) to make the kind a runtime per-buffer tag (behind a finite,
closed `Host`/`Device` switch) without losing the `AGENTS.md` §3.5 /
`../architecture/storage_kind_and_location.md` symmetry guarantees.

## 2. Two-level model and the buffer foundation

- **Dense block (leaf).** A lightweight `TensorView` whose mdspec records
  extents, mapping, accessor semantics, and immediate or deferred data identity.
  A block never spans devices (no intra-block slicing across devices). The
  selected backend acquires the execution-domain lease and resolves an mdspan.
- **Block-sparse container (`BlockTensor`).** A structured collection of block
  *views* plus the symmetry metadata (which sectors exist) and the layout (where
  each block lives, how it is stored, and its coupling factor). The container is the
  tensor; the blocks are its populated sectors.

Storage is owned by **buffers** the async runtime hazard-tracks; blocks are
`layout_stride` views into those buffers. This is the `block_sparse_tensor.md`
storage model and carries its three consequences: dual views over one buffer
(individual blocks *and* a wider coalesced operand), buffer-with-subviews hazard
granularity, and **token-pins-storage lifetime** (buffer lifetime released on
completion-token completion, not coroutine return — mandatory for deferred-sync
device execution, per the two-clocks rule in `../architecture/ordering_and_backend_lowering.md`).
The "completion token" is concrete, not an abstraction: the CUDA event recorded at
submission (or the MPI request). Managing it is the `BlockTensorStorage` policy's
job — an async-capable policy frees a device buffer only after the event signals
the buffer is no longer in use, and then frees asynchronously.

`BlockTensor` sits directly on the buffer-with-subviews foundation, not on the
user-facing dense `Tensor` owner. Block access materializes a TensorView over a
storage record so ordinary dense dispatch can retain leaf backend policy until
mdspec normalization. Dense `Tensor` is a sibling owner over the same buffer
kind, not a required part of the container representation.

## 3. Morphism boundary and space kinds

**Terminology.** An individual ordered index occurrence is a **leg**; the total
leg count is the tensor's **order**. The word "rank" is avoided on the tensor
side — it survives only in mdspan's own `rank()` (which means the order) — and
**"MPI rank" is always written qualified**, never bare.

`BlockTensor` is a morphism with an ordered `Domain<...>` and ordered
`Codomain<...>`. These boundary value templates are implemented for concrete
`Space` factors. They preserve factor types and order, expose structural space
values read-only, and permit explicit per-leg label changes. Empty boundaries
represent the tensor unit.

Membership in either boundary is independent of whether the object is
represented as a concrete space or `Dual<S>`. The generic `Dual<S>` adaptor
preserves basis occurrences and dimensions while dualizing quantum-number
observations; `DualSpace` exposes that status to generic code. The obsolete
`CoBlockSpace`, `CoLocalSpace`, and `CoQNumSpace`
proposal conflated these two axes and is not part of this design. See
[Spaces, Duals, and Tensor Morphisms](spaces_duals_and_morphisms.md) for the
canonical rules, including wire bending and all-out boundary orientation.

Each concrete space models `Space` directly:

| Space decomposition | Sectors | Degeneracy | Grounded on | Example use |
|---|---|---|---|---|
| `BlockSpace` | many `(QNum, dim)` | dim ≥ 1, varies | implemented `BlockSpace` | MPS virtual bond |
| `IrregularSpace` | ordered `(QNum, dim)`, repeats allowed | dim ≥ 1, varies | implemented `IrregularSpace` | symmetry-projected bond before repacking |
| `LocalSpace` | ordered QNums (repeats allowed) | 1 per explicit state | implemented `LocalSpace` | physical site index |
| `QNumSpace` | exactly one QNum | 1 | implemented `QNumSpace` | irrep label of an irreducible operator |
| `DenseSpace` | — | dense extent | implemented `DenseSpace` | plain dense or bundled axis |

The structural portion of a space is immutable, while its string leg label may
be changed. Exact equality includes both structure and label. Worked tensor
aliases are:

```cpp
// MPS A-tensor: left_bond tensor physical -> right_bond
template<class T, CompleteBlockStorage Storage>
using AMatrix = BlockTensor<
    T,
    Domain<BlockSpace, LocalSpace>,
    Codomain<BlockSpace>,
    Storage>;

// MPO site: left auxiliary tensor ket -> right auxiliary tensor bra
template<class T, SparseBlockStorage Storage>
using MpoSite = BlockTensor<
    T,
    Domain<LocalSpace, LocalSpace>,
    Codomain<LocalSpace, LocalSpace>,
    Storage>;

// Irreducible tensor operator carrying one explicit operator irrep
template<class T, BlockTensorStorage Storage>
using IrreducibleOperator = BlockTensor<
    T,
    Domain<BlockSpace, QNumSpace>,
    Codomain<BlockSpace>,
    Storage>;
```

The alias fixes space kinds and boundary order. Each tensor value carries the
actual space values, so the same `AMatrix` type describes every site while
adjacent tensors carry equal copies of the intended labeled bond space.

Boundary legs have two independent storage properties:

| Space | Block-key coordinate | Dense-block axis |
|---|---|---|
| `BlockSpace` | sector index | sector degeneracy |
| `IrregularSpace` | stored block index | block dimension |
| `LocalSpace` | state index | none |
| `QNumSpace` | none | none |
| `DenseSpace` | none | dense extent |

`LocalSpace` is the natural **coalescing axis** (`block_coalescing.md`): small,
regular, and interleavable. An individual logical block omits that trivial
dense axis; a coalesced internal view introduces or widens the corresponding
axis across several LocalSpace-key values. Coalescing remains a per-device
layout detail and never changes the logical leg structure.

`QNumSpace` stores an **irrep `QNum`**, not a one-dimensional dense space. In
the abelian case the irrep is a 1-D charge shift, but for a non-abelian
irreducible operator the single
`QNum` names an irrep of dimension > 1 whose internal structure is a fusion concern
(Wigner–Eckart: the stored data is the reduced matrix element on the `BlockSpace`
legs; the `QNumSpace` leg contributes via coupling). Degeneracy is 1; irrep
dimension is fusion's job (§8). A `QNumSpace` has neither a selectable block-key
coordinate nor a dense-block axis; its fixed charge remains in the boundary
metadata and selection rule.

The **`DenseSpace` leg** is an ordinary index with no symmetry attached — the degenerate
case, but a load-bearing one. Its motivating use is *an array of tensors of the same
shape*: block-Lanczos vectors, a bundled MPS, or a set of tangent vectors. The
conventional `std::vector<BlockTensor<…>>` is poor here — it replicates the
(identical) layout metadata once per element and leaves the per-element buffers
uncoalesced. A `DenseSpace` leg instead adds one ordinary index to a *single* tensor:
the layout metadata is shared across the bundle, and the storage can coalesce across
it. So `DenseSpace` is both the dense-tensor degenerate case and the right tool for
batched/bundled symmetric tensors.

## 4. The BlockTensor template and its type family

```cpp
template <
    class T,
    class DomainType,
    class CodomainType,
    BlockTensorStorage Storage>
class BlockTensor;
```

- `T` — element scalar type (real or complex).
- `DomainType` and `CodomainType` — ordered boundary space kinds. The tensor's
  order and kinds are **always compile-time** — the order is
  part of the mdspan type (its `rank()`), so there is no real choice here. The
  concrete space values are runtime metadata. A runtime-order tensor for
  the Python surface is a later, *different class* behind the same dispatch
  (plausibly a variant over a small maximum order, or an mdspan extension with
  runtime rank), not a mode of `BlockTensor`.
- `Storage` — the container storage policy (`BlockTensorStorage`, §1/§5),
  whose concrete initial implementations are defined by the prototype design.

The first public template deliberately has no speculative category parameter;
it is fixed to the bosonic abelian scope. The general `BlockTensor` will gain a
compile-time `Category` parameter when a second categorical scope is
implemented. Its exact concept and data interface remain deferred until that
implementation, while the opaque block key retains the required coupling
extension point.

A small type family shares `DomainType`/`CodomainType`/`Storage`, varying only
on ownership and view-state (mirroring the existing `Tensor` / `TensorView`
split):

- **`BlockTensor`** — owning; holds the buffers.
- **`BlockTensorView`** — non-owning; carries op-state (§7) and possibly rebound
  per-block scalars over another tensor's buffers.
- **`ReplicatedBlockTensor`** — implicitly **const**; its blocks are **not
  necessarily uniquely stored**. This does *not* mean every block is copied to
  every MPI rank / device: each block has a canonical location (optionally assigned
  via a `BlockSpace → device / MPI rank` layout at commit), and a device or node that needs a
  non-resident block pulls it from the canonical location on demand and may cache
  it. Intended for environment/MPO tensors whose blocks may be needed in more than
  one place. Constructed by `std::move`-ing a finished `BlockTensor` into it — the
  commit point after which blocks are immutable — so replicas need no coherence
  protocol: the immutable-⇒-free-replication invariant of the persistent-object
  model (`../backends/mpi/persistent_dispatch.md`). It takes the **same**
  boundary and storage types as its source, and its implementation may
  resemble `BlockTensor`, but it is a separate class: mutable-vs-immutable is
  the load-bearing distinction, and the type is what enforces it.

Mutability invariant: a **mutable** block has a location-set of size 1 (exactly one
device); a block may acquire a location-set > 1 (replicated) only once committed
immutable via `ReplicatedBlockTensor`. Taking a write handle requires a singleton
location.

## 5. Storage: the BlockTensorStorage policy

`Storage` is the **container-level** policy (§1). At compile time it fixes (a) the
form of the per-block metadata record (§6), (b) the `TensorStorage` of the dense
blocks, and (c) whether and how the per-block data is wrapped for the async
runtime. Distribution belongs here, and only here:

- **`HostOnly`** — blocks live only in host-addressable memory; the record is
  `(buffer, offset, strides)` with no location fields. The lean default (no device
  types compiled in), and a meaningful *semantic* statement for tensors that
  should never go to device — e.g. a Hamiltonian MPO that is replicated to all
  MPI ranks and never resident on CUDA.
- **`HostOrDevice`** — buffers are a small **closed variant** (`HostBuffer |
  DeviceBuffer`); admits heterogeneous residence (the "tail on CPU" case). Kernel
  dispatch switches on each buffer's kind at block / coalesced-group granularity — a
  finite, closed switch whose arms call fully-typed kernels (`contract_blocks<T,
  CpuBackend>` / `<T, CudaBackend>`), preserving
  `kernel_accepts_types` / `try_kernel` per arm.
- **`Mpi<X>`** — blocks distributed over MPI ranks; `X` is the **node-local**
  policy (`Host`, `Cuda`, `HostOrDevice`). The record adds the owning MPI rank — plus
  the device ordinal when `X` involves a device — and the blocks' leaf
  `TensorStorage` derives from the innermost kind: an `Mpi<Cuda>` tensor stores
  its blocks on the MPI processes with `TensorStorage == Cuda`.

Dispatch reads all of this as **traits** of the policy (`is_distributed_v`, the
record type, the leaf storage type), never by matching the template structure of
the policy's name — so new policies stay additive. And because the first
distributed workload is a replicated host MPO contracted against a distributed
state, **mixed-policy operands are the normal case, not a corner**: the MPI
dispatch predicate is "at least one operand distributed, the rest
distributed-or-replicated", not `is_distributed` on every operand.

**Outer and per-block async are separate layers.** `Async<BlockTensor>` is an
optional owner-level value for cases where tensor structure is itself pending.
It is not required for CPU block parallelism, CUDA execution, or MPI
distribution when boundary and block structure are already immediate. The
container policy decides whether each block is immediate, represented by an
`Async<Tensor>`, or managed by a compact `AsyncArray`-like hazard table. CUDA
and MPI policies likewise select their local scheduling and completion
mechanisms. MPI remains a container-storage capability, not an implication that
the whole `BlockTensor` is an async value.

Distribution is never a *leaf* storage kind: a dense tensor striped over MPI ranks
is conceivable but off-roadmap — dense blocks are sized to be resident on one node.
**MPI is a `BlockTensorStorage` capability; CUDA vs CPU is the memory kind.**

Sequencing: the first MPI need is **immutable host replication** (MPO / environment
tensors, `HostOnly`, made available across MPI ranks via `ReplicatedBlockTensor`) — the
easy persistent-object path — ahead of mutable distributed blocks and ahead of
device residence. So `HostOnly` → MPI replication → `HostOrDevice` / `Mpi<…>`
residence.

## 6. The layout: per-block record

The layout maps each populated block to a record consumed by tensor storage,
dispatch, scheduling, MPI, and the placement planner (it is the single object that
threads all of them — `block_sparse_tensor.md`):

```
block (sector tuple, +coupling key) → (buffer, offset, strides, location, coupling_factor)
```

- `(buffer, offset, strides)` — the `layout_stride` view into shared storage.
- `location` — placement fields whose presence is policy-typed (§1): absent for
  single-node `HostOnly`, owning MPI rank for `Mpi<Host>`, `(MPI rank, device
  ordinal)` for `Mpi<Cuda>`. Runtime values, per block; the planner's *output*.
- `coupling_factor` — a scalar in `T`'s field (§7, §8).

Rebalancing the device split after construction is "rewrite the location/buffer map
+ schedule explicit transfers" — a cross-location move is always an explicit
scheduled async operation (`../architecture/storage_kind_and_location.md`), never implicit. This is
why **> 1 buffer per device must be allowed**: rebalancing and grouping route blocks
into target buffers.

**LocalSpace interleave.** The memory plan should, by default, store blocks that
share `BlockSpace` sectors but differ in `LocalSpace` value interleaved/contiguous,
so the per-device `LocalSpace` coalescing (`block_coalescing.md`) is zero-copy when
chosen. The arrangement must *allow* this from the start (the interleave-vs-pack
tradeoff); it stays an internal perf detail.

**Global structure vs local residency (distributed case).** Block *structure* is
replicated on every MPI rank; block *storage* is local. Replicated everywhere — so block
identity is globally deterministic and tags agree without communication
(`../architecture/ordering_and_backend_lowering.md`) — are the morphism
boundary, concrete space values and labels, structural sector dimensions, the
fusion/coupling layout, and the
**location map** (block → owning `(MPI rank, device)`). Local to each node is only the
buffer binding `(buffer, offset, strides)`, materialized for blocks actually resident
there (owned, or cached read-only replicas). A block "not present on this node"
therefore needs no explicit absent-label: it is simply omitted from the local sparse
buffer map, while the global location map still records where it lives — which is
what lets a node know to fetch it. So a per-node `BlockTensor` carries storage and
per-block records only for present blocks, over a globally-replicated shape.

## 7. Views: transpose, conjugate, scale

Numerical conjugation, compatible boundary transformations, adjoint, and scaling
can be **views** of the same underlying buffers when the category's lowering
permits it. Such a view combines a compile-time morphism-boundary transform,
lazy runtime op-state, and per-block scalars rather than copying data.

We are **inspired by** the C++26 `std::linalg` view adaptors (`transposed`,
`conjugated`, `conjugate_transposed`, `scaled`) but do **not** use them: they are
designed for the general user and do not give enough control over the backend BLAS
library. We roll our own view adaptors so we own the lowering.

- **Compile-time part.** Numerical `conj` preserves the morphism boundary.
  `repartition` bends selected legs across the domain/codomain boundary and
  toggles their explicit space/`Dual<S>` status. `adjoint` reverses the
  morphism according to the category. Permutations and transposes update the
  ordered boundary through the recoupling/braiding rules rather than raw
  `LegList` manipulation.
- **Runtime part.** A lazy per-operand **op-state** and the per-block scalars. The
  op-state defers the actual data transpose/conjugate to the kernel.

The implemented bosonic edge-repartition slice bends only a leftmost or
rightmost factor. It owns a transformed canonical key index which maps to the
source tensor's unchanged physical block bindings. A moved dense axis is a
`layout_stride` permutation over the same data handle. Its numerical factor is
one; later categorical factors belong beside that key/binding metadata rather
than in rewritten payload values. Existing payload elements may be mutated
through a writable view, but replacing or structurally modifying its source
invalidates that view and any view transitively built from it.

The same mapped-view implementation now provides `permute<Axis...>` for
bosonic factor exchanges within domain and codomain. Axis positions are
flattened domain-then-codomain positions, labels move with their factors, and
cross-boundary movement remains an explicit `repartition`. A permutation may
resort its logical key index and expose permuted `layout_stride` blocks, but it
does not move payload.

The op-state lowers directly to the backend BLAS `op` when the provider can
represent it. The baseline Fortran BLAS set is `N`/`T`/`C`; conjugate-only is a
provider extension, for example OpenBLAS `CblasConjNoTrans` or its
develop-branch Fortran GEMM `R` spelling, or else requires materialization:

| op | meaning |
|---|---|
| `N` | normal |
| `T` | transpose |
| `C` | conjugate-transpose |
| `R` | conjugate only (no transpose) — provider extension or prepared fallback |

The per-block scalar lowers to the GEMM **`alpha`** — and it is genuinely
**per-block**, not one uniform alpha as in stdBLAS `scaled`, because the phases are
`QNum`-dependent. Category-defined adjoint, repartition, and recoupling factors
are read through these scalars when they reduce to blockwise phases; a
normalization-convention change is just a different set of per-block scalars
over the same data.

**GEMM lowering.** A contraction `A ⊗ B → C` lowers to per-block GEMMs with `op`
from each operand's op-state and `alpha = s_A · s_B / s_C`. A freshly produced output
block takes `s_C = 1` and folds the rest into its data; accumulation into an existing
`C` with a fixed convention uses the full ratio.

The first synchronous host `contract<left_axis, right_axis>` path implements
one adjacent pair: the rightmost codomain factor of the left operand and the
leftmost domain factor of the right operand. It requires exact space equality,
matches logical sector or local-state coordinates, constructs only sparse
result blocks with stored contributions, and invokes ordinary dense kernel
dispatch for each dense block product. The returned block
TensorViews retain leaf backend selection until fixed operands are normalized
to mdspecs at the ordinary dense operation boundary. `BlockSpace` contracts one degeneracy
axis; coordinate-only spaces produce an outer product or scalar accumulation
without stored dimension-one axes. This first BlockTensor overload requires
immediate host blocks with the default mdspan accessor even though the dense
operation itself uses normal backend dispatch.

`ParallelSeparateSparseBlockStorage` is the first lightweight parallel CPU
policy. It uses the same immediate dense blocks as separate serial storage, but
groups every contribution to one result block into one synchronous scheduler
batch item. Distinct result blocks may execute concurrently; accumulation
within a block remains ordered. Batch items usually invoke immediate dense
operations and do not create coroutine frames. Nested scheduling and
`get_wait()` remain supported for composability but are not the normal lowering.

The first per-block async lowering uses `AsyncSeparateSparseBlockStorage`. The
planner still produces a symmetry-keyed logical worklist, then storage lowering
binds each item to stable input and output ordinals. Rank-two dense blocks lower
through the existing async matrix-product dispatch. Distinct output blocks have
independent timelines; repeated contributions to one output block use its
ordered writer epochs. The operation returns an immediate `BlockTensor` whose
individual block values may remain pending and whose default storage policy is
inherited from the left input. The selected output policy supplies the backend
list, and submission requires an active async scheduler. This first lowering
does not consume mapped input views. It does not require or return
`Async<BlockTensor>`.

**Determinism.** The scalars are deterministically recomputable on every MPI rank from
the block's `QNum`s and the view's op-state, so they do not threaten the
deterministic per-edge tag agreement that SPMD MPI requires
(`../architecture/ordering_and_backend_lowering.md`).

**Read/write.** Scaled and conjugated views are **input-only** (read-only): writing
through them is ill-defined without un-scaling on store. A transposed view (pure
stride flip, no scaling) may be assignable.

## 8. Forward-compatibility seams

The progression is multiplicity-free abelian (coupling multiplicity μ ∈ {0,1}) →
three-leg **outer multiplicity** (μ a scalar count) → N-leg **fusion trees**
(intermediate labels + per-vertex μ) → **braided** (R/F moves). Each step *adds* to
the block's coupling data. Six seams keep all four cases additive rather than a
rewrite:

1. **Block key is an opaque, extensible structure** — not a flat `QNum` tuple. Now:
   `key = (sector index per leg)`. A `CouplingDescriptor` slot is *empty* for
   abelian, a single μ index for three-leg multiplicity, a fusion tree beyond three
   legs. The
   container's block map keys on `(sector tuple, coupling)`, so multiplicity adds
   *blocks* without a structural change. Do not bake "one block per sector tuple."
2. **Selection rule = a `Symmetry` fusion query**, not a hardcoded charge sum.
   Express legality as `dim Hom(q_0 ⊗ … ⊗ q_{n-1} → 1)`. Abelian returns 0 or 1 (and
   *is* the charge-balance check); non-abelian returns the coupling multiplicity. The
   rule lives on the existing `Symmetry` handle.
3. **Permute / transpose / adjoint route through a recoupling hook** — even when it
   is the identity for abelian. Never raw axis reorder, so braiding R-moves / F-moves
   slot in later as a drop-in `Symmetry::recouple(...)`.
4. **Block data shape leaves room for a coupling/fusion extent** distinct from the
   degeneracy extents. A block is logically `degeneracy dims × fusion-multiplicity`;
   the abelian fusion extent is 1 and collapses to a plain dense block, but the shape
   descriptor must not *assume* it absent.
5. **Morphism side and object duality are explicit and independent** (§3) —
   ordered `Domain`/`Codomain` boundaries and concrete spaces/`Dual<S>` provide exactly
   the information that fusion, wire bending, and adjoint operations need.
6. **`QNumSpace` stores an irrep, not a 1-D dense space** (§3) — so a non-abelian irrep's
   internal dimension is a fusion concern, not a baked-in degeneracy of 1.

**The per-block scalar is the abelian shadow of seams #1/#4.** In the
multiplicity-free abelian case there is no fusion-multiplicity dimension to act on,
so the entire recoupling/F-move effect collapses to one scalar per block — exactly
the §7 coupling factor. Later, with μ > 1, it generalizes to a small `μ×μ` recoupling
matrix on the fusion-multiplicity extent; the scalar is the 1×1 case. So
`block.coupling_factor()` returns a scalar today and can return/apply a small matrix
later. Adjoint, repartition, recoupling, and normalization views may therefore
present different coupling factors over the same buffer; numerical conjugation
remains a separate accessor/op-state transformation.

**Bake the interface, trivial values first.** The lowering consumes `op` + per-block
`alpha` from day one; the `HostOnly` first pass can ship with all scalars ≡ 1 and
only transpose implemented, adding category-defined phases and non-trivial coupling
factors as *data*, not a code-path change. Richer scopes add the planned
compile-time `Category` parameter. Its capabilities are not one linear tier:
associator data, fusion multiplicity, and symmetric or general braiding impose
different representation and operation requirements.

## 9. Block SVD: decomposition, selection, and materialization

A block SVD is a staged operation. It does not first construct complete output
`BlockTensor` values and then truncate their sectors or packed storage:

```text
factorize required logical blocks
    -> select sector-labelled singular subspaces
    -> materialize one or more output tensor sets
```

The factorization stage produces an intermediate decomposition object. It owns
or retains the per-block provider results needed to construct the left and
right singular factors, together with the singular values. Every singular
triplet remains identified by:

- the logical source-block or grouped factorization key;
- the resulting bond charge sector and any coupling-channel data;
- its index within the dense block factorization.

Selection operates on that metadata-bearing spectrum. A truncation policy may
compare singular values across sectors, but it must return selected logical
triplets rather than an unlabelled dense index set. It may also partition the
spectrum into several disjoint selections, for example kept states, discarded
states, and a null space.

Only materialization constructs the resulting `BlockTensor` values. It first
derives the output bond spaces and exact per-sector extents from the selection,
then chooses storage and allocates the final buffers. It finally copies or
moves the selected columns, singular values, and rows from the intermediate
block results. A packed storage policy therefore allocates only the blocks and
extents that the result will retain; no final tensor keeps an unused tail from
an earlier, larger block layout.

This staging is the primary interface. Convenience operations may combine
factorization, a standard truncation policy, and materialization for common
cases, but they must lower to the same stages rather than implement eager
construction followed by in-place structural truncation. Materializing several
partitions together may share scans and allocation planning while still
producing structurally independent output tensors.

The intermediate decomposition is symmetry-aware state, not a dense fallback.
Its block keys, sector charges, coupling descriptors, and boundary orientation
remain available until materialization. Missing blocks represented as implicit
zero must also remain distinguishable when a requested partition includes their
null spaces. Provider-specific temporary storage and asynchronous execution are
implementation choices for the later SVD design; neither may erase this
metadata.

## 10. Decisions made

- Name `BlockTensor`; general order-N with a typed morphism boundary over
  `BlockSpace`/`IrregularSpace`/`LocalSpace`/`QNumSpace`/`DenseSpace`
  decompositions.
  `Domain`/`Codomain` is independent of concrete-space/`Dual<S>` status; there are no
  `Co...` leg kinds. `DenseSpace` is both the dense degenerate case and the
  batched/bundled-tensor tool (block Lanczos, tangent vectors), avoiding
  `vector<BlockTensor>`'s metadata duplication.
- Leg vocabulary: an individual ordered index occurrence is a **leg**; the count
  is the tensor's **order**; "rank" is reserved for MPI (always written "MPI
  rank") and for mdspan's own `rank()`.
- Two-level: lightweight `mdspan` dense-block leaf (one device, no cross-device
  slicing) + block-sparse container of `layout_stride` views over shared buffers,
  on the buffer-with-subviews + completion-token foundation.
- `BlockTensor<T, DomainType, CodomainType, Storage>`; boundary order and
  decomposition kinds are compile-time, while concrete space values and their
  labels are runtime metadata. A runtime-order Python-facing tensor is a later,
  separate class behind the same dispatch.
- Type family over shared boundary and storage types: owning `BlockTensor`,
  non-owning `BlockTensorView`, and const `ReplicatedBlockTensor` (separate
  class; move-in commit; canonical block location + on-demand cached replicas;
  mutable ⇒ singleton location / replicated ⇒ immutable).
- Two storage policies: leaf `TensorStorage` (`Cpu`/`Cuda` memory kind; MPI never a
  leaf kind) and container `BlockTensorStorage` = the `Storage` parameter
  (`HostOnly` first, then `HostOrDevice` closed variant, then `Mpi<X>` with `X` the
  node-local policy). The container policy fixes the per-block record form, the
  blocks' leaf storage, and the async wrapping; distribution is a compile-time
  capability of the container policy, placement is runtime; dispatch reads policy
  traits, never template names.
- The first template has no public category parameter and is fixed to the
  bosonic abelian scope. The general template will gain a compile-time
  `Category` parameter when the second scope is implemented. Its opaque block
  key already leaves room for a later `CouplingDescriptor`.
- Per-block layout record `(buffer, offset, strides, location, coupling_factor)`;
  per-block location; > 1 buffer per device allowed; cross-location moves explicit;
  `LocalSpace` interleave permitted for zero-copy coalescing. Distributed: structure
  + location map replicated everywhere, local sparse buffer map for present blocks
  only ("not present" = omitted, no explicit label).
- Compatible transforms are views = compile-time morphism-boundary transform +
  lazy op-state + per-block scalars; **inspired by but not using** C++26
  `std::linalg`; op set `N`/`T`/`C` plus provider-specific or prepared `R`;
  per-block `alpha`; scaled/conjugated views read-only, transposed possibly
  assignable.
- Six forward-compat seams; the per-block scalar is the abelian recoupling case;
  bake the `op`/`alpha` interface now with trivial values, add factors as data.

## 11. Open questions

- **Where the `CouplingDescriptor` / fusion query physically lives** — extend the
  existing `Symmetry` handle (current lean), or a separate `FusionCategory` object
  that `Symmetry` refers to (more honest for braided categories, heavier now).
  The bosonic abelian prototype needs none of it, so this is settled when the
  first nontrivial tier is added.
- **`Category` capability representation** — the compile-time parameter is
  planned, but its concept and concrete names remain open. The real structure
  has independent associator, fusion-multiplicity, and braiding capabilities,
  not a single abelian/non-abelian ladder. Whether the parameter carries the
  corresponding `CouplingDescriptor` type directly also remains open.
- **`ReplicatedBlockTensor` surface** — a near-mirror of `BlockTensor` (it takes the
  same template parameters) or also a thinner read-only handle alias for the common
  case.
- **Hazard granularity** — buffer-granularity (conservative, simple) vs sub-range
  overlap tracking (precise, more parallel) for mixed coalesced/uncoalesced access to
  one buffer (`block_sparse_tensor.md`). Per-element epoch queues alone cannot order a
  coalesced wide view against the individual block views it overlaps.
- **`ReplicatedBlockTensor` commit semantics** — "move in a *finished* tensor" needs a
  definition of finished under the async runtime: all pending epochs on every block
  drained *and* all completion events (the per-buffer tokens — CUDA events recorded
  at submission) signaled (deferred device sync — the two-clocks rule in
  `../architecture/ordering_and_backend_lowering.md`). Is the commit an explicit
  quiesce/await-all barrier, or is the replicated tensor itself an async value whose
  readers simply await the last writes? The `std::move` alone does not provide either.
- **Block-index identity for MPI** — the canonical, replication-stable identity used
  to derive tags (sector `QNum`s + per-leg sector indices?) and how it survives
  truncation / growth across DMRG sweeps.
- **Layout memory-plan generality** — arbitrary per-block strides vs a constrained
  interleave/contiguous descriptor that guarantees common coalescing groups are
  stride-addressable (`block_coalescing.md`).
- **Closed-variant configurability** — how the `HostOrDevice` buffer variant is gated
  by build features so a CPU-only build stays lean.
- **The name `BlockTensor`** — the container must be carefully distinguished from
  the dense *blocks* it contains (which are ordinary `Tensor`s), and the current
  name describes the storage format rather than the semantic guarantee (symmetry).
  `SymmetricTensor`/`SymTensor` name the semantics but collide with the established
  mathematical meaning of "symmetric tensor" (permutation-invariant, `T_ij = T_ji`)
  — a real hazard in a linear-algebra library. Other candidates: `GradedTensor`,
  `InvariantTensor`, `QTensor`. Renaming is cheap until implementation starts.
