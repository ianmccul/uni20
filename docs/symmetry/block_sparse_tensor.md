# Block-Sparse Tensors and Layout

**Status:** active design note. The current branch does not provide the complete
symmetry-aware block-sparse Tensor described here.

This is a draft design note. It records the intended design of the uni20
block-sparse tensor and its layout abstraction, the data model the rest of the
execution stack operates on. It is design direction, not a description of current
implemented behavior — there is no real tensor class yet beyond embryonic
experiments.

Related notes:

- `docs/symmetry/block_tensor_prototype.md` — the first host-only bosonic
  abelian implementation contract.
- `docs/symmetry/block_tensor.md` — the symmetry-typed `BlockTensor` design that refines this note.
- `docs/architecture/storage_kind_and_location.md` — storage memory kind (type) vs location (runtime).
- `docs/architecture/ordering_and_backend_lowering.md` — ordering ownership; two-clocks lifetime rule.
- `docs/backends/cuda/epoch_design_draft.md` — GPU per-buffer hazard model.
- `docs/symmetry/block_coalescing.md` — single-axis GEMM grouping over the layout.
- `docs/architecture/execution.md` — how tensor, dispatch, and scheduling compose.
- `docs/symmetry/qnum.md` — quantum numbers and symmetry.
- `docs/architecture/backend_dispatch.md` — kernel dispatch the tensor feeds.

## Summary

The block-sparse tensor is the missing data model, and its **layout** is the
linchpin of the whole stack. The design is two-level: a lightweight dense block
(an `mdspan` leaf living on one device) and a block-sparse container of such
blocks. The **layout** object carries two things at once — *which device / MPI
rank owns each block* and *how blocks are arranged in memory* (coalescing-aware
strides) — and it is the single object that tensor storage, kernel dispatch,
device scheduling, MPI distribution, and the placement planner all consume.
(Which location fields the per-block record carries is policy-typed in the
refined design — see `block_tensor.md` §1/§6.) Blocks are
strided views into shared buffer storage tracked by the async runtime; symmetry
metadata is part of the type. The tensor is **policy-free mechanism**: it can map
any block to any device. Placement — including "tail on CPU" — is the planner's
policy, layered on top (see `../architecture/execution.md`).

## Two-level structure

- **Dense block (leaf).** A lightweight `mdspan` (namespace `stdex::`) over dense
  storage that lives on exactly one device. This is mostly C++ detail — extents,
  `layout_stride`, accessor/memory-space policy — not a heavyweight object. A
  block never spans devices.
- **Block-sparse container.** A structured collection of blocks plus the symmetry
  metadata that says which sectors exist and the layout that says where each block
  lives and how it is stored. The container is the "tensor"; the blocks are its
  populated sectors.

## Typed legs

Tensors are general order-N, but the legs are **typed**, because different leg
kinds carry different sparsity:

- **BlockSpace** — symmetry-decomposed virtual index (e.g. an MPS bond). Block-
  sparse: only some charge sectors are populated, with varying sector dimensions.
- **IrregularSpace** — ordered segmented virtual index whose `QNum`s may repeat,
  for example after projecting out part of a symmetry without immediately
  repacking the existing blocks.
- **LocalSpace** — a small dense physical index (e.g. 2 for spin-½, 4 for a
  Hubbard site). Treated as *inherently sparse* through the selection rule: for a
  fixed combination of the other legs, only some local values are allowed. It is
  dense-but-small and regular, which makes it the natural coalescing axis (see
  `block_coalescing.md`).
- **QNumSpace** — one fixed irrep coordinate with degeneracy extent one. It is
  used for an explicit operator or boundary charge and contributes that charge
  to the selection rule.
- **DenseSpace** — an ordinary dense index where neither block-sparsity nor a
  selection rule applies. Mostly relevant for the dense-tensor degenerate case.

The current DMRG blocks are a three-leg prototype `A(i, s, j)`: `i`/`j` BlockSpace
virtual bonds, `s` a LocalSpace physical index.

### Selection rule is part of the type

The selection rule (U(1): `q_column = q_row + q_local`; `q_bra = q_ket +
q_operator`; `q_left_virtual + q_ket = q_right_virtual + q_bra`) defines which
sector combinations are nonzero. It is a property of the tensor type, not an
optimization. There is **no dense fallback** on a symmetry-typed path: dense
reference helpers must be explicitly named (`to_dense_reference`) and must not feed
back into symmetry-typed state. Any operation that would silently drop
`LocalSpace`/`BlockSpace`/`IrregularSpace`/`QNum`/leg-orientation, or flatten a block-sparse tensor
to dense on a symmetry path, is a correctness bug. Coalescing (below) does *not*
violate this: it is a kernel-execution lowering that round-trips through the
correct sector structure, never a representation change.

## Storage model: blocks as views into shared buffers

Storage is owned by **buffers** that the async runtime hazard-tracks; a tensor's
blocks are **`layout_stride` views into those buffers**, each with its own
`(offset, strides)`. This has three consequences that must be designed in from the
start because they are expensive to retrofit:

1. **Dual views over one buffer.** The same bytes are simultaneously addressable
   as "individual blocks" (for symmetry-correct assembly and scatter) and as a
   wider "coalesced operand" (for a single GEMM). This is what makes single-axis
   coalescing zero-copy when the layout cooperates — see `block_coalescing.md`.
2. **Hazard tracking is buffer-with-subviews, not per-block-allocation.** A
   coalesced op reads/writes a strided region spanning several logical blocks, so
   the epoch/buffer layer must reason about a buffer and its sub-ranges, not just
   whole-object RAW/WAR. Plan-time coalescing keeps this simple: the coalesced op
   *is* the node, so individual-block ops don't run concurrently against the same
   region. Where coalesced and uncoalesced ops share a buffer, conservative
   buffer-granularity hazarding is correct.
3. **Token-pins-storage lifetime.** Following the two-clocks rule in
   `../architecture/ordering_and_backend_lowering.md`: buffer storage lifetime is owned by the
   pending-completion token (released on completion, not on coroutine return).
   This is mandatory for the deferred-sync ("launch returns immediately, sync at
   next access") execution mode, where no coroutine frame is left holding the
   inputs alive.

## The layout object (the linchpin)

A uni20 **layout** is `(block-index → device / MPI-rank map) + (coalescing-aware
memory arrangement)`. One object, consumed everywhere:

| Consumer | What it reads from the layout |
|---|---|
| Tensor storage | block → buffer/offset/strides (memory arrangement) |
| Kernel dispatch | block location → backend/device selection |
| Device scheduling | which device's scheduler/stream an op targets |
| MPI | the distribution, and deterministic block identity for tag derivation |
| Placement planner | the device / MPI-rank map is the planner's *output* |

Because the same object threads all of these, getting it right makes the pieces
compose, and getting it wrong makes them fight at every boundary. The layout type
should be designed first, before the tensor that contains it.

## Distribution from day one

The device / MPI-rank map exists even in the single-device case (everything
trivially on device 0), so single-device, multi-GPU, and multi-MPI-rank are the
same code path with different maps — not a CPU-first design with distribution
bolted on later. For MPI, block identity must be globally deterministic so both
MPI ranks derive the same tag for an edge without communicating (graph construction, including
reverse-mode AD, must be replicated/deterministic — see
`../architecture/ordering_and_backend_lowering.md`).

## Decisions made

- Two-level model: lightweight dense `mdspan` block + block-sparse container.
- Typed legs: `BlockSpace` / `IrregularSpace` / `LocalSpace` / `QNumSpace` /
  `DenseSpace`; general order-N; `LocalSpace` is a distinct kind, treated as
  sparse via the selection rule.
- The selection rule and all symmetry metadata are part of the type; no dense
  fallback on a symmetry path.
- Blocks are `layout_stride` views into shared buffer storage; storage is owned by
  buffers, not by blocks.
- Hazard tracking is buffer-with-subviews; storage lifetime is pinned by the
  completion token.
- The layout is a single object = device / MPI-rank map + coalescing-aware memory
  plan, consumed by tensor/dispatch/scheduling/MPI/planner.
- The tensor is policy-free mechanism; placement is the planner's policy.

## Open questions

- ~~**LocalSpace representation.**~~ *Resolved:* `LocalSpace` is a first-class
  immutable `Space` model containing an ordered list of local-state `QNum`s.
  Repeated charges and state order are preserved.
- **Layout memory plan generality.** How much memory-arrangement freedom does the
  layout expose — arbitrary per-block strides, or a constrained interleave/
  contiguous descriptor that guarantees the common coalescing groups are
  stride-addressable? (See the interleaved-vs-contiguous tradeoff in
  `block_coalescing.md`.)
- **Hazard granularity.** Buffer-granularity (conservative, simple) vs sub-range
  overlap tracking (precise, more parallel) for mixed coalesced/uncoalesced access
  to one buffer.
- **Block-index identity for MPI.** What canonical, replication-stable identity is
  used to derive tags (sector quantum numbers + per-leg sector indices?), and how it
  survives truncation/growth across DMRG sweeps.
- ~~**How leg orientation is represented**~~ — *resolved* in
  [Spaces, Duals, and Tensor Morphisms](spaces_duals_and_morphisms.md):
  concrete-space/`Dual<S>` object duality is independent of an occurrence's position
  in the ordered `Domain` or `Codomain`. Moving between sides is explicit wire
  bending, not a metadata flip.
