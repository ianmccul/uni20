# Storage Memory Kind vs Location

**Status:** current dense CUDA storage/location semantics plus active design for
future distributed and block-sparse placement.

This note records both the implemented dense CUDA model and the design direction
for how future Uni20 tensor storage should model *where data lives*. Sections on
distributed and block-sparse placement remain design work.

Related notes:

- `docs/architecture/ordering_and_backend_lowering.md` — who owns ordering; events/MPI as lowerings.
- `docs/backends/cuda/epoch_design_draft.md` — GPU per-buffer hazard model and the explicit
  host/device transfer model.
- `docs/architecture/kernel_dispatch.md` — compile-time type eligibility through
  `kernel_accepts_types` and runtime acceptance through `try_kernel`.
- `docs/symmetry/qnum.md` — symmetry/QNum and block-key invariants.
- `docs/symmetry/block_tensor.md` — the container-level refinement: the
  `TensorStorage` / `BlockTensorStorage` two-policy split and policy-typed
  per-block records (§1, §5–§6 there).

## Summary

"Where does this data live" is two independent axes, and they must not be
collapsed into one:

- **Memory kind** — host / device / unified. A **compile-time** property of the
  storage type. It selects which kernels are legal through
  `kernel_accepts_types`. It is also what keeps symmetry/QNum type guarantees
  intact.
- **Location** — which device ordinal or process owns a storage object. A
  **runtime** value attached to storage. For deferred dense device spans, the
  data descriptor identifies storage whose resources carry the device identity;
  the eventual leased pointer need not encode it. Distributed block placement
  is container metadata rather than a dense mdspan accessor concern.

Conflating them breaks the design in one of two ways:

- If location is encoded in the *type*, you cannot support multi-GPU or MPI: the
  device ordinal and the MPI rank are not knowable at compile time.
- If kind is pushed to *runtime* (type-erased storage), you lose compile-time kernel
  dispatch **and** the symmetry/QNum guarantees that uni20 treats as correctness
  invariants (see `docs/symmetry/qnum.md`).

So: kind is a type, location is a value.

## Current state

- **The kind axis is implemented.** `Tensor` is parameterized by a
  `StoragePolicy` (`VectorStorage` by default; see `src/uni20/tensor/`).
  `CudaStorage` is the current device-resident policy. Other memory kinds
  should use the same type-level mechanism rather than becoming runtime tags.
- **CUDA Tensor placement is implemented for owning dense tensors.**
  `CudaStorage` owns a typed `CudaBuffer`, while `CudaTensor::mdspec()`
  carries a `CudaBufferView` descriptor, mapping, and eventual pointer accessor.
  The descriptor identifies the buffer and element offset without exposing a
  usable data handle. The buffer resolves its device through the
  `DeviceResources` it borrows. A scoped process-wide CUDA runtime owns one
  canonical resource set for each enrolled device; ordinary Tensor construction
  uses the configured default. The default backend selector remains stateless.
- **The prototype validates the runtime-location model.** The vendored
  TensorContraction engine models location entirely at runtime:
  `DeviceMatrixView::deviceId_` (an `int`), device-local execution resources,
  and a `MatrixHeader` POD that is "safe to send via MPI as raw bytes," with
  per-block placement layouts chosen at runtime. This is the "data lives on a
  device" model in practice, with multi-GPU + MPI placement as runtime
  decisions.

## Interaction with symmetry / block-sparse tensors

Block identity (the QNum sector of each block) is the **logical key** — it is
host/type-level and governs which contractions are legal. Location is a **runtime
annotation** on each block's storage. The two must coexist without either erasing
the other:

- A placement decision may *move* a block between devices/ranks, but must never drop
  its QNum/block key.
- This mirrors uni20's symmetry-metadata rule for block-sparse paths: carry logical
  block keys and never silently drop sector metadata (see `docs/symmetry/qnum.md`). The
  kind-vs-location split is the storage-side counterpart of that invariant.

A block-sparse tensor therefore needs *per-block* location, not a single
whole-tensor location, because different sectors can be placed on different
devices / MPI ranks. `../symmetry/block_tensor.md` §1 refines this into a two-policy split:
the leaf `TensorStorage` carries the compile-time kind, and the container's
`BlockTensorStorage` policy fixes which location fields the per-block record
carries (none for single-node `HostOnly`, MPI rank and device ordinal for
`Mpi<Cuda>`) — distribution as a compile-time *capability*, placement as runtime
*values*.

## Relationship to the ordering model

- **Kind** selects which backend lowering applies (host kernel vs CUDA kernel vs
  unified path).
- **Location** feeds the *placement policy*, which sits **above** the hazard/ordering
  layer described in `ordering_and_backend_lowering.md`. The ordering layer must
  remain correct for any placement the policy chooses; placement decides *where*
  blocks live and *when* to communicate, and emits ordinary operations into the
  async DAG.

## Cross-location moves are explicit operations

Changing a tensor's location is never implicit. A host↔device transfer or a
cross-rank transfer is a scheduled operation in the async DAG — a read of the source
storage and a write of the destination storage — exactly as in the explicit
upload/readback model of `../backends/cuda/epoch_design_draft.md` and the MPI lowering of
`ordering_and_backend_lowering.md`. The fallback for a *missing device kernel* is
likewise an explicit scheduled `D2H → host kernel → H2D` path, never a hidden
blocking host round-trip (same principle as "no silent dense fallback").

Unified memory, if supported later, is a distinct **kind** (e.g.
`Tensor<T, Rank, UnifiedStorage>`) with its own coherence rules — not an
implicit behavior of device storage.

## Selector Boundary

The backend selector chooses an ordered set of implementations. It does not
duplicate intrinsic operand placement. Explicit selector overrides may still
carry genuine operation context such as a CUDA stream, MPI communicator,
workspace policy, algorithm option, or multiprecision setting.

## Open questions

- How is location represented? Candidates: an `(MPI rank, device_ordinal)` pair, or
  an opaque `Device` handle that the scheduler resolves. The representation must be
  cheap to attach to every block.
- How should a future block-storage policy reuse `CudaBufferView` and
  `CudaStorage` without forcing one allocation per logical block?
- How does per-block location coexist with whole-tensor APIs that today assume a
  single storage (slicing, views, assignment semantics)?
- How does location participate in (de)serialization for checkpointing and for the
  POD-header MPI transfer path?
