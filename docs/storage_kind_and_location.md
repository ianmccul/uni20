# Storage Memory Kind vs Location

This is a draft design note. It records design direction for how Uni20 tensor
storage should model *where data lives*. It is not a description of current
implemented behavior.

Related notes:

- `docs/ordering_and_backend_lowering.md` — who owns ordering; events/MPI as lowerings.
- `docs/gpu_epoch_design_draft.md` — GPU per-buffer hazard model and the explicit
  host/device transfer model.
- `docs/backend_dispatch.md` — compile-time capability / runtime `try_*` dispatch.
- `docs/qnum.md` — symmetry/QNum and block-key invariants.

## Summary

"Where does this data live" is two independent axes, and they must not be
collapsed into one:

- **Memory kind** — host / device / unified. A **compile-time** property of the
  storage type. It selects which kernels are legal and drives the `maybe_can_*`
  capability traits in `backend_dispatch.md`. It is also what keeps symmetry/QNum
  type guarantees intact.
- **Location** — which device ordinal, which MPI rank. A **runtime** value attached
  to the storage. It drives scheduling, placement, and communication.

Conflating them breaks the design in one of two ways:

- If location is encoded in the *type*, you cannot support multi-GPU or MPI: the
  device ordinal and the rank are not knowable at compile time.
- If kind is pushed to *runtime* (type-erased storage), you lose compile-time kernel
  dispatch **and** the symmetry/QNum guarantees that uni20 treats as correctness
  invariants (see `docs/qnum.md`).

So: kind is a type, location is a value.

## Current state

- **Kind axis exists.** `BasicTensor` is parameterized by a `StoragePolicy`
  (`VectorStorage` by default; see `src/uni20/tensor/`). A `GpuStorage` policy is
  the intended mechanism for device-resident tensors, and `backend_dispatch.md`
  already lists "memory-space or storage policy" as a compile-time capability and
  "memory is resident on the required device" as a runtime check.
- **Location axis is missing.** There is no runtime location field on Uni20 storage
  today. This is the concrete gap.
- **The prototype validates the runtime-location model.** The vendored
  TensorContraction engine models location entirely at runtime:
  `DeviceMatrixView::deviceId_` (an `int`), `CudaDeviceContext`, and a
  `MatrixHeader` POD that is "safe to send via MPI as raw bytes," with per-block
  placement layouts chosen at runtime. This is the "data lives on a device" model in
  practice, with multi-GPU + MPI placement as runtime decisions.

## Interaction with symmetry / block-sparse tensors

Block identity (the QNum sector of each block) is the **logical key** — it is
host/type-level and governs which contractions are legal. Location is a **runtime
annotation** on each block's storage. The two must coexist without either erasing
the other:

- A placement decision may *move* a block between devices/ranks, but must never drop
  its QNum/block key.
- This mirrors uni20's symmetry-metadata rule for block-sparse paths: carry logical
  block keys and never silently drop sector metadata (see `docs/qnum.md`). The
  kind-vs-location split is the storage-side counterpart of that invariant.

A block-sparse tensor therefore needs *per-block* location, not a single
whole-tensor location, because different sectors can be placed on different
devices/ranks.

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
upload/readback model of `gpu_epoch_design_draft.md` and the MPI lowering of
`ordering_and_backend_lowering.md`. The fallback for a *missing device kernel* is
likewise an explicit scheduled `D2H → host kernel → H2D` path, never a hidden
blocking host round-trip (same principle as "no silent dense fallback").

Unified memory, if supported later, is a distinct **kind** (e.g.
`Tensor<T, UnifiedStorage>`) with its own coherence rules — not an implicit behavior
of device storage.

## Open questions

- How is location represented? Candidates: a `(rank, device_ordinal)` pair, or an
  opaque `Device` handle that the scheduler resolves. The representation must be
  cheap to attach to every block.
- How does a compile-time `GpuStorage` policy carry a runtime device id — a policy
  type with a runtime member, versus a separate location field on the storage object?
- How does per-block location coexist with whole-tensor APIs that today assume a
  single storage (slicing, views, assignment semantics)?
- How does location participate in (de)serialization for checkpointing and for the
  POD-header MPI transfer path?
