# Uni20 Architecture Status: AI Guidance

- **Audience:** remote assistants, coding agents, and reviewers
- **Authority:** non-normative summary
- **Status:** implementation-status snapshot; expected to evolve
- **Canonical sources:** `docs/about.md`, `docs/architecture/overview.md`,
  `docs/roadmap.md`, `docs/tensor/operations.md`, canonical subsystem guides,
  the source tree, and tests

This file is for questions about project maturity, active design seams, and what should or should not be described as stable.

## File-level answer rule

- State what is implemented today.
- Separate implemented behavior from roadmap material.
- Treat the Tensor -> linalg dispatch -> CPU/BLAS/LAPACK path and its Async
  wrappers as one implemented vertical slice.
- Treat other subsystems more cautiously unless specific files are inspected.

## src/uni20/async/

### STATUS

- `src/uni20/async/` is the most mature subsystem.

### SAFE CLAIMS

- `Async<T>`, `ReadBuffer<T>`, `WriteBuffer<T>`, `EpochQueue`, and `EpochContext` are established core types.
- `DebugScheduler`, `TbbScheduler`, and `TbbNumaScheduler` exist and are active scheduler types.
- Async ordering and coroutine safety are primary architectural constraints.

### DO NOT CLAIM

- Do not describe the async surface as fully simplified or permanently settled.

### RELATED

- `../async/README.md`
- `../async/runtime_model.md`

## reverse-mode AD

### STATUS

- Reverse-mode AD is usable and conceptually mature.

### SAFE CLAIMS

- Reverse-mode AD uses `Var<T>` and `ReverseValue<T>`.
- Reverse-mode AD is integrated with the async runtime.
- Current reverse accumulation is deterministic.

### DO NOT CLAIM

- Do not describe the reverse-mode API as fully cleaned up.
- Do not describe Uni20 AD as tape-replay-based.

### DESIGN SEAMS

- `ReverseValue<T>` still has overlapping API surface.
- API simplification is still a legitimate topic.

### RELATED

- `reverse_mode_ad.md`

## CPU / BLAS path

### STATUS

- CPU / BLAS support is usable.

### SAFE CLAIMS

- CPU linear algebra exists.
- BLAS integration exists.
- BLAS vendor selection uses `UNI20_BLAS_VENDOR` and then `BLA_VENDOR`.

### DO NOT CLAIM

- Do not use CPU / BLAS support as evidence that GPU support is equally mature.

## tensor / TensorView

### STATUS

- Dense Tensor/view ownership, generated tensors, materialization, reshape, and
  initial backend-dispatched linalg operations are implemented.
- Async owner-retaining conjugation/reshape aliases, matrix products, and
  self-adjoint `eigh` are implemented.
- General slicing, CUDA/distributed storage, and symmetry-aware `BlockTensor`
  semantics remain active design work.

### SAFE CLAIMS

- Readable, mutable, and rank-constrained Tensor-view concepts are implemented.
- `Tensor<T, Rank, ...>` is the concrete composition-based owner and models
  both concepts; `BasicTensor<T, Extents, ...>` aliases a specialization with
  explicit mixed or static extents.
- `Tensor` resolves mutable or const mdspans directly. Read-only
  `ConjugatedTensorView` and `ConstTensorView` adaptors are implemented; general
  slice/external-storage adaptors are not.
- `copy` and `make_tensor` dispatch `copy_op`; lazy `conj(tensor)` does not
  allocate.
- `assign_transform` and `transform_inplace` dispatch callable-carrying
  elementwise operation values. The CPU reference backend supports arbitrary
  rank, variadic inputs, and non-strided accessor-respecting fallback.
- Their all-async overloads retain the callable and input readers in one
  coroutine. Overwrite can construct an empty output; update uses one writer
  for existing output values and rejects an input on the same epoch queue.
  Mutable async aliases are supported as fixed-shape outputs without descriptor
  rebinding.
- `async::conj` and async reshape views retain the parent owner and share its
  exact epoch queue.
- All-async elementwise transforms, `assign_product`, and `add_product` lower
  through their synchronous Tensor front ends. Preserving and consuming async
  `eigh` return independent async outputs.
- Implemented alias lifetime and assignment rules are documented; future slice
  descriptors still need operation-specific semantics.
- Default backend selectors are storage-derived candidate lists. Backend values
  may still carry explicit operation options, but operand location belongs to
  the mdspan accessor/handle rather than the selector.

### DO NOT CLAIM

- Do not generalize implemented conjugation/reshape lifetime rules to arbitrary
  future views.
- Do not claim that the async runtime proves arbitrary overlapping-storage
  alias safety.
- Do not generalize the implemented all-async dense matrix-product wrapper to
  CUDA, MPI, mixed sync/async operands, or unrelated output semantics.
- Do not claim that `std::type_list` exists or that `unique_tuple_cat_t` is a
  standard C++ metafunction.

### DESIGN SEAMS

- Independent-value write-proxy assignment follows the stored type's ordinary
  assignment expression. Async aliases use ADL `assign_through`; tensor element
  copying remains an explicit backend-dispatched operation.
- Shared async alias descriptors must remain consistent with their retained
  lifetime owner and epoch queue; retargeting requires a new alias.
- Future slices may initially share whole-owner ordering. Add subrange hazards
  only when conservative ordering becomes a demonstrated bottleneck.
- Candidate tensor roles are `Tensor` as owning value, `TensorRef` as
  proposed write-through non-owning lvalue, and resolved mdspan-like views as
  leaf-kernel arguments.
- Candidate backend state model: backend entries are values. Stateless tags are
  fine for CPU/BLAS/LAPACK, while CUDA/MPI entries may carry device, stream, or
  communicator state. A selector may deduplicate shared state internally, but
  that should not appear as a required extra parameter in every leaf-kernel CPO.

### RELATED

- `async_runtime.md`
- `tensor_dispatch_design.md`
- `../roadmap.md`
- `../tensor/dispatch_and_view_semantics_draft.md`
- `../async/tensor_lifetime_and_dispatch_draft.md`
- `../architecture/kernel_dispatch.md`

## expression layer

### STATUS

- There is no general Tensor expression/fusion layer. Explicit Tensor operations
  and Async operation wrappers are the current model.

### SAFE CLAIMS

- Explicit operations exist.
- Expression or fusion work is deferred until a measured operation sequence
  benefits from it.

### DO NOT CLAIM

- Do not claim that fusion or lazy expression lowering is already a mature subsystem.

## presentation / Python display

### STATUS

- C++ presentation formatting is used by trace, kernel-dispatch diagnostics,
  async task/DAG diagnostics, and runnable examples.
- Python tensor display and Jupyter rich display are roadmap material.

### SAFE CLAIMS

- Presentation is a semantic formatting layer, not a compute subsystem.
- Future Python tensor display should be preview-first.
- Future Jupyter display should be a renderer adapter over presentation data.

### DO NOT CLAIM

- Do not claim that HTML or Jupyter rendering is currently implemented.
- Do not claim that Python tensor `repr` is currently safe for large tensors.
- Do not claim that exhaustive mdspan formatting is appropriate as a default Python tensor display.

### RELATED

- `presentation_and_python.md`
- `../diagnostics/presentation.md`
- `../python/bindings.md`

## GPU / heterogeneous execution

### STATUS

- GPU / heterogeneous execution is partial.

### SAFE CLAIMS

- CUDA-related directories and types exist.
- Full CUDA execution is not complete.
- Full cuSOLVER execution is not complete.
- Future CUDA scheduler and device-resource design notes live in
  `cuda_scheduler_notes.md`; treat them as roadmap, not implemented behavior.

### DO NOT CLAIM

- Do not describe GPU support as complete unless inspected code proves it.

## build / CMake

### STATUS

- Uni20 uses modern CMake.

### SAFE CLAIMS

- Uni20 uses both system-package discovery and `FetchContent`.
- Some dependency handling is centralized in `cmake/Uni20Dependencies.cmake`.
- Not every dependency path necessarily uses the same helper.

### DO NOT CLAIM

- Do not infer dependency wiring without inspecting CMake.
- Do not infer imported targets without inspecting CMake.
- Do not infer transitive linkage without inspecting CMake.

## testing status

### STATUS

- Async, Tensor/linalg dispatch, and Krylov vertical slices have substantial
  focused tests. Coverage still varies by backend and optional scalar/provider
  configuration.

### SAFE CLAIMS

- `tests/async/`, `tests/linalg/`, `tests/tensor/`, and `tests/krylov/` encode
  important current behavior for their implemented slices.

### DO NOT CLAIM

- Do not assume equal test maturity across non-async subsystems.

## final caution

- Avoid overselling tensor maturity.
- Avoid overselling GPU maturity.
- Treat async ordering, coroutine safety, and lifetime rules as primary constraints.

## Related detailed docs

- `../roadmap.md`
- `../architecture/overview.md`
- `../async/README.md`
- `../async/runtime_model.md`
