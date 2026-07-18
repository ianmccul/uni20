# Uni20 AI Retrieval Glossary

- **Audience:** remote assistants, coding agents, and reviewers
- **Authority:** non-normative terminology index
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-18
- **Canonical sources:** linked subsystem documentation, source, and tests

This glossary is intentionally compact. Detailed invariants belong in the
subsystem guidance rather than being duplicated here.

## Async

### `Async<T>`
User-facing async value containing shared storage and an epoch timeline.
`Async<T>()` is unconstructed; `Async<T>(args...)` is constructed.

### `shared_storage<T>`
Reference-counted storage whose control-block validity and contained-object
construction state are separate.

### `EpochQueue`
One causal timeline with conceptual order
`writer_n -> readers_n -> writer_{n+1}`. Scheduler timing does not define legality.

### `ReadBuffer<T>`
Shared read capability for one epoch. Plain await is borrowed; `transfer()` is owning.

### `WriteBuffer<T>`
Exclusive mutable capability for one write epoch. It can inspect/mutate an existing
value or construct/replace/move an independent value.

### await path
Adaptor such as `maybe()`, `or_cancel()`, `storage()`, `take()`, or `transfer()`.
It changes await behavior without creating another capability or epoch.

### async alias
`Async<View>` descriptor that retains a parent owner and shares the parent's exact queue.

### `is_async_alias<T>`
Trait classifying structural async aliases versus independent async values.

### async rebind
Independent-value assignment that moves a handle to fresh storage and a fresh queue.
Aliases cannot rebind.

### write-through assignment
Alias assignment through ADL `assign_through`; descriptor, owner, and queue remain unchanged.

### cancellation
Terminal absence surfaced explicitly by `or_cancel()` as `task_cancelled`.
It is not unconstructed storage and not an exception.

## AD

### `Var<T>`
User-facing reverse-mode variable owning a forward `Async<T>` value and reverse
`ReverseValue<T>` channel.

### `ReverseValue<T>`
Async gradient accumulation channel with reverse ordering/finalization state.

### `backprop()`
Exposes/finalizes a gradient's async channel. It does not replay a tape.

### gradient finalization
Signal that no more contributions will be attached. Retained named intermediates
may require explicit `grad.finalize()` in the current API.

### gradient materialization
A gradient becomes concrete after seeding and reverse propagation; it is not eager.

### Wirtinger `dL/dz*`
Complex-gradient convention used by Uni20 for real scalar losses.

## Tensor

### `Tensor`
Concrete owning dense Tensor with compile-time rank and runtime extents by default.

### `BasicTensor`
Extents-first alias for a `Tensor` specialization with mixed/static extents.

### `TensorView`
Readable tensor-level concept exposing extents, `mdspan()`, and backend selection.
It is not a base class.

### `MutableTensorView`
Tensor-view refinement whose resolved mdspan permits writes.

### resolved mdspan
Short-lived leaf-kernel operand containing handle, mapping, extents, and accessor semantics.

### `GeneratedTensor`
Compact layout-neutral readable Tensor whose accessor generates values.

### semantic view
A view such as lazy conjugation whose accessor changes observed values; generally read-only.

### overwrite output
Destination whose old values do not participate. A resizable owner may change shape/storage.

### update output
One read/write operand whose old values participate. Async code enrolls one writer.

### aliasing
Overlapping storage. `Async` queue order does not automatically prove overlap safety
across distinct objects.

## Dispatch

### operation value
Dispatch key that may be an empty tag or carry immutable options/callable state.

### backend selector
Ordered backend candidate value used by Tensor-level dispatch. Storage determines
the default domain; do not duplicate operand placement as unrelated selector state.

### type probe
Compile-time capability classification for exact argument types.

### clean decline
Runtime refusal before mutation, submission, commitment, or externally visible side effect.

### terminal backend failure
Failure after work starts or a provider reports an operation error. It must not trigger fallback.

### dynamic dispatch boundary
Runtime-erased entry point for Python/plugin-like callers that must remain callable
when static operation support is absent.

## Presentation and CUDA

### semantic glyph
Renderer-independent status/layout token mapped by terminal/plain/ASCII adapters.

### mdspan preview
Bounded, deterministic display that marks elision; distinct from exhaustive formatting.

### CUDA completion
Provisional representation of device-work completion. Current implementation details
must not be treated as a settled public contract.

### CUDA stream/resource primitive
Experimental low-level mechanism for device execution and resource ownership.
Inspect current source and maintainer decisions before relying on lifecycle semantics.

### CUDA buffer access primitive
Experimental mechanism for expressing device-buffer dependencies. It must remain
consistent with Uni20 async causality, but its final design is unresolved.

## Safety terms

### coroutine safety rule
Uni20 async coroutine lambdas must be captureless and `static`; pass state as parameters.

### symmetry metadata
Quantum-number, local/block-space, and leg-orientation information that is part of
the mathematical object and must not be silently erased.
