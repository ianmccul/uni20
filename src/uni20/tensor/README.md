# src/uni20/tensor

This directory contains the owning dense tensor and tensor-level concepts.
Tensor objects own shape, layout, storage, and execution policy, while lower
kernels operate on resolved mdspans.

## Contents

- `basic_tensor.hpp`: concrete composition-based `Tensor` owner and the
  extents-first `BasicTensor` alias.
- `access.hpp`: generic RAII mdspan and tensor leases plus host acquisition.
- `cuda_access.hpp`: CUDA descriptor resolution through blocking or
  stream-ordered `CudaBuffer` access.
- `tensor.hpp`: named
  `ColumnMajorTensor`, `RowMajorTensor`, `StridedTensor`, and `ScalarTensor`
  aliases, the host `DenseMatrix` alias, and CUDA-enabled `CudaTensor` and
  `CudaMatrix` aliases.
- `conjugate.hpp`: read-only tensor view backed by the lazy conjugating mdspan
  accessor.
- `generated.hpp`: compact generated tensors and the `full`, `zeros`, `ones`,
  and generalized `eye` factories.
- `async.hpp`: async tensor aliases that retain parent storage and share its
  epoch queue.
- `copy_into.hpp`: backend-dispatched copies into existing mdspan or tensor outputs.
- `copy.hpp`: inferred `make_tensor(...)` materialization and owning reshape support.
- `conjugate_inplace.hpp`: backend-dispatched eager conjugation of mutable
  tensor storage.
- `transform.hpp`: backend-dispatched variadic elementwise overwrite and update
  operations for mdspan and Tensor operands. Their all-async Tensor overloads
  live in [`linalg/async/`](../linalg/async/).
- Fixed-output GEMM and matrix-product overwrite/update operations live in
  [`linalg/`](../linalg/); their all-async Tensor overloads live in
  [`linalg/async/`](../linalg/async/).
- `reductions.hpp`: storage-preserving full and partial sums, host-result sums,
  inner products, and stable Euclidean norms. All-async sum overloads live in
  [`linalg/async/`](../linalg/async/).
- `concepts.hpp`: tensor views and their immediate, mutable, owning, strided,
  and rank-constrained refinements.
- `output.hpp`: fixed-output validation and resizable-output shape preparation.
- `reshape.hpp`: explicit no-copy, in-place, and owning reshape operations.
- `shape.hpp`: checked runtime-extents construction shared by tensor factories.
- `layout.hpp`: layout construction helpers.

## Notes

- Keep dense tensor behavior distinct from symmetry-aware block tensor behavior.
- `Tensor<Element, Rank, ...>` is the concrete owning class, has runtime
  extents on every axis by default, and defaults to `ColumnMajor`.
  `BasicTensor<Element, Extents, ...>` is its extents-first alias for mixed or
  static mdspan extents; it does not introduce another implementation type.
  Use the named layout aliases when the physical order is part of the local
  contract; use `StridedTensor` only when an explicit stride mapping is needed.
- Both forms have compile-time rank because they are mdspan-based. A future
  runtime-rank tensor requires a separate descriptor and type rather than a
  second meaning for `Tensor`.
- Every specialization models the tensor-level concepts directly.
- Rank-zero `ScalarTensor` owners default-construct their one logical element
  and use `scalar[]` for ordinary host-accessible indexing.
- `OwningTensor` is an explicit opt-in classification for types whose move
  operation transfers the storage and lifetime exposed through `mdspan()`.
  Non-owning descriptors such as `ConstTensorView` and
  `ConjugatedTensorView` deliberately do not model it.
- Ownership alone does not make an expression consumable. Value operations may
  transfer storage only from a mutable owning rvalue. Passing such an rvalue
  grants permission to reuse its allocation but does not guarantee reuse when
  the layout, accessor, storage policy, or backend is incompatible.
- Moving an owning tensor follows ordinary C++ lifetime rules: existing
  non-owning tensor views and mdspans into the transferred storage must not be
  used afterward. Uni20 does not track synchronous views to prevent this.
- `Tensor::release_storage()` transfers the concrete policy-selected container
  out of an owning rvalue. `Tensor::adopt_storage(...)` installs
  a mapping over a transferred container without reallocating it. Adoption
  requires `storage.size() >= mapping.required_span_size()`; padding and an
  unused storage tail are preserved intentionally.
- `rebind_layout_type<Layout>` preserves a tensor's element type, extents,
  storage policy, and accessor factory while changing its mapping policy. This
  is the type-level counterpart to releasing storage and adopting it under a
  compatible mapping.
- `Tensor(view)` and CTAD through `BasicTensor(view)` eagerly materialize a
  readable tensor view through backend-dispatched copy and deduce runtime
  extents. An explicitly specialized `BasicTensor<Element, Extents, ...>`
  instead preserves those requested static or mixed extents. CTAD preserves a
  canonical physical source layout and otherwise selects column-major storage.
  A named layout alias is deducible only when that inferred layout matches it.
- `make_tensor(view)` provides the same inferred materialization as an
  operation, while `make_tensor<Layout>(view)` forces the physical result
  layout at compile time. The selector-taking form also accepts a bare mdspan.
  Materialization is an operation and must remain eligible for backend
  dispatch, including future BLAS matrix-copy extensions.
- A tensor-level object exposes a storage-derived backend selector plus
  synchronous extents metadata. Every `TensorView` exposes metadata through
  `mdspec_of()`. An `ImmediateTensorView` also exposes `mdspan()` because its
  data handle needs no acquisition. A descriptor-backed `TensorView` instead
  exposes `mdspec()` until an acquisition operation resolves its data handle.
  Element and accessor semantics determine whether either representation is
  writable; owning tensors overload the available observer on constness.
- `Tensor::mdspec()` prefers the corresponding immediate read or write
  handle and returns an ordinary mdspan in that case. Descriptor metadata is
  selected only when no immediate handle is available. Readable and writable
  capabilities are independent.
- `TensorView` covers both immediately accessible and descriptor-backed
  tensors. Domain-explicit operations use explicit completion suffixes:
  `acquire_host_read_access_sync` returns an RAII TensorView lease directly,
  while `acquire_cuda_write_access_async` returns an awaitable that yields one.
  The lease's mdspan accessor is valid in the named execution domain. There are
  no unsuffixed acquisition defaults.
  Immediate host lvalue views use borrowed no-op leases and do not need a
  public `storage()` observer. These immediate leases store only a pointer to
  the source view and forward its mdspan and metadata. Acquisition never
  transfers data between host and CUDA domains; `copy` is the explicit bridge.
  Lease and access-state release is idempotent; moving either transfers the
  active lifetime and leaves the source inactive. See
  [Mdspec](../../../docs/tensor/mdspec.md).
- `CudaTensor<T, Rank>` uses the installed CUDA runtime's default device
  when constructed from extents alone. Passing an explicit
  `cuda::DeviceResources` selects another enrolled device or an isolated test
  resource set. Its storage is a move-only `CudaBuffer<T>`.
  `mdspec()` exposes a `CudaBufferView` descriptor, the tensor mapping,
  and the actual pointer accessor without exposing a pointer. Tensor-level
  acquisition resolves this to a lease with a `T*` or `T const*` mdspan.
  An owning rvalue read moves its buffer into an owning access state; non-owning
  deferred views remain lvalue-only. Stream-ordered access installs predecessor
  waits and publishes completion when the lease ends. `CudaTensor` deliberately
  does not expose `mdspan()` and therefore models `TensorView`, not
  `ImmediateTensorView`. Fixed CUDA GEMM dispatch receives normalized mdspec
  descriptors; the cuBLAS backend validates them and acquires the referenced
  buffers.
- Tensor objects deliberately do not model Uni20's mdspan concepts.
  `mdspec_of(tensor)` selects `.mdspec()` when available and otherwise returns
  `.mdspan()` unchanged. `ImmediateTensorView` explicitly requires the latter;
  `TensorView` accepts either representation. The concepts explicitly
  reject objects that directly model the corresponding mdspan representation.
  Fixed-output operation frontends select
  a backend list from tensor policy, then dispatch normalized
  `MdspecLike` operands. Blocking backends acquire mdspan leases before
  entering an existing mdspan implementation; descriptor-native backends may
  interpret unresolved metadata directly. Replaceable-output operation tags
  retain tensor or shared-storage outputs until the selected backend prepares
  them.
- Generated tensors own compact generator state rather than an element buffer.
  They model readable `ImmediateTensorView` but not `StridedImmediateTensorView`; their synthetic
  `GeneratedLayout` is not a physical storage order. `GeneratedStorage` is
  backend-neutral when an operation also has concrete storage operands.
- `reshape_view` is the strict no-copy operation, `reshape_inplace` changes an
  owning tensor descriptor without reallocating, and plain `reshape` returns
  an owner with ordinary copy-or-move value semantics. See [Tensor Creation and
  Reshape](../../../docs/tensor/creation_and_reshape.md) for the complete
  contracts.
- Automatic `reshape_view` preserves a static `ColumnMajor` or `RowMajor`
  source layout. `reshape_view_left` and `reshape_view_right` explicitly select
  the interpretation of a compatible `layout_stride` mapping; plain reshape
  never guesses that order.
- `DenseMatrix<T>` is `Tensor<T, 2, VectorStorage, ColumnMajor>`; use
  `DenseMatrix<T, RowMajor>` when row-major ownership is preferred. Matrix-level
  linalg front ends accept either form and resolve mdspans internally.
- Tensor operations should lower to dense primitives only after storage, layout,
  backend, and any symmetry metadata have been resolved by the appropriate
  higher layer.
- See [Tensor Operations](../../../docs/tensor/operations.md) for the canonical operation vocabulary,
  ownership/output contracts, and current Async support matrix.
- `async::conj(Async<Tensor>)` returns an `Async<ConjugatedTensorView>` rather
  than materializing values. The alias remains a tensor-level object whose
  `mdspan()` resolves the conjugating accessor after the shared epoch is ready.
- `async::reshape_view(Async<Tensor>, ...)` returns an owner-retaining
  structural alias on the parent's exact epoch queue. It may be formed before
  the parent value is constructed and resolves its mdspan only after the shared
  epoch is readable.

## Related Documentation

- [Source tree map](../)
- [Tensor documentation index](../../../docs/tensor/)
- [Async storage and alias lifetime](../../../docs/async/storage.md)
- [Kernel dispatch](../../../docs/architecture/kernel_dispatch.md)
