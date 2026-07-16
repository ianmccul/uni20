# src/uni20/tensor

This directory contains the owning dense tensor and tensor-level concepts.
Tensor objects own shape, layout, storage, and execution policy, while lower
kernels operate on resolved mdspans.

## Contents

- `basic_tensor.hpp`: concrete composition-based `Tensor` owner and the
  extents-first `BasicTensor` alias.
- `tensor.hpp`: named
  `ColumnMajorTensor`, `RowMajorTensor`, `StridedTensor`, and `ScalarTensor`
  aliases, and the host `DenseMatrix` alias.
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
- `reductions.hpp`: storage-preserving full and partial sums, host-result sums,
  inner products, and stable Euclidean norms. All-async sum overloads live in
  [`linalg/async/`](../linalg/async/).
- `concepts.hpp`: readable, mutable, owning, strided, and rank-constrained
  tensor-level concepts.
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
  synchronous extents metadata and `mdspan()`.
  Element and accessor semantics determine whether the returned span is mutable;
  owning tensors overload `mdspan()` on constness.
- Tensor objects deliberately do not model Uni20's mdspan concepts. Leaf
  kernels receive the mdspans returned by those accessors.
- Generated tensors own compact generator state rather than an element buffer.
  They model readable `TensorView` but not `StridedTensorView`; their synthetic
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
