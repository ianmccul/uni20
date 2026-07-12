# `src/uni20/tensor`

This directory contains the owning dense tensor and tensor-level concepts.
Tensor objects own shape, layout, storage, and execution policy, while lower
kernels operate on resolved mdspans.

## Contents

- `basic_tensor.hpp`: configurable composition-based owner parameterized by an
  mdspan extents type.
- `tensor.hpp`: the general-purpose runtime-extents `Tensor` alias and the
  column-major-by-default host `DenseMatrix` alias.
- `conjugate.hpp`: read-only tensor view backed by the lazy conjugating mdspan
  accessor.
- `copy.hpp`: backend-dispatched copies and `make_tensor(...)` materialization.
- `concepts.hpp`: readable, mutable, strided, and rank-constrained tensor-level
  concepts.
- `output.hpp`: fixed-output validation and resizable-output shape preparation.
- `layout.hpp`: layout construction helpers.

## Notes

- Keep dense tensor behavior distinct from symmetry-aware block tensor behavior.
- `BasicTensor<Element, Extents, ...>` is the configurable owner for mixed or
  static mdspan extents. `Tensor<Element, Rank, ...>` is the ordinary owning
  tensor and has runtime extents on every axis.
- Both forms have compile-time rank because they are mdspan-based. A future
  runtime-rank tensor requires a separate descriptor and type rather than a
  second meaning for `Tensor`.
- Both owning forms model the tensor-level concepts directly.
- Use `make_tensor(view)` for inferred materialization rather than a
  view-taking `Tensor` constructor. Materialization is an operation and must
  remain eligible for backend dispatch, including future BLAS matrix-copy
  extensions.
- A tensor-level object exposes a storage-derived backend selector plus
  synchronous extents metadata and `mdspan()`.
  Element and accessor semantics determine whether the returned span is mutable;
  owning tensors overload `mdspan()` on constness.
- Tensor objects deliberately do not model Uni20's mdspan concepts. Leaf
  kernels receive the mdspans returned by those accessors.
- `DenseMatrix<T>` is `Tensor<T, 2, VectorStorage, ColumnMajor>`; use
  `DenseMatrix<T, RowMajor>` when row-major ownership is preferred. Matrix-level
  linalg front ends accept either form and resolve mdspans internally.
- Tensor operations should lower to dense primitives only after storage, layout,
  backend, and any symmetry metadata have been resolved by the appropriate
  higher layer.
