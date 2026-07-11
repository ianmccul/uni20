# `src/uni20/tensor`

This directory contains the owning dense tensor, tensor-view concepts, and a
small concrete non-owning adaptor. Tensor objects own shape/layout/storage
policy, while lower kernels operate on resolved mdspans.

## Contents

- `basic_tensor.hpp`: composition-based owning tensor implementation
  parameterized by element, extents, storage policy, layout, and accessor
  factory.
- `tensor.hpp`: rank-convenience alias for `BasicTensor`.
- `tensor_view.hpp`: readable, mutable, and rank-constrained Tensor-view
  concepts plus the non-owning `BasicTensorView` adaptor.
- `layout.hpp`: layout construction helpers.

## Notes

- Keep dense tensor behavior distinct from symmetry-aware block tensor behavior.
- `BasicTensor` models both tensor-view concepts but does not inherit from
  `BasicTensorView`.
- A tensor view exposes a storage-derived backend selector plus `mdspan()`.
  Element and accessor semantics determine whether the returned span is mutable;
  owning tensors overload `mdspan()` on constness.
- Tensor-view objects deliberately do not model Uni20's mdspan concepts. Leaf
  kernels receive the mdspans returned by those accessors.
- Tensor operations should lower to dense primitives only after storage, layout,
  backend, and any symmetry metadata have been resolved by the appropriate
  higher layer.
