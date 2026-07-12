# `src/uni20/tensor`

This directory contains the owning dense tensor and tensor-level concepts.
Tensor objects own shape, layout, storage, and execution policy, while lower
kernels operate on resolved mdspans.

## Contents

- `basic_tensor.hpp`: composition-based owning tensor implementation
  parameterized by element, extents, storage policy, layout, and accessor
  factory.
- `tensor.hpp`: rank-convenience alias for `BasicTensor`.
- `concepts.hpp`: readable, mutable, strided, and rank-constrained tensor-level
  concepts.
- `layout.hpp`: layout construction helpers.

## Notes

- Keep dense tensor behavior distinct from symmetry-aware block tensor behavior.
- `BasicTensor` models the tensor-level concepts directly and owns its storage.
- A tensor-level object exposes a storage-derived backend selector plus
  `mdspan()`.
  Element and accessor semantics determine whether the returned span is mutable;
  owning tensors overload `mdspan()` on constness.
- Tensor objects deliberately do not model Uni20's mdspan concepts. Leaf
  kernels receive the mdspans returned by those accessors.
- Tensor operations should lower to dense primitives only after storage, layout,
  backend, and any symmetry metadata have been resolved by the appropriate
  higher layer.
