# `src/uni20/tensor`

This directory contains the basic dense tensor and tensor-view types. Tensor
code owns shape/layout/storage policy at the object level, while lower kernel
layers operate on resolved views.

## Contents

- `basic_tensor.hpp`: owning tensor implementation parameterized by element,
  extents, storage policy, layout, and accessor factory.
- `tensor.hpp`: rank-convenience alias for `BasicTensor`.
- `tensor_view.hpp`: non-owning tensor view type.
- `layout.hpp`: layout construction helpers.

## Notes

- Keep dense tensor behavior distinct from symmetry-aware block tensor behavior.
- Tensor operations should lower to dense primitives only after storage, layout,
  backend, and any symmetry metadata have been resolved by the appropriate
  higher layer.
